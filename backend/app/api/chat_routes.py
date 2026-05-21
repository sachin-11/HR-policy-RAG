"""Chat API routes."""

from __future__ import annotations

import json as _json
import logging
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.graph import HRPolicyAgent
from app.agent.nodes import (
    build_initial_state,
    classify_intent_node,
    execute_tools_node,
    retrieve_context_node,
)
from app.agent.prompts import NO_CONTEXT_ANSWER, build_rag_prompt
from app.agent.tools import ToolOrchestrator
from app.security.auth import AuthClaims, get_current_user
from app.security.permissions import UserContext, authorize_chat_request, PermissionDenied
from app.security.pii import contains_prompt_injection, mask_pii
from app.agent.llm import LLMClient, build_llm_client
from app.config import get_settings
from app.rag.embeddings import build_embedding_provider
from app.rag.retriever import RagRetriever
from app.rag.vector_store import build_vector_store, infer_local_store_embedding_dimension
from app.observability.logging import log_event
from app.sessions.store import get_session_store


router = APIRouter(tags=["chat"])
chat_logger = logging.getLogger("app.api.chat")


class ApprovedToolAction(BaseModel):
    """One sensitive tool step the user acknowledged in the Human Approval UI."""

    tool_name: str = Field(min_length=1, max_length=64)
    action: str = Field(min_length=1, max_length=64)


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(min_length=1, max_length=4000)
    user_id: str | None = None
    session_id: str | None = Field(
        default=None,
        description="Conversation session ID. Omit on first turn; reuse returned session_id for follow-ups.",
    )
    country: str | None = None
    employee_type: str | None = None
    access_level: str | None = "employee"
    department: str | None = None
    policy_type: str | None = None
    top_k: int = Field(default=8, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    approved_tool_actions: list[ApprovedToolAction] = Field(
        default_factory=list,
        description="User-confirmed sensitive tool actions from a prior response (e.g. email send).",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("message must not be blank")
        return clean_value


class SourceCitation(BaseModel):
    """Citation returned with a chat answer."""

    chunk_id: str
    title: str | None = None
    source: str | None = None
    section_title: str | None = None
    score: float
    policy_type: str | None = None
    country: str | None = None
    employee_type: str | None = None


class ChatResponse(BaseModel):
    """Chat response payload."""

    answer: str
    sources: list[SourceCitation]
    used_context: bool
    needs_human_confirmation: bool = False
    intent: str | None = None
    tool_results: list[dict[str, object]] = Field(default_factory=list)
    approval_required_actions: list[dict[str, object]] = Field(default_factory=list)
    session_id: str | None = Field(
        default=None,
        description="Session ID to pass in the next request for multi-turn conversation.",
    )


@lru_cache
def get_default_retriever() -> RagRetriever:
    """Build the default retriever from settings."""

    settings = get_settings()
    embedding_provider_name = "openai" if settings.openai_api_key else "mock"
    vector_store_dir = Path(settings.vector_store_dir)
    mock_dimension: int | None = None
    if embedding_provider_name == "mock" and settings.vector_store_provider.lower().strip() in {
        "local",
        "local_json",
        "json",
    }:
        mock_dimension = infer_local_store_embedding_dimension(vector_store_dir)
        if mock_dimension is not None and mock_dimension != 64:
            chat_logger.warning(
                "Using mock embeddings with dimension %s from local vector store; "
                "set OPENAI_API_KEY for embeddings that match OpenAI-indexed vectors.",
                mock_dimension,
            )
    embedding_provider = build_embedding_provider(
        embedding_provider_name,
        openai_api_key=settings.openai_api_key,
        openai_embedding_model=settings.openai_embedding_model,
        mock_dimension=mock_dimension,
    )
    vector_store = build_vector_store(
        settings.vector_store_provider,
        directory=vector_store_dir,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_index_name=settings.pinecone_index_name,
        pinecone_namespace=settings.pinecone_namespace,
        dimension=embedding_provider.dimension,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )
    return RagRetriever(embedding_provider=embedding_provider, vector_store=vector_store)


@lru_cache
def get_default_llm_client() -> LLMClient:
    """Build the default LLM client from settings."""

    settings = get_settings()
    return build_llm_client(
        settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        openai_chat_model=settings.openai_chat_model,
    )


def get_retriever() -> RagRetriever:
    """FastAPI dependency for the retriever."""

    return get_default_retriever()


def get_llm_client() -> LLMClient:
    """FastAPI dependency for the LLM client."""

    return get_default_llm_client()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: AuthClaims = Depends(get_current_user),
    retriever: RagRetriever = Depends(get_retriever),
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    """Answer an HR policy question using retrieved context and citations."""

    if contains_prompt_injection(request.message):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt injection detected in request text.",
        )

    safe_message = mask_pii(request.message)
    # Do not treat the placeholder "anonymous" user id as an account-scoped target; that
    # incorrectly triggers PermissionDenied for unauthenticated /chat calls.
    target_user_id = request.user_id
    if target_user_id is None and current_user.role != "anonymous":
        target_user_id = current_user.user_id
    user_context = UserContext(
        user_id=current_user.user_id,
        role=current_user.role,
        country=current_user.country,
        employee_type=current_user.employee_type,
        department=current_user.department,
    )

    try:
        safe_filters = authorize_chat_request(
            user_context=user_context,
            target_user_id=target_user_id,
            requested_filters={
                "country": request.country,
                "employee_type": request.employee_type,
                "department": request.department,
                "access_level": request.access_level,
                "policy_type": request.policy_type,
            },
        )
    except PermissionDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    # Load or create conversation session for multi-turn history.
    session_store = get_session_store()
    session = session_store.get_or_create(request.session_id, user_id=target_user_id)

    agent = HRPolicyAgent(retriever=retriever, llm_client=llm_client)
    started = time.perf_counter()
    approved_payload = [item.model_dump() for item in request.approved_tool_actions]
    state = agent.run(
        user_message=safe_message,
        user_id=target_user_id,
        filters=safe_filters,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        approved_tool_actions=approved_payload,
        conversation_history=session.to_prompt_block(),
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    log_event(
        chat_logger,
        event="chat.agent_run",
        duration_ms=duration_ms,
        intent=state.get("intent"),
        used_context=state.get("used_context", False),
        needs_human_confirmation=state.get("needs_human_confirmation", False),
        sources_count=len(state.get("sources", [])),
        tool_results_count=len(state.get("tool_results", [])),
        session_id=session.session_id,
    )

    # Persist this turn to the session for future follow-ups.
    final_answer = state["final_answer"]
    session_store.add_turn(session.session_id, safe_message, final_answer)

    return ChatResponse(
        answer=final_answer,
        sources=[SourceCitation.model_validate(source) for source in state.get("sources", [])],
        used_context=state.get("used_context", False),
        needs_human_confirmation=state.get("needs_human_confirmation", False),
        intent=state.get("intent"),
        tool_results=state.get("tool_results", []),
        approval_required_actions=state.get("approval_required_actions", []),
        session_id=session.session_id,
    )


# ── streaming helpers ──────────────────────────────────────────────────────────

_STREAM_REDUCER_FIELDS: frozenset[str] = frozenset(
    {"tool_results", "approval_required_actions", "sources", "errors"}
)


def _merge_for_stream(state: dict, delta: dict) -> dict:
    merged = dict(state)
    for key, value in delta.items():
        if key in _STREAM_REDUCER_FIELDS:
            merged[key] = list(merged.get(key) or []) + list(value or [])
        else:
            merged[key] = value
    return merged


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: AuthClaims = Depends(get_current_user),
    retriever: RagRetriever = Depends(get_retriever),
    llm_client: LLMClient = Depends(get_llm_client),
) -> StreamingResponse:
    """Streaming chat — returns Server-Sent Events (text/event-stream).

    Each event is ``data: <json>\\n\\n``. Event types:
    - ``{"type": "token", "text": "..."}``   — partial answer chunk
    - ``{"type": "done",  "sources": [...], "session_id": "..."}``  — final metadata
    - ``{"type": "error", "message": "..."}`` — on failure
    """

    if contains_prompt_injection(request.message):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt injection detected in request text.",
        )

    safe_message = mask_pii(request.message)
    target_user_id = request.user_id
    if target_user_id is None and current_user.role != "anonymous":
        target_user_id = current_user.user_id
    user_context = UserContext(
        user_id=current_user.user_id,
        role=current_user.role,
        country=current_user.country,
        employee_type=current_user.employee_type,
        department=current_user.department,
    )

    try:
        safe_filters = authorize_chat_request(
            user_context=user_context,
            target_user_id=target_user_id,
            requested_filters={
                "country": request.country,
                "employee_type": request.employee_type,
                "department": request.department,
                "access_level": request.access_level,
                "policy_type": request.policy_type,
            },
        )
    except PermissionDenied as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    store = get_session_store()
    session = store.get_or_create(request.session_id, user_id=target_user_id)
    history_block = session.to_prompt_block()
    approved_payload = [item.model_dump() for item in request.approved_tool_actions]

    async def event_stream():
        try:
            state = build_initial_state(
                user_message=safe_message,
                user_id=target_user_id,
                filters=safe_filters,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                approved_tool_actions=approved_payload,
                conversation_history=history_block,
            )

            state = _merge_for_stream(state, classify_intent_node(state))

            if state.get("intent") == "action_request":
                orchestrator = ToolOrchestrator(llm_client=llm_client)
                state = _merge_for_stream(state, execute_tools_node(state, orchestrator))

            state = _merge_for_stream(state, retrieve_context_node(state, retriever))

            if not state.get("used_context"):
                pending_approvals = state.get("approval_required_actions") or []
                if pending_approvals:
                    no_ctx_text = (
                        "I've prepared a draft email for you. "
                        "Please review it in the approval panel and confirm to send."
                    )
                else:
                    no_ctx_text = NO_CONTEXT_ANSWER
                yield f"data: {_json.dumps({'type': 'token', 'text': no_ctx_text})}\n\n"
                store.add_turn(session.session_id, safe_message, no_ctx_text)
                yield f"data: {_json.dumps({'type': 'done', 'sources': [], 'intent': state.get('intent'), 'used_context': False, 'needs_human_confirmation': True, 'approval_required_actions': pending_approvals, 'session_id': session.session_id})}\n\n"
                return

            prompt = build_rag_prompt(
                user_message=state["user_message"],
                context=state.get("context", ""),
                conversation_history=state.get("conversation_history", ""),
            )

            full_answer = ""
            async for token in llm_client.stream_generate(prompt):
                full_answer += token
                yield f"data: {_json.dumps({'type': 'token', 'text': token})}\n\n"

            full_answer = full_answer.strip() or NO_CONTEXT_ANSWER
            store.add_turn(session.session_id, safe_message, full_answer)

            sources = [
                SourceCitation.model_validate(s).model_dump(mode="json")
                for s in state.get("sources", [])
            ]
            pending_approvals = state.get("approval_required_actions") or []
            yield f"data: {_json.dumps({'type': 'done', 'sources': sources, 'intent': state.get('intent'), 'used_context': True, 'needs_human_confirmation': bool(pending_approvals), 'approval_required_actions': pending_approvals, 'session_id': session.session_id})}\n\n"

        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
