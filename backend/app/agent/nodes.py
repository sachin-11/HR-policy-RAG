"""Workflow nodes for the HR policy agent.

Each node receives the full AgentState and returns only the fields it changed
(a partial delta).  LangGraph merges deltas into the shared state; list fields
with `operator.add` reducers (errors, tool_results, etc.) are appended rather
than replaced.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.llm import ExtractiveLLMClient, LLMClient
from app.agent.prompts import NO_CONTEXT_ANSWER, build_rag_prompt
from app.agent.state import AgentState, Intent
from app.agent.tools import ToolOrchestrator, tool_results_to_answer_block
from app.observability.logging import log_event
from app.rag.retriever import RagRetriever, build_source_citations, dedupe_source_citations_for_display, format_retrieved_context


logger = logging.getLogger("app.agent.nodes")

MAX_RETRIEVAL_RETRIES = 2  # how many times to retry retrieval if grading fails

ACTION_KEYWORDS = {"draft", "email", "mail", "apply", "create", "submit", "ticket", "send"}
POLICY_KEYWORDS = {
    "policy",
    "leave",
    "sick",
    "maternity",
    "reimbursement",
    "insurance",
    "benefit",
    "wfh",
    "work from home",
    "contractor",
}


# ── nodes ──────────────────────────────────────────────────────────────────────


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    """Classify user intent. Returns only the `intent` field."""

    message = state.get("user_message", "").lower()
    intent: Intent = "unknown"
    if any(keyword in message for keyword in ACTION_KEYWORDS):
        intent = "action_request"
    elif any(keyword in message for keyword in POLICY_KEYWORDS):
        intent = "policy_qa"
    elif message.strip():
        intent = "general_hr"

    return {"intent": intent}


def rewrite_query_node(state: AgentState, llm_client: LLMClient) -> dict[str, Any]:
    """Rewrite the user message into a concise vector-search query.

    On retries, generates an alternative query using synonyms so the vector
    store has a better chance of finding relevant chunks.
    """
    message = state["user_message"]
    retry_count = state.get("retry_count", 0)
    prev_query = state.get("search_query") or message

    # ExtractiveLLMClient has no real text-generation capability — skip rewriting.
    if isinstance(llm_client, ExtractiveLLMClient):
        return {"search_query": message}

    if retry_count == 0:
        prompt = (
            "Convert this HR question into a short keyword search query for a vector database.\n"
            "Keep only key HR terms. Remove conversational words like 'what', 'how', 'can I'.\n"
            f"Question: {message}\n"
            "Search query (one line only):"
        )
    else:
        prompt = (
            f"The search query '{prev_query}' returned no useful HR policy documents.\n"
            f"Original question: {message}\n"
            "Write a different search query using synonyms or rephrased terms (one line only):"
        )

    try:
        rewritten = llm_client.generate_freeform(prompt).strip()
        # Strip quotes, trailing labels, extra newlines
        rewritten = rewritten.split("\n")[0].strip().strip("\"'").strip()
        if not rewritten or len(rewritten) > 300:
            rewritten = message
    except Exception:
        rewritten = message

    log_event(
        logger,
        event="rag.query_rewrite",
        original=message,
        rewritten=rewritten,
        retry_count=retry_count,
    )
    return {"search_query": rewritten}


def retrieve_context_node(state: AgentState, retriever: RagRetriever) -> dict[str, Any]:
    """Retrieve RAG context using the (possibly rewritten) search query."""

    # Use rewritten query when available; fall back to original user message.
    search_query = state.get("search_query") or state["user_message"]

    try:
        retrieval = retriever.retrieve(
            search_query,
            filters=state.get("filters") or {},
            top_k=state.get("top_k"),
            score_threshold=state.get("score_threshold"),
        )
    except Exception as exc:
        return _error_delta(f"retrieval_error: {exc}")

    context = format_retrieved_context(retrieval.chunks)
    chunk_ids = [chunk.id for chunk in retrieval.chunks]
    log_event(
        logger,
        event="rag.retrieval",
        chunk_ids=chunk_ids,
        chunk_count=len(chunk_ids),
        search_query=search_query,
        metadata_filter=retrieval.metadata_filter,
    )

    return {
        "retrieved_chunks": retrieval.chunks,
        "retrieval_metadata_filter": retrieval.metadata_filter,
        "context": context,
        "used_context": retrieval.has_context,
    }


def grade_documents_node(state: AgentState, llm_client: LLMClient) -> dict[str, Any]:
    """Grade retrieved chunks for relevance to the user question.

    Filters retrieved_chunks to only relevant ones and rebuilds the context
    string.  If no chunk passes grading, increments retry_count so the router
    can loop back to rewrite_query_node for another attempt.
    """
    chunks = state.get("retrieved_chunks") or []
    question = state["user_message"]
    retry_count = state.get("retry_count", 0)

    if not chunks:
        log_event(logger, event="rag.grading", total=0, relevant=0, retry_count=retry_count)
        return {"used_context": False, "retry_count": retry_count + 1}

    if isinstance(llm_client, ExtractiveLLMClient):
        relevant = _keyword_grade(question, chunks)
    else:
        relevant = _llm_grade_relevance(question, chunks, llm_client)

    log_event(
        logger,
        event="rag.grading",
        total=len(chunks),
        relevant=len(relevant),
        retry_count=retry_count,
    )

    if relevant:
        context = format_retrieved_context(relevant)
        citations = dedupe_source_citations_for_display(build_source_citations(relevant))
        return {
            "retrieved_chunks": relevant,
            "context": context,
            "sources": citations,
            "used_context": True,
        }

    # No relevant chunks — signal retry by incrementing counter
    return {
        "retrieved_chunks": [],
        "context": "",
        "sources": [],
        "used_context": False,
        "retry_count": retry_count + 1,
    }


def execute_tools_node(state: AgentState, tool_orchestrator: ToolOrchestrator) -> dict[str, Any]:
    """Run safe tools and collect approval-required actions."""

    try:
        raw_approved = state.get("approved_tool_actions") or []
        approved_pairs = {
            (str(a["tool_name"]), str(a["action"]))
            for a in raw_approved
            if a.get("tool_name") and a.get("action")
        }
        tool_results, approvals = tool_orchestrator.run(
            user_id=state.get("user_id"),
            user_message=state["user_message"],
            intent=state.get("intent"),
            approved_pairs=approved_pairs,
        )
    except Exception as exc:
        return _error_delta(f"tool_error: {exc}")

    updated_filters = dict(state.get("filters") or {})
    profile_result = next(
        (r for r in tool_results if r.tool_name == "employee_profile"), None
    )
    if profile_result:
        profile = profile_result.output
        updated_filters.setdefault("country", profile.get("country"))
        updated_filters.setdefault("employee_type", profile.get("employee_type"))
        updated_filters.setdefault("access_level", profile.get("access_level"))

    return {
        "filters": {k: v for k, v in updated_filters.items() if v is not None},
        # These fields use operator.add reducers — node returns only new items
        "tool_results": [r.model_dump(mode="json") for r in tool_results],
        "approval_required_actions": [a.model_dump(mode="json") for a in approvals],
    }


def generate_answer_node(state: AgentState, llm_client: LLMClient) -> dict[str, Any]:
    """Generate the final answer using retrieved context."""

    if not state.get("used_context"):
        return {
            "final_answer": NO_CONTEXT_ANSWER,
            "needs_human_confirmation": True,
        }

    prompt = build_rag_prompt(
        user_message=state["user_message"],
        context=state.get("context", ""),
        conversation_history=state.get("conversation_history", ""),
    )
    try:
        answer = llm_client.generate(prompt).strip() or NO_CONTEXT_ANSWER
    except Exception as exc:
        return _error_delta(f"generation_error: {exc}")

    model_name = getattr(llm_client, "model", None)
    log_event(
        logger,
        event="llm.completion",
        provider=type(llm_client).__name__,
        model=model_name,
        prompt_chars=len(prompt),
        completion_chars=len(answer),
        prompt_tokens_estimated=max(len(prompt) // 4, 0),
        completion_tokens_estimated=max(len(answer) // 4, 0),
        total_tokens_estimated_placeholder=None,
        cost_usd_estimated=None,
    )

    return {
        "prompt": prompt,
        "final_answer": answer,
        "needs_human_confirmation": False,
    }


def validate_response_node(state: AgentState) -> dict[str, Any]:
    """Validate response safety and completeness at the workflow boundary."""

    answer = state.get("final_answer", "").strip()
    new_errors: list[str] = []
    needs_confirmation = state.get("needs_human_confirmation", False)

    if not answer:
        answer = NO_CONTEXT_ANSWER
        needs_confirmation = True
        new_errors.append("empty_answer")

    tool_results_list = state.get("tool_results") or []
    any_blocked_pending = any(
        bool(r.get("blocked")) and bool(r.get("requires_approval"))
        for r in tool_results_list
    )
    email_sent = any(
        r.get("tool_name") == "email_draft"
        and r.get("action") == "send"
        and bool((r.get("output") or {}).get("sent"))
        for r in tool_results_list
    )
    if state.get("approval_required_actions") or any_blocked_pending:
        needs_confirmation = True

    tool_answer_block = tool_results_to_answer_block(tool_results_list)

    if email_sent:
        # Show only the sent email content — no generic "not found" noise
        answer = tool_answer_block or "Your email has been sent successfully."
    elif any_blocked_pending and answer == NO_CONTEXT_ANSWER:
        # Don't say "information not found" when we have a pending email draft waiting for approval
        answer = (
            "I've prepared a draft email for you. "
            "Please review it in the approval panel and confirm to send."
        )
    elif tool_answer_block and tool_answer_block not in answer:
        answer = f"{answer}\n\n{tool_answer_block}" if answer else tool_answer_block

    return {
        "final_answer": answer,
        "needs_human_confirmation": needs_confirmation,
        # errors reducer appends — node returns only newly discovered errors
        "errors": new_errors,
    }


# ── grading helpers ────────────────────────────────────────────────────────────


_STOPWORDS = frozenset({
    "what", "is", "the", "how", "do", "i", "a", "an", "are", "of", "for",
    "in", "to", "my", "can", "will", "be", "me", "at", "on", "it", "this",
    "that", "get", "about", "which", "when", "where", "who", "does",
})


def _keyword_grade(question: str, chunks: list) -> list:
    """Offline relevance check: keyword overlap between question and chunk text."""
    q_words = {w for w in question.lower().split() if w not in _STOPWORDS and len(w) > 2}
    if not q_words:
        return chunks  # nothing to filter on

    relevant = [
        chunk for chunk in chunks
        if q_words & set((getattr(chunk, "text", "") or "").lower().split())
    ]
    # If keyword filter is too aggressive, keep all chunks rather than returning empty
    return relevant or chunks


def _llm_grade_relevance(question: str, chunks: list, llm_client: LLMClient) -> list:
    """Single LLM call to identify which chunks can help answer the question."""
    if not chunks:
        return []

    chunk_list = "\n".join(
        f"[{i}] {(getattr(c, 'text', '') or '')[:200].replace(chr(10), ' ')}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"HR Question: {question}\n\n"
        f"Retrieved document chunks:\n{chunk_list}\n\n"
        "List the index numbers of chunks that can help answer this HR question.\n"
        "Reply with comma-separated numbers only (e.g. '0,2'). If none are relevant, reply 'none'."
    )

    try:
        response = llm_client.generate_freeform(prompt).strip().lower()
        if "none" in response:
            return []
        indices = {
            int(x.strip())
            for x in re.split(r"[,\s]+", response)
            if x.strip().isdigit()
        }
        if not indices:
            # LLM returned unrecognised format — keep all chunks to avoid blocking the answer
            return chunks
        return [chunks[i] for i in sorted(indices) if i < len(chunks)]
    except Exception:
        return chunks  # on error keep all chunks rather than blocking the answer


# ── helpers ────────────────────────────────────────────────────────────────────


def _error_delta(message: str) -> dict[str, Any]:
    """Return a minimal error delta that keeps the workflow runnable."""
    return {
        "errors": [message],
        "final_answer": NO_CONTEXT_ANSWER,
        "used_context": False,
        "needs_human_confirmation": True,
    }


def build_initial_state(
    *,
    user_message: str,
    user_id: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
    approved_tool_actions: list[dict[str, Any]] | None = None,
    conversation_history: str = "",
) -> AgentState:
    """Build a normalized initial workflow state."""

    return {
        "user_id": user_id,
        "user_message": user_message.strip(),
        "filters": filters or {},
        "top_k": top_k,
        "score_threshold": score_threshold,
        "approved_tool_actions": list(approved_tool_actions or []),
        "conversation_history": conversation_history,
        "retrieved_chunks": [],
        "retrieval_metadata_filter": {},
        "tool_results": [],
        "approval_required_actions": [],
        "context": "",
        "prompt": "",
        "final_answer": "",
        "sources": [],
        "used_context": False,
        "needs_human_confirmation": False,
        "errors": [],
        "search_query": "",
        "retry_count": 0,
    }
