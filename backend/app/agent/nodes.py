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
from app.agent.tool_definitions import TOOL_NAME_TO_INTENT, TOOL_SCHEMAS
from app.agent.tools import (
    ApprovalRequiredAction,
    EmailDraftTool,
    EmailDraftKind,
    EmployeeProfileTool,
    HRTicketTool,
    ToolOrchestrator,
    ToolResult,
    extract_email_from_message,
    tool_results_to_answer_block,
)
from app.agent.sub_agents import SupervisorAgent
from app.cache.semantic_cache import get_semantic_cache
from app.config import get_settings
from app.memory.long_term_memory import (
    build_memory_context,
    extract_and_save_preferences,
    record_interaction,
)
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


def llm_tool_calling_node(state: AgentState, llm_client: LLMClient) -> dict[str, Any]:
    """Use OpenAI function calling to decide tools + execute them in one node.

    Falls back to keyword-based classify_intent_node when the LLM does not
    support tool calling (e.g. ExtractiveLLMClient in offline tests).
    """
    if not getattr(llm_client, "supports_tool_calling", False):
        return classify_intent_node(state)

    user_message = state.get("user_message", "")
    try:
        result = llm_client.call_with_tools(user_message, TOOL_SCHEMAS)
    except Exception as exc:
        log_event(logger, event="agent.tool_calling.error", error=str(exc))
        return classify_intent_node(state)

    tool_calls = result.get("tool_calls", [])
    if not tool_calls:
        # LLM chose no tools → treat as general HR question
        return {"intent": "general_hr"}

    log_event(logger, event="agent.tool_calling.selected", tools=[t["name"] for t in tool_calls])

    # Determine intent from first tool called
    first_tool = tool_calls[0]["name"]
    intent: Intent = TOOL_NAME_TO_INTENT.get(first_tool, "general_hr")

    # If it's a policy question, just set intent and let RAG pipeline handle it
    if first_tool == "answer_policy_question":
        query = tool_calls[0].get("arguments", {}).get("query", user_message)
        return {"intent": "policy_qa", "search_query": query}

    # Execute action tools (email / ticket) based on LLM's structured decision
    tool_results: list[dict] = []
    approval_required: list[dict] = []
    settings = get_settings()

    profile_tool = EmployeeProfileTool()
    email_tool = EmailDraftTool()
    ticket_tool = HRTicketTool()

    profile_result = profile_tool.run(state.get("user_id"))
    tool_results.append(profile_result.model_dump(mode="json"))
    employee_profile = profile_result.output

    raw_approved = state.get("approved_tool_actions") or []
    approved_pairs = {
        (str(a["tool_name"]), str(a["action"]))
        for a in raw_approved
        if a.get("tool_name") and a.get("action")
    }

    for call in tool_calls:
        name = call["name"]
        args = call.get("arguments", {})

        if name == "send_email":
            action = args.get("action", "send")
            recipient_type = args.get("recipient_type", "manager")
            recipient_email = args.get("recipient_email") or extract_email_from_message(user_message)
            message_context = args.get("message_context", user_message)

            if recipient_type == "custom" and recipient_email:
                kind: EmailDraftKind = "custom_recipient"
            elif recipient_type == "hr":
                kind = "hr_sick_leave"
            else:
                kind = "manager_leave"

            hr_contact = settings.hr_contact_email.strip() if recipient_type == "hr" else None

            if action == "send":
                # Custom recipient → send directly; others need approval
                if kind == "custom_recipient" and recipient_email:
                    res = email_tool.send(
                        user_message=message_context,
                        employee_profile=employee_profile,
                        approved=True,
                        kind=kind,
                        custom_recipient_email=recipient_email,
                        llm_client=llm_client,
                    )
                    tool_results.append(res.model_dump(mode="json"))
                elif ("email_draft", "send") in approved_pairs:
                    res = email_tool.send(
                        user_message=message_context,
                        employee_profile=employee_profile,
                        approved=True,
                        kind=kind,
                        hr_contact_email=hr_contact,
                        llm_client=llm_client,
                    )
                    tool_results.append(res.model_dump(mode="json"))
                else:
                    draft_res = email_tool.run(
                        user_message=message_context,
                        employee_profile=employee_profile,
                        kind=kind,
                        hr_contact_email=hr_contact,
                        llm_client=llm_client,
                    )
                    draft_text = draft_res.output.get("draft", "")
                    subject = draft_res.output.get("subject", "")
                    recipient = draft_res.output.get("recipient") or employee_profile.get("manager_email") or ""
                    blocked = ToolResult(
                        tool_name="email_draft",
                        action="send",
                        output={"draft": draft_text, "subject": subject, "recipient": recipient,
                                "kind": kind, "pending_approval": True},
                        success=False, blocked=True, requires_approval=True,
                        message="Email draft ready — awaiting your approval to send.",
                    )
                    tool_results.append(blocked.model_dump(mode="json"))
                    approval_required.append(ApprovalRequiredAction(
                        tool_name="email_draft", action="send",
                        reason="Review the email below and confirm to send.",
                        input={"message": message_context, "draft": draft_text,
                               "subject": subject, "recipient": recipient},
                    ).model_dump(mode="json"))
            else:
                res = email_tool.run(
                    user_message=message_context,
                    employee_profile=employee_profile,
                    kind=kind,
                    hr_contact_email=hr_contact,
                    custom_recipient_email=recipient_email if kind == "custom_recipient" else None,
                    llm_client=llm_client,
                )
                tool_results.append(res.model_dump(mode="json"))

        elif name == "create_hr_ticket":
            action = args.get("action", "draft")
            description = args.get("description", user_message)
            if action == "create":
                if ("hr_ticket", "create") in approved_pairs:
                    res = ticket_tool.create(user_message=description, approved=True)
                else:
                    res = ticket_tool.create(user_message=description, approved=False)
                    approval_required.append(ApprovalRequiredAction(
                        tool_name="hr_ticket", action="create",
                        reason="Creating an HR ticket changes enterprise records and needs your approval.",
                        input={"description": description},
                    ).model_dump(mode="json"))
            else:
                res = ticket_tool.draft(user_message=description)
            tool_results.append(res.model_dump(mode="json"))

    updated_filters = dict(state.get("filters") or {})
    updated_filters.setdefault("country", employee_profile.get("country"))
    updated_filters.setdefault("employee_type", employee_profile.get("employee_type"))
    updated_filters.setdefault("access_level", employee_profile.get("access_level"))

    return {
        "intent": intent,
        "tool_results": tool_results,
        "approval_required_actions": approval_required,
        "filters": {k: v for k, v in updated_filters.items() if v is not None},
    }


MAX_REACT_ITERATIONS = 3  # prevent infinite loops


def supervisor_node(state: AgentState, llm_client: LLMClient) -> dict[str, Any]:
    """Route user message to the best specialist sub-agent.

    Sets `specialist` and `specialist_system_prompt` in state so that
    generate_answer_node can use the specialist's focused system prompt.
    """
    supervisor = SupervisorAgent(llm_client)
    specialist_name, specialist_cfg = supervisor.route(state.get("user_message", ""))
    return {
        "specialist": specialist_name,
        "specialist_system_prompt": specialist_cfg.get("system_prompt", ""),
    }


def react_loop_node(state: AgentState, llm_client: LLMClient) -> dict[str, Any]:
    """ReAct: Reason → Act → Observe, repeat until done or max iterations.

    After each tool execution the LLM observes the result and decides:
    - 'done'          → proceed to RAG answer generation
    - 'retry_email'   → try sending email again (e.g. after a failure)
    - 'escalate'      → create HR ticket instead of email
    - 'search_policy' → query RAG for policy info before acting
    """
    if not getattr(llm_client, "supports_tool_calling", False):
        return {}  # skip — keyword path handles this

    tool_results = list(state.get("tool_results") or [])
    user_message = state.get("user_message", "")
    iterations = state.get("react_iterations", 0)

    if iterations >= MAX_REACT_ITERATIONS or not tool_results:
        return {"react_iterations": iterations}

    # If email was already sent or pending approval → don't escalate, just proceed
    email_sent = any(
        r.get("tool_name") == "email_draft"
        and (r.get("output") or {}).get("sent")
        for r in tool_results
    )
    email_pending = any(
        r.get("tool_name") == "email_draft" and r.get("requires_approval")
        for r in tool_results
    )
    if email_sent or email_pending:
        return {"react_iterations": iterations + 1, "react_decision": "done"}

    # Build observation summary for the LLM
    obs_lines = []
    for r in tool_results:
        tool = r.get("tool_name", "")
        action = r.get("action", "")
        success = r.get("success", True)
        blocked = r.get("blocked", False)
        msg = r.get("message", "")
        output = r.get("output", {})
        if blocked:
            obs_lines.append(f"- {tool}.{action}: PENDING APPROVAL — {msg}")
        elif not success:
            error = output.get("error", msg)
            obs_lines.append(f"- {tool}.{action}: FAILED — {error}")
        else:
            obs_lines.append(f"- {tool}.{action}: SUCCESS — {msg}")

    observation = "\n".join(obs_lines)

    reasoning_prompt = (
        f"You are an HR assistant. The user asked: \"{user_message}\"\n\n"
        f"Tool execution results so far:\n{observation}\n\n"
        "Based on these results, what should happen next?\n"
        "Reply with exactly one of these words:\n"
        "- 'done'          if the task is complete or pending human approval\n"
        "- 'escalate'      if the action failed and an HR ticket should be raised instead\n"
        "- 'search_policy' if you need HR policy info before responding\n"
        "- 'retry'         if a transient error occurred and the action should be retried\n"
        "Your single-word decision:"
    )

    try:
        decision = llm_client.generate_freeform(reasoning_prompt).strip().lower().split()[0]
    except Exception:
        decision = "done"

    log_event(logger, event="agent.react.decision", decision=decision, iteration=iterations)

    if decision == "escalate":
        # Only escalate to ticket if no email tool already ran (avoid duplicate actions)
        email_already_ran = any(r.get("tool_name") == "email_draft" for r in tool_results)
        if not email_already_ran:
            ticket_tool = HRTicketTool()
            ticket_res = ticket_tool.draft(user_message=user_message)
            return {
                "tool_results": [ticket_res.model_dump(mode="json")],
                "react_iterations": iterations + 1,
                "react_decision": decision,
            }
        return {"react_iterations": iterations + 1, "react_decision": "done"}

    if decision == "search_policy":
        return {
            "react_iterations": iterations + 1,
            "react_decision": decision,
            "used_context": False,  # force RAG to run
        }

    # 'done' or 'retry' or unrecognised → proceed
    return {"react_iterations": iterations + 1, "react_decision": decision}


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


def cache_store_node(state: AgentState, retriever: RagRetriever) -> dict[str, Any]:
    """Store the final answer in the semantic cache after successful RAG generation."""
    # Only cache policy Q&A answers, not action results or cache-served answers
    if state.get("cache_hit") or state.get("intent") == "action_request":
        return {}
    answer = state.get("final_answer", "")
    if not answer or answer == NO_CONTEXT_ANSWER:
        return {}
    try:
        embedder = getattr(retriever, "embedding_provider", None)
        if embedder is None:
            return {}

        class _EmbedAdapter:
            def embed(self, text: str) -> list[float]:  # type: ignore[override]
                return embedder.embed_text(text)

        cache = get_semantic_cache()
        cache.store(
            query=state.get("user_message", ""),
            answer=answer,
            embedder=_EmbedAdapter(),
            sources=state.get("sources", []),
        )
    except Exception as exc:
        log_event(logger, event="cache.store_node_error", error=str(exc))
    return {}


def retrieve_context_node(state: AgentState, retriever: RagRetriever) -> dict[str, Any]:
    """Retrieve RAG context using the (possibly rewritten) search query.

    Checks semantic cache first — if a similar query was answered recently,
    returns the cached answer without hitting the vector store or LLM.
    """
    # Semantic cache check (only on first attempt, not retries)
    if state.get("retry_count", 0) == 0 and state.get("intent") != "action_request":
        cache = get_semantic_cache()
        embedder = getattr(retriever, "embedding_provider", None)
        if embedder is not None:
            class _EmbedAdapter:
                def embed(self, text: str) -> list[float]:  # type: ignore[override]
                    return embedder.embed_text(text)
            hit = cache.lookup(state.get("user_message", ""), _EmbedAdapter())
            if hit:
                return {
                    "final_answer": hit["answer"],
                    "sources": hit.get("sources", []),
                    "used_context": True,
                    "cache_hit": True,
                }

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

    user_id = state.get("user_id")
    user_message = state.get("user_message", "")

    # Save preferences learned from this message
    extract_and_save_preferences(user_id or "", user_message, llm_client)

    if not state.get("used_context"):
        return {
            "final_answer": NO_CONTEXT_ANSWER,
            "needs_human_confirmation": True,
        }

    memory_context = build_memory_context(user_id or "")
    prompt = build_rag_prompt(
        user_message=user_message,
        context=state.get("context", ""),
        conversation_history=state.get("conversation_history", ""),
        memory_context=memory_context,
        specialist_system_prompt=state.get("specialist_system_prompt", ""),
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

    # Store in semantic cache for future similar queries (skip action_request turns)
    if state.get("intent") != "action_request" and not state.get("cache_hit"):
        try:
            from app.rag.retriever import RagRetriever  # avoid circular at module level
            cache = get_semantic_cache()
            # We don't have direct embedder access here — stored lazily on next retrieve call
            # so we mark answer for deferred caching via state
            pass
        except Exception:
            pass

    return {
        "prompt": prompt,
        "final_answer": answer,
        "needs_human_confirmation": False,
        "cache_hit": False,
    }


def validate_response_node(state: AgentState) -> dict[str, Any]:
    """Validate response safety and completeness at the workflow boundary."""

    answer = state.get("final_answer", "").strip()
    new_errors: list[str] = []
    needs_confirmation = state.get("needs_human_confirmation", False)
    user_id = state.get("user_id") or ""

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
        # Record sent email in long-term memory
        for r in tool_results_list:
            if r.get("tool_name") == "email_draft" and r.get("action") == "send" and (r.get("output") or {}).get("sent"):
                recipient = (r.get("output") or {}).get("recipient", "recipient")
                record_interaction(user_id, action="email_sent",
                                   summary=f"Sent email to {recipient}",
                                   metadata={"recipient": recipient})
                break
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
        "react_iterations": 0,
        "react_decision": "",
        "specialist": "",
        "specialist_system_prompt": "",
        "cache_hit": False,
    }
