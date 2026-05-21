"""State definitions for the HR policy agent workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


Intent = Literal["policy_qa", "action_request", "general_hr", "unknown"]


class AgentState(TypedDict, total=False):
    """Shared state passed between agent workflow nodes.

    Fields annotated with `operator.add` are *reducers*: LangGraph appends
    node-returned lists instead of replacing them.  All other fields use the
    default replace-on-update semantics.
    """

    # ── inputs ─────────────────────────────────────────────────────────────
    user_id: str | None
    user_message: str
    intent: Intent
    filters: dict[str, Any]
    top_k: int
    score_threshold: float | None
    approved_tool_actions: list[dict[str, Any]]
    conversation_history: str  # formatted prior-turn block passed to prompt

    # ── retrieval ──────────────────────────────────────────────────────────
    retrieved_chunks: list[Any]
    retrieval_metadata_filter: dict[str, Any]
    context: str
    used_context: bool

    # ── agentic rag ────────────────────────────────────────────────────────
    search_query: str  # rewritten query used for vector search (may differ from user_message)
    retry_count: int   # number of retrieval retries attempted so far

    # ── accumulated lists (reducer = append, not replace) ──────────────────
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    approval_required_actions: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]

    # sources is replaced (not appended) so retried retrievals overwrite stale citations
    sources: list[dict[str, Any]]

    # ── output ─────────────────────────────────────────────────────────────
    prompt: str
    final_answer: str
    needs_human_confirmation: bool
