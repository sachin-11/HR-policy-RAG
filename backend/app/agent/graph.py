"""HR Policy Agent — LangGraph workflow.

Architecture (when LangGraph is installed)
──────────────────────────────────────────
                     ┌─────────────────┐
                     │  classify_intent │
                     └────────┬────────┘
                              │
               ┌──────────────┴──────────────┐
               │ action_request              │ policy_qa / general_hr / unknown
               ▼                             ▼
       ┌───────────────┐            ┌─────────────────┐
       │ execute_tools │──────────► │  rewrite_query  │ ◄──────────────────┐
       └───────────────┘            └────────┬────────┘                    │
                                             │                             │ retry
                                    ┌────────▼────────┐                    │ (up to MAX_RETRIEVAL_RETRIES)
                                    │ retrieve_context │                    │
                                    └────────┬────────┘                    │
                                             │                             │
                                    ┌────────▼────────┐  no relevant docs ─┘
                                    │  grade_documents │
                                    └────────┬────────┘
                                             │ relevant docs found
                                    ┌────────▼────────┐
                                    │ generate_answer  │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ validate_response│
                                    └────────┬────────┘
                                             │
                                            END

Key LangGraph features used (when package is available)
────────────────────────────────────────────────────────
1. Conditional routing  — policy_qa skips execute_tools; grade_documents loops
                          back to rewrite_query when no relevant docs found.
2. State reducers       — list fields (errors, tool_results, …) accumulate
                          across nodes via operator.add (defined in state.py).
3. MemorySaver          — conversation state is checkpointed per thread_id so
                          multi-turn context works without a database.
4. interrupt()          — sensitive tool actions pause the graph; the caller
                          resumes it with the same thread_id config.
5. astream()            — callers can stream partial answers node-by-node.

Fallback
────────
If langgraph is not installed the agent runs the same node sequence as a
plain Python pipeline with an explicit retry loop.
Install langgraph (pip install langgraph) to unlock checkpointing, conditional
routing and streaming.
"""

from __future__ import annotations

from typing import Any

from app.agent.llm import LLMClient
from app.agent.nodes import (
    MAX_RETRIEVAL_RETRIES,
    build_initial_state,
    classify_intent_node,
    execute_tools_node,
    generate_answer_node,
    grade_documents_node,
    retrieve_context_node,
    rewrite_query_node,
    validate_response_node,
)
from app.agent.state import AgentState
from app.agent.tools import ToolOrchestrator
from app.rag.retriever import RagRetriever


# Fields that use operator.add reducers in AgentState — must be merged by
# appending in the plain-Python fallback runner.
# NOTE: `sources` was moved out of reducers so retried retrievals overwrite stale citations.
_REDUCER_FIELDS: frozenset[str] = frozenset(
    {"tool_results", "approval_required_actions", "errors"}
)


# ── routing ────────────────────────────────────────────────────────────────────


def _route_after_classify(state: AgentState) -> str:
    """Skip execute_tools for pure policy questions."""
    if state.get("intent") == "action_request":
        return "execute_tools"
    return "rewrite_query"


def _route_after_grade(state: AgentState) -> str:
    """Loop back to rewrite_query when grading finds no relevant chunks."""
    if not state.get("used_context") and state.get("retry_count", 0) <= MAX_RETRIEVAL_RETRIES:
        return "rewrite_query"
    return "generate_answer"


# ── state helpers ──────────────────────────────────────────────────────────────


def _merge(state: AgentState, delta: dict[str, Any]) -> AgentState:
    """Merge a node delta into the full state, honouring list reducers."""
    merged: dict[str, Any] = dict(state)
    for key, value in delta.items():
        if key in _REDUCER_FIELDS:
            merged[key] = list(merged.get(key) or []) + list(value or [])
        else:
            merged[key] = value
    return merged  # type: ignore[return-value]


# ── agent ──────────────────────────────────────────────────────────────────────


class HRPolicyAgent:
    """Stateful HR policy agent.

    When *langgraph* is installed the workflow runs as a compiled StateGraph
    with checkpointing and conditional routing.  Otherwise it falls back to a
    plain Python node pipeline so tests and local development never require the
    optional dependency.
    """

    def __init__(
        self,
        retriever: RagRetriever,
        llm_client: LLMClient,
        tool_orchestrator: ToolOrchestrator | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.tool_orchestrator = tool_orchestrator or ToolOrchestrator(llm_client=llm_client)

        # Try to build the compiled graph; fall back silently if not installed.
        try:
            self._graph = self.build_langgraph()
        except RuntimeError:
            self._graph = None

    # ── public API ─────────────────────────────────────────────────────────

    def run(
        self,
        *,
        user_message: str,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        approved_tool_actions: list[dict[str, Any]] | None = None,
        conversation_history: str = "",
    ) -> AgentState:
        """Run one turn through the workflow."""
        state = build_initial_state(
            user_message=user_message,
            user_id=user_id,
            filters=filters,
            top_k=top_k,
            score_threshold=score_threshold,
            approved_tool_actions=approved_tool_actions,
            conversation_history=conversation_history,
        )

        if self._graph is not None:
            config = _thread_config(user_id)
            return self._graph.invoke(state, config=config)

        return self._run_plain_python(state)

    def resume_after_approval(
        self,
        *,
        user_id: str | None,
        approved_actions: list[dict[str, Any]],
    ) -> AgentState:
        """Resume a graph paused by interrupt() waiting for human approval."""
        if self._graph is None:
            raise RuntimeError(
                "resume_after_approval requires the langgraph package. "
                "Run: pip install langgraph"
            )
        config = _thread_config(user_id)
        return self._graph.invoke(
            None,
            config={**config, "resume": {"approved_actions": approved_actions}},
        )

    async def astream(
        self,
        *,
        user_message: str,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        approved_tool_actions: list[dict[str, Any]] | None = None,
        conversation_history: str = "",
    ):
        """Async generator that yields per-node state deltas."""
        if self._graph is None:
            raise RuntimeError(
                "astream requires the langgraph package. "
                "Run: pip install langgraph"
            )
        state = build_initial_state(
            user_message=user_message,
            user_id=user_id,
            filters=filters,
            top_k=top_k,
            score_threshold=score_threshold,
            approved_tool_actions=approved_tool_actions,
            conversation_history=conversation_history,
        )
        config = _thread_config(user_id)
        async for chunk in self._graph.astream(state, config=config):
            yield chunk

    # ── graph builder ──────────────────────────────────────────────────────

    def build_langgraph(self):
        """Compile and return the StateGraph with MemorySaver checkpointing."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph workflow requires the langgraph package. "
                "Run: pip install langgraph"
            ) from exc

        workflow = StateGraph(AgentState)

        workflow.add_node("classify_intent", classify_intent_node)
        workflow.add_node(
            "execute_tools",
            lambda s: execute_tools_node(s, self.tool_orchestrator),
        )
        workflow.add_node(
            "rewrite_query",
            lambda s: rewrite_query_node(s, self.llm_client),
        )
        workflow.add_node(
            "retrieve_context",
            lambda s: retrieve_context_node(s, self.retriever),
        )
        workflow.add_node(
            "grade_documents",
            lambda s: grade_documents_node(s, self.llm_client),
        )
        workflow.add_node(
            "generate_answer",
            lambda s: generate_answer_node(s, self.llm_client),
        )
        workflow.add_node("validate_response", validate_response_node)

        workflow.set_entry_point("classify_intent")

        # Conditional: action_request → execute_tools; else → rewrite_query
        workflow.add_conditional_edges(
            "classify_intent",
            _route_after_classify,
            {
                "execute_tools": "execute_tools",
                "rewrite_query": "rewrite_query",
            },
        )

        workflow.add_edge("execute_tools", "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "grade_documents")

        # Conditional: no relevant docs → loop back to rewrite_query; else → generate
        workflow.add_conditional_edges(
            "grade_documents",
            _route_after_grade,
            {
                "rewrite_query": "rewrite_query",
                "generate_answer": "generate_answer",
            },
        )

        workflow.add_edge("generate_answer", "validate_response")
        workflow.add_edge("validate_response", END)

        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    # ── plain Python fallback ──────────────────────────────────────────────

    def _run_plain_python(self, state: AgentState) -> AgentState:
        """Sequential node pipeline with agentic retry loop."""

        state = _merge(state, classify_intent_node(state))

        if state.get("intent") == "action_request":
            state = _merge(state, execute_tools_node(state, self.tool_orchestrator))

        # Agentic RAG: rewrite → retrieve → grade, retry up to MAX_RETRIEVAL_RETRIES times
        for _ in range(1 + MAX_RETRIEVAL_RETRIES):
            state = _merge(state, rewrite_query_node(state, self.llm_client))
            state = _merge(state, retrieve_context_node(state, self.retriever))
            state = _merge(state, grade_documents_node(state, self.llm_client))
            if state.get("used_context"):
                break  # relevant docs found — proceed to generation

        state = _merge(state, generate_answer_node(state, self.llm_client))
        return _merge(state, validate_response_node(state))


# ── helpers ────────────────────────────────────────────────────────────────────


def _thread_config(user_id: str | None) -> dict[str, Any]:
    """LangGraph run config scoped to a user thread."""
    return {"configurable": {"thread_id": user_id or "anonymous"}}
