"""Tests for Agentic RAG nodes: rewrite_query, grade_documents, and retry loop."""

from __future__ import annotations

from app.agent.graph import HRPolicyAgent
from app.agent.llm import ExtractiveLLMClient, LLMClient
from app.agent.nodes import (
    MAX_RETRIEVAL_RETRIES,
    build_initial_state,
    grade_documents_node,
    rewrite_query_node,
)
from app.rag.retriever import RetrievalResponse, RetrievedChunk


# ── fakes ──────────────────────────────────────────────────────────────────────


class FakeLLMClient(LLMClient):
    """Records all prompts and returns a fixed answer."""

    def __init__(self, answer: str = "0") -> None:
        self.answer = answer
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return self.answer

    def generate_freeform(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return self.answer


class FakeRetriever:
    """Returns a preconfigured response; tracks how many times retrieve was called."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.call_count = 0
        self.queries: list[str] = []

    def retrieve(self, query: str, *, filters=None, top_k=None, score_threshold=None) -> RetrievalResponse:
        self.call_count += 1
        self.queries.append(query)
        return RetrievalResponse(query=query, chunks=self.chunks, metadata_filter={})


def _chunk(text: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=text,
        score=0.9,
        metadata={"title": "Policy", "source": "policy.md"},
    )


# ── rewrite_query_node ─────────────────────────────────────────────────────────


def test_rewrite_query_first_attempt_uses_original_message_format() -> None:
    """First attempt should produce a short keyword query (no retry context in prompt)."""
    llm = FakeLLMClient(answer="sick leave India duration")
    state = build_initial_state(user_message="How many sick leave days do I get in India?")

    delta = rewrite_query_node(state, llm)

    assert delta["search_query"] == "sick leave India duration"
    assert llm.call_count == 1
    assert "retry" not in llm.prompts[0].lower()


def test_rewrite_query_retry_prompt_mentions_previous_failure() -> None:
    """On retry (retry_count > 0) the prompt should mention the previous query failed."""
    llm = FakeLLMClient(answer="maternity leave parental policy")
    state = build_initial_state(user_message="How much maternity leave do I get?")
    state = {**state, "search_query": "maternity leave duration", "retry_count": 1}

    delta = rewrite_query_node(state, llm)

    assert delta["search_query"] == "maternity leave parental policy"
    # Prompt must explain that previous query failed
    assert "maternity leave duration" in llm.prompts[0]
    assert "no useful" in llm.prompts[0].lower() or "returned no" in llm.prompts[0].lower()


def test_rewrite_query_skips_rewriting_for_extractive_client() -> None:
    """ExtractiveLLMClient has no generation capability — original message should pass through."""
    state = build_initial_state(user_message="What is the sick leave policy?")

    delta = rewrite_query_node(state, ExtractiveLLMClient())

    assert delta["search_query"] == "What is the sick leave policy?"


def test_rewrite_query_sanitizes_llm_output() -> None:
    """Quoted or multiline LLM responses should be cleaned up."""
    llm = FakeLLMClient(answer='"sick leave policy"\nextra line here')
    state = build_initial_state(user_message="What is sick leave?")

    delta = rewrite_query_node(state, llm)

    assert delta["search_query"] == "sick leave policy"
    assert "\n" not in delta["search_query"]
    assert '"' not in delta["search_query"]


# ── grade_documents_node ───────────────────────────────────────────────────────


def test_grade_documents_accepts_relevant_chunks() -> None:
    """LLM returns index '0' → chunk passes grading, used_context stays True."""
    llm = FakeLLMClient(answer="0")
    chunk = _chunk("Sick leave: employees get 12 days per year.")
    state = {**build_initial_state(user_message="How many sick leave days?"), "retrieved_chunks": [chunk]}

    delta = grade_documents_node(state, llm)

    assert delta["used_context"] is True
    assert len(delta["retrieved_chunks"]) == 1
    assert delta["retrieved_chunks"][0].id == "c1"


def test_grade_documents_filters_irrelevant_chunks() -> None:
    """LLM returns 'none' → all chunks rejected, retry_count increments."""
    llm = FakeLLMClient(answer="none")
    chunk = _chunk("Company cafeteria opens at 9am.")
    state = {**build_initial_state(user_message="What is the sick leave policy?"), "retrieved_chunks": [chunk]}

    delta = grade_documents_node(state, llm)

    assert delta["used_context"] is False
    assert delta["retrieved_chunks"] == []
    assert delta["retry_count"] == 1  # incremented from 0


def test_grade_documents_increments_retry_count_on_empty_retrieval() -> None:
    """Zero chunks returned → retry_count should increment."""
    state = {**build_initial_state(user_message="Some HR question?"), "retrieved_chunks": [], "retry_count": 0}

    delta = grade_documents_node(state, ExtractiveLLMClient())

    assert delta["used_context"] is False
    assert delta["retry_count"] == 1


def test_grade_documents_keyword_match_offline() -> None:
    """ExtractiveLLMClient mode uses keyword overlap — relevant chunk should pass."""
    chunk = _chunk("Sick leave policy: employees must notify their manager.")
    state = {**build_initial_state(user_message="What is the sick leave policy?"), "retrieved_chunks": [chunk]}

    delta = grade_documents_node(state, ExtractiveLLMClient())

    assert delta["used_context"] is True
    assert len(delta["retrieved_chunks"]) == 1


def test_grade_documents_keyword_mismatch_offline() -> None:
    """ExtractiveLLMClient: chunk with no keyword overlap is filtered but all chunks kept as fallback."""
    chunk = _chunk("The cafeteria serves lunch from noon.")
    state = {**build_initial_state(user_message="maternity leave policy"), "retrieved_chunks": [chunk]}

    delta = grade_documents_node(state, ExtractiveLLMClient())

    # Keyword fallback keeps all chunks rather than blocking the answer
    assert delta["used_context"] is True


# ── retry loop integration ─────────────────────────────────────────────────────


def test_agent_retries_retrieval_when_grading_fails() -> None:
    """When grading returns no relevant chunks, the agent retries retrieve up to MAX times."""
    retriever = FakeRetriever(chunks=[])  # always empty → grade will always fail
    llm = FakeLLMClient(answer="sick leave query")  # rewrite returns this
    agent = HRPolicyAgent(retriever=retriever, llm_client=llm)

    state = agent.run(user_message="What is the sick leave policy?")

    # Should have tried (1 + MAX_RETRIEVAL_RETRIES) times
    assert retriever.call_count == 1 + MAX_RETRIEVAL_RETRIES
    assert state["used_context"] is False
    assert state["needs_human_confirmation"] is True


def test_agent_uses_rewritten_query_for_retrieval() -> None:
    """The rewritten search_query (not the original message) should be passed to retriever."""
    retriever = FakeRetriever(chunks=[_chunk("Sick leave policy content.")])
    llm = FakeLLMClient(answer="0")  # grading: index 0 is relevant
    agent = HRPolicyAgent(retriever=retriever, llm_client=llm)

    # Patch rewrite so we know exactly what query goes to retriever
    original_rewrite = llm.generate_freeform

    def controlled_rewrite(prompt: str) -> str:
        if "Convert this HR question" in prompt or "returned no useful" in prompt:
            return "sick leave policy India"
        return original_rewrite(prompt)

    llm.generate_freeform = controlled_rewrite  # type: ignore[method-assign]

    agent.run(user_message="kya main sick leave le sakta hoon?")

    assert retriever.queries[0] == "sick leave policy India"


def test_agent_stops_retrying_once_relevant_docs_found() -> None:
    """Once grading finds relevant chunks, the loop must NOT continue retrying."""
    retriever = FakeRetriever(chunks=[_chunk("Leave policy: 12 days sick leave.")])
    llm = FakeLLMClient(answer="0")  # always grade index 0 as relevant
    agent = HRPolicyAgent(retriever=retriever, llm_client=llm)

    agent.run(user_message="What is the sick leave policy?")

    # Should only retrieve once — no retry needed
    assert retriever.call_count == 1


def test_agent_retry_uses_different_prompt_on_second_attempt() -> None:
    """Second rewrite attempt should include context about the failed first query."""
    retriever = FakeRetriever(chunks=[])  # always empty to force retry
    rewrite_prompts: list[str] = []

    class TrackingLLM(LLMClient):
        def generate(self, prompt: str) -> str:
            return "0"

        def generate_freeform(self, prompt: str) -> str:
            rewrite_prompts.append(prompt)
            return "alternative query"

    agent = HRPolicyAgent(retriever=retriever, llm_client=TrackingLLM())
    agent.run(user_message="What is the bereavement leave policy?")

    # At minimum two rewrite calls — second should mention previous failure
    assert len(rewrite_prompts) >= 2
    assert any("returned no" in p.lower() or "no useful" in p.lower() for p in rewrite_prompts[1:])
