from app.agent.graph import HRPolicyAgent
from app.agent.llm import LLMClient
from app.agent.nodes import build_initial_state, classify_intent_node, validate_response_node
from app.rag.retriever import RetrievalResponse, RetrievedChunk


class FakeRetriever:
    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.last_query = ""
        self.last_filters = None

    def retrieve(self, query: str, *, filters=None, top_k=None, score_threshold=None) -> RetrievalResponse:
        self.last_query = query
        self.last_filters = filters
        return self.response


class FakeLLMClient(LLMClient):
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.answer


def test_classify_intent_node_detects_policy_question() -> None:
    state = build_initial_state(user_message="What is the sick leave policy?")

    updated = classify_intent_node(state)

    assert updated["intent"] == "policy_qa"


def test_classify_intent_node_detects_action_request() -> None:
    state = build_initial_state(user_message="Draft an email for sick leave")

    updated = classify_intent_node(state)

    assert updated["intent"] == "action_request"


def test_agent_workflow_generates_answer_with_context() -> None:
    retriever = FakeRetriever(
        RetrievalResponse(
            query="sick leave",
            chunks=[
                RetrievedChunk(
                    id="chunk-1",
                    content="Employees should inform their manager for sick leave.",
                    score=0.9,
                    metadata={
                        "title": "Sick Leave Policy",
                        "source": "leave.md",
                        "section_title": "Sick Leave",
                        "country": "India",
                        "policy_type": "leave",
                    },
                )
            ],
        )
    )
    llm = FakeLLMClient("Inform your manager and submit the leave request.")
    agent = HRPolicyAgent(retriever=retriever, llm_client=llm)

    state = agent.run(
        user_message="What is the sick leave policy?",
        user_id="emp_123",
        filters={"country": "India", "policy_type": "leave"},
    )

    assert state["intent"] == "policy_qa"
    assert state["final_answer"] == "Inform your manager and submit the leave request."
    assert state["used_context"] is True
    assert state["needs_human_confirmation"] is False
    assert state["sources"][0]["chunk_id"] == "chunk-1"
    assert "Approved HR policy context:" in llm.last_prompt
    assert retriever.last_filters["country"] == "India"
    assert retriever.last_filters["policy_type"] == "leave"


def test_agent_workflow_returns_no_context_fallback() -> None:
    agent = HRPolicyAgent(
        retriever=FakeRetriever(RetrievalResponse(query="unknown", chunks=[])),
        llm_client=FakeLLMClient("Should not be used"),
    )

    state = agent.run(user_message="Unknown benefit?")

    assert state["used_context"] is False
    assert state["needs_human_confirmation"] is True
    assert "could not find" in state["final_answer"]


def test_action_request_draft_email_does_not_force_human_confirmation() -> None:
    agent = HRPolicyAgent(
        retriever=FakeRetriever(
            RetrievalResponse(
                query="draft email",
                chunks=[
                    RetrievedChunk(
                        id="chunk-1",
                        content="Employees should inform their manager before sick leave.",
                        score=0.9,
                        metadata={"title": "Sick Leave", "source": "leave.md"},
                    )
                ],
            )
        ),
        llm_client=FakeLLMClient("Here is a draft email."),
    )

    state = agent.run(user_message="Draft an email for sick leave")

    assert state["intent"] == "action_request"
    assert state["needs_human_confirmation"] is False


def test_action_request_send_email_requires_confirmation_until_approved() -> None:
    retriever = FakeRetriever(
        RetrievalResponse(
            query="send email",
            chunks=[
                RetrievedChunk(
                    id="chunk-1",
                    content="Use the HR portal for leave requests.",
                    score=0.9,
                    metadata={"title": "Leave", "source": "leave.md"},
                )
            ],
        )
    )
    agent = HRPolicyAgent(retriever=retriever, llm_client=FakeLLMClient("You may email your manager per policy."))

    state = agent.run(user_message="Send an email to my manager requesting sick leave")

    assert state["intent"] == "action_request"
    assert state["needs_human_confirmation"] is True
    assert len(state["approval_required_actions"]) == 1
    assert state["approval_required_actions"][0]["tool_name"] == "email_draft"
    assert state["approval_required_actions"][0]["action"] == "send"

    state_ok = agent.run(
        user_message="Send an email to my manager requesting sick leave",
        approved_tool_actions=[{"tool_name": "email_draft", "action": "send"}],
    )
    assert state_ok["needs_human_confirmation"] is False
    assert state_ok["approval_required_actions"] == []
    email_send = next(
        r for r in state_ok["tool_results"] if r.get("tool_name") == "email_draft" and r.get("action") == "send"
    )
    assert email_send.get("success") is True
    assert email_send.get("blocked") is False


def test_validate_response_node_fills_empty_answer() -> None:
    state = validate_response_node({"user_message": "hello", "final_answer": "", "errors": []})

    assert state["needs_human_confirmation"] is True
    assert "empty_answer" in state["errors"]


def test_build_langgraph_reports_missing_optional_dependency() -> None:
    agent = HRPolicyAgent(
        retriever=FakeRetriever(RetrievalResponse(query="unknown", chunks=[])),
        llm_client=FakeLLMClient("answer"),
    )

    try:
        graph = agent.build_langgraph()
    except RuntimeError as exc:
        assert "langgraph package" in str(exc)
    else:
        assert graph is not None
