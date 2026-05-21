from fastapi import status
from fastapi.testclient import TestClient

from app.agent.llm import LLMClient
from app.api.chat_routes import get_llm_client, get_retriever
from app.config import Settings
from app.main import create_app
from app.security.auth import create_access_token
from app.rag.retriever import RetrievalResponse, RetrievedChunk


class FakeRetriever:
    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.last_filters = None

    def retrieve(self, query: str, *, filters=None, top_k=None, score_threshold=None) -> RetrievalResponse:
        self.last_filters = filters
        return self.response


class FakeLLMClient(LLMClient):
    def __init__(self, answer: str = "You can take sick leave after informing your manager.") -> None:
        self.answer = answer
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.answer


def build_client(fake_retriever: FakeRetriever, fake_llm: FakeLLMClient) -> TestClient:
    app = create_app(Settings(app_env="test", app_debug=False))
    app.dependency_overrides[get_retriever] = lambda: fake_retriever
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    return TestClient(app)


def test_chat_endpoint_returns_answer_with_citations() -> None:
    fake_retriever = FakeRetriever(
        RetrievalResponse(
            query="What is sick leave process?",
            metadata_filter={"country": "India", "policy_type": "leave"},
            chunks=[
                RetrievedChunk(
                    id="chunk-1",
                    content="Employees can take sick leave after informing their manager.",
                    score=0.91,
                    metadata={
                        "title": "Sick Leave Policy India",
                        "source": "leave_policy_india.md",
                        "section_title": "Sick Leave",
                        "policy_type": "leave",
                        "country": "India",
                        "employee_type": "full_time",
                    },
                )
            ],
        )
    )
    fake_llm = FakeLLMClient()
    client = build_client(fake_retriever, fake_llm)
    token = create_access_token(
        {
            "user_id": "emp_123",
            "role": "employee",
            "country": "India",
            "employee_type": "full_time",
            "department": "HR",
        }
    )

    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": "What is sick leave process?",
            "country": "India",
            "employee_type": "full_time",
            "policy_type": "leave",
            "top_k": 3,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["answer"] == "You can take sick leave after informing your manager."
    assert body["used_context"] is True
    assert body["needs_human_confirmation"] is False
    assert body["intent"] == "policy_qa"
    assert body["sources"] == [
        {
            "chunk_id": "chunk-1",
            "title": "Sick Leave Policy India",
            "source": "leave_policy_india.md",
            "section_title": "Sick Leave",
            "score": 0.91,
            "policy_type": "leave",
            "country": "India",
            "employee_type": "full_time",
        }
    ]
    assert "Approved HR policy context:" in fake_llm.last_prompt
    assert fake_retriever.last_filters["country"] == "India"


def test_chat_endpoint_returns_no_context_fallback() -> None:
    fake_retriever = FakeRetriever(RetrievalResponse(query="unknown", chunks=[]))
    fake_llm = FakeLLMClient(answer="Should not be used")
    client = build_client(fake_retriever, fake_llm)

    response = client.post("/chat", json={"message": "Does this policy exist?"})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["used_context"] is False
    assert body["sources"] == []
    assert body["needs_human_confirmation"] is True
    assert body["intent"] == "policy_qa"
    assert "could not find" in body["answer"]
    # LLM is called for query rewriting but NOT for answer generation when no context found
    assert "Approved HR policy context:" not in fake_llm.last_prompt


def test_versioned_chat_endpoint_is_available() -> None:
    fake_retriever = FakeRetriever(
        RetrievalResponse(
            query="reimbursement",
            chunks=[
                RetrievedChunk(
                    id="chunk-2",
                    content="Employees may request reimbursement with a valid invoice.",
                    score=0.8,
                    metadata={"title": "Reimbursement Policy", "source": "reimbursement_policy.txt"},
                )
            ],
        )
    )
    client = build_client(fake_retriever, FakeLLMClient(answer="Submit a valid invoice."))

    response = client.post("/api/v1/chat", json={"message": "How do I claim reimbursement?"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["answer"] == "Submit a valid invoice."


def test_chat_endpoint_validates_empty_message() -> None:
    client = build_client(FakeRetriever(RetrievalResponse(query="", chunks=[])), FakeLLMClient())

    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
