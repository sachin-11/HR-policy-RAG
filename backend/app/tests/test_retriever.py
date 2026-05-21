from app.rag.embeddings import MockEmbeddingProvider
from app.rag.retriever import (
    RagRetriever,
    RetrievalFilters,
    build_source_citations,
    dedupe_source_citations_for_display,
    format_retrieved_context,
    normalize_filters,
)
from app.rag.vector_store import SearchResult, VectorRecord, VectorStore


class StaticEmbeddingProvider(MockEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimension=3)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        mapping = {
            "leave query": [1.0, 0.0, 0.0],
            "reimbursement query": [0.0, 1.0, 0.0],
        }
        return [mapping.get(text, [0.0, 0.0, 1.0]) for text in texts]


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: list[VectorRecord] = [
            VectorRecord(
                id="leave-1",
                text="Employees can take sick leave after informing their manager.",
                embedding=[1.0, 0.0, 0.0],
                metadata={
                    "title": "Sick Leave Policy India",
                    "source": "leave_policy.md",
                    "section_title": "Sick Leave",
                    "country": "India",
                    "employee_type": "full_time",
                    "access_level": "employee",
                    "policy_type": "leave",
                },
            ),
            VectorRecord(
                id="reimbursement-1",
                text="Employees may request laptop reimbursement with a valid invoice.",
                embedding=[0.0, 1.0, 0.0],
                metadata={
                    "title": "Reimbursement Policy",
                    "source": "reimbursement_policy.txt",
                    "country": "US",
                    "access_level": "employee",
                    "policy_type": "reimbursement",
                },
            ),
        ]

    def upsert(self, records: list[VectorRecord]) -> None:
        self.records.extend(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        results = []
        for record in self.records:
            if metadata_filter and any(record.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            score = sum(left * right for left, right in zip(query_embedding, record.embedding, strict=True))
            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    score=score,
                    metadata=record.metadata,
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def build_retriever() -> RagRetriever:
    return RagRetriever(
        embedding_provider=StaticEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        default_top_k=2,
    )


def test_retriever_returns_top_k_results() -> None:
    retriever = build_retriever()

    response = retriever.retrieve("leave query", top_k=1)

    assert response.has_context is True
    assert len(response.chunks) == 1
    assert response.chunks[0].id == "leave-1"
    assert response.chunks[0].score == 1.0


def test_retriever_applies_metadata_filters() -> None:
    retriever = build_retriever()

    response = retriever.retrieve(
        "leave query",
        filters=RetrievalFilters(country="India", policy_type="leave"),
    )

    assert len(response.chunks) == 1
    assert response.metadata_filter == {"country": "India", "policy_type": "leave"}
    assert response.chunks[0].metadata["country"] == "India"


def test_retriever_applies_score_threshold() -> None:
    retriever = build_retriever()

    response = retriever.retrieve("leave query", score_threshold=1.1)

    assert response.has_context is False
    assert response.chunks == []


def test_retriever_rejects_empty_query() -> None:
    retriever = build_retriever()

    try:
        retriever.retrieve("   ")
    except ValueError as exc:
        assert "query must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_format_retrieved_context_includes_source_labels() -> None:
    response = build_retriever().retrieve("leave query", top_k=1)

    context = format_retrieved_context(response.chunks)

    assert "[Source 1]" in context
    assert "Sick Leave Policy India > Sick Leave" in context
    assert "Employees can take sick leave" in context


def test_retrieve_context_returns_formatted_context() -> None:
    retriever = build_retriever()

    context = retriever.retrieve_context("reimbursement query", filters={"policy_type": "reimbursement"})

    assert "Reimbursement Policy" in context
    assert "valid invoice" in context


def test_normalize_filters_removes_none_values() -> None:
    assert normalize_filters({"country": "India", "employee_type": None}) == {"country": "India"}


def test_build_source_citations_returns_compact_metadata() -> None:
    response = build_retriever().retrieve("leave query", top_k=1)

    citations = build_source_citations(response.chunks)

    assert citations == [
        {
            "chunk_id": "leave-1",
            "title": "Sick Leave Policy India",
            "source": "leave_policy.md",
            "section_title": "Sick Leave",
            "score": 1.0,
            "policy_type": "leave",
            "country": "India",
            "employee_type": "full_time",
        }
    ]


def test_dedupe_source_citations_collapses_same_file() -> None:
    citations = [
        {
            "chunk_id": "a",
            "title": "Handbook",
            "source": "demo.md",
            "section_title": "Leave",
            "score": 0.5,
            "policy_type": None,
            "country": None,
            "employee_type": None,
        },
        {
            "chunk_id": "b",
            "title": "Handbook",
            "source": "demo.md",
            "section_title": "FAQ",
            "score": 0.9,
            "policy_type": None,
            "country": None,
            "employee_type": None,
        },
        {
            "chunk_id": "c",
            "title": "Other",
            "source": "other.md",
            "section_title": None,
            "score": 0.7,
            "policy_type": None,
            "country": None,
            "employee_type": None,
        },
    ]
    deduped = dedupe_source_citations_for_display(citations)
    assert len(deduped) == 2
    assert deduped[0]["chunk_id"] == "b"
    assert deduped[0]["source"] == "demo.md"
    assert deduped[1]["source"] == "other.md"
