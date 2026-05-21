"""Basic RAG retriever with metadata filtering and context formatting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import SearchResult, VectorStore


DEFAULT_TOP_K = 5


class RetrievalFilters(BaseModel):
    """User/document filters applied during retrieval."""

    country: str | None = None
    employee_type: str | None = None
    access_level: str | None = None
    department: str | None = None
    policy_type: str | None = None

    def to_metadata_filter(self) -> dict[str, Any]:
        """Convert non-empty fields to vector-store metadata filters."""

        return {key: value for key, value in self.model_dump().items() if value is not None}


class RetrievedChunk(BaseModel):
    """Retriever output for one relevant chunk."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source_label(self) -> str:
        """Human-readable source label for prompts and citations."""

        title = self.metadata.get("title") or "Untitled"
        source = self.metadata.get("source") or self.metadata.get("file_name") or "unknown source"
        section = self.metadata.get("section_title")
        if section:
            return f"{title} > {section} ({source})"
        return f"{title} ({source})"


class RetrievalResponse(BaseModel):
    """Full retriever response."""

    query: str
    chunks: list[RetrievedChunk]
    metadata_filter: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)


class RagRetriever:
    """Embeds user queries, searches a vector store, and formats RAG context."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | dict[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Retrieve relevant chunks for a user query."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")

        resolved_top_k = top_k or self.default_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        metadata_filter = normalize_filters(filters)
        query_embedding = self.embedding_provider.embed_text(clean_query)
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=resolved_top_k,
            metadata_filter=metadata_filter or None,
        )
        threshold = self.score_threshold if score_threshold is None else score_threshold
        chunks = [to_retrieved_chunk(result) for result in raw_results if passes_threshold(result, threshold)]

        return RetrievalResponse(query=clean_query, chunks=chunks, metadata_filter=metadata_filter)

    def retrieve_context(
        self,
        query: str,
        *,
        filters: RetrievalFilters | dict[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        max_chars: int = 4000,
    ) -> str:
        """Retrieve and format context for an LLM prompt."""

        response = self.retrieve(
            query=query,
            filters=filters,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return format_retrieved_context(response.chunks, max_chars=max_chars)


def normalize_filters(filters: RetrievalFilters | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize filter input into vector-store metadata filters."""

    if filters is None:
        return {}
    if isinstance(filters, RetrievalFilters):
        return filters.to_metadata_filter()
    return {key: value for key, value in filters.items() if value is not None}


def to_retrieved_chunk(result: SearchResult) -> RetrievedChunk:
    """Convert vector-store result into retriever output."""

    return RetrievedChunk(
        id=result.id,
        content=result.text,
        score=result.score,
        metadata=result.metadata,
    )


def passes_threshold(result: SearchResult, threshold: float | None) -> bool:
    """Return true when result score satisfies the threshold."""

    return threshold is None or result.score >= threshold


def format_retrieved_context(chunks: list[RetrievedChunk], max_chars: int = 4000) -> str:
    """Format retrieved chunks into prompt context with source labels."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if not chunks:
        return ""

    parts: list[str] = []
    current_length = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[Source {index}]\n"
            f"Title: {chunk.source_label}\n"
            f"Score: {chunk.score:.4f}\n"
            f"Content:\n{chunk.content.strip()}\n"
        )
        if current_length + len(block) > max_chars:
            break
        parts.append(block)
        current_length += len(block)

    return "\n".join(parts).strip()


def build_source_citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Build compact source citations for API responses."""

    citations: list[dict[str, Any]] = []
    for chunk in chunks:
        citations.append(
            {
                "chunk_id": chunk.id,
                "title": chunk.metadata.get("title"),
                "source": chunk.metadata.get("source"),
                "section_title": chunk.metadata.get("section_title"),
                "score": chunk.score,
                "policy_type": chunk.metadata.get("policy_type"),
                "country": chunk.metadata.get("country"),
                "employee_type": chunk.metadata.get("employee_type"),
            }
        )
    return citations


def dedupe_source_citations_for_display(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per source document for chat UI (multiple chunks often share the same file)."""

    if len(citations) <= 1:
        return list(citations)

    def doc_key(c: dict[str, Any]) -> str:
        source = (c.get("source") or "").strip()
        title = (c.get("title") or "").strip()
        if source or title:
            return f"{source}\x00{title}"
        return str(c.get("chunk_id"))

    # Highest-similarity chunk wins per document; output sorted by score descending.
    sorted_cits = sorted(citations, key=lambda c: float(c.get("score") or 0.0), reverse=True)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for citation in sorted_cits:
        key = doc_key(citation)
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out
