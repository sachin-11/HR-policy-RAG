"""Vector store abstractions for local and Pinecone-backed retrieval."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.rag.chunking import DocumentChunk


class VectorRecord(BaseModel):
    """A vector-store record for one document chunk."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Vector search result."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update vector records."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search records by vector similarity."""


class LocalJsonVectorStore(VectorStore):
    """Small local JSON vector store for tests and MVP development."""

    def __init__(self, directory: str | Path, file_name: str = "vectors.json") -> None:
        self.directory = Path(directory)
        self.path = self.directory / file_name
        self.directory.mkdir(parents=True, exist_ok=True)

    def upsert(self, records: list[VectorRecord]) -> None:
        existing = {record.id: record for record in self._read_records()}
        for record in records:
            existing[record.id] = record
        self._write_records(list(existing.values()))

    def replace_all(self, records: list[VectorRecord]) -> None:
        """Drop the entire local index and write only these records (full re-index)."""

        self._write_records(records)

    def record_count(self) -> int:
        """Return how many vector records are persisted on disk."""

        return len(self._read_records())

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for record in self._read_records():
            if metadata_filter and not metadata_matches(record.metadata, metadata_filter):
                continue
            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    score=cosine_similarity(query_embedding, record.embedding),
                    metadata=record.metadata,
                )
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def _read_records(self) -> list[VectorRecord]:
        if not self.path.exists():
            return []
        raw_records = json.loads(self.path.read_text(encoding="utf-8"))
        return [VectorRecord.model_validate(record) for record in raw_records]

    def _write_records(self, records: list[VectorRecord]) -> None:
        payload = [record.model_dump(mode="json") for record in records]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PineconeVectorStore(VectorStore):
    """Pinecone vector store wrapper.

    This class is optional and only imports Pinecone when used.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "default",
        dimension: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        if not api_key:
            raise ValueError("Pinecone API key is required")
        if not index_name:
            raise ValueError("Pinecone index name is required")

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError("Pinecone vector store requires the pinecone package.") from exc

        self.namespace = namespace
        self.client = Pinecone(api_key=api_key)
        listed_indexes = self.client.list_indexes()
        existing_indexes = set(listed_indexes.names()) if hasattr(listed_indexes, "names") else {
            index["name"] for index in listed_indexes
        }
        if index_name not in existing_indexes:
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self.client.Index(index_name)

    def upsert(self, records: list[VectorRecord]) -> None:
        vectors = [
            {
                "id": record.id,
                "values": record.embedding,
                "metadata": {"text": record.text, **record.metadata},
            }
            for record in records
        ]
        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=metadata_filter,
        )
        results: list[SearchResult] = []
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        for match in matches:
            match_data = match if isinstance(match, dict) else match.to_dict()
            metadata = dict(match_data.get("metadata", {}))
            text = str(metadata.pop("text", ""))
            results.append(
                SearchResult(
                    id=match_data["id"],
                    text=text,
                    score=float(match_data.get("score", 0.0)),
                    metadata=metadata,
                )
            )
        return results


def build_vector_records(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> list[VectorRecord]:
    """Build vector records from chunks and embeddings."""

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    return [
        VectorRecord(
            id=chunk.id,
            text=chunk.content,
            embedding=embedding,
            metadata=chunk.metadata.model_dump(mode="json"),
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def infer_local_store_embedding_dimension(
    directory: str | Path,
    *,
    file_name: str = "vectors.json",
) -> int | None:
    """Return embedding length from the first record in a local JSON vector store.

    Used so offline/mock query embeddings match vectors produced by OpenAI indexing.
    """

    path = Path(directory) / file_name
    if not path.exists():
        return None
    try:
        raw_records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not raw_records:
        return None
    first = raw_records[0]
    if not isinstance(first, dict):
        return None
    embedding = first.get("embedding")
    if isinstance(embedding, list) and embedding:
        return len(embedding)
    return None


def build_vector_store(
    provider: str,
    *,
    directory: str | Path,
    pinecone_api_key: str = "",
    pinecone_index_name: str = "hr-policy-assistant",
    pinecone_namespace: str = "local",
    dimension: int = 1536,
    pinecone_cloud: str = "aws",
    pinecone_region: str = "us-east-1",
) -> VectorStore:
    """Build a vector store by provider name."""

    normalized_provider = provider.lower().strip()
    if normalized_provider in {"local", "local_json", "json"}:
        return LocalJsonVectorStore(directory)
    if normalized_provider == "pinecone":
        return PineconeVectorStore(
            api_key=pinecone_api_key,
            index_name=pinecone_index_name,
            namespace=pinecone_namespace,
            dimension=dimension,
            cloud=pinecone_cloud,
            region=pinecone_region,
        )
    raise ValueError(f"Unsupported vector store provider: {provider}")


def metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return true if all expected metadata fields match exactly."""

    for key, value in expected.items():
        if value is None:
            continue
        if metadata.get(key) != value:
            return False
    return True


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity."""

    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
