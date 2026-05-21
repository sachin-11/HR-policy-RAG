"""Embedding providers for RAG chunks."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


DEFAULT_MOCK_EMBEDDING_DIMENSION = 64


class EmbeddingProvider(ABC):
    """Abstract embedding provider used by indexing and retrieval."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""

    def embed_text(self, text: str) -> list[float]:
        """Embed one text."""

        return self.embed_texts([text])[0]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embeddings for tests and offline development."""

    def __init__(self, dimension: int = DEFAULT_MOCK_EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [normalize_vector(hash_text_to_vector(text, self.dimension)) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider.

    This import is lazy so tests do not need OpenAI installed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAIEmbeddingProvider")
        self.api_key = api_key
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI embeddings require the openai package.") from exc

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def build_embedding_provider(
    provider: str,
    *,
    openai_api_key: str = "",
    openai_embedding_model: str = "text-embedding-3-small",
    mock_dimension: int | None = None,
) -> EmbeddingProvider:
    """Build an embedding provider by name."""

    normalized_provider = provider.lower().strip()
    if normalized_provider in {"mock", "local", "local_mock"}:
        dimension = mock_dimension if mock_dimension is not None else DEFAULT_MOCK_EMBEDDING_DIMENSION
        return MockEmbeddingProvider(dimension=dimension)
    if normalized_provider == "openai":
        return OpenAIEmbeddingProvider(api_key=openai_api_key, model=openai_embedding_model)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def hash_text_to_vector(text: str, dimension: int) -> list[float]:
    """Convert text into a deterministic pseudo-random vector."""

    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimension:
                break
        counter += 1
    return values


def normalize_vector(vector: list[float]) -> list[float]:
    """L2-normalize a vector."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
