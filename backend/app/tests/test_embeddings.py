import pytest

from app.rag.embeddings import MockEmbeddingProvider, build_embedding_provider, normalize_vector


def test_mock_embedding_provider_returns_deterministic_vectors() -> None:
    provider = MockEmbeddingProvider(dimension=8)

    first = provider.embed_text("leave policy")
    second = provider.embed_text("leave policy")

    assert first == second
    assert len(first) == 8


def test_mock_embedding_provider_embeds_multiple_texts() -> None:
    provider = MockEmbeddingProvider(dimension=6)

    embeddings = provider.embed_texts(["leave", "reimbursement"])

    assert len(embeddings) == 2
    assert all(len(embedding) == 6 for embedding in embeddings)
    assert embeddings[0] != embeddings[1]


def test_build_embedding_provider_supports_mock() -> None:
    provider = build_embedding_provider("mock")

    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimension == 64


def test_build_embedding_provider_mock_accepts_dimension_override() -> None:
    provider = build_embedding_provider("mock", mock_dimension=1536)

    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimension == 1536


def test_build_openai_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        build_embedding_provider("openai")


def test_normalize_vector_handles_zero_vector() -> None:
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
