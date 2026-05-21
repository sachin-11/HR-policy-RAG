"""Full-document indexing pipeline shared by CLI scripts and admin APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.rag.chunking import chunk_documents
from app.rag.embeddings import build_embedding_provider
from app.rag.loaders import load_documents_from_directory
from app.rag.vector_store import LocalJsonVectorStore, build_vector_records, build_vector_store


def run_full_index(settings: Settings) -> dict[str, Any]:
    """Load raw docs, chunk, embed, and upsert into the configured vector store.

    Returns counts and provider metadata for observability and admin status UIs.
    """

    raw_docs_dir = Path(settings.raw_docs_dir)
    vector_store_dir = Path(settings.vector_store_dir)

    documents = load_documents_from_directory(raw_docs_dir)
    chunks = chunk_documents(documents)

    embedding_provider_name = "openai" if settings.openai_api_key else "mock"
    embedding_provider = build_embedding_provider(
        embedding_provider_name,
        openai_api_key=settings.openai_api_key,
        openai_embedding_model=settings.openai_embedding_model,
    )
    embeddings = embedding_provider.embed_texts([chunk.content for chunk in chunks])
    records = build_vector_records(chunks, embeddings)

    vector_store = build_vector_store(
        settings.vector_store_provider,
        directory=vector_store_dir,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_index_name=settings.pinecone_index_name,
        pinecone_namespace=settings.pinecone_namespace,
        dimension=embedding_provider.dimension,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )
    if isinstance(vector_store, LocalJsonVectorStore):
        vector_store.replace_all(records)
    else:
        vector_store.upsert(records)

    if isinstance(vector_store, LocalJsonVectorStore):
        total_vectors = vector_store.record_count()
    else:
        total_vectors = len(records)

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "vector_count": total_vectors,
        "embedding_provider": embedding_provider_name,
        "vector_store_provider": settings.vector_store_provider,
    }


def clear_cached_retriever() -> None:
    """Invalidate the cached RAG retriever so new vectors are visible to /chat."""

    from app.api.chat_routes import get_default_retriever

    get_default_retriever.cache_clear()
