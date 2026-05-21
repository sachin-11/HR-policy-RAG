"""Example command to retrieve formatted RAG context from the local vector store."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.rag.embeddings import build_embedding_provider
from app.rag.retriever import RagRetriever, RetrievalFilters
from app.rag.vector_store import build_vector_store, infer_local_store_embedding_dimension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve HR policy context for a query.")
    parser.add_argument("query", help="User query to retrieve context for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--country", default=None)
    parser.add_argument("--employee-type", default=None)
    parser.add_argument("--access-level", default=None)
    parser.add_argument("--policy-type", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    embedding_provider_name = "openai" if settings.openai_api_key else "mock"
    vector_store_dir = Path(settings.vector_store_dir)
    mock_dimension: int | None = None
    if embedding_provider_name == "mock" and settings.vector_store_provider.lower().strip() in {
        "local",
        "local_json",
        "json",
    }:
        mock_dimension = infer_local_store_embedding_dimension(vector_store_dir)
    embedding_provider = build_embedding_provider(
        embedding_provider_name,
        openai_api_key=settings.openai_api_key,
        openai_embedding_model=settings.openai_embedding_model,
        mock_dimension=mock_dimension,
    )
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
    retriever = RagRetriever(embedding_provider=embedding_provider, vector_store=vector_store)
    context = retriever.retrieve_context(
        args.query,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        filters=RetrievalFilters(
            country=args.country,
            employee_type=args.employee_type,
            access_level=args.access_level,
            policy_type=args.policy_type,
        ),
    )

    if not context:
        print("No relevant context found.")
        return
    print(context)


if __name__ == "__main__":
    main()
