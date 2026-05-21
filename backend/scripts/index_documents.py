"""Index local HR documents into the configured vector store."""

from __future__ import annotations

from app.config import get_settings
from app.rag.indexing import run_full_index


def main() -> None:
    settings = get_settings()
    counts = run_full_index(settings)

    print(f"Loaded documents: {counts['document_count']}")
    print(f"Created chunks: {counts['chunk_count']}")
    print(f"Indexed vectors: {counts['vector_count']}")
    print(f"Vector store provider: {counts['vector_store_provider']}")


if __name__ == "__main__":
    main()
