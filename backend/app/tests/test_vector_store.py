from pathlib import Path
import json

import pytest

from app.rag.chunking import chunk_document
from app.rag.embeddings import MockEmbeddingProvider
from app.rag.loaders import load_document
from app.rag.vector_store import (
    LocalJsonVectorStore,
    VectorRecord,
    build_vector_records,
    build_vector_store,
    cosine_similarity,
    infer_local_store_embedding_dimension,
)


def test_build_vector_records_requires_matching_lengths(tmp_path: Path) -> None:
    path = tmp_path / "leave.md"
    path.write_text("# Leave\n\nLeave content.", encoding="utf-8")
    chunks = chunk_document(load_document(path))

    with pytest.raises(ValueError, match="same length"):
        build_vector_records(chunks, [])


def test_local_json_vector_store_upserts_and_searches_with_filter(tmp_path: Path) -> None:
    first = tmp_path / "leave.md"
    second = tmp_path / "reimbursement.md"
    first.write_text(
        """---
country: India
policy_type: leave
---
# Leave

Sick leave policy for employees.
""",
        encoding="utf-8",
    )
    second.write_text(
        """---
country: US
policy_type: reimbursement
---
# Reimbursement

Laptop reimbursement policy.
""",
        encoding="utf-8",
    )

    chunks = chunk_document(load_document(first)) + chunk_document(load_document(second))
    embedding_provider = MockEmbeddingProvider(dimension=16)
    records = build_vector_records(chunks, embedding_provider.embed_texts([chunk.content for chunk in chunks]))
    store = LocalJsonVectorStore(tmp_path / "vectors")

    store.upsert(records)
    results = store.search(
        embedding_provider.embed_text("sick leave"),
        top_k=3,
        metadata_filter={"country": "India"},
    )

    assert len(results) == 1
    assert results[0].metadata["country"] == "India"
    assert "Sick leave" in results[0].text


def test_local_json_replace_all_drops_stale_vectors(tmp_path: Path) -> None:
    store = LocalJsonVectorStore(tmp_path / "store")
    store.upsert(
        [
            VectorRecord(id="stale-1", text="old policy", embedding=[1.0, 0.0, 0.0], metadata={"source": "old.md"}),
            VectorRecord(id="stale-2", text="old policy 2", embedding=[0.0, 1.0, 0.0], metadata={}),
        ]
    )
    assert store.record_count() == 2

    store.replace_all(
        [
            VectorRecord(
                id="only-new", text="Sick leave 10 days per FAQ", embedding=[0.0, 0.0, 1.0], metadata={"source": "book.pdf"}
            ),
        ]
    )
    assert store.record_count() == 1
    remaining = store._read_records()
    assert remaining[0].id == "only-new"


def test_build_vector_store_supports_local_json(tmp_path: Path) -> None:
    store = build_vector_store("local_json", directory=tmp_path)

    assert isinstance(store, LocalJsonVectorStore)


def test_cosine_similarity_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_infer_local_store_embedding_dimension_reads_first_record(tmp_path: Path) -> None:
    store_dir = tmp_path / "vs"
    store_dir.mkdir()
    payload = [{"id": "a", "text": "x", "embedding": [0.1, 0.2, 0.3], "metadata": {}}]
    (store_dir / "vectors.json").write_text(json.dumps(payload), encoding="utf-8")

    assert infer_local_store_embedding_dimension(store_dir) == 3


def test_infer_local_store_embedding_dimension_missing_file(tmp_path: Path) -> None:
    assert infer_local_store_embedding_dimension(tmp_path / "empty") is None
