from pathlib import Path

import pytest

from app.rag.chunking import (
    DocumentChunk,
    chunk_document,
    chunk_documents,
    chunk_to_embedding_record,
    split_into_sections,
    split_text_with_overlap,
)
from app.rag.loaders import load_document


def test_split_into_sections_uses_markdown_headings() -> None:
    sections = split_into_sections("# Leave Policy\n\nRules.\n\n## Sick Leave\n\nDetails.")

    assert len(sections) == 2
    assert sections[0].title == "Leave Policy"
    assert sections[1].title == "Sick Leave"
    assert "Details." in sections[1].content


def test_chunk_document_preserves_metadata_and_section_titles(tmp_path: Path) -> None:
    path = tmp_path / "leave_policy.md"
    path.write_text(
        """---
title: Leave Policy India
source: leave_policy.md
department: HR
policy_type: leave
country: India
employee_type: full_time
access_level: employee
updated_at: 2026-04-01
---

# Leave Policy India

Employees can apply for annual leave.

## Sick Leave

Employees can apply for sick leave after informing their manager.
""",
        encoding="utf-8",
    )
    document = load_document(path)

    chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 2
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert chunks[0].metadata.document_id == document.id
    assert chunks[0].metadata.title == "Leave Policy India"
    assert chunks[0].metadata.policy_type == "leave"
    assert chunks[0].metadata.country == "India"
    assert chunks[0].metadata.section_title == "Leave Policy India"
    assert chunks[1].metadata.section_title == "Sick Leave"
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[1].metadata.chunk_index == 1


def test_long_section_uses_character_fallback_with_overlap(tmp_path: Path) -> None:
    path = tmp_path / "long_policy.md"
    path.write_text("# Long Policy\n\n" + ("Policy detail sentence. " * 40), encoding="utf-8")
    document = load_document(path)

    chunks = chunk_document(document, chunk_size=180, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 180 for chunk in chunks)
    assert chunks[0].metadata.section_title == "Long Policy"
    assert chunks[1].metadata.start_char < chunks[0].metadata.end_char


def test_chunk_documents_combines_multiple_documents(tmp_path: Path) -> None:
    first = tmp_path / "leave.md"
    second = tmp_path / "reimbursement.txt"
    first.write_text("# Leave\n\nLeave content.", encoding="utf-8")
    second.write_text("Reimbursement Policy\n\nExpense content.", encoding="utf-8")

    documents = [load_document(first), load_document(second)]
    chunks = chunk_documents(documents)

    assert len(chunks) == 2
    assert {chunk.metadata.file_name for chunk in chunks} == {"leave.md", "reimbursement.txt"}


def test_split_text_with_overlap_validates_empty_text() -> None:
    assert split_text_with_overlap("   ", chunk_size=100, chunk_overlap=10) == []


def test_invalid_chunk_settings_raise_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nContent.", encoding="utf-8")
    document = load_document(path)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        chunk_document(document, chunk_size=100, chunk_overlap=100)


def test_chunk_to_embedding_record_has_vector_store_shape(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nContent.", encoding="utf-8")
    chunk = chunk_document(load_document(path))[0]

    record = chunk_to_embedding_record(chunk)

    assert record["id"] == chunk.id
    assert record["text"] == chunk.content
    assert record["metadata"]["document_id"] == chunk.metadata.document_id
    assert record["metadata"]["section_title"] == "Policy"
