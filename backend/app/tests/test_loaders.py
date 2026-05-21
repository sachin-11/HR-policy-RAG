from pathlib import Path

import pytest

from app.rag.loaders import (
    LoadedDocument,
    document_to_source_metadata,
    load_document,
    load_documents_from_directory,
    normalize_text,
    split_front_matter,
)


def test_load_markdown_document_extracts_front_matter(tmp_path: Path) -> None:
    policy_path = tmp_path / "leave_policy.md"
    policy_path.write_text(
        """---
title: Sick Leave Policy India
source: hr_leave_policy_india_2026.md
department: HR
policy_type: leave
country: India
employee_type: full_time
access_level: employee
updated_at: 2026-04-01
---

# Sick Leave Policy India

Employees can take sick leave after informing their manager.
""",
        encoding="utf-8",
    )

    document = load_document(policy_path)

    assert isinstance(document, LoadedDocument)
    assert document.content.startswith("# Sick Leave Policy India")
    assert document.metadata.title == "Sick Leave Policy India"
    assert document.metadata.source == "hr_leave_policy_india_2026.md"
    assert document.metadata.department == "HR"
    assert document.metadata.policy_type == "leave"
    assert document.metadata.country == "India"
    assert document.metadata.employee_type == "full_time"
    assert document.metadata.access_level == "employee"
    assert document.metadata.updated_at == "2026-04-01"


def test_load_text_document_infers_title_and_policy_type(tmp_path: Path) -> None:
    policy_path = tmp_path / "reimbursement_policy.txt"
    policy_path.write_text(
        "Laptop Reimbursement Policy\n\nEmployees may request reimbursement.",
        encoding="utf-8",
    )

    document = load_document(policy_path)

    assert document.metadata.title == "Reimbursement Policy"
    assert document.metadata.policy_type == "reimbursement"
    assert document.metadata.department == "HR"
    assert "Employees may request reimbursement." in document.content


def test_load_documents_from_directory_loads_supported_files_only(tmp_path: Path) -> None:
    (tmp_path / "leave_policy.md").write_text("# Leave Policy\n\nLeave rules.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Internal FAQ\n\nUseful answer.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("not,supported", encoding="utf-8")

    documents = load_documents_from_directory(tmp_path)

    assert len(documents) == 2
    assert {doc.metadata.file_name for doc in documents} == {"leave_policy.md", "notes.txt"}


def test_split_front_matter_returns_metadata_and_body() -> None:
    body, metadata = split_front_matter(
        """---
title: Leave Policy
country: India
---

# Body
"""
    )

    assert metadata == {"title": "Leave Policy", "country": "India"}
    assert body.strip() == "# Body"


def test_normalize_text_collapses_extra_spaces_and_blank_lines() -> None:
    assert normalize_text("Hello   world\r\n\r\n\r\nNext\tline") == "Hello world\n\nNext line"


def test_document_to_source_metadata_returns_citation_fields(tmp_path: Path) -> None:
    policy_path = tmp_path / "leave_policy.md"
    policy_path.write_text("# Leave Policy\n\nLeave rules.", encoding="utf-8")
    document = load_document(policy_path)

    source_metadata = document_to_source_metadata(document)

    assert source_metadata["document_id"] == document.id
    assert source_metadata["title"] == "Leave Policy"
    assert source_metadata["source"] == "leave_policy.md"
    assert "access_level" in source_metadata


def test_unsupported_document_type_raises_clear_error(tmp_path: Path) -> None:
    unsupported = tmp_path / "policy.csv"
    unsupported.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(unsupported)
