"""Local document loaders for HR policy RAG ingestion."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


class DocumentMetadata(BaseModel):
    """Metadata used later for permissions, filtering, citations, and audits."""

    source: str
    title: str
    file_name: str
    file_path: str
    file_type: str
    policy_type: str | None = None
    department: str | None = None
    country: str | None = None
    employee_type: str | None = None
    access_level: str = "employee"
    updated_at: str | None = None
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LoadedDocument(BaseModel):
    """Normalized document object returned by all loaders."""

    id: str
    content: str
    metadata: DocumentMetadata


def load_documents_from_directory(directory: str | Path) -> list[LoadedDocument]:
    """Load every supported local document from a directory recursively."""

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    documents: list[LoadedDocument] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_document(path))
    return documents


def load_document(path: str | Path) -> LoadedDocument:
    """Load one supported document file."""

    document_path = Path(path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document file does not exist: {document_path}")
    if not document_path.is_file():
        raise IsADirectoryError(f"Expected a file, got directory: {document_path}")

    suffix = document_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        raw_content = document_path.read_text(encoding="utf-8")
        content, front_matter = split_front_matter(raw_content)
    elif suffix == ".txt":
        content = document_path.read_text(encoding="utf-8")
        front_matter = {}
    elif suffix == ".pdf":
        content = load_pdf_text(document_path)
        front_matter = {}
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

    normalized_content = normalize_text(content)
    metadata = build_metadata(document_path, normalized_content, front_matter)
    document_id = build_document_id(document_path, normalized_content)

    return LoadedDocument(id=document_id, content=normalized_content, metadata=metadata)


def load_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF loading requires pypdf. Install backend requirements first.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def split_front_matter(content: str) -> tuple[str, dict[str, str]]:
    """Split simple YAML-like Markdown front matter.

    This intentionally supports only `key: value` pairs because we only need
    stable metadata for local learning docs right now.
    """

    if not content.startswith("---"):
        return content, {}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return content, {}

    metadata_text = parts[1]
    body = parts[2]
    metadata: dict[str, str] = {}

    for line in metadata_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        clean_key = key.strip().lower().replace("-", "_")
        clean_value = value.strip().strip('"').strip("'")
        if clean_key and clean_value:
            metadata[clean_key] = clean_value

    return body, metadata


def build_metadata(path: Path, content: str, front_matter: dict[str, str]) -> DocumentMetadata:
    """Build document metadata from front matter and file information."""

    title = front_matter.get("title") or infer_title(content, path)
    policy_type = front_matter.get("policy_type") or infer_policy_type(path)
    department = front_matter.get("department") or "HR"

    return DocumentMetadata(
        source=front_matter.get("source") or path.name,
        title=title,
        file_name=path.name,
        file_path=str(path),
        file_type=path.suffix.lower().lstrip("."),
        policy_type=policy_type,
        department=department,
        country=front_matter.get("country"),
        employee_type=front_matter.get("employee_type"),
        access_level=front_matter.get("access_level", "employee"),
        updated_at=front_matter.get("updated_at"),
    )


def infer_title(content: str, path: Path) -> str:
    """Infer document title from the first Markdown heading or filename."""

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def infer_policy_type(path: Path) -> str | None:
    """Infer a coarse policy type from filename keywords."""

    name = path.stem.lower()
    keyword_map = {
        "leave": "leave",
        "sick": "leave",
        "maternity": "leave",
        "insurance": "benefits",
        "benefit": "benefits",
        "reimbursement": "reimbursement",
        "expense": "reimbursement",
        "onboarding": "onboarding",
        "work_from_home": "work_from_home",
        "wfh": "work_from_home",
    }
    for keyword, policy_type in keyword_map.items():
        if keyword in name:
            return policy_type
    return None


def normalize_text(content: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = content.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_document_id(path: Path, content: str) -> str:
    """Build a stable document ID from path and content."""

    digest = hashlib.sha256(f"{path.as_posix()}::{content}".encode("utf-8")).hexdigest()
    return digest[:16]


def document_to_source_metadata(document: LoadedDocument) -> dict[str, Any]:
    """Return citation-friendly metadata for later API responses."""

    metadata = document.metadata
    return {
        "document_id": document.id,
        "title": metadata.title,
        "source": metadata.source,
        "policy_type": metadata.policy_type,
        "department": metadata.department,
        "country": metadata.country,
        "employee_type": metadata.employee_type,
        "access_level": metadata.access_level,
        "updated_at": metadata.updated_at,
    }
