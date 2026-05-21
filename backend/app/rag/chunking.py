"""Chunk loaded HR documents into retrieval-ready units."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.rag.loaders import LoadedDocument


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


class ChunkMetadata(BaseModel):
    """Metadata attached to every retrieval chunk."""

    document_id: str
    chunk_index: int
    source: str
    title: str
    file_name: str
    file_path: str
    file_type: str
    section_title: str | None = None
    policy_type: str | None = None
    department: str | None = None
    country: str | None = None
    employee_type: str | None = None
    access_level: str = "employee"
    updated_at: str | None = None
    start_char: int = 0
    end_char: int = 0


class DocumentChunk(BaseModel):
    """A document chunk ready for embedding and retrieval."""

    id: str
    content: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class Section:
    """Internal section representation used before fallback splitting."""

    title: str | None
    content: str
    start_char: int
    end_char: int


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk multiple loaded documents."""

    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk a loaded document using sections first, then fallback splitting."""

    validate_chunk_params(chunk_size, chunk_overlap)

    sections = split_into_sections(document.content)
    chunks: list[DocumentChunk] = []

    for section in sections:
        pieces = split_text_with_overlap(section.content, chunk_size, chunk_overlap)
        section_offset = 0

        for piece in pieces:
            relative_start = section.content.find(piece, section_offset)
            if relative_start == -1:
                relative_start = section_offset
            relative_end = relative_start + len(piece)
            section_offset = max(relative_start + 1, relative_end - chunk_overlap)

            chunk_index = len(chunks)
            start_char = section.start_char + relative_start
            end_char = section.start_char + relative_end
            chunks.append(
                DocumentChunk(
                    id=build_chunk_id(document.id, chunk_index, piece),
                    content=piece,
                    metadata=build_chunk_metadata(
                        document=document,
                        chunk_index=chunk_index,
                        section_title=section.title,
                        start_char=start_char,
                        end_char=end_char,
                    ),
                )
            )

    return chunks


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk sizing settings."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def split_into_sections(content: str) -> list[Section]:
    """Split Markdown-like text into sections by headings.

    If no headings are present, the whole document becomes one section. This
    keeps plain text and PDF output compatible with the same chunking pipeline.
    """

    matches = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", content))
    if not matches:
        return [Section(title=None, content=content.strip(), start_char=0, end_char=len(content.strip()))]

    sections: list[Section] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section_content = content[start:end].strip()
        if not section_content:
            continue
        sections.append(
            Section(
                title=match.group(1).strip(),
                content=section_content,
                start_char=start,
                end_char=end,
            )
        )
    return sections


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into character chunks while preferring paragraph boundaries."""

    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        target_end = min(start + chunk_size, len(clean_text))
        end = find_best_split(clean_text, start, target_end)
        piece = clean_text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean_text):
            break
        start = max(0, end - chunk_overlap)
        while start < len(clean_text) and clean_text[start].isspace():
            start += 1

    return chunks


def find_best_split(text: str, start: int, target_end: int) -> int:
    """Find a split point near target_end, preferring paragraph/sentence boundaries."""

    if target_end >= len(text):
        return len(text)

    window = text[start:target_end]
    for separator in ("\n\n", "\n", ". ", "; ", ", "):
        index = window.rfind(separator)
        if index >= max(1, int(len(window) * 0.45)):
            return start + index + len(separator.rstrip())

    return target_end


def build_chunk_metadata(
    document: LoadedDocument,
    chunk_index: int,
    section_title: str | None,
    start_char: int,
    end_char: int,
) -> ChunkMetadata:
    """Build chunk metadata while preserving document metadata."""

    metadata = document.metadata
    return ChunkMetadata(
        document_id=document.id,
        chunk_index=chunk_index,
        source=metadata.source,
        title=metadata.title,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        file_type=metadata.file_type,
        section_title=section_title,
        policy_type=metadata.policy_type,
        department=metadata.department,
        country=metadata.country,
        employee_type=metadata.employee_type,
        access_level=metadata.access_level,
        updated_at=metadata.updated_at,
        start_char=start_char,
        end_char=end_char,
    )


def build_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Build a stable chunk ID."""

    digest = hashlib.sha256(f"{document_id}:{chunk_index}:{content}".encode("utf-8")).hexdigest()
    return f"{document_id}-{chunk_index}-{digest[:10]}"


def chunk_to_embedding_record(chunk: DocumentChunk) -> dict[str, Any]:
    """Convert a chunk into a simple vector-store friendly record."""

    return {
        "id": chunk.id,
        "text": chunk.content,
        "metadata": chunk.metadata.model_dump(mode="json"),
    }
