"""Admin document upload, listing, and reindexing API routes."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.rag.indexing import clear_cached_retriever, run_full_index
from app.rag.loaders import SUPPORTED_EXTENSIONS
from app.rag.vector_store import LocalJsonVectorStore
from app.security.auth import AuthClaims, get_admin_user

router = APIRouter(tags=["documents"])

_INDEX_LOCK = threading.Lock()
_reindex_running = False

STATUS_FILENAME = "indexing_status.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class DocumentListItem(BaseModel):
    """One file in the raw document directory."""

    name: str
    path: str
    size_bytes: int
    modified_at: str


class DocumentListResponse(BaseModel):
    """List of ingested source files."""

    documents: list[DocumentListItem]
    raw_docs_dir: str


class IndexingStatusPayload(BaseModel):
    """Persisted and live indexing job status for the admin UI."""

    job_state: str = Field(default="idle", description="idle | running | success | failed")
    started_at: str | None = None
    finished_at: str | None = None
    message: str = ""
    error_detail: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    vector_count: int | None = None
    vector_store_provider: str = ""
    embedding_provider: str = ""


class UploadResponse(BaseModel):
    """Response after uploading a raw document."""

    message: str
    file_name: str
    size_bytes: int


class ReindexQueuedResponse(BaseModel):
    """Acknowledgement when a background indexing job is scheduled."""

    message: str
    job_state: str = "running"


def _status_path(settings: Settings) -> Path:
    return Path(settings.processed_data_dir) / STATUS_FILENAME


def _read_status(settings: Settings) -> IndexingStatusPayload:
    path = _status_path(settings)
    if not path.exists():
        return IndexingStatusPayload()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return IndexingStatusPayload.model_validate(raw)
    except (json.JSONDecodeError, ValueError):
        return IndexingStatusPayload(message="Could not read indexing status file; showing defaults.")


def _write_status(settings: Settings, payload: IndexingStatusPayload) -> None:
    path = _status_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _live_vector_count(settings: Settings) -> int | None:
    normalized = settings.vector_store_provider.lower().strip()
    if normalized not in {"local", "local_json", "json"}:
        return None
    directory = Path(settings.vector_store_dir)
    store = LocalJsonVectorStore(directory)
    return store.record_count()


def _list_raw_documents(settings: Settings) -> list[DocumentListItem]:
    root = Path(settings.raw_docs_dir)
    if not root.exists():
        return []

    items: list[DocumentListItem] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        items.append(
            DocumentListItem(
                name=path.name,
                path=relative,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            )
        )

    return items


def _safe_filename(upload_name: str | None) -> str:
    if not upload_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A file name is required.")
    candidate = Path(upload_name).name
    if not candidate or candidate in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file name.")
    return candidate


def _execute_reindex(settings: Settings) -> None:
    global _reindex_running

    started_at = datetime.now(UTC).isoformat()
    _write_status(
        settings,
        IndexingStatusPayload(
            job_state="running",
            started_at=started_at,
            finished_at=None,
            message="Indexing in progress...",
            error_detail=None,
        ),
    )
    try:
        counts = run_full_index(settings)
        clear_cached_retriever()
        finished_at = datetime.now(UTC).isoformat()
        _write_status(
            settings,
            IndexingStatusPayload(
                job_state="success",
                started_at=started_at,
                finished_at=finished_at,
                message="Indexing completed successfully.",
                error_detail=None,
                document_count=counts["document_count"],
                chunk_count=counts["chunk_count"],
                vector_count=counts["vector_count"],
                vector_store_provider=str(counts["vector_store_provider"]),
                embedding_provider=str(counts["embedding_provider"]),
            ),
        )
    except Exception as exc:  # pragma: no cover - broad catch logs operator-facing errors
        finished_at = datetime.now(UTC).isoformat()
        _write_status(
            settings,
            IndexingStatusPayload(
                job_state="failed",
                started_at=started_at,
                finished_at=finished_at,
                message="Indexing failed.",
                error_detail=str(exc),
            ),
        )
    finally:
        with _INDEX_LOCK:
            _reindex_running = False


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    _: AuthClaims = Depends(get_admin_user),
    settings: Settings = Depends(get_settings),
) -> DocumentListResponse:
    """List discoverable HR policy files from the configured raw document directory."""

    documents = _list_raw_documents(settings)
    return DocumentListResponse(documents=documents, raw_docs_dir=settings.raw_docs_dir)


@router.get("/documents/indexing-status", response_model=IndexingStatusPayload)
async def indexing_status(
    _: AuthClaims = Depends(get_admin_user),
    settings: Settings = Depends(get_settings),
) -> IndexingStatusPayload:
    """Return the latest indexing job status, enriched with live vector counts when available."""

    payload = _read_status(settings)
    if payload.job_state != "running":
        live_count = _live_vector_count(settings)
        if live_count is not None:
            payload = payload.model_copy(update={"vector_count": live_count})
    return payload


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    _: AuthClaims = Depends(get_admin_user),
    settings: Settings = Depends(get_settings),
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload a new policy document into the raw document directory."""

    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    raw_root = Path(settings.raw_docs_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    destination = raw_root / filename

    total_size = 0
    chunk = await file.read(MAX_UPLOAD_BYTES + 1)
    total_size += len(chunk)
    if total_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES} bytes.",
        )

    destination.write_bytes(chunk)

    return UploadResponse(
        message="Upload stored. Run re-index to embed new content.",
        file_name=filename,
        size_bytes=total_size,
    )


@router.post("/documents/reindex", response_model=ReindexQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_reindex(
    background_tasks: BackgroundTasks,
    _: AuthClaims = Depends(get_admin_user),
    settings: Settings = Depends(get_settings),
) -> ReindexQueuedResponse:
    """Chunk, embed, and upsert all raw documents in the background."""

    global _reindex_running

    with _INDEX_LOCK:
        if _reindex_running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An indexing job is already running.",
            )
        _reindex_running = True

    background_tasks.add_task(_execute_reindex, settings)
    return ReindexQueuedResponse(message="Indexing job started.", job_state="running")
