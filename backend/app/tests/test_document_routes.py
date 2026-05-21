"""Tests for admin document routes."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.security.auth import create_access_token


def _admin_headers() -> dict[str, str]:
    token = create_access_token({"user_id": "admin-user", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


def _build_client(tmp_path: Path) -> TestClient:
    raw_dir = tmp_path / "raw_docs"
    processed_dir = tmp_path / "processed"
    vector_dir = processed_dir / "vector_store"
    raw_dir.mkdir()
    processed_dir.mkdir()

    settings = Settings(
        app_env="test",
        app_debug=False,
        raw_docs_dir=str(raw_dir),
        processed_data_dir=str(processed_dir),
        vector_store_dir=str(vector_dir),
        openai_api_key="",
        vector_store_provider="local_json",
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_documents_requires_admin(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/api/v1/documents")

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_list_documents_returns_uploaded_files(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    raw_dir = tmp_path / "raw_docs"
    (raw_dir / "sample.txt").write_text("Hello policy world.", encoding="utf-8")

    response = client.get("/api/v1/documents", headers=_admin_headers())

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["raw_docs_dir"] == str(raw_dir)
    assert len(body["documents"]) == 1
    assert body["documents"][0]["name"] == "sample.txt"
    assert body["documents"][0]["size_bytes"] > 0


def test_upload_and_reindex_flow(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    files = {"file": ("note.txt", b"Remote work rules for IT.", "text/plain")}
    upload = client.post("/api/v1/documents/upload", headers=_admin_headers(), files=files)
    assert upload.status_code == status.HTTP_200_OK
    assert upload.json()["file_name"] == "note.txt"

    reindex = client.post("/api/v1/documents/reindex", headers=_admin_headers())
    assert reindex.status_code == status.HTTP_202_ACCEPTED

    status_response = client.get("/api/v1/documents/indexing-status", headers=_admin_headers())
    assert status_response.status_code == status.HTTP_200_OK
    status_body = status_response.json()
    assert status_body["job_state"] == "success"
    assert status_body["document_count"] >= 1
    assert status_body["chunk_count"] >= 1
    assert status_body["vector_count"] is not None
