"""Tests for observability logging and middleware."""

from __future__ import annotations

import json
import logging

from fastapi import status
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.observability.logging import JsonFormatter, configure_logging, reset_request_id, set_request_id


def test_json_formatter_merges_json_payload_log_message() -> None:
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='{"event":"sample","answer":42}',
        args=(),
        exc_info=None,
    )

    output = fmt.format(record)
    data = json.loads(output)

    assert data["event"] == "sample"
    assert data["answer"] == 42
    assert data["level"] == "INFO"


def test_json_formatter_includes_request_id_from_context() -> None:
    fmt = JsonFormatter()
    token = set_request_id("correlation-xyz")
    try:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='{"event":"has_rid"}',
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
    finally:
        reset_request_id(token)

    data = json.loads(output)

    assert data["request_id"] == "correlation-xyz"


def test_middleware_sets_response_request_id_header() -> None:
    client = TestClient(create_app(Settings(app_env="test", app_debug=False, log_level="INFO")))

    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert "x-request-id" in response.headers
    request_id = response.headers["x-request-id"]
    assert len(request_id) > 8


def test_middleware_respects_incoming_request_id_header() -> None:
    client = TestClient(create_app(Settings(app_env="test", app_debug=False, log_level="INFO")))

    response = client.get("/health", headers={"X-Request-ID": "client-provided-id"})

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["x-request-id"] == "client-provided-id"


def test_configure_logging_is_idempotent_per_process() -> None:
    settings = Settings(app_env="test", log_level="WARNING")
    configure_logging(settings)
    app_logger = logging.getLogger("app")
    handler_count = len(app_logger.handlers)
    configure_logging(settings)
    assert len(app_logger.handlers) == handler_count
