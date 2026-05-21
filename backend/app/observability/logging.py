"""Structured JSON logging and request-scoped context."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextvars import Token
from datetime import UTC, datetime
from typing import Any

from app.config import Settings

request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return the active request correlation ID, if any."""

    return request_id_ctx.get()


def set_request_id(value: str) -> Token:
    """Bind a request ID for the current execution context."""

    return request_id_ctx.set(value)


def reset_request_id(token: Token) -> None:
    request_id_ctx.reset(token)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for ingestion by log agents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid

        message = record.getMessage()
        if message:
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                payload["message"] = message
            else:
                if isinstance(parsed, dict):
                    payload.update(parsed)
                else:
                    payload["message"] = message
        return json.dumps(payload, default=str, ensure_ascii=False)


def log_event(logger: logging.Logger, **fields: Any) -> None:
    """Log a structured payload (merged into the JSON line by JsonFormatter)."""

    logger.info(json.dumps(fields, default=str, ensure_ascii=False))


def configure_logging(settings: Settings) -> None:
    """Attach JSON logging to the ``app`` logger namespace."""

    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    app_logger = logging.getLogger("app")
    app_logger.handlers.clear()
    app_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())

    app_logger.addHandler(handler)
    app_logger.propagate = False
