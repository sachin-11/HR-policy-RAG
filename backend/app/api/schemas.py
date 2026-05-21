"""Shared API schemas."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    """Standard error payload returned by the API."""

    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None)


class ErrorResponse(BaseModel):
    """Envelope for API errors."""

    error: ApiError


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    environment: str
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MessageResponse(BaseModel):
    """Generic message response for simple endpoints."""

    message: str
