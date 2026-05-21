"""Fake JWT-style authentication for MVP."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings

Role = Literal["anonymous", "employee", "manager", "hr", "admin"]


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("utf-8")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class AuthClaims(BaseModel):
    """User claims stored in the fake JWT token."""

    user_id: str = Field(default="anonymous")
    role: Role = Field(default="anonymous")
    country: str = Field(default="India")
    employee_type: str = Field(default="full_time")
    department: str = Field(default="HR")
    exp: int | None = None


def create_access_token(data: dict[str, Any], expires_in: int | None = None) -> str:
    """Create a fake JWT-style token for local development or tests.

    If ``expires_in`` is omitted, uses ``jwt_access_token_expire_seconds`` from settings (default 24 hours).
    """

    settings = get_settings()
    ttl = settings.jwt_access_token_expire_seconds if expires_in is None else expires_in
    header = {"alg": settings.jwt_algorithm, "typ": "JWT"}
    now = datetime.now(timezone.utc)
    payload = {
        **data,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl)).timestamp()),
    }
    encoded_header = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    signature = hmac.new(
        settings.jwt_secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    encoded_signature = _base64url_encode(signature)
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def decode_access_token(token: str) -> AuthClaims:
    """Decode and validate a fake JWT-style token."""

    settings = get_settings()
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
    except ValueError as exc:
        raise ValueError("Invalid token format.") from exc

    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    expected_signature = hmac.new(
        settings.jwt_secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    actual_signature = _base64url_decode(encoded_signature)
    if not hmac.compare_digest(expected_signature, actual_signature):
        raise ValueError("Invalid token signature.")

    try:
        payload = json.loads(_base64url_decode(encoded_payload).decode("utf-8"))
    except ValueError as exc:
        raise ValueError("Invalid token payload.") from exc

    exp = payload.get("exp")
    if exp is not None and int(datetime.now(timezone.utc).timestamp()) > int(exp):
        raise ValueError("Token has expired.")

    return AuthClaims.model_validate(payload)


def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuthClaims:
    """Resolve the current user from an Authorization header token."""

    if authorization is None:
        return AuthClaims()

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer token.",
        )

    token = authorization[len("Bearer ") :].strip()
    try:
        return decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
        )


def get_authenticated_user(user: AuthClaims = Depends(get_current_user)) -> AuthClaims:
    """Require an authenticated user instead of anonymous access."""

    if user.role == "anonymous" or user.user_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required for this operation.",
        )
    return user


def get_admin_user(user: AuthClaims = Depends(get_current_user)) -> AuthClaims:
    """Require an admin role for document management endpoints."""

    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges are required for this operation.",
        )
    return user
