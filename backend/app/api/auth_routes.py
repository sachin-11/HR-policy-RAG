"""Auth routes — admin token generation."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.security.auth import create_access_token

router = APIRouter(tags=["auth"])


class AdminTokenRequest(BaseModel):
    password: str


class AdminTokenResponse(BaseModel):
    token: str
    expires_in: int


@router.post("/auth/admin-token", response_model=AdminTokenResponse)
async def generate_admin_token(
    body: AdminTokenRequest,
    settings: Settings = Depends(get_settings),
) -> AdminTokenResponse:
    """Return a fresh admin JWT when the correct admin password is supplied.

    Set ADMIN_PASSWORD in your .env to override the default.
    """
    if not hmac.compare_digest(body.password, settings.admin_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect admin password.",
        )

    token = create_access_token(
        {"user_id": "admin", "role": "admin"},
    )
    return AdminTokenResponse(
        token=token,
        expires_in=settings.jwt_access_token_expire_seconds,
    )
