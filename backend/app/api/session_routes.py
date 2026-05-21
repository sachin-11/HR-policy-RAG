"""Session management API routes.

GET  /sessions/{session_id}  — fetch conversation history for a session
DELETE /sessions/{session_id} — clear (forget) a session
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.sessions.store import ConversationMessage, get_session_store


router = APIRouter(tags=["sessions"])


class SessionHistoryResponse(BaseModel):
    """Conversation history for a session."""

    session_id: str
    user_id: str | None
    messages: list[ConversationMessage]
    created_at: datetime
    updated_at: datetime
    message_count: int


class SessionClearedResponse(BaseModel):
    """Confirmation that a session was cleared."""

    session_id: str
    cleared: bool


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Return the full conversation history for the given session ID."""

    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or has expired.",
        )
    return SessionHistoryResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        messages=session.messages,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
    )


@router.delete("/sessions/{session_id}", response_model=SessionClearedResponse)
async def clear_session(session_id: str) -> SessionClearedResponse:
    """Delete a session and all its conversation history."""

    store = get_session_store()
    cleared = store.clear_session(session_id)
    return SessionClearedResponse(session_id=session_id, cleared=cleared)
