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


class SessionInfo(BaseModel):
    """Metadata/summary of a conversation session for the sidebar."""

    session_id: str
    updated_at: datetime
    message_count: int
    title: str


class SessionListResponse(BaseModel):
    """List of all active conversation sessions."""

    sessions: list[SessionInfo]


def get_session_title(messages: list[ConversationMessage]) -> str:
    """Helper to extract a title from the first user message."""
    for msg in messages:
        if msg.role == "user":
            content = msg.content.strip()
            if len(content) > 30:
                return content[:27] + "..."
            return content
    return "New Chat"


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """Return all active conversation sessions sorted by updated_at descending."""
    store = get_session_store()
    sessions = store.get_all_sessions()
    
    session_infos = []
    for s in sessions:
        title = get_session_title(s.messages)
        session_infos.append(
            SessionInfo(
                session_id=s.session_id,
                updated_at=s.updated_at,
                message_count=len(s.messages),
                title=title,
            )
        )
    return SessionListResponse(sessions=session_infos)



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
