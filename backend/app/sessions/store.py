"""In-memory conversation session store.

Stores per-session message history in a thread-safe dict.
Sessions expire after SESSION_TTL_SECONDS of inactivity.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


SESSION_TTL_SECONDS = 7200  # 2 hours
MAX_HISTORY_TURNS = 10      # keep last 10 user/assistant pairs = 20 messages


Role = Literal["user", "assistant"]


class ConversationMessage(BaseModel):
    """One turn in a conversation."""

    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationSession(BaseModel):
    """Full session state for one user conversation."""

    session_id: str
    user_id: str | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append(ConversationMessage(role=role, content=content))
        self.updated_at = datetime.now(UTC)
        # Keep only the last MAX_HISTORY_TURNS pairs
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def is_expired(self) -> bool:
        elapsed = (datetime.now(UTC) - self.updated_at).total_seconds()
        return elapsed > SESSION_TTL_SECONDS

    def to_prompt_block(self) -> str:
        """Format history as a readable block for LLM context."""
        if not self.messages:
            return ""
        lines = ["Conversation history:"]
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)


class SessionStore:
    """Thread-safe in-memory store for conversation sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str | None = None) -> ConversationSession:
        """Create a new session and return it."""
        session = ConversationSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Return the session or None if not found / expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired():
                del self._sessions[session_id]
                return None
            return session

    def get_or_create(self, session_id: str | None, user_id: str | None = None) -> ConversationSession:
        """Return existing session or create a new one."""
        if session_id:
            session = self.get_session(session_id)
            if session is not None:
                return session
        return self.create_session(user_id=user_id)

    def add_turn(self, session_id: str, user_message: str, assistant_answer: str) -> None:
        """Append a user+assistant message pair to the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired():
                return
            session.add_message("user", user_message)
            session.add_message("assistant", assistant_answer)

    def clear_session(self, session_id: str) -> bool:
        """Delete session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def get_all_sessions(self) -> list[ConversationSession]:
        """Return all active (non-expired) sessions sorted by updated_at descending."""
        with self._lock:
            # Purge expired first to ensure we only return active ones
            expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
            for sid in expired:
                del self._sessions[sid]
            
            return sorted(
                self._sessions.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )

    def purge_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Module-level singleton shared across the app process.
_store: SessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Return the process-wide SessionStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore()
    return _store
