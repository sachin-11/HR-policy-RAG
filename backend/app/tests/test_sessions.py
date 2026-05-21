"""Tests for the session store and session API routes."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.sessions.store import (
    SESSION_TTL_SECONDS,
    ConversationSession,
    SessionStore,
    get_session_store,
)


# ── SessionStore unit tests ────────────────────────────────────────────────────


def test_create_and_get_session():
    store = SessionStore()
    session = store.create_session(user_id="u1")
    assert session.user_id == "u1"
    assert session.session_id

    fetched = store.get_session(session.session_id)
    assert fetched is not None
    assert fetched.session_id == session.session_id


def test_get_unknown_session_returns_none():
    store = SessionStore()
    assert store.get_session("does-not-exist") is None


def test_get_or_create_uses_existing():
    store = SessionStore()
    session = store.create_session(user_id="u2")
    same = store.get_or_create(session.session_id, user_id="u2")
    assert same.session_id == session.session_id


def test_get_or_create_makes_new_when_missing():
    store = SessionStore()
    session = store.get_or_create(None, user_id="u3")
    assert session.session_id
    assert store.session_count == 1


def test_add_turn_appends_messages():
    store = SessionStore()
    session = store.create_session()
    store.add_turn(session.session_id, "What is sick leave?", "You get 10 days per year.")
    updated = store.get_session(session.session_id)
    assert updated is not None
    assert len(updated.messages) == 2
    assert updated.messages[0].role == "user"
    assert updated.messages[1].role == "assistant"


def test_add_turn_on_missing_session_is_noop():
    store = SessionStore()
    store.add_turn("ghost-session", "hello", "world")  # must not raise


def test_clear_session_returns_true_when_exists():
    store = SessionStore()
    session = store.create_session()
    assert store.clear_session(session.session_id) is True
    assert store.get_session(session.session_id) is None


def test_clear_session_returns_false_when_missing():
    store = SessionStore()
    assert store.clear_session("never-existed") is False


def test_max_history_trim():
    session = ConversationSession(session_id="s1")
    for i in range(12):
        session.add_message("user", f"question {i}")
        session.add_message("assistant", f"answer {i}")
    # MAX_HISTORY_TURNS=10 → 20 messages max
    assert len(session.messages) == 20


def test_to_prompt_block_empty():
    session = ConversationSession(session_id="s1")
    assert session.to_prompt_block() == ""


def test_to_prompt_block_formatted():
    session = ConversationSession(session_id="s1")
    session.add_message("user", "What is leave?")
    session.add_message("assistant", "You get 10 days.")
    block = session.to_prompt_block()
    assert "Conversation history:" in block
    assert "User: What is leave?" in block
    assert "Assistant: You get 10 days." in block


def test_purge_expired(monkeypatch):
    store = SessionStore()
    session = store.create_session()
    # Force the session to appear expired
    monkeypatch.setattr(
        "app.sessions.store.SESSION_TTL_SECONDS", -1, raising=False
    )
    # Rebuild with patched TTL by manipulating updated_at directly
    from datetime import UTC, datetime, timedelta
    store._sessions[session.session_id].updated_at = datetime.now(UTC) - timedelta(seconds=SESSION_TTL_SECONDS + 1)
    removed = store.purge_expired()
    assert removed == 1
    assert store.session_count == 0


# ── Session API route tests ────────────────────────────────────────────────────


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def seeded_session():
    """Create a real session with one turn and return its session_id."""
    store = get_session_store()
    session = store.create_session(user_id="test-user")
    store.add_turn(session.session_id, "What is maternity leave?", "26 weeks paid.")
    return session.session_id


def test_get_session_history_ok(client, seeded_session):
    resp = client.get(f"/sessions/{seeded_session}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == seeded_session
    assert data["message_count"] == 2
    assert data["messages"][0]["role"] == "user"


def test_get_session_history_not_found(client):
    resp = client.get("/sessions/no-such-session")
    assert resp.status_code == 404


def test_delete_session_ok(client, seeded_session):
    resp = client.delete(f"/sessions/{seeded_session}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cleared"] is True

    # Session should now be gone
    resp2 = client.get(f"/sessions/{seeded_session}")
    assert resp2.status_code == 404


def test_delete_session_not_found(client):
    resp = client.delete("/sessions/ghost-session-xyz")
    assert resp.status_code == 200
    assert resp.json()["cleared"] is False


def test_versioned_session_route(client, seeded_session):
    resp = client.get(f"/api/v1/sessions/{seeded_session}")
    assert resp.status_code == 200
