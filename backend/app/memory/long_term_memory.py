"""Long-term user memory — persists facts about users across sessions.

Stores two kinds of memory per user:
  - preferences  : things the user has told us (language, department, leave type)
  - interactions : summary of past significant interactions (emails sent, tickets raised)

Storage: JSON file per user in MEMORY_DIR (default ./data/memory/).
In production, swap the JSON backend for Redis or a DB table.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.observability.logging import log_event

logger = logging.getLogger("app.memory")

MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "./data/memory"))
MAX_INTERACTIONS = 20  # keep last N interactions per user


def _user_path(user_id: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    return MEMORY_DIR / f"{safe_id}.json"


def _load(user_id: str) -> dict[str, Any]:
    path = _user_path(user_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"user_id": user_id, "preferences": {}, "interactions": []}


def _save(user_id: str, data: dict[str, Any]) -> None:
    path = _user_path(user_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── public API ────────────────────────────────────────────────────────────────


def get_user_memory(user_id: str) -> dict[str, Any]:
    """Return the full memory record for a user."""
    return _load(user_id)


def update_preference(user_id: str, key: str, value: Any) -> None:
    """Store or update a user preference (e.g. language='hindi', leave_type='sick')."""
    data = _load(user_id)
    data["preferences"][key] = value
    _save(user_id, data)
    log_event(logger, event="memory.preference.updated", user_id=user_id, key=key)


def record_interaction(
    user_id: str,
    *,
    action: str,
    summary: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a significant interaction to the user's history.

    Examples:
        record_interaction(uid, action="email_sent", summary="Sent sick leave email to manager")
        record_interaction(uid, action="ticket_created", summary="Raised HR ticket #HR-001")
    """
    data = _load(user_id)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "summary": summary,
        **(metadata or {}),
    }
    interactions: list = data.setdefault("interactions", [])
    interactions.append(entry)
    # Keep only the most recent MAX_INTERACTIONS
    data["interactions"] = interactions[-MAX_INTERACTIONS:]
    _save(user_id, data)
    log_event(logger, event="memory.interaction.recorded", user_id=user_id, action=action)


def build_memory_context(user_id: str) -> str:
    """Return a compact text block injected into the RAG prompt as user context."""
    if not user_id:
        return ""
    data = _load(user_id)
    prefs = data.get("preferences", {})
    interactions = data.get("interactions", [])

    lines: list[str] = []
    if prefs:
        pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items())
        lines.append(f"User preferences: {pref_str}")
    if interactions:
        recent = interactions[-3:]  # last 3 interactions for context
        lines.append("Recent interactions:")
        for i in recent:
            lines.append(f"  - [{i['ts'][:10]}] {i['summary']}")

    return "\n".join(lines)


def extract_and_save_preferences(
    user_id: str,
    user_message: str,
    llm_client: Any | None = None,
) -> None:
    """Use LLM (or heuristics) to extract preferences from the user's message and save them."""
    if not user_id:
        return

    msg = user_message.lower()

    # Heuristic: detect language preference
    hindi_markers = ("hindi", "हिंदी", "mujhe", "mera", "karo", "chahiye", "aap")
    if any(m in msg for m in hindi_markers):
        update_preference(user_id, "preferred_language", "hindi")

    # Heuristic: detect leave type preference
    if "sick" in msg or "fever" in msg or "bukhar" in msg or "ill" in msg:
        update_preference(user_id, "last_leave_type", "sick")
    elif "casual" in msg or "personal" in msg:
        update_preference(user_id, "last_leave_type", "casual")
    elif "maternity" in msg or "paternity" in msg:
        update_preference(user_id, "last_leave_type", "maternity/paternity")
