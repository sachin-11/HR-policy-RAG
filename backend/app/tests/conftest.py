"""Shared pytest hooks for backend tests."""

from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache_for_isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real SMTP sends during tests when developers have credentials in `.env`."""

    get_settings.cache_clear()
    monkeypatch.setenv("SMTP_USER", "")
    monkeypatch.setenv("SMTP_PASS", "")
    yield
    get_settings.cache_clear()
