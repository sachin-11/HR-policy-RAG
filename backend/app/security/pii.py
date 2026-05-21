"""PII masking and prompt injection guard utilities."""

from __future__ import annotations

import re
from typing import Iterable

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d -]{7,}\d)\b")
ID_PATTERN = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")
CARD_PATTERN = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")

PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "forget all previous instructions",
    "disregard all earlier directions",
    "do not follow the earlier instructions",
    "please ignore",
    "you are instructed to",
    "you must",
    "break the rules",
)

PLACEHOLDERS = {
    EMAIL_PATTERN: "[EMAIL_REDACTED]",
    PHONE_PATTERN: "[PHONE_REDACTED]",
    ID_PATTERN: "[ID_REDACTED]",
    CARD_PATTERN: "[CARD_REDACTED]",
}


def mask_pii(text: str) -> str:
    """Replace common PII patterns with redaction placeholders."""

    if not text:
        return text

    redacted = text
    for pattern, placeholder in PLACEHOLDERS.items():
        redacted = pattern.sub(placeholder, redacted)
    return redacted


def contains_prompt_injection(text: str) -> bool:
    """Detect obvious prompt injection phrases in user-provided text."""

    normalized = (text or "").strip().lower()
    return any(phrase in normalized for phrase in PROMPT_INJECTION_PATTERNS)


def summarize_guard_rules() -> str:
    """Return a short summary of prompt injection guard rules."""

    return (
        "Prompt injection guard checks for phrases like 'ignore previous instructions', "
        " 'forget all previous instructions', and other explicit override attempts. "
        "User input containing suspicious directives is rejected." 
    )
