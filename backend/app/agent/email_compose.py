"""Turn free-form employee notes (any language) into professional leave emails."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.agent.llm import LLMClient

ComposeKind = Literal["manager_leave", "hr_sick_leave"]

_NOTE_START = "BEGIN_EMPLOYEE_NOTE"
_NOTE_END = "END_EMPLOYEE_NOTE"


def build_email_compose_prompt(
    *,
    raw_user_message: str,
    email_kind: ComposeKind,
    manager_name: str,
) -> str:
    recipient_line = (
        "The email is addressed to the HR team (use greeting: Dear HR Team,)."
        if email_kind == "hr_sick_leave"
        else f"The email is addressed to my manager named {manager_name} (use greeting: Hi {manager_name},)."
    )
    leave_focus = (
        "Focus on sick leave / medical reasons and workplace tone."
        if email_kind == "hr_sick_leave"
        else "Infer leave type (sick, casual, personal, etc.) from the note when possible."
    )

    return (
        "You write concise workplace emails in English.\n\n"
        f"{recipient_line}\n"
        f"{leave_focus}\n"
        "Convert the employee's situation into a polite, professional message.\n"
        "Do not quote meta-instructions such as 'send email to manager' — only the employee's real situation.\n\n"
        f"{_NOTE_START}\n{raw_user_message.strip()}\n{_NOTE_END}\n\n"
        "Respond in exactly this format (no markdown fences):\n"
        "SUBJECT: <short subject line>\n\n"
        "BODY:\n"
        "<email body only: greeting line, 2–5 sentences, closing line Regards>\n"
    )


def extract_employee_note_from_compose_prompt(prompt: str) -> str:
    """Used by offline extractive fallback."""

    match = re.search(
        rf"{re.escape(_NOTE_START)}\s*(?P<note>.*?)\s*{re.escape(_NOTE_END)}",
        prompt,
        flags=re.DOTALL,
    )
    if match:
        return match.group("note").strip()
    return prompt.strip()[:2000]


def infer_compose_kind_from_prompt(prompt: str) -> ComposeKind:
    if "addressed to the HR team" in prompt:
        return "hr_sick_leave"
    return "manager_leave"


def infer_manager_name_from_compose_prompt(prompt: str) -> str:
    match = re.search(r"manager named\s+(.+?)\s+\(use greeting", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Manager"


def parse_subject_and_body(llm_text: str) -> tuple[str | None, str | None]:
    """Parse SUBJECT:/BODY: blocks from model output."""

    clean = llm_text.strip().strip("`").strip()
    subject: str | None = None
    m_sub = re.search(r"(?im)^SUBJECT:\s*(.+?)\s*$", clean)
    if m_sub:
        subject = m_sub.group(1).strip()

    m_body = re.search(r"(?is)BODY:\s*(.+)\Z", clean)
    body = m_body.group(1).strip() if m_body else None
    return subject, body


def _infer_sick_focus(note_lower: str, kind: ComposeKind) -> bool:
    if kind == "hr_sick_leave":
        return True
    sick_markers = (
        "sick",
        "fever",
        "ill",
        "unwell",
        "medical",
        "doctor",
        "hospital",
        "flu",
        "bukhar",
        "sar dard",
        "sir dard",
        "tabiyat",
        "khansi",
        "तबीयत",
        "बुखार",
        "दर्द",
    )
    return any(m in note_lower for m in sick_markers)


def fallback_compose_leave_email(
    *,
    raw_note: str,
    email_kind: ComposeKind,
    manager_name: str,
) -> tuple[str, str]:
    """Deterministic English email when no chat model is available."""

    note = raw_note.strip()
    nl = note.lower()

    phrases: list[str] = []
    hint_map = (
        ("bukhar", "I have a fever"),
        ("बुखार", "I have a fever"),
        ("sar dard", "I have a headache"),
        ("sir dard", "I have a headache"),
        ("sar mein dard", "I have a headache"),
        ("सिर दर्द", "I have a headache"),
        ("tabiyat", "I am feeling unwell"),
        ("तबीयत", "I am feeling unwell"),
        ("khansi", "I have a cough"),
        ("खांसी", "I have a cough"),
        ("nausea", "I am experiencing nausea"),
        ("chakkar", "I feel dizzy / unwell"),
    )
    for needle, english in hint_map:
        if needle in nl or needle in note:
            phrases.append(english)

    sick = _infer_sick_focus(nl, email_kind)

    if email_kind == "hr_sick_leave":
        subject = "Sick leave — employee notification"
        if phrases:
            situation = " ".join(dict.fromkeys(phrases)) + ". "
        elif sick:
            situation = "I am unwell and unable to work today. "
        else:
            situation = ""
        core = (
            f"{situation}I would like to formally notify HR and request sick leave in line with company policy.\n\n"
            f"Original note from me for context: {note}"
        )
        body = (
            "Dear HR Team,\n\n"
            + core
            + "\n\nPlease let me know if you need a medical certificate or any other documentation.\n\nRegards"
        )
        return subject, body

    subject = "Leave request — sick leave" if sick else "Leave request"
    greeting = f"Hi {manager_name},\n\n"
    if phrases:
        detail = " ".join(dict.fromkeys(phrases)) + ". "
        middle = (
            f"{detail}I would like to request leave and will update you on my availability as soon as possible.\n\n"
            f"Further context: {note}"
        )
    else:
        middle = (
            "I am writing to request leave from work. Below is a brief description of my situation:\n\n"
            f"{note}\n\n"
            "Please let me know if you require any additional information or documentation."
        )
    body = greeting + middle + "\n\nRegards"
    return subject, body


def compose_leave_email(
    llm_client: LLMClient | None,
    *,
    raw_user_message: str,
    email_kind: ComposeKind,
    manager_name: str,
) -> tuple[str, str]:
    """Return (subject, full body plain text) for SMTP / draft display."""

    manager_display = (manager_name or "Manager").strip()
    if llm_client is None:
        return fallback_compose_leave_email(
            raw_note=raw_user_message,
            email_kind=email_kind,
            manager_name=manager_display,
        )

    prompt = build_email_compose_prompt(
        raw_user_message=raw_user_message,
        email_kind=email_kind,
        manager_name=manager_display,
    )
    try:
        raw = llm_client.generate_freeform(prompt)
    except Exception:
        return fallback_compose_leave_email(
            raw_note=raw_user_message,
            email_kind=email_kind,
            manager_name=manager_display,
        )

    parsed_subject, parsed_body = parse_subject_and_body(raw)
    if parsed_subject and parsed_body:
        return parsed_subject.strip(), parsed_body.strip()

    return fallback_compose_leave_email(
        raw_note=raw_user_message,
        email_kind=email_kind,
        manager_name=manager_display,
    )
