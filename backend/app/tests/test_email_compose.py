"""Tests for LLM-assisted leave email composition."""

from __future__ import annotations

from app.agent.email_compose import compose_leave_email, fallback_compose_leave_email


def test_fallback_compose_detects_hindi_sick_symptoms_for_manager() -> None:
    subject, body = fallback_compose_leave_email(
        raw_note="aaj mujhe bukhar aur sar dard hai, kal se leave chahiye",
        email_kind="manager_leave",
        manager_name="Priya Manager",
    )

    assert "sick leave" in subject.lower()
    assert "Priya Manager" in body
    assert "fever" in body.lower() or "headache" in body.lower()


def test_compose_with_fake_llm_parse_success() -> None:
    from app.agent.llm import LLMClient

    class Mini(LLMClient):
        def generate(self, prompt: str) -> str:
            return ""

        def generate_freeform(self, prompt: str) -> str:
            return (
                "SUBJECT: Leave — medical\n\n"
                "BODY:\n"
                "Hi Alex,\n\n"
                "I am unwell and need sick leave today.\n\nRegards"
            )

    subject, body = compose_leave_email(
        Mini(),
        raw_user_message="ignore meta",
        email_kind="manager_leave",
        manager_name="Alex",
    )

    assert subject == "Leave — medical"
    assert "unwell" in body.lower()
