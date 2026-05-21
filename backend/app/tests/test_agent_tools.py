import pytest

from app.agent.tools import (
    EmailDraftTool,
    EmployeeProfileTool,
    HRTicketTool,
    ToolOrchestrator,
    user_message_targets_hr_team,
)


def test_user_message_targets_hr_team_detects_common_phrases() -> None:
    assert user_message_targets_hr_team("send email to HR about leave") is True
    assert user_message_targets_hr_team("draft an email to hr") is True
    assert user_message_targets_hr_team("contact human resources please") is True
    assert user_message_targets_hr_team("Send an email to my manager requesting a leave.") is False


def test_email_draft_hr_sick_leave_template() -> None:
    result = EmailDraftTool().run(
        user_message="Starting tomorrow, fever.",
        kind="hr_sick_leave",
        hr_contact_email="hr-inbox@example.com",
    )
    assert "hr-inbox@example.com" in result.output["draft"]
    assert "Sick leave" in result.output["draft"]
    assert "Dear HR Team" in result.output["draft"]
    result = EmployeeProfileTool().run(user_id="emp_123")

    assert result.tool_name == "employee_profile"
    assert result.output["user_id"] == "emp_123"
    assert result.output["department"] == "Engineering"
    assert result.message == "Employee profile loaded for permission-aware retrieval."


def test_email_draft_tool_creates_draft() -> None:
    result = EmailDraftTool().run(
        user_message="I need sick leave starting next week.",
        employee_profile={"manager_name": "Asha Manager"},
    )

    assert result.tool_name == "email_draft"
    assert result.action == "draft"
    assert result.success is True
    assert "Subject:" in result.output["draft"]
    assert "Hi Asha Manager" in result.output["draft"]


def test_email_send_tool_requires_approval() -> None:
    result = EmailDraftTool().send(
        user_message="Please send this leave request to my manager.",
        employee_profile={"manager_name": "Asha Manager"},
        approved=False,
    )

    assert result.tool_name == "email_draft"
    assert result.action == "send"
    assert result.success is False
    assert result.blocked is True
    assert result.requires_approval is True
    assert "requires approval" in result.message.lower()


def test_hr_ticket_tool_create_is_blocked_without_approval() -> None:
    result = HRTicketTool().create(
        user_message="Please create a ticket to investigate my reimbursement claim.",
        approved=False,
    )

    assert result.tool_name == "hr_ticket"
    assert result.action == "create"
    assert result.success is False
    assert result.blocked is True
    assert result.requires_approval is True
    assert "requires approval" in result.message.lower()


def test_tool_orchestrator_blocks_sensitive_actions_and_returns_approval_requests() -> None:
    orchestrator = ToolOrchestrator()
    results, approvals = orchestrator.run(
        user_id="emp_123",
        user_message="Send an email to my manager requesting a leave.",
        intent="action_request",
    )

    email_send = next((result for result in results if result.tool_name == "email_draft" and result.action == "send"), None)
    assert email_send is not None
    assert email_send.blocked is True
    assert email_send.requires_approval is True

    approval_actions = [approval for approval in approvals if approval.tool_name == "email_draft" and approval.action == "send"]
    assert len(approval_actions) == 1
    assert "sensitive action" in approval_actions[0].reason.lower()


def test_tool_orchestrator_executes_email_send_after_approval() -> None:
    orchestrator = ToolOrchestrator()
    results, approvals = orchestrator.run(
        user_id="emp_123",
        user_message="Send an email to my manager requesting a leave.",
        intent="action_request",
        approved_pairs={("email_draft", "send")},
    )

    assert approvals == []
    email_send = next(result for result in results if result.tool_name == "email_draft" and result.action == "send")
    assert email_send.blocked is False
    assert email_send.success is True
    assert email_send.output.get("sent") is True


def test_tool_orchestrator_hr_email_send_uses_hr_inbox_and_sick_leave(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("HR_CONTACT_EMAIL", "hrfixture@example.com")
    get_settings.cache_clear()
    try:
        orchestrator = ToolOrchestrator()
        results, _ = orchestrator.run(
            user_id="emp_123",
            user_message="send email to hr about sick leave",
            intent="action_request",
            approved_pairs={("email_draft", "send")},
        )
        email_send = next(r for r in results if r.tool_name == "email_draft" and r.action == "send")
        assert email_send.success is True
        assert email_send.output.get("recipient") == "hrfixture@example.com"
        assert "Sick leave" in email_send.output.get("draft", "")
    finally:
        get_settings.cache_clear()


def test_tool_orchestrator_creates_hr_ticket_approval_request() -> None:
    orchestrator = ToolOrchestrator()
    results, approvals = orchestrator.run(
        user_id="emp_123",
        user_message="Create a ticket for reimbursement support.",
        intent="action_request",
    )

    ticket_create = next((result for result in results if result.tool_name == "hr_ticket" and result.action == "create"), None)
    assert ticket_create is not None
    assert ticket_create.blocked is True
    assert ticket_create.requires_approval is True

    approval_actions = [approval for approval in approvals if approval.tool_name == "hr_ticket" and approval.action == "create"]
    assert len(approval_actions) == 1
    assert "enterprise records" in approval_actions[0].reason.lower()
