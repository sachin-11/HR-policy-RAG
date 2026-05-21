"""Agent tools and approval controls."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.email_compose import compose_leave_email
from app.agent.llm import LLMClient
from app.config import get_settings
from app.mail.smtp_client import send_plain_text_email, smtp_is_configured
from app.observability.logging import log_event

logger = logging.getLogger("app.agent.tools")


ToolName = Literal["employee_profile", "email_draft", "hr_ticket"]

EmailDraftKind = Literal["manager_leave", "hr_sick_leave"]


def user_message_targets_hr_team(text: str) -> bool:
    """Return True when the user is asking to email HR (not only their line manager)."""

    lowered = text.lower().strip()
    padded = f" {lowered} "
    if any(
        needle in padded
        for needle in (
            " to hr ",
            " email hr",
            " e-mail hr",
            " mail hr",
            " human resources",
            " hr department",
            " hr team",
            " contact hr",
            " reach hr ",
            " write to hr",
        )
    ):
        return True
    return bool(re.search(r"\b(send|draft)\s+(?:an\s+)?(?:e-?mail\s+)?to\s+hr\b", lowered))


class EmployeeProfile(BaseModel):
    """Minimal employee profile used for metadata filters."""

    user_id: str
    country: str
    employee_type: str
    department: str = "HR"
    access_level: str = "employee"
    manager_name: str | None = None
    manager_email: str | None = None


class ToolRequest(BaseModel):
    """A planned tool action."""

    tool_name: ToolName
    action: str
    input: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    reason: str | None = None


class ToolResult(BaseModel):
    """Result returned by a tool."""

    tool_name: ToolName
    action: str
    output: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    blocked: bool = False
    requires_approval: bool = False
    message: str = ""


class ApprovalRequiredAction(BaseModel):
    """Action that must be approved by a human/user before execution."""

    tool_name: ToolName
    action: str
    reason: str
    input: dict[str, Any] = Field(default_factory=dict)


class EmployeeProfileTool:
    """Local mock employee profile lookup for MVP."""

    DEFAULT_PROFILE = EmployeeProfile(
        user_id="anonymous",
        country="India",
        employee_type="full_time",
        department="HR",
        manager_name="Reporting Manager",
        manager_email="rajeshsachin786@gmail.com",
    )

    SAMPLE_PROFILES = {
        "emp_123": EmployeeProfile(
            user_id="emp_123",
            country="India",
            employee_type="full_time",
            department="Engineering",
            manager_name="Asha Manager",
            manager_email="asha.manager@example.com",
        ),
        "contractor_001": EmployeeProfile(
            user_id="contractor_001",
            country="India",
            employee_type="contractor",
            department="Engineering",
            manager_name="Contractor Manager",
            manager_email="contractor.manager@example.com",
        ),
    }

    def run(self, user_id: str | None) -> ToolResult:
        profile = self.SAMPLE_PROFILES.get(user_id or "", self.DEFAULT_PROFILE)
        return ToolResult(
            tool_name="employee_profile",
            action="lookup",
            output=profile.model_dump(),
            message="Employee profile loaded for permission-aware retrieval.",
        )


class EmailDraftTool:
    """Create an email draft and block send operations until approved."""

    def run(
        self,
        user_message: str,
        employee_profile: dict[str, Any] | None = None,
        *,
        kind: EmailDraftKind = "manager_leave",
        hr_contact_email: str | None = None,
        llm_client: LLMClient | None = None,
    ) -> ToolResult:
        manager_display = (employee_profile or {}).get("manager_name") or "Manager"
        subject, body_plain = compose_leave_email(
            llm_client,
            raw_user_message=user_message,
            email_kind=kind,
            manager_name=manager_display,
        )

        if kind == "hr_sick_leave":
            hr_to = (hr_contact_email or "").strip()
            to_line = f"To: {hr_to}\n\n" if hr_to else ""
            draft = f"{to_line}Subject: {subject}\n\n{body_plain}"
            return ToolResult(
                tool_name="email_draft",
                action="draft",
                output={"draft": draft, "kind": kind, "recipient": hr_to or None, "subject": subject},
                message="Email draft created for HR (sick leave). It has not been sent.",
            )

        manager_email = (employee_profile or {}).get("manager_email")
        to_line = f"To: {manager_email}\n\n" if manager_email else ""
        draft = f"{to_line}Subject: {subject}\n\n{body_plain}"
        return ToolResult(
            tool_name="email_draft",
            action="draft",
            output={"draft": draft, "kind": kind, "subject": subject},
            message="Email draft created. It has not been sent.",
        )

    def send(
        self,
        user_message: str,
        employee_profile: dict[str, Any] | None = None,
        approved: bool = False,
        *,
        kind: EmailDraftKind = "manager_leave",
        hr_contact_email: str | None = None,
        llm_client: LLMClient | None = None,
    ) -> ToolResult:
        if not approved:
            return ToolResult(
                tool_name="email_draft",
                action="send",
                output={"message": user_message, "kind": kind},
                success=False,
                blocked=True,
                requires_approval=True,
                message="Sending email is a sensitive action and requires approval before execution.",
            )

        manager_display = (employee_profile or {}).get("manager_name") or "Manager"
        subject, body_plain = compose_leave_email(
            llm_client,
            raw_user_message=user_message,
            email_kind=kind,
            manager_name=manager_display,
        )

        if kind == "hr_sick_leave":
            hr_to = (hr_contact_email or "").strip()
            if not hr_to:
                return ToolResult(
                    tool_name="email_draft",
                    action="send",
                    output={"draft": "", "sent": False, "kind": kind},
                    success=False,
                    message="Cannot send email: HR_CONTACT_EMAIL / hr_contact_email is not configured.",
                )
            to_line = f"To: {hr_to}\n\n"
            draft = f"{to_line}Subject: {subject}\n\n{body_plain}"

            if smtp_is_configured():
                try:
                    send_plain_text_email(
                        to_addr=hr_to,
                        subject=subject,
                        body=body_plain,
                    )
                except Exception as exc:
                    log_event(logger, event="agent.email.smtp_error", error=str(exc))
                    return ToolResult(
                        tool_name="email_draft",
                        action="send",
                        output={"draft": draft, "sent": False, "error": str(exc), "kind": kind},
                        success=False,
                        message=f"SMTP send failed: {exc}",
                    )

                return ToolResult(
                    tool_name="email_draft",
                    action="send",
                    output={"draft": draft, "sent": True, "recipient": hr_to, "via": "smtp", "kind": kind},
                    message="Sick leave email sent to HR via SMTP.",
                )

            return ToolResult(
                tool_name="email_draft",
                action="send",
                output={"draft": draft, "sent": True, "recipient": hr_to, "kind": kind},
                message="Email send executed after approval (SMTP not configured — simulated send only).",
            )

        manager_email = (employee_profile or {}).get("manager_email")
        if not manager_email or not str(manager_email).strip():
            return ToolResult(
                tool_name="email_draft",
                action="send",
                output={"draft": "", "sent": False},
                success=False,
                message="Cannot send email: recipient (manager_email) is missing from the employee profile.",
            )

        to_line = f"To: {manager_email}\n\n"
        draft = f"{to_line}Subject: {subject}\n\n{body_plain}"

        if smtp_is_configured():
            try:
                send_plain_text_email(
                    to_addr=str(manager_email).strip(),
                    subject=subject,
                    body=body_plain,
                )
            except Exception as exc:
                log_event(logger, event="agent.email.smtp_error", error=str(exc))
                return ToolResult(
                    tool_name="email_draft",
                    action="send",
                    output={"draft": draft, "sent": False, "error": str(exc)},
                    success=False,
                    message=f"SMTP send failed: {exc}",
                )

            return ToolResult(
                tool_name="email_draft",
                action="send",
                output={"draft": draft, "sent": True, "recipient": manager_email, "via": "smtp"},
                message="Email sent via SMTP.",
            )

        return ToolResult(
            tool_name="email_draft",
            action="send",
            output={"draft": draft, "sent": True},
            message="Email send executed after approval (SMTP not configured — simulated send only).",
        )


class HRTicketTool:
    """Draft or create HR tickets. Create is blocked until approval exists."""

    def draft(self, user_message: str) -> ToolResult:
        return ToolResult(
            tool_name="hr_ticket",
            action="draft",
            output={
                "title": "HR policy support request",
                "description": user_message,
                "status": "draft",
            },
            message="HR ticket draft created. It has not been submitted.",
        )

    def create(self, user_message: str, approved: bool = False) -> ToolResult:
        if not approved:
            return ToolResult(
                tool_name="hr_ticket",
                action="create",
                output={"description": user_message},
                success=False,
                blocked=True,
                requires_approval=True,
                message="HR ticket creation requires approval before execution.",
            )
        return ToolResult(
            tool_name="hr_ticket",
            action="create",
            output={"ticket_id": "HR-DRAFT-001", "status": "created"},
            message="HR ticket created after approval.",
        )


class ToolOrchestrator:
    """Plan and execute safe MVP tools for the agent."""

    def __init__(
        self,
        employee_profile_tool: EmployeeProfileTool | None = None,
        email_draft_tool: EmailDraftTool | None = None,
        hr_ticket_tool: HRTicketTool | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.employee_profile_tool = employee_profile_tool or EmployeeProfileTool()
        self.email_draft_tool = email_draft_tool or EmailDraftTool()
        self.hr_ticket_tool = hr_ticket_tool or HRTicketTool()
        self.llm_client = llm_client

    def run(
        self,
        *,
        user_id: str | None,
        user_message: str,
        intent: str | None,
        approved_pairs: set[tuple[str, str]] | None = None,
    ) -> tuple[list[ToolResult], list[ApprovalRequiredAction]]:
        results: list[ToolResult] = []
        approvals: list[ApprovalRequiredAction] = []
        approved = approved_pairs or set()

        profile_result = self.employee_profile_tool.run(user_id)
        results.append(profile_result)
        employee_profile = profile_result.output

        if intent != "action_request":
            return results, approvals

        message = user_message.lower()
        settings = get_settings()
        hr_inbox = settings.hr_contact_email.strip()
        targets_hr = user_message_targets_hr_team(user_message)
        email_kind: EmailDraftKind = "hr_sick_leave" if targets_hr else "manager_leave"
        hr_contact_kwarg = hr_inbox if targets_hr else None

        has_mail = "email" in message or "mail" in message
        if "send" in message and has_mail:
            if ("email_draft", "send") in approved:
                results.append(
                    self.email_draft_tool.send(
                        user_message=user_message,
                        employee_profile=employee_profile,
                        approved=True,
                        kind=email_kind,
                        hr_contact_email=hr_contact_kwarg,
                        llm_client=self.llm_client,
                    )
                )
            else:
                # Compose the draft first so the approval modal can show a preview
                draft_result = self.email_draft_tool.run(
                    user_message=user_message,
                    employee_profile=employee_profile,
                    kind=email_kind,
                    hr_contact_email=hr_contact_kwarg,
                    llm_client=self.llm_client,
                )
                draft_text = draft_result.output.get("draft", "")
                subject = draft_result.output.get("subject", "")
                recipient = draft_result.output.get("recipient") or (employee_profile or {}).get("manager_email") or ""
                results.append(ToolResult(
                    tool_name="email_draft",
                    action="send",
                    output={
                        "draft": draft_text,
                        "subject": subject,
                        "recipient": recipient,
                        "kind": email_kind,
                        "pending_approval": True,
                    },
                    success=False,
                    blocked=True,
                    requires_approval=True,
                    message="Email draft ready — awaiting your approval to send.",
                ))
                approvals.append(
                    ApprovalRequiredAction(
                        tool_name="email_draft",
                        action="send",
                        reason="Review the email below and confirm to send.",
                        input={
                            "message": user_message,
                            "draft": draft_text,
                            "subject": subject,
                            "recipient": recipient,
                        },
                    )
                )
        elif "draft" in message and has_mail:
            results.append(
                self.email_draft_tool.run(
                    user_message=user_message,
                    employee_profile=employee_profile,
                    kind=email_kind,
                    hr_contact_email=hr_contact_kwarg,
                    llm_client=self.llm_client,
                )
            )

        if "ticket" in message:
            if any(keyword in message for keyword in ("create", "submit", "raise")):
                if ("hr_ticket", "create") in approved:
                    results.append(self.hr_ticket_tool.create(user_message=user_message, approved=True))
                else:
                    blocked_result = self.hr_ticket_tool.create(user_message=user_message, approved=False)
                    results.append(blocked_result)
                    approvals.append(
                        ApprovalRequiredAction(
                            tool_name="hr_ticket",
                            action="create",
                            reason="Creating an HR ticket changes enterprise records and needs user approval.",
                            input={"description": user_message},
                        )
                    )
            else:
                results.append(self.hr_ticket_tool.draft(user_message=user_message))

        for result in results:
            log_event(
                logger,
                event="agent.tool.call",
                tool_name=result.tool_name,
                action=result.action,
                success=result.success,
                blocked=result.blocked,
                requires_approval=result.requires_approval,
            )

        return results, approvals


def tool_results_to_answer_block(results: list[dict[str, Any]]) -> str:
    """Format safe tool outputs for the final answer."""

    blocks: list[str] = []
    for result in results:
        if result.get("blocked"):
            continue
        tool_name = result.get("tool_name")
        action = result.get("action")
        output = result.get("output") or {}
        if tool_name == "email_draft" and "draft" in output:
            if action == "send" and output.get("sent"):
                recipient = output.get("recipient", "recipient")
                via = output.get("via")
                header = f"**Email sent to {recipient}**" + (f" via {via}" if via else "") + "."
                blocks.append(f"{header}\n\n{output['draft']}")
            else:
                blocks.append(f"Email draft:\n{output['draft']}")
        elif tool_name == "hr_ticket" and action == "draft":
            blocks.append(
                "HR ticket draft:\n"
                f"Title: {output.get('title')}\n"
                f"Description: {output.get('description')}\n"
                f"Status: {output.get('status')}"
            )
    return "\n\n".join(blocks).strip()
