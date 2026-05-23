"""Specialist sub-agents for the HR assistant multi-agent system.

Architecture:
    SupervisorAgent
    ├── LeaveAgent       — leave policies, sick/casual/maternity rules
    ├── EmailAgent       — composing and sending emails
    ├── TicketAgent      — creating/drafting HR support tickets
    └── BenefitsAgent    — insurance, reimbursements, WFH, payroll

The SupervisorAgent receives the user message, decides which specialist to
invoke, calls it, and returns the combined result.  Each specialist has a
focused system prompt and its own RAG retrieval scope.
"""

from __future__ import annotations

import logging
from typing import Any

from app.observability.logging import log_event

logger = logging.getLogger("app.agent.sub_agents")


# ── specialist definitions ─────────────────────────────────────────────────────

SPECIALIST_REGISTRY: dict[str, dict[str, Any]] = {
    "leave_agent": {
        "description": "Handles all leave-related questions: sick leave, casual leave, maternity/paternity, leave balance, leave application process.",
        "system_prompt": (
            "You are the Leave Policy Specialist for this company. "
            "Answer only leave-related HR questions using the provided context. "
            "Be specific about day counts, eligibility, and procedures."
        ),
        "keywords": {"leave", "sick", "casual", "maternity", "paternity", "annual", "holiday", "absence", "pto"},
    },
    "email_agent": {
        "description": "Composes and sends professional workplace emails on behalf of the employee.",
        "system_prompt": (
            "You are an Email Composition Specialist. "
            "Help employees write and send professional emails to managers or HR."
        ),
        "keywords": {"email", "mail", "send", "write", "compose", "draft"},
    },
    "ticket_agent": {
        "description": "Creates and manages HR support tickets for issues needing follow-up.",
        "system_prompt": (
            "You are an HR Ticket Specialist. "
            "Help employees raise, track, and describe HR support tickets."
        ),
        "keywords": {"ticket", "grievance", "complaint", "issue", "raise", "submit", "support"},
    },
    "benefits_agent": {
        "description": "Handles insurance, reimbursements, WFH policy, payroll, and employee benefits.",
        "system_prompt": (
            "You are the Benefits & Compensation Specialist. "
            "Answer questions about health insurance, reimbursements, WFH policy, and payroll."
        ),
        "keywords": {"insurance", "reimbursement", "wfh", "work from home", "payroll", "salary", "benefit", "allowance"},
    },
}


# ── supervisor ─────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """Routes user messages to the appropriate specialist sub-agent.

    Routing strategy:
    1. LLM-based routing when the LLM supports tool calling (OpenAI).
    2. Keyword-based fallback for offline / ExtractiveLLMClient.
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client
        self._routing_tools = self._build_routing_tools()

    def route(self, user_message: str) -> tuple[str, dict[str, Any]]:
        """Return (specialist_name, specialist_config) for the given message."""
        if getattr(self.llm_client, "supports_tool_calling", False):
            specialist = self._llm_route(user_message)
        else:
            specialist = self._keyword_route(user_message)

        config = SPECIALIST_REGISTRY.get(specialist, SPECIALIST_REGISTRY["leave_agent"])
        log_event(logger, event="supervisor.routed", specialist=specialist, message_len=len(user_message))
        return specialist, config

    def get_specialist_prompt(self, specialist_name: str) -> str:
        return SPECIALIST_REGISTRY.get(specialist_name, {}).get("system_prompt", "")

    # ── private ────────────────────────────────────────────────────────────

    def _llm_route(self, user_message: str) -> str:
        try:
            result = self.llm_client.call_with_tools(user_message, self._routing_tools)
            calls = result.get("tool_calls", [])
            if calls:
                return calls[0].get("arguments", {}).get("specialist", "leave_agent")
        except Exception as exc:
            log_event(logger, event="supervisor.llm_route_error", error=str(exc))
        return self._keyword_route(user_message)

    def _keyword_route(self, user_message: str) -> str:
        msg = user_message.lower()
        scores: dict[str, int] = {}
        for name, cfg in SPECIALIST_REGISTRY.items():
            scores[name] = sum(1 for kw in cfg["keywords"] if kw in msg)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "leave_agent"

    def _build_routing_tools(self) -> list[dict]:
        specialist_names = list(SPECIALIST_REGISTRY.keys())
        descriptions = "\n".join(
            f"- {name}: {cfg['description']}"
            for name, cfg in SPECIALIST_REGISTRY.items()
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "route_to_specialist",
                    "description": (
                        "Choose the best specialist sub-agent to handle the user's HR request.\n"
                        f"Available specialists:\n{descriptions}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "specialist": {
                                "type": "string",
                                "enum": specialist_names,
                                "description": "The specialist agent that should handle this request.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "One-line reason for choosing this specialist.",
                            },
                        },
                        "required": ["specialist"],
                    },
                },
            }
        ]
