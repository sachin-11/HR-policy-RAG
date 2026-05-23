"""OpenAI function-calling schemas for the HR agent.

These are passed to the LLM so it can decide which tools to invoke
instead of relying on keyword matching.
"""

from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send or draft a professional email on behalf of the employee. "
                "Use when the user wants to send any email — to their manager, HR team, "
                "or a specific email address they mention."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["send", "draft"],
                        "description": "'send' to dispatch immediately, 'draft' to just compose for review.",
                    },
                    "recipient_type": {
                        "type": "string",
                        "enum": ["manager", "hr", "custom"],
                        "description": (
                            "'manager' for line manager, 'hr' for HR team, "
                            "'custom' when a specific email address is mentioned."
                        ),
                    },
                    "recipient_email": {
                        "type": "string",
                        "description": "Specific email address — required when recipient_type is 'custom'.",
                    },
                    "message_context": {
                        "type": "string",
                        "description": "The actual reason/content the employee wants to communicate.",
                    },
                },
                "required": ["action", "recipient_type", "message_context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_hr_ticket",
            "description": (
                "Create or draft an HR support ticket for issues needing follow-up: "
                "grievances, policy clarifications, payroll disputes, or any formal HR request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "draft"],
                        "description": "'create' to submit the ticket, 'draft' to preview it.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the issue or request for the HR ticket.",
                    },
                },
                "required": ["action", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_policy_question",
            "description": (
                "Search approved HR policy documents and answer questions about: "
                "leave policies, sick leave, WFH rules, reimbursements, benefits, "
                "maternity/paternity leave, contractor rules, insurance, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The HR policy question to look up.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# Map LLM tool name → internal intent
TOOL_NAME_TO_INTENT: dict[str, str] = {
    "send_email": "action_request",
    "create_hr_ticket": "action_request",
    "answer_policy_question": "policy_qa",
}
