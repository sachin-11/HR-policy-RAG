"""Role-based permissions and metadata filtering for HR policy access."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["anonymous", "employee", "manager", "hr", "admin"]


class UserContext(BaseModel):
    """Authenticated user context for permission checks."""

    user_id: str = Field(default="anonymous")
    role: Role = Field(default="anonymous")
    country: str = Field(default="India")
    employee_type: str = Field(default="full_time")
    department: str = Field(default="HR")


class PermissionDenied(Exception):
    """Raised when a user is not allowed to perform an operation."""


def authorize_chat_request(
    user_context: UserContext,
    target_user_id: str | None,
    requested_filters: dict[str, Any],
) -> dict[str, Any]:
    """Validate requested metadata filters against the user's role and context."""

    filters = {key: value for key, value in requested_filters.items() if value is not None}

    if user_context.role == "anonymous":
        # access_level may be sent by clients as a UI default; it is not user-identifying metadata.
        if target_user_id is not None or any(
            requested_filters.get(field) is not None
            for field in ("country", "employee_type", "department")
        ):
            raise PermissionDenied("Authentication is required to access user-specific metadata or account-specific data.")
        return {}

    if user_context.role in ("admin", "hr"):
        return filters

    if target_user_id is not None and target_user_id != user_context.user_id:
        raise PermissionDenied("Users may only request information for their own account.")

    if user_context.role == "manager":
        if filters.get("department") and filters["department"] != user_context.department:
            raise PermissionDenied("Managers may only access their own department data.")
        if filters.get("country") and filters["country"] != user_context.country:
            raise PermissionDenied("Managers may only access data for their own country.")
    else:
        if filters.get("department") and filters["department"] != user_context.department:
            raise PermissionDenied("Employees may only access their own department data.")
        if filters.get("country") and filters["country"] != user_context.country:
            raise PermissionDenied("Employees may only access their own country data.")
        if filters.get("employee_type") and filters["employee_type"] != user_context.employee_type:
            raise PermissionDenied("Employees may only access their own employee type data.")

    safe_filters = {
        "country": user_context.country,
        "employee_type": user_context.employee_type,
        "department": user_context.department,
        **{k: v for k, v in filters.items() if k not in {"country", "employee_type", "department"}},
    }

    if user_context.role == "manager":
        safe_filters["country"] = filters.get("country", user_context.country)
        safe_filters["department"] = filters.get("department", user_context.department)
        if "employee_type" in filters:
            safe_filters["employee_type"] = filters["employee_type"]

    return safe_filters
