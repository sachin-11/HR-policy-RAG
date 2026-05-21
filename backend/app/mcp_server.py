"""HR Policy Assistant — MCP Server.

Exposes HR tools as MCP (Model Context Protocol) endpoints so any
MCP-compatible client (Claude Desktop, Claude Code, Cursor, etc.) can
use the HR Policy RAG search, email drafting, and ticket creation without
going through the FastAPI HTTP layer.

Usage
─────
Install the extra dependency first:
    pip install "mcp[cli]"

Run the server (stdio transport — standard for local MCP):
    python -m app.mcp_server

Configure Claude Desktop (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "hr-policy": {
          "command": "python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "C:/sachinProjects/hr_policy_assistant/backend",
          "env": { "PYTHONPATH": "." }
        }
      }
    }
"""

from __future__ import annotations

import os
import uuid
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit(
        "MCP package not installed. Run: pip install \"mcp[cli]\""
    ) from exc

from app.config import get_settings
from app.rag.embeddings import build_embedding_provider
from app.rag.retriever import RagRetriever, format_retrieved_context
from app.rag.vector_store import build_vector_store


# ── MCP server init ────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="HR Policy Assistant",
    instructions=(
        "You have access to the company HR policy knowledge base. "
        "Use search_hr_policy to answer leave, reimbursement, insurance, or WFH questions. "
        "Use draft_leave_email to help employees write leave request emails. "
        "Use create_hr_ticket to raise HR support requests."
    ),
)

# ── lazy-initialised retriever ─────────────────────────────────────────────────

_retriever: RagRetriever | None = None


def _get_retriever() -> RagRetriever:
    global _retriever
    if _retriever is None:
        settings = get_settings()
        embedding_provider = build_embedding_provider(
            provider=settings.llm_provider,
            openai_api_key=settings.openai_api_key,
            openai_embedding_model=settings.openai_embedding_model,
        )
        vector_store = build_vector_store(
            provider=settings.vector_store_provider,
            directory=settings.vector_store_dir,
            pinecone_api_key=settings.pinecone_api_key,
            pinecone_index_name=settings.pinecone_index_name,
            pinecone_namespace=settings.pinecone_namespace,
            pinecone_cloud=settings.pinecone_cloud,
            pinecone_region=settings.pinecone_region,
        )
        _retriever = RagRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
    return _retriever


# ── tools ──────────────────────────────────────────────────────────────────────


@mcp.tool()
def search_hr_policy(
    query: str,
    country: str = "",
    policy_type: str = "",
    employee_type: str = "",
    top_k: int = 5,
) -> str:
    """Search the HR policy knowledge base and return relevant policy information.

    Args:
        query:         What you want to know (e.g. "sick leave rules India").
        country:       Filter by country code or name (e.g. "India", "US"). Leave blank for all.
        policy_type:   Filter by category: leave | reimbursement | insurance | wfh | benefits.
        employee_type: Filter by employment type: full_time | contractor | intern.
        top_k:         Number of document chunks to retrieve (1–10, default 5).
    """
    filters: dict[str, Any] = {}
    if country:
        filters["country"] = country
    if policy_type:
        filters["policy_type"] = policy_type
    if employee_type:
        filters["employee_type"] = employee_type

    try:
        retriever = _get_retriever()
        result = retriever.retrieve(query, filters=filters, top_k=min(max(top_k, 1), 10))
    except Exception as exc:
        return f"Retrieval error: {exc}"

    if not result.chunks:
        return (
            "No relevant HR policy documents found for your query. "
            "Try rephrasing or removing filters."
        )

    context = format_retrieved_context(result.chunks)
    sources = "\n".join(
        f"- {chunk.metadata.get('title', 'Untitled')}  [{chunk.metadata.get('source', '')}]"
        for chunk in result.chunks
    )
    return f"{context}\n\n**Sources:**\n{sources}"


@mcp.tool()
def draft_leave_email(
    employee_name: str,
    manager_name: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    reason: str = "",
    manager_email: str = "",
) -> str:
    """Draft a professional leave request email ready to send to a manager.

    Args:
        employee_name: Full name of the employee requesting leave.
        manager_name:  Full name of the reporting manager.
        leave_type:    Type of leave: sick | annual | maternity | paternity | emergency.
        start_date:    Leave start date (e.g. "2026-06-01").
        end_date:      Leave end date (e.g. "2026-06-05").
        reason:        Brief reason (optional, can be omitted for privacy).
        manager_email: Manager's email address (used in the To: field if provided).
    """
    to_line = f"To: {manager_email}" if manager_email else ""
    reason_line = f"\nReason: {reason}" if reason else ""

    subject = f"Leave Request - {leave_type.title()} Leave ({start_date} to {end_date})"
    body = f"""Dear {manager_name},

I hope this message finds you well.

I would like to request {leave_type} leave from {start_date} to {end_date} (inclusive).{reason_line}

I will ensure all pending tasks are completed or appropriately handed over before my leave begins. I am happy to discuss the work handover plan if required.

Please let me know if you need any additional information or documentation.

Regards,
{employee_name}"""

    lines = ["---"]
    if to_line:
        lines.append(to_line)
    lines += [
        f"Subject: {subject}",
        "---",
        body,
        "---",
    ]
    return "\n".join(lines)


@mcp.tool()
def create_hr_ticket(
    title: str,
    description: str,
    category: str = "general",
    priority: str = "medium",
    employee_id: str = "",
) -> str:
    """Raise an HR support ticket for issues that need HR team attention.

    Args:
        title:       Short summary of the issue (max 100 chars).
        description: Full description of the problem or request.
        category:    Ticket category: leave | reimbursement | payroll | policy | general.
        priority:    Priority level: low | medium | high | urgent.
        employee_id: Employee ID or email (optional, for tracking).
    """
    ticket_id = f"HR-{uuid.uuid4().hex[:6].upper()}"
    return f"""HR Ticket Created
-----------------------------
Ticket ID   : {ticket_id}
Title       : {title}
Category    : {category}
Priority    : {priority}
Employee    : {employee_id or "not specified"}
Status      : Open
-----------------------------
Description:
{description}
-----------------------------
The HR team will respond within 2 business days.
For urgent matters, contact hr@company.com directly."""


# ── resources ──────────────────────────────────────────────────────────────────


@mcp.resource("hr://policies/list")
def list_hr_policies() -> str:
    """List all HR policy documents currently loaded in the knowledge base."""
    settings = get_settings()
    raw_docs_dir = settings.raw_docs_dir

    try:
        files = sorted(
            f for f in os.listdir(raw_docs_dir)
            if f.endswith((".md", ".txt", ".pdf"))
        )
    except FileNotFoundError:
        return f"Documents directory not found: {raw_docs_dir}"

    if not files:
        return "No HR policy documents found. Upload documents via the Admin Panel."

    lines = [f"Available HR Policy Documents ({len(files)} files):", ""]
    lines += [f"  {i+1}. {name}" for i, name in enumerate(files)]
    return "\n".join(lines)


@mcp.resource("hr://policies/{document_name}")
def get_hr_policy_document(document_name: str) -> str:
    """Read the raw content of a specific HR policy document by filename."""
    settings = get_settings()
    doc_path = os.path.join(settings.raw_docs_dir, document_name)

    # Prevent path traversal
    resolved = os.path.realpath(doc_path)
    allowed_dir = os.path.realpath(settings.raw_docs_dir)
    if not resolved.startswith(allowed_dir):
        return "Access denied: invalid document path."

    if not os.path.isfile(resolved):
        return f"Document '{document_name}' not found."

    try:
        with open(resolved, encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Error reading document: {exc}"


# ── entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
