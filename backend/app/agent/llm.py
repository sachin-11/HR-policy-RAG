"""LLM client abstractions for chat answer generation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.agent.prompts import NO_CONTEXT_ANSWER


class LLMClient(ABC):
    """Abstract chat completion client."""

    supports_tool_calling: bool = False

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate from RAG-shaped prompts (policy Q&A)."""

    def generate_freeform(self, prompt: str) -> str:
        """Generate from standalone instructions (e.g. email drafting).

        Defaults to :meth:`generate`; providers whose ``generate`` only supports
        RAG-shaped prompts should override this method.
        """
        return self.generate(prompt)

    def call_with_tools(self, user_message: str, tools: list[dict]) -> dict:
        """Call LLM with function-calling tools. Returns {tool_calls, text}.

        Base implementation returns empty tool_calls so callers fall back to
        keyword-based routing when the provider does not support tool calling.
        """
        return {"tool_calls": [], "text": ""}

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens. Default: yield the full answer in one shot."""
        yield self.generate(prompt)


class ExtractiveLLMClient(LLMClient):
    """Local fallback client for offline development.

    It does not call an LLM. It extracts the first context block so the API can
    be exercised before OpenAI/Azure OpenAI is configured.
    """

    def generate(self, prompt: str) -> str:
        context_match = re.search(
            r"Approved HR policy context:\n(?P<context>.*?)\n\nEmployee question:",
            prompt,
            flags=re.DOTALL,
        )
        if not context_match:
            return NO_CONTEXT_ANSWER

        context = context_match.group("context").strip()
        content_match = re.search(r"Content:\n(?P<content>.*?)(?:\n\n\[Source|\Z)", context, flags=re.DOTALL)
        if not content_match:
            return NO_CONTEXT_ANSWER

        answer = content_match.group("content").strip()
        return answer or NO_CONTEXT_ANSWER

    def generate_freeform(self, prompt: str) -> str:
        """Offline email drafting using deterministic English templates."""

        from app.agent.email_compose import (
            extract_employee_note_from_compose_prompt,
            fallback_compose_leave_email,
            infer_compose_kind_from_prompt,
            infer_manager_name_from_compose_prompt,
        )

        note = extract_employee_note_from_compose_prompt(prompt)
        kind = infer_compose_kind_from_prompt(prompt)
        manager_name = infer_manager_name_from_compose_prompt(prompt)
        subject, body = fallback_compose_leave_email(raw_note=note, email_kind=kind, manager_name=manager_name)
        return f"SUBJECT: {subject}\n\nBODY:\n{body}"


class OpenAIChatClient(LLMClient):
    """OpenAI chat client with lazy import."""

    supports_tool_calling: bool = True

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAIChatClient")
        self.api_key = api_key
        self.model = model

    def call_with_tools(self, user_message: str, tools: list[dict]) -> dict:
        """Use OpenAI function calling to let the LLM decide which tools to invoke."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package required") from exc

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an HR assistant. Based on the user's request, "
                        "call the appropriate tool(s). For email requests, always use "
                        "send_email. For policy questions, use answer_policy_question. "
                        "Extract the exact email address if mentioned by the user."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        tool_calls = []
        for tc in message.tool_calls or []:
            import json
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool_calls.append({"name": tc.function.name, "arguments": args})
        return {"tool_calls": tool_calls, "text": message.content or ""}

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI chat requires the openai package.") from exc

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI chat requires the openai package.") from exc

        client = AsyncOpenAI(api_key=self.api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


def build_llm_client(provider: str, *, openai_api_key: str = "", openai_chat_model: str = "gpt-4o-mini") -> LLMClient:
    """Build an LLM client by provider name."""

    normalized_provider = provider.lower().strip()
    if normalized_provider in {"local", "mock", "extractive"}:
        return ExtractiveLLMClient()
    if normalized_provider == "openai":
        if not openai_api_key:
            return ExtractiveLLMClient()
        return OpenAIChatClient(api_key=openai_api_key, model=openai_chat_model)
    raise ValueError(f"Unsupported LLM provider: {provider}")
