"""LangChain adapter for using a ChatGPT Codex subscription.

This provider is deliberately separate from the OpenAI API provider. It talks
to the local Codex app-server through ``openai-codex`` and therefore uses the
account created by ``codex login`` rather than ``OPENAI_API_KEY`` billing.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Sequence
from contextlib import suppress
from copy import deepcopy
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai_codex import ApprovalMode, Codex, Sandbox
from openai_codex.generated.v2_all import ReasoningEffort
from pydantic import BaseModel, PrivateAttr

from .base_client import BaseLLMClient
from .validators import validate_model

_CODEX_INSTRUCTIONS = """You are the language-model backend for TradingAgents.
Use only the conversation supplied in the request. Do not inspect files, run
shell commands, modify files, browse the web, or use Codex built-in tools.
TradingAgents executes any explicitly declared external tool calls itself.
When an output schema is supplied, return only data that matches that schema.
"""


def _create_codex_runtime() -> Codex:
    """Create the SDK runtime (kept as a seam for isolated tests)."""
    return Codex()


def _json_loads(value: str) -> Any:
    """Parse JSON, tolerating a Markdown fence around otherwise valid output."""
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 : -3].strip()
    return json.loads(text)


def _codex_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON Schema to the strict subset accepted by Codex.

    OpenAI structured outputs require every object property to be listed in
    ``required`` and disallow free-form extra properties. Pydantic represents
    optional fields as nullable properties with defaults, so make those fields
    required-but-nullable and remove unsupported ``default`` annotations.
    """
    normalized = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            properties = node.get("properties")
            if node.get("type") == "object" and isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(normalized)
    return normalized


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _serialize_messages(messages: list[BaseMessage]) -> str:
    """Serialize LangChain history without giving Codex access to local state."""
    sections: list[str] = []
    for message in messages:
        role = getattr(message, "type", "message")
        payload: dict[str, Any] = {"content": _message_content(message)}
        if isinstance(message, AIMessage) and message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        if getattr(message, "tool_call_id", None):
            payload["tool_call_id"] = message.tool_call_id
        sections.append(
            f"<{role}>\n{json.dumps(payload, ensure_ascii=False, default=str)}\n</{role}>"
        )
    return "\n\n".join(sections)


def _schema_for_tool_calls(tools: list[dict[str, Any]]) -> dict[str, Any]:
    names = [tool.get("function", tool)["name"] for tool in tools]
    # Codex structured output intentionally supports a restricted JSON Schema
    # subset (notably, no oneOf). Keep the call envelope strict and carry the
    # provider-specific argument object as JSON text; ToolNode performs the
    # normal schema validation after this adapter decodes it.
    item_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": names},
            "arguments_json": {
                "type": "string",
                "description": "A JSON object string matching the selected tool's parameters.",
            },
        },
        "required": ["name", "arguments_json"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Final response text, or an empty string when requesting tools.",
            },
            "tool_calls": {
                "type": "array",
                "description": "External tools TradingAgents should execute before the next turn.",
                "items": item_schema,
            },
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


class CodexChatModel(BaseChatModel):
    """A minimal LangChain chat model backed by the local Codex app-server."""

    model_name: str
    cwd: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None

    _codex: Codex | None = PrivateAttr(default=None)
    _account_checked: bool = PrivateAttr(default=False)
    _runtime_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @property
    def _llm_type(self) -> str:
        return "codex_subscription"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "service_tier": self.service_tier,
        }

    def _ensure_runtime(self) -> Codex:
        if self._codex is None:
            self._codex = _create_codex_runtime()
        if not self._account_checked:
            response = self._codex.account()
            # The SDK wraps the account union in GetAccountResponse.account;
            # accepting the unwrapped shape as well keeps this tolerant of
            # older SDK releases.
            account = getattr(response, "account", response)
            root = getattr(account, "root", account)
            account_type = getattr(root, "type", None)
            if account_type != "chatgpt":
                self.close()
                raise RuntimeError(
                    "Codex provider requires a ChatGPT subscription login. "
                    "Run `codex login` and choose ChatGPT authentication; "
                    "use provider='openai' for OPENAI_API_KEY billing."
                )
            self._account_checked = True
        return self._codex

    def close(self) -> None:
        with self._runtime_lock:
            if self._codex is not None:
                self._codex.close()
                self._codex = None
                self._account_checked = False

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        output_schema = kwargs.pop("output_schema", None)
        tools = kwargs.pop("codex_tools", None)
        tool_choice = kwargs.pop("codex_tool_choice", None)

        prompt = _serialize_messages(messages)
        if stop:
            prompt += "\n\nStop generating before any of these sequences: " + json.dumps(stop)

        if tools is not None:
            output_schema = _schema_for_tool_calls(tools)
            prompt += (
                "\n\nAvailable external tools:\n"
                + json.dumps(tools, ensure_ascii=False, default=str)
                + "\nReturn content with an empty tool_calls list when you can answer now. "
                "When external data is needed, return an empty content string and one or more "
                "tool calls. Never invent tool results."
            )
            if tool_choice:
                prompt += f"\nTool choice requested by the caller: {tool_choice!r}."

        effort = self.reasoning_effort
        if effort is not None:
            try:
                resolved_effort = ReasoningEffort(effort)
            except ValueError as exc:
                supported = ", ".join(item.value for item in ReasoningEffort)
                raise ValueError(
                    f"Unsupported Codex reasoning effort {effort!r}; choose one of: {supported}"
                ) from exc
        else:
            resolved_effort = None

        with self._runtime_lock:
            codex = self._ensure_runtime()
            thread = codex.thread_start(
                approval_mode=ApprovalMode.deny_all,
                base_instructions=_CODEX_INSTRUCTIONS,
                cwd=self.cwd or os.getcwd(),
                developer_instructions=_CODEX_INSTRUCTIONS,
                ephemeral=True,
                model=self.model_name,
                sandbox=Sandbox.read_only,
                service_name="tradingagents",
                service_tier=self.service_tier,
            )
            result = thread.run(prompt, effort=resolved_effort, output_schema=output_schema)

        response = result.final_response
        if response is None:
            raise RuntimeError("Codex completed the turn without a final response")

        tool_calls: list[dict[str, Any]] = []
        content = response
        if tools is not None:
            payload = _json_loads(response)
            content = payload.get("content", "")
            tool_calls = [
                {
                    "name": call["name"],
                    "args": _json_loads(call.get("arguments_json", "{}")),
                    "id": f"codex_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
                for call in payload.get("tool_calls", [])
            ]

        usage_metadata = None
        usage = getattr(getattr(result, "usage", None), "last", None)
        if usage is not None:
            usage_metadata = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
            response_metadata={"model_name": self.model_name, "provider": "codex"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(codex_tools=formatted, codex_tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Codex structured-output options: {unsupported}")

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = _codex_output_schema(schema.model_json_schema())
            parser = PydanticOutputParser(pydantic_object=schema)

            def parse(message: AIMessage) -> Any:
                try:
                    parsed = parser.parse(message.content)
                except Exception as exc:
                    if include_raw:
                        return {"raw": message, "parsed": None, "parsing_error": exc}
                    raise
                if include_raw:
                    return {"raw": message, "parsed": parsed, "parsing_error": None}
                return parsed

        elif isinstance(schema, dict):
            json_schema = _codex_output_schema(schema)

            def parse(message: AIMessage) -> Any:
                try:
                    parsed = _json_loads(message.content)
                except Exception as exc:
                    if include_raw:
                        return {"raw": message, "parsed": None, "parsing_error": exc}
                    raise
                if include_raw:
                    return {"raw": message, "parsed": parsed, "parsing_error": None}
                return parsed

        else:
            raise TypeError("Codex structured output requires a Pydantic model or JSON schema dict")

        return self.bind(output_schema=json_schema) | RunnableLambda(parse)


class CodexClient(BaseLLMClient):
    """Factory wrapper for the ChatGPT-subscription Codex provider."""

    provider = "codex"

    def __init__(self, model: str, base_url: str | None = None, **kwargs: Any):
        if base_url:
            raise ValueError(
                "The Codex provider does not use an OpenAI-compatible base URL; "
                "authenticate with `codex login` instead."
            )
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> CodexChatModel:
        self.warn_if_unknown_model()
        supported = {
            key: value
            for key, value in self.kwargs.items()
            if key in {"callbacks", "cwd", "reasoning_effort", "service_tier"}
        }
        return CodexChatModel(model_name=self.model, **supported)

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
