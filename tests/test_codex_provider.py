"""Unit tests for the ChatGPT-subscription Codex provider."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients import codex_client as codex_module
from tradingagents.llm_clients.codex_client import CodexChatModel, CodexClient
from tradingagents.llm_clients.factory import create_llm_client


class FakeThread:
    def __init__(self, runtime):
        self.runtime = runtime

    def run(self, prompt, **kwargs):
        self.runtime.runs.append((prompt, kwargs))
        response = self.runtime.responses.pop(0)
        usage = SimpleNamespace(
            last=SimpleNamespace(input_tokens=12, output_tokens=7, total_tokens=19)
        )
        return SimpleNamespace(final_response=response, usage=usage)


class FakeCodex:
    def __init__(self, responses, account_type="chatgpt"):
        self.responses = list(responses)
        self.account_type = account_type
        self.starts = []
        self.runs = []
        self.closed = False

    def account(self):
        account = SimpleNamespace(root=SimpleNamespace(type=self.account_type))
        return SimpleNamespace(account=account)

    def thread_start(self, **kwargs):
        self.starts.append(kwargs)
        return FakeThread(self)

    def close(self):
        self.closed = True


def install_fake(monkeypatch, responses, account_type="chatgpt"):
    fake = FakeCodex(responses, account_type=account_type)
    monkeypatch.setattr(codex_module, "_create_codex_runtime", lambda: fake)
    return fake


@pytest.mark.unit
def test_factory_creates_native_codex_client_without_base_url():
    client = create_llm_client("codex", "gpt-5.6-luna", reasoning_effort="low")
    assert isinstance(client, CodexClient)
    llm = client.get_llm()
    assert isinstance(llm, CodexChatModel)
    assert llm.reasoning_effort == "low"


@pytest.mark.unit
def test_codex_rejects_openai_compatible_base_url():
    with pytest.raises(ValueError, match="does not use an OpenAI-compatible base URL"):
        create_llm_client("codex", "gpt-5.6-luna", "https://api.openai.com/v1")


@pytest.mark.unit
def test_invoke_uses_chatgpt_login_and_locked_down_thread(monkeypatch, tmp_path):
    fake = install_fake(monkeypatch, ["concise answer"])
    llm = CodexChatModel(
        model_name="gpt-5.6-luna",
        cwd=str(tmp_path),
        reasoning_effort="low",
    )

    message = llm.invoke([HumanMessage(content="Analyze ACME")])

    assert message.content == "concise answer"
    assert message.usage_metadata == {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
    }
    assert fake.starts[0]["approval_mode"].value == "deny_all"
    assert fake.starts[0]["sandbox"].value == "read-only"
    assert fake.starts[0]["ephemeral"] is True
    assert fake.starts[0]["model"] == "gpt-5.6-luna"
    assert fake.starts[0]["cwd"] == str(tmp_path)
    assert fake.runs[0][1]["effort"].value == "low"
    assert "Analyze ACME" in fake.runs[0][0]


@pytest.mark.unit
def test_non_chatgpt_login_is_rejected(monkeypatch):
    fake = install_fake(monkeypatch, ["unused"], account_type="apiKey")
    llm = CodexChatModel(model_name="gpt-5.6-luna")

    with pytest.raises(RuntimeError, match="requires a ChatGPT subscription login"):
        llm.invoke("hello")

    assert fake.closed is True
    assert not fake.runs


@pytest.mark.unit
def test_bound_tools_return_langchain_tool_calls(monkeypatch):
    fake = install_fake(
        monkeypatch,
        [json.dumps({"content": "", "tool_calls": [
            {"name": "lookup_price", "arguments_json": '{"ticker": "NVDA"}'}
        ]})],
    )

    @tool
    def lookup_price(ticker: str) -> str:
        """Look up a ticker's latest price."""
        return ticker

    llm = CodexChatModel(model_name="gpt-5.6-luna")
    message = llm.bind_tools([lookup_price]).invoke("What is NVDA trading at?")

    assert message.content == ""
    assert message.tool_calls[0]["name"] == "lookup_price"
    assert message.tool_calls[0]["args"] == {"ticker": "NVDA"}
    assert message.tool_calls[0]["id"].startswith("codex_")
    schema = fake.runs[0][1]["output_schema"]
    item = schema["properties"]["tool_calls"]["items"]
    assert item["properties"]["name"]["enum"] == ["lookup_price"]
    assert "oneOf" not in item


@pytest.mark.unit
def test_pydantic_structured_output_is_parsed(monkeypatch):
    fake = install_fake(monkeypatch, ['{"rating": 4, "summary": "bullish"}'])

    class Assessment(BaseModel):
        rating: int
        summary: str

    llm = CodexChatModel(model_name="gpt-5.6-sol")
    result = llm.with_structured_output(Assessment).invoke("Rate the setup")

    assert result == Assessment(rating=4, summary="bullish")
    schema = fake.runs[0][1]["output_schema"]
    assert set(schema["required"]) == {"rating", "summary"}


@pytest.mark.unit
def test_optional_pydantic_fields_are_normalized_for_strict_schema(monkeypatch):
    fake = install_fake(monkeypatch, ['{"action": "Hold", "target": null}'])

    class Proposal(BaseModel):
        action: str
        target: float | None = None

    llm = CodexChatModel(model_name="gpt-5.6-sol")
    result = llm.with_structured_output(Proposal).invoke("Decide")

    assert result == Proposal(action="Hold", target=None)
    schema = fake.runs[0][1]["output_schema"]
    assert schema["required"] == ["action", "target"]
    assert "default" not in schema["properties"]["target"]
    assert schema["additionalProperties"] is False
