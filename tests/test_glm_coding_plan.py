"""GLM Coding Plan routing through CLI choices, the client and model tiers."""

import json
from unittest.mock import Mock

import httpx
import pytest

from cli import prompts
from tradingagents.llm_clients.factory import create_llm_client, create_tier_client

CODING_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"
STANDARD_URL = "https://open.bigmodel.cn/api/paas/v4/"
INTERNATIONAL_URL = "https://api.z.ai/api/paas/v4/"


@pytest.mark.parametrize("provider,url", [("glm", INTERNATIONAL_URL), ("glm-cn", CODING_URL)])
def test_region_selection_matches_default(monkeypatch, provider, url):
    def select_region(*args, **kwargs):
        choice = next(c for c in kwargs["choices"] if c.value[0] == provider)
        return Mock(ask=Mock(return_value=choice.value))

    monkeypatch.setattr(prompts.questionary, "select", select_region)
    assert prompts.ask_glm_region() == (provider, url)
    assert prompts.provider_default_url(provider.upper()) == url


@pytest.mark.parametrize(
    "provider,base_url,expected_url,expected_key",
    [
        ("glm-cn", None, CODING_URL, "cn-test-key"),
        ("glm-cn", CODING_URL, CODING_URL, "cn-test-key"),
        ("glm-cn", STANDARD_URL, STANDARD_URL, "cn-test-key"),
        ("glm", None, INTERNATIONAL_URL, "international-test-key"),
    ],
)
def test_chat_request_uses_regional_endpoint_and_key(
    monkeypatch, provider, base_url, expected_url, expected_key
):
    monkeypatch.setenv("ZHIPU_CN_API_KEY", "cn-test-key")
    monkeypatch.setenv("ZHIPU_API_KEY", "international-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "glm-test",
            "object": "chat.completion",
            "created": 0,
            "model": "glm-4.7",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        llm = create_llm_client(
            provider, "glm-4.7", base_url=base_url, http_client=http_client
        ).get_llm()
        assert llm.invoke("Reply ok").content == "ok"

    assert len(requests) == 1
    assert str(requests[0].url) == expected_url + "chat/completions"
    assert requests[0].headers["authorization"] == f"Bearer {expected_key}"
    assert json.loads(requests[0].content)["model"] == "glm-4.7"


@pytest.mark.parametrize("menu_url", [None, CODING_URL])
def test_backend_override_wins_for_env_and_interactive_selection(menu_url):
    assert prompts.resolve_backend_url("glm-cn", menu_url) == CODING_URL
    assert prompts.resolve_backend_url("glm-cn", menu_url, STANDARD_URL) == STANDARD_URL


def test_mainland_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("ZHIPU_CN_API_KEY", raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "international-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    with pytest.raises(ValueError, match="ZHIPU_CN_API_KEY"):
        create_llm_client("glm-cn", "glm-4.7").get_llm()


@pytest.mark.parametrize("tier", ["quick", "deep"])
def test_a_glm_cn_tier_uses_coding_plan_without_the_main_provider_url(monkeypatch, tier):
    monkeypatch.setenv("ZHIPU_CN_API_KEY", "cn-test-key")
    config = {
        "llm_provider": "openai",
        "backend_url": "https://api.openai.com/v1",
        f"{tier}_think_provider": "glm-cn",
        f"{tier}_think_llm": "glm-4.7",
    }

    llm = create_tier_client(config, tier).get_llm()

    assert llm.openai_api_base == CODING_URL
    assert llm.openai_api_key.get_secret_value() == "cn-test-key"
