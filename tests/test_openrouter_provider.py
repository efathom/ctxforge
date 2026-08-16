"""
Contract-style tests for the OpenRouter provider.

These tests do NOT hit the network.
"""

import pytest

from ctxforge.config.base import EngineConfig, LLMConfig, LLMProviderType
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import registry
from ctxforge.llm.openrouter_provider import (
    OPENROUTER_BASE_URL,
    OpenRouterConfig,
    OpenRouterLLMProvider,
)
from ctxforge.protocols.llm import ChatMessage


class _FakeUsage:
    prompt_tokens = 5
    completion_tokens = 7
    total_tokens = 12


class _FakeChoice:
    class _Message:
        content = "hello from fake"

    message = _Message()
    finish_reason = "stop"


class _FakeChatResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()

    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from fake",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class _FakeCompletions:
    async def create(self, **kwargs):
        return _FakeChatResponse()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def test_openrouter_registered():
    assert registry.get_llm("openrouter") is OpenRouterLLMProvider


def test_openrouter_config_defaults():
    cfg = OpenRouterConfig(api_key="sk-or-test")
    assert cfg.base_url == OPENROUTER_BASE_URL
    assert cfg.http_referer is None
    assert cfg.site_title is None


@pytest.mark.asyncio
async def test_openrouter_chat_contract(monkeypatch):
    cfg = OpenRouterConfig(api_key="sk-or-test", model="openai/gpt-4o-mini")
    provider = OpenRouterLLMProvider(cfg)
    assert provider.name == "openrouter"
    assert provider.default_model == "openai/gpt-4o-mini"

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from fake"
    assert resp.model == "openai/gpt-4o-mini"
    assert resp.total_tokens == 12


def test_factory_builds_openrouter_provider():
    config = EngineConfig(
        llm=LLMConfig(
            provider=LLMProviderType.OPENROUTER,
            model="openai/gpt-4o-mini",
            api_key="sk-or-test",
            extra_params={"http_referer": "https://example.com", "site_title": "My App"},
        )
    )
    provider = EngineFactory()._create_llm_provider(config)
    assert isinstance(provider, OpenRouterLLMProvider)
    assert provider.default_model == "openai/gpt-4o-mini"
    assert provider._config.base_url == OPENROUTER_BASE_URL
    assert provider._config.http_referer == "https://example.com"
    assert provider._config.site_title == "My App"
