"""
Contract-style tests for Azure OpenAI providers.

These tests do NOT hit the network. They patch the provider's client.
"""

import pytest

from ctxforge.llm.azure_openai_provider import (
    AzureOpenAIConfig,
    AzureOpenAIEmbeddingProvider,
    AzureOpenAILLMProvider,
)
from ctxforge.protocols.llm import ChatMessage


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeChatResponse:
    class _Choice:
        class _Message:
            def __init__(self, content: str):
                self.content = content

        def __init__(self, content: str):
            self.message = self._Message(content)
            self.finish_reason = "stop"

    def __init__(self, content: str):
        self.choices = [self._Choice(content)]
        self.usage = _FakeUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7)


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding):
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbeddingItem(i, e) for i, e in enumerate(embeddings)]
        self.usage = _FakeUsage(total_tokens=9)


class _FakeChatCompletions:
    def __init__(self, fail_once: bool = False):
        self._fail_once = fail_once
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self._fail_once and self.calls == 1:
            # Simulate a transient 500 from the gateway (name-based retry in provider).
            class InternalServerError(Exception):
                pass

            raise InternalServerError("Error code: 500 - internal server error")
        return _FakeChatResponse("hello from azure fake")


class _FakeChat:
    def __init__(self, fail_once: bool = False):
        self.completions = _FakeChatCompletions(fail_once=fail_once)


class _FakeEmbeddings:
    def __init__(self, fail_once: bool = False):
        self._fail_once = fail_once
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self._fail_once and self.calls == 1:
            class InternalServerError(Exception):
                pass

            raise InternalServerError("Error code: 500 - internal server error")
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = [[float(i), 0.0, 0.0] for i in range(len(inputs))]
        return _FakeEmbeddingResponse(embeddings)


class _FakeClient:
    def __init__(self, fail_once: bool = False):
        self.chat = _FakeChat(fail_once=fail_once)
        self.embeddings = _FakeEmbeddings(fail_once=fail_once)


@pytest.mark.asyncio
async def test_azure_openai_llm_provider_chat_contract(monkeypatch):
    cfg = AzureOpenAIConfig(
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment="gpt-4o-mini",
        embedding_deployment="text-embedding-3-small",
    )
    provider = AzureOpenAILLMProvider(cfg)

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from azure fake"
    assert resp.model == "gpt-4o-mini"
    assert resp.total_tokens == 7
    assert resp.input_tokens == 3
    assert resp.output_tokens == 4


@pytest.mark.asyncio
async def test_azure_openai_llm_provider_retries_transient_500(monkeypatch):
    cfg = AzureOpenAIConfig(
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment="gpt-4o-mini",
        embedding_deployment="text-embedding-3-small",
        max_retries=2,
    )
    provider = AzureOpenAILLMProvider(cfg)

    fake_client = _FakeClient(fail_once=True)

    async def _fake_get_client():
        return fake_client

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from azure fake"
    assert fake_client.chat.completions.calls == 2


@pytest.mark.asyncio
async def test_azure_openai_embedding_provider_contract(monkeypatch):
    cfg = AzureOpenAIConfig(
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment="gpt-4o-mini",
        embedding_deployment="embed-deploy",
    )
    provider = AzureOpenAIEmbeddingProvider(cfg)

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.embed(["a", "b"])
    assert resp.model == "embed-deploy"
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) == 3

    single = await provider.embed_single("x")
    assert len(single) == 3


@pytest.mark.asyncio
async def test_azure_openai_embedding_provider_retries_transient_500(monkeypatch):
    cfg = AzureOpenAIConfig(
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment="gpt-4o-mini",
        embedding_deployment="embed-deploy",
        max_retries=2,
    )
    provider = AzureOpenAIEmbeddingProvider(cfg)

    fake_client = _FakeClient(fail_once=True)

    async def _fake_get_client():
        return fake_client

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.embed(["a", "b"])
    assert resp.model == "embed-deploy"
    assert fake_client.embeddings.calls == 2


