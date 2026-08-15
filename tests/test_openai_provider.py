"""
Contract-style tests for OpenAI providers.

These tests do NOT hit the network. They patch the provider's client.
"""

import pytest

from ctxforge.llm.openai_provider import OpenAIConfig, OpenAIEmbeddingProvider, OpenAILLMProvider
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
        self.usage = _FakeUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12)


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding):
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbeddingItem(i, e) for i, e in enumerate(embeddings)]
        self.usage = _FakeUsage(total_tokens=9)


class _FakeChatCompletions:
    async def create(self, **kwargs):
        return _FakeChatResponse("hello from fake")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeEmbeddings:
    async def create(self, **kwargs):
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        # embed with simple deterministic vectors
        embeddings = [[float(i), 0.0, 0.0] for i in range(len(inputs))]
        return _FakeEmbeddingResponse(embeddings)


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()
        self.embeddings = _FakeEmbeddings()


@pytest.mark.asyncio
async def test_openai_llm_provider_chat_contract(monkeypatch):
    cfg = OpenAIConfig(api_key="sk-test", model="gpt-4", embedding_model="text-embedding-3-small")
    provider = OpenAILLMProvider(cfg)

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from fake"
    assert resp.model == "gpt-4"
    assert resp.total_tokens == 12
    assert resp.input_tokens == 5
    assert resp.output_tokens == 7


@pytest.mark.asyncio
async def test_openai_embedding_provider_contract(monkeypatch):
    cfg = OpenAIConfig(api_key="sk-test", model="gpt-4", embedding_model="text-embedding-3-small")
    provider = OpenAIEmbeddingProvider(cfg)

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.embed(["a", "b"])
    assert resp.model == "text-embedding-3-small"
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) == 3

    single = await provider.embed_single("x")
    assert len(single) == 3


