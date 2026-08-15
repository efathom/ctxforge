"""
Contract-style tests for the local (sentence-transformers) embedding provider
and the OpenAI embedding provider's base_url override.

These tests do NOT download models or hit the network.
"""

import pytest

from ctxforge.engine.registry import registry
from ctxforge.llm.local_embedding_provider import (
    LocalEmbeddingConfig,
    LocalEmbeddingProvider,
)
from ctxforge.llm.openai_provider import OpenAIConfig, OpenAIEmbeddingProvider


class _FakeModel:
    """Minimal stand-in for a sentence_transformers.SentenceTransformer."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def encode(self, texts, **kwargs):
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def test_local_embedding_registered():
    assert registry.get_embedding("local") is LocalEmbeddingProvider
    assert registry.get_embedding("sentence_transformers") is LocalEmbeddingProvider
    assert registry.get_embedding("huggingface") is LocalEmbeddingProvider


def test_local_embedding_config_coercion():
    provider = LocalEmbeddingProvider(
        LocalEmbeddingConfig(model="BAAI/bge-small-en-v1.5", normalize_embeddings=False)
    )
    assert provider.default_model == "BAAI/bge-small-en-v1.5"
    assert provider.name == "local"
    assert provider._normalize is False

    # Should also accept a dict.
    provider2 = LocalEmbeddingProvider({"model": "intfloat/e5-small-v2", "batch_size": 8})
    assert provider2.default_model == "intfloat/e5-small-v2"
    assert provider2._batch_size == 8


@pytest.mark.asyncio
async def test_local_embedding_embed_contract(monkeypatch):
    provider = LocalEmbeddingProvider(LocalEmbeddingConfig(model="fake/model"))
    monkeypatch.setattr(provider, "_load_model", lambda: _FakeModel(dim=4))

    resp = await provider.embed(["hello", "world"])
    assert resp.model == "fake/model"
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) == 4
    assert provider.embedding_dimension == 4

    single = await provider.embed_single("hello")
    assert len(single) == 4


@pytest.mark.asyncio
async def test_local_embedding_empty_input(monkeypatch):
    provider = LocalEmbeddingProvider(LocalEmbeddingConfig(model="fake/model"))
    resp = await provider.embed([])
    assert resp.embeddings == []
    assert resp.total_tokens == 0


@pytest.mark.asyncio
async def test_openai_embedding_base_url(monkeypatch):
    import sys
    import types

    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Inject a fake `openai` module so the test doesn't require the optional
    # `openai` dependency to be installed.
    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    cfg = OpenAIConfig(
        api_key="sk-test",
        embedding_model="text-embedding-3-small",
        base_url="http://localhost:8080/v1",
    )
    provider = OpenAIEmbeddingProvider(cfg)
    await provider._get_client()

    assert captured.get("base_url") == "http://localhost:8080/v1"
    assert captured.get("api_key") == "sk-test"


@pytest.mark.asyncio
async def test_local_embedding_get_dimension(monkeypatch):
    provider = LocalEmbeddingProvider(LocalEmbeddingConfig(model="fake/model"))
    monkeypatch.setattr(provider, "_load_model", lambda: _FakeModel(dim=384))

    assert provider.embedding_dimension == 0  # not loaded yet
    dim = await provider.get_dimension()
    assert dim == 384
    assert provider.embedding_dimension == 384


@pytest.mark.asyncio
async def test_factory_derives_local_dimension(monkeypatch):
    from ctxforge.engine.factory import EngineFactory
    from ctxforge.llm.mock_provider import MockEmbeddingProvider

    factory = EngineFactory()

    provider = LocalEmbeddingProvider(LocalEmbeddingConfig(model="fake/model"))
    monkeypatch.setattr(provider, "_load_model", lambda: _FakeModel(dim=384))
    assert await factory._derive_embedding_dimension(provider) == 384

    # Providers without get_dimension() -> 0 (configured dimension is authoritative).
    assert await factory._derive_embedding_dimension(MockEmbeddingProvider(dimension=1536)) == 0
