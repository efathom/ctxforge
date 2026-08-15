"""
Tests for vectorstore-backed memory retrieval and indexing hooks.

Uses an in-test FakeVectorStore (no external services).
"""

import pytest

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.core.memory import MemoryFactory, MemoryItem, MemoryType
from ctxforge.engine.factory import EngineFactory
from ctxforge.retrieval.indexers.memory import MemoryIndexer
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class FakeVectorStore:
    """Minimal IVectorStore fake for MemoryIndexer."""

    def __init__(self):
        self._vectors = {}  # namespace -> id -> (embedding, metadata, content)
        self.initialized = False

    @property
    def name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 3

    @property
    def metric(self):
        return "cosine"

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        return None

    async def upsert(self, vectors, namespace=None):
        ns = namespace or ""
        self._vectors.setdefault(ns, {})
        for v in vectors:
            self._vectors[ns][v.id] = (v.embedding, v.metadata, v.content)
        return len(vectors)

    async def query(self, embedding, top_k=10, namespace=None, filters=None, include_embedding=False, include_metadata=True):
        from ctxforge.vectorstores.protocol import VectorQueryResult

        ns = namespace or ""
        items = list(self._vectors.get(ns, {}).items())
        # super-simplified similarity: dot product on first element only
        scored = []
        for vid, (emb, meta, content) in items:
            score = 1.0 if emb and embedding and emb[0] == embedding[0] else 0.0
            scored.append(VectorQueryResult(id=vid, score=score, metadata=meta, content=content))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def delete(self, ids, namespace=None, delete_all=False):
        ns = namespace or ""
        if delete_all:
            self._vectors.pop(ns, None)
            return 0
        for _id in ids:
            self._vectors.get(ns, {}).pop(_id, None)
        return len(ids)

    async def delete_namespace(self, namespace):
        self._vectors.pop(namespace or "", None)
        return True


class FakeEmbeddingProvider:
    """Deterministic embedder: first token length -> embedding[0]."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake"

    @property
    def embedding_dimension(self) -> int:
        return 3

    async def embed(self, texts, model=None, **kwargs):
        from ctxforge.protocols.llm import EmbeddingResponse
        embeddings = [[float(len(t)), 0.0, 0.0] for t in texts]
        return EmbeddingResponse(embeddings=embeddings, model=model or self.default_model, total_tokens=0, latency_ms=0.0)

    async def embed_single(self, text, model=None, **kwargs):
        resp = await self.embed([text], model=model, **kwargs)
        return resp.embeddings[0]


@pytest.mark.asyncio
async def test_vectorstore_semantic_retrieval_and_indexing_hooks():
    config = TESTING_CONFIG.merge_with(
        {
            "retrieval": {"strategy": "semantic"},
            "storage": {
                "memory": {
                    "backend": "chromadb",  # enables vectorstore path (we override with fake)
                }
            },
        }
    )

    factory = EngineFactory()
    engine = await factory.create(
        config=config,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    mem1 = MemoryFactory.semantic_memory(user_id="user_1", content="abc")  # len=3
    mem2 = MemoryFactory.semantic_memory(user_id="user_1", content="abcdef")  # len=6

    await engine.add_memory(mem1)
    await engine.add_memory(mem2)

    # Query should match mem1 based on our fake similarity rule
    ctx = await engine.prepare_context(
        session_id="s1",
        user_id="user_1",
        user_input="abc",
        include_history=False,
        include_memories=True,
        max_memories=5,
    )

    assert len(ctx.memories) >= 1
    assert any(m.content == "abc" for m in ctx.memories)


@pytest.mark.asyncio
async def test_indexer_uses_restatement_for_embedding():
    """Verify that MemoryIndexer indexes restatement when available."""
    vs = FakeVectorStore()
    ep = FakeEmbeddingProvider()
    indexer = MemoryIndexer(vector_store=vs, embedding_provider=ep)

    mem = MemoryItem(
        user_id="u1",
        content="He likes coffee",
        type=MemoryType.SEMANTIC,
        restatement="Bob likes coffee",
    )
    await indexer.index_item(mem, scope_id="u1")

    # The stored content in the vector store should be based on the restatement
    stored = vs._vectors.get("u1", {})
    assert len(stored) == 1
    record_data = list(stored.values())[0]
    # record_data is (embedding, metadata, content)
    # The embedding was generated from the restatement-based indexable content
    # FakeEmbeddingProvider uses len(text) as embedding[0]
    restatement_text = "Bob likes coffee | Type: semantic"
    raw_text = "He likes coffee | Type: semantic"
    embedding = record_data[0]
    assert embedding[0] == float(len(restatement_text))
    assert embedding[0] != float(len(raw_text))


@pytest.mark.asyncio
async def test_indexer_falls_back_to_content():
    """Verify that MemoryIndexer falls back to content when no restatement."""
    vs = FakeVectorStore()
    ep = FakeEmbeddingProvider()
    indexer = MemoryIndexer(vector_store=vs, embedding_provider=ep)

    mem = MemoryItem(
        user_id="u1",
        content="User likes tea",
        type=MemoryType.SEMANTIC,
    )
    await indexer.index_item(mem, scope_id="u1")

    stored = vs._vectors.get("u1", {})
    assert len(stored) == 1
    record_data = list(stored.values())[0]
    content_text = "User likes tea | Type: semantic"
    assert record_data[0][0] == float(len(content_text))


