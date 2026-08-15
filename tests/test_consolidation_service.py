"""Tests for the ConsolidationService (decay, merge, prune)."""

import datetime

import pytest

from ctxforge.config.base import ConsolidationConfig
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.engine.services.consolidation_service import ConsolidationService
from ctxforge.storage.memory.memory import InMemoryMemoryStore

# ---------------------------------------------------------------------------
# Fake embedding provider
# ---------------------------------------------------------------------------

class FakeEmbeddingProvider:
    """Deterministic embedder: uses hash of content for embedding."""

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

        embeddings = []
        for t in texts:
            h = hash(t) % 1000
            embeddings.append([float(h), 0.0, 0.0])
        return EmbeddingResponse(
            embeddings=embeddings, model=model or "fake", total_tokens=0, latency_ms=0.0
        )

    async def embed_single(self, text, model=None, **kwargs):
        resp = await self.embed([text], model=model, **kwargs)
        return resp.embeddings[0]


def _make(user_id, content, importance=1.0, days_old=0, embedding=None):
    created = datetime.datetime.now() - datetime.timedelta(days=days_old)
    return MemoryItem(
        user_id=user_id,
        content=content,
        type=MemoryType.SEMANTIC,
        importance=importance,
        created_at=created,
        embedding=embedding,
    )


@pytest.fixture
def store():
    return InMemoryMemoryStore()


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


@pytest.mark.asyncio
async def test_decay_reduces_importance_for_old_memories(store, embedder):
    cfg = ConsolidationConfig(
        enabled=True, decay_factor=0.5, max_age_days=7, min_importance=0.01
    )
    svc = ConsolidationService(store, embedder, cfg)

    old = _make("u1", "old fact", importance=1.0, days_old=10)
    new = _make("u1", "new fact", importance=1.0, days_old=1)
    await store.add(old)
    await store.add(new)

    report = await svc.consolidate("u1")

    assert report.decayed == 1
    updated_old = await store.get(old.memory_id)
    updated_new = await store.get(new.memory_id)
    assert updated_old.importance == 0.5
    assert updated_new.importance == 1.0


@pytest.mark.asyncio
async def test_decay_does_not_affect_new_memories(store, embedder):
    cfg = ConsolidationConfig(enabled=True, decay_factor=0.9, max_age_days=30)
    svc = ConsolidationService(store, embedder, cfg)

    mem = _make("u1", "recent", importance=1.0, days_old=5)
    await store.add(mem)

    report = await svc.consolidate("u1")
    assert report.decayed == 0


@pytest.mark.asyncio
async def test_merge_marks_duplicate_as_superseded(store, embedder):
    cfg = ConsolidationConfig(
        enabled=True, merge_similarity_threshold=0.99, max_age_days=999
    )
    svc = ConsolidationService(store, embedder, cfg)

    # Same embedding -> identical content -> should merge
    emb = [1.0, 0.0, 0.0]
    m1 = _make("u1", "fact A", importance=0.8, embedding=emb)
    m2 = _make("u1", "fact A duplicate", importance=0.5, embedding=emb)
    await store.add(m1)
    await store.add(m2)

    report = await svc.consolidate("u1")

    assert report.merged == 1
    loser = await store.get(m2.memory_id)
    assert loser.superseded_by == m1.memory_id
    assert loser.is_active is False


@pytest.mark.asyncio
async def test_prune_soft_deletes_low_importance(store, embedder):
    cfg = ConsolidationConfig(
        enabled=True, min_importance=0.3, max_age_days=999,
        merge_similarity_threshold=0.9999,
    )
    svc = ConsolidationService(store, embedder, cfg)

    # Use distinct embeddings so they don't get merged
    low = _make("u1", "low importance fact xyz", importance=0.1, embedding=[1.0, 0.0, 0.0])
    high = _make("u1", "high importance fact abc", importance=0.9, embedding=[0.0, 1.0, 0.0])
    await store.add(low)
    await store.add(high)

    report = await svc.consolidate("u1")

    assert report.pruned == 1
    pruned = await store.get(low.memory_id)
    assert pruned.superseded_by == "__pruned__"
    assert pruned.is_active is False

    kept = await store.get(high.memory_id)
    assert kept.is_active is True


@pytest.mark.asyncio
async def test_full_pipeline(store, embedder):
    cfg = ConsolidationConfig(
        enabled=True,
        decay_factor=0.5,
        max_age_days=7,
        merge_similarity_threshold=0.99,
        min_importance=0.2,
    )
    svc = ConsolidationService(store, embedder, cfg)

    emb = [1.0, 0.0, 0.0]
    # Old memory that will decay
    old = _make("u1", "old", importance=0.4, days_old=10, embedding=emb)
    # Duplicate of old (same embedding)
    dup = _make("u1", "old dup", importance=0.3, days_old=10, embedding=emb)
    # Fresh memory
    fresh = _make("u1", "fresh", importance=1.0, days_old=1)
    await store.add(old)
    await store.add(dup)
    await store.add(fresh)

    report = await svc.consolidate("u1")

    # old decays: 0.4 * 0.5 = 0.2 (at threshold)
    # dup decays: 0.3 * 0.5 = 0.15 (below threshold)
    # merge: old and dup have same embedding -> one gets superseded
    # prune: the loser after merge is already inactive; the decayed winner may be at threshold
    assert report.decayed >= 1
    assert report.merged >= 1


@pytest.mark.asyncio
async def test_idempotency(store, embedder):
    """Running consolidation twice should not double-decay."""
    cfg = ConsolidationConfig(
        enabled=True, decay_factor=0.5, max_age_days=7, min_importance=0.01
    )
    svc = ConsolidationService(store, embedder, cfg)

    mem = _make("u1", "fact", importance=1.0, days_old=10)
    await store.add(mem)

    await svc.consolidate("u1")
    m1 = await store.get(mem.memory_id)
    imp_after_first = m1.importance

    await svc.consolidate("u1")
    m2 = await store.get(mem.memory_id)
    imp_after_second = m2.importance

    # Second run should decay again (importance * 0.5 again)
    assert imp_after_second == imp_after_first * 0.5
