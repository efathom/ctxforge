from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.protocols.context import IndexSearchResult
from ctxforge.retrieval.retrievers.vectorstore_temporal import VectorStoreTemporalRetriever
from ctxforge.storage.memory.memory import InMemoryMemoryStore


class _FakeIndexer:
    def __init__(self, results):
        self._results = results

    async def search(self, query: str, scope_id: str, limit: int, min_score: float = 0.0):
        # Ignore query/scope_id for this unit test; return deterministic candidates
        return self._results[:limit]


@pytest.mark.asyncio
async def test_temporal_vectorstore_prefers_recency_when_semantic_equal():
    store = InMemoryMemoryStore()

    now = datetime.now(timezone.utc)
    old = MemoryItem(
        memory_id="m_old",
        user_id="u1",
        content="old",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
        created_at=now - timedelta(days=30),
        updated_at=now - timedelta(days=30),
    )
    new = MemoryItem(
        memory_id="m_new",
        user_id="u1",
        content="new",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
        created_at=now - timedelta(days=1),
        updated_at=now - timedelta(days=1),
    )
    await store.add(old)
    await store.add(new)

    # Semantic scores equal; recency should break the tie
    indexer = _FakeIndexer(
        [
            IndexSearchResult(item_id="m_old", score=0.8),
            IndexSearchResult(item_id="m_new", score=0.8),
        ]
    )
    retriever = VectorStoreTemporalRetriever(
        memory_store=store,
        indexer=indexer,
        semantic_weight=0.7,
        recency_weight=0.3,
        half_life_days=7.0,
    )

    results = await retriever.retrieve("q", "u1")
    assert results[0].memory.memory_id == "m_new"


@pytest.mark.asyncio
async def test_temporal_vectorstore_min_score_filters_results():
    store = InMemoryMemoryStore()
    now = datetime.now(timezone.utc)
    mem = MemoryItem(
        memory_id="m1",
        user_id="u1",
        content="x",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
        created_at=now - timedelta(days=1),
        updated_at=now - timedelta(days=1),
    )
    await store.add(mem)

    indexer = _FakeIndexer([IndexSearchResult(item_id="m1", score=0.1)])
    retriever = VectorStoreTemporalRetriever(memory_store=store, indexer=indexer, semantic_weight=1.0, recency_weight=0.0)

    results = await retriever.retrieve("q", "u1", config=None)
    assert len(results) == 1

    # With a higher min_score, it should be filtered out
    from ctxforge.protocols.retriever import RetrievalConfig

    results2 = await retriever.retrieve("q", "u1", config=RetrievalConfig(limit=5, min_score=0.5))
    assert results2 == []


