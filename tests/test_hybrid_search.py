"""Tests for hybrid_search (semantic + keyword via RRF) on MemoryService."""

import pytest

from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.engine.services.memory_service import MemoryService
from ctxforge.retrieval.ranking import reciprocal_rank_fusion
from ctxforge.storage.memory.memory import InMemoryMemoryStore

# ------------------------------------------------------------------
# RRF unit tests
# ------------------------------------------------------------------

class TestReciprocalRankFusion:

    def test_single_list(self):
        items = ["a", "b", "c"]
        result = reciprocal_rank_fusion([items], key_fn=lambda x: x)
        assert result == ["a", "b", "c"]

    def test_two_lists_merge(self):
        list1 = ["a", "b", "c"]
        list2 = ["b", "c", "d"]
        result = reciprocal_rank_fusion([list1, list2], key_fn=lambda x: x)
        # "b" appears in both lists at good ranks -> should be first or second
        assert "b" in result[:2]
        assert len(result) == 4

    def test_limit(self):
        result = reciprocal_rank_fusion(
            [["a", "b", "c"], ["d", "e", "f"]],
            key_fn=lambda x: x,
            limit=3,
        )
        assert len(result) == 3

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([], key_fn=lambda x: x)
        assert result == []

    def test_deduplication(self):
        result = reciprocal_rank_fusion(
            [["a", "b"], ["a", "b"]],
            key_fn=lambda x: x,
        )
        assert len(result) == 2


# ------------------------------------------------------------------
# MemoryService.hybrid_search tests
# ------------------------------------------------------------------

def _make(user_id, content, keywords):
    return MemoryItem(
        user_id=user_id,
        content=content,
        type=MemoryType.SEMANTIC,
        keywords=keywords,
    )


@pytest.fixture
def store():
    return InMemoryMemoryStore()


@pytest.fixture
def service(store):
    return MemoryService(memory_store=store)


@pytest.mark.asyncio
async def test_hybrid_search_without_keywords(service, store):
    """When no keywords are given, hybrid_search falls back to semantic."""
    await store.add(_make("u1", "Alice likes coffee", ["coffee"]))
    results = await service.hybrid_search(user_id="u1", query="coffee")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_hybrid_search_merges_results(service, store):
    """Hybrid search merges semantic and keyword results."""
    # This memory matches keyword but not semantic (no word overlap with query)
    await store.add(_make("u1", "Bob prefers espresso", ["coffee", "espresso"]))
    # This memory matches semantic (word overlap with query)
    await store.add(_make("u1", "Alice likes coffee a lot", ["tea"]))

    results = await service.hybrid_search(
        user_id="u1", query="coffee", keywords=["coffee"]
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_hybrid_search_respects_limit(service, store):
    for i in range(10):
        await store.add(_make("u1", f"item {i} coffee", ["coffee"]))

    results = await service.hybrid_search(
        user_id="u1", query="coffee", keywords=["coffee"], limit=3
    )
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_keyword_search_delegates(service, store):
    await store.add(_make("u1", "Alice coffee", ["coffee"]))
    results = await service.keyword_search(user_id="u1", keywords=["coffee"])
    assert len(results) == 1
