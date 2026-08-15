"""Tests for the salience-aware retriever."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.protocols.retriever import RetrievalConfig
from ctxforge.retrieval.retrievers.salience import (
    SalienceRetriever,
    compute_salience_score,
)

# ---------------------------------------------------------------------------
# compute_salience_score unit tests
# ---------------------------------------------------------------------------

def test_compute_salience_score_zero_access():
    score = compute_salience_score(
        similarity=0.9, access_count=0, accessed_at=None
    )
    assert score == 0.0


def test_compute_salience_score_with_access():
    now = datetime.now(timezone.utc)
    score = compute_salience_score(
        similarity=1.0, access_count=1, accessed_at=now
    )
    expected = 1.0 * math.log(2) * 1.0  # recency ≈ 1.0 (just now)
    assert abs(score - expected) < 0.01


def test_compute_salience_score_recency_decay():
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    score = compute_salience_score(
        similarity=1.0, access_count=1, accessed_at=thirty_days_ago,
        half_life_days=30.0,
    )
    # reinforcement = log(2) ≈ 0.693, recency ≈ 0.5
    expected = 1.0 * math.log(2) * 0.5
    assert abs(score - expected) < 0.05


def test_compute_salience_score_no_accessed_at():
    score = compute_salience_score(
        similarity=0.8, access_count=5, accessed_at=None
    )
    expected = 0.8 * math.log(6) * 1.0
    assert abs(score - expected) < 0.01


def test_compute_salience_score_naive_datetime():
    naive_now = datetime.now()  # no tzinfo
    score = compute_salience_score(
        similarity=1.0, access_count=1, accessed_at=naive_now
    )
    # Should not raise; recency should be close to 1
    assert score > 0.5


# ---------------------------------------------------------------------------
# SalienceRetriever integration tests
# ---------------------------------------------------------------------------

def _make_memory(content, embedding, access_count, accessed_at=None, memory_id=None):
    mem = MemoryItem(
        user_id="u1",
        content=content,
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        embedding=embedding,
        access_count=access_count,
        accessed_at=accessed_at,
    )
    if memory_id:
        mem.memory_id = memory_id
    return mem


@pytest.mark.asyncio
async def test_salience_retriever_retrieve():
    now = datetime.now(timezone.utc)
    memories = [
        _make_memory("cold start", [1.0, 0.0], access_count=0, accessed_at=None),
        _make_memory("active recent", [1.0, 0.0], access_count=5, accessed_at=now),
        _make_memory("active old", [1.0, 0.0], access_count=5,
                      accessed_at=now - timedelta(days=60)),
    ]

    store = AsyncMock()
    store.get_by_user = AsyncMock(return_value=memories)

    async def embed(text):
        return [1.0, 0.0]

    retriever = SalienceRetriever(store, embed, half_life_days=30.0)
    results = await retriever.retrieve("query", "u1")

    # "active recent" should be first (high access + recent)
    assert results[0].memory.content == "active recent"
    # "cold start" should not appear (score=0)
    assert all(r.memory.content != "cold start" or r.score == 0.0 for r in results)


@pytest.mark.asyncio
async def test_salience_retriever_filters():
    now = datetime.now(timezone.utc)
    memories = [
        _make_memory("semantic", [1.0, 0.0], access_count=3, accessed_at=now),
    ]
    memories[0].type = MemoryType.EPISODIC

    store = AsyncMock()
    store.get_by_user = AsyncMock(return_value=memories)

    async def embed(text):
        return [1.0, 0.0]

    retriever = SalienceRetriever(store, embed)
    config = RetrievalConfig(memory_types=[MemoryType.SEMANTIC])
    results = await retriever.retrieve("query", "u1", config)
    # Episodic memory should be filtered out
    assert len(results) == 0


@pytest.mark.asyncio
async def test_salience_retriever_retrieve_related():
    now = datetime.now(timezone.utc)
    ref = _make_memory("reference", [1.0, 0.0], access_count=2,
                        accessed_at=now, memory_id="ref-1")
    other = _make_memory("related", [0.9, 0.1], access_count=3,
                          accessed_at=now, memory_id="other-1")

    store = AsyncMock()
    store.get = AsyncMock(return_value=ref)
    store.get_by_user = AsyncMock(return_value=[ref, other])

    async def embed(text):
        return [1.0, 0.0]

    retriever = SalienceRetriever(store, embed)
    results = await retriever.retrieve_related("ref-1", "u1", limit=5)
    # Should not include ref-1 itself
    assert all(r.memory.memory_id != "ref-1" for r in results)
