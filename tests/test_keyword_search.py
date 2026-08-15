"""Tests for keyword_search on InMemoryMemoryStore."""

import pytest

from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.storage.memory.memory import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


def _make(user_id, content, keywords, persons=None, locations=None, topics=None):
    return MemoryItem(
        user_id=user_id,
        content=content,
        type=MemoryType.SEMANTIC,
        keywords=keywords,
        persons=persons or [],
        locations=locations or [],
        topics=topics or [],
    )


@pytest.mark.asyncio
async def test_keyword_search_basic(store):
    await store.add(_make("u1", "Alice likes coffee", ["coffee", "preference"]))
    await store.add(_make("u1", "Bob likes tea", ["tea", "preference"]))

    results = await store.keyword_search("u1", ["coffee"])
    assert len(results) == 1
    assert "coffee" in results[0].content.lower()


@pytest.mark.asyncio
async def test_keyword_search_overlap_ranking(store):
    await store.add(_make("u1", "Alice coffee morning", ["coffee"]))
    await store.add(_make("u1", "Alice coffee tea", ["coffee", "tea", "morning"]))

    results = await store.keyword_search("u1", ["coffee", "morning"])
    # Second item has 2 keyword overlaps (coffee + morning), first has 1
    assert len(results) == 2
    assert results[0].keywords == ["coffee", "tea", "morning"]
    assert results[1].keywords == ["coffee"]


@pytest.mark.asyncio
async def test_keyword_search_no_match(store):
    await store.add(_make("u1", "Alice likes coffee", ["coffee"]))
    results = await store.keyword_search("u1", ["python"])
    assert results == []


@pytest.mark.asyncio
async def test_keyword_search_respects_user_id(store):
    await store.add(_make("u1", "Alice likes coffee", ["coffee"]))
    await store.add(_make("u2", "Bob likes coffee", ["coffee"]))

    results = await store.keyword_search("u1", ["coffee"])
    assert len(results) == 1
    assert results[0].user_id == "u1"


@pytest.mark.asyncio
async def test_keyword_search_with_persons_filter(store):
    await store.add(_make("u1", "Alice coffee", ["coffee"], persons=["Alice"]))
    await store.add(_make("u1", "Bob coffee", ["coffee"], persons=["Bob"]))

    results = await store.keyword_search(
        "u1", ["coffee"], filters={"persons": ["Alice"]}
    )
    assert len(results) == 1
    assert results[0].persons == ["Alice"]


@pytest.mark.asyncio
async def test_keyword_search_with_topics_filter(store):
    await store.add(_make("u1", "travel Paris", ["paris"], topics=["travel"]))
    await store.add(_make("u1", "work Paris", ["paris"], topics=["work"]))

    results = await store.keyword_search(
        "u1", ["paris"], filters={"topics": ["travel"]}
    )
    assert len(results) == 1
    assert results[0].topics == ["travel"]


@pytest.mark.asyncio
async def test_keyword_search_limit(store):
    for i in range(10):
        await store.add(_make("u1", f"item {i}", ["common"]))

    results = await store.keyword_search("u1", ["common"], limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_keyword_search_case_insensitive(store):
    await store.add(_make("u1", "Coffee lover", ["Coffee"]))
    results = await store.keyword_search("u1", ["coffee"])
    assert len(results) == 1
