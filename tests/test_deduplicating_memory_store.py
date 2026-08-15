import pytest

from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.extraction.consolidation.deduplicator import DeduplicationConsolidator
from ctxforge.storage.memory.deduplicating import DeduplicatingMemoryStore
from ctxforge.storage.memory.memory import InMemoryMemoryStore


class _FakeConsolidator(DeduplicationConsolidator):
    def __init__(self, result):
        # Don't call parent init (we only need consolidate())
        self._result = result

    async def consolidate(self, new_items, existing_items):
        return self._result


@pytest.mark.asyncio
async def test_add_skips_duplicate_when_consolidator_returns_empty():
    base = InMemoryMemoryStore()
    store = DeduplicatingMemoryStore(base, consolidator=_FakeConsolidator(result=[]))

    mem = MemoryItem(
        memory_id="m1",
        user_id="u1",
        content="hello",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
    )
    out = await store.add(mem)
    assert out == ""


@pytest.mark.asyncio
async def test_add_updates_existing_when_marked_is_update():
    base = InMemoryMemoryStore()
    # Seed existing memory
    existing = MemoryItem(
        memory_id="m_existing",
        user_id="u1",
        content="hello",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
    )
    await base.add(existing)

    updated = MemoryItem(
        memory_id="m_new",
        user_id="u1",
        content="hello (better)",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.95,
        metadata={"is_update": True, "updates_memory_id": "m_existing"},
    )

    store = DeduplicatingMemoryStore(base, consolidator=_FakeConsolidator(result=[updated]))
    out = await store.add(updated)

    assert out == "m_existing"
    loaded = await base.get("m_existing")
    assert loaded is not None
    assert loaded.content == "hello (better)"
    assert "is_update" not in loaded.metadata
    assert "updates_memory_id" not in loaded.metadata


@pytest.mark.asyncio
async def test_add_batch_mixes_kept_duplicates_and_updates():
    base = InMemoryMemoryStore()
    await base.add(
        MemoryItem(
            memory_id="m_existing",
            user_id="u1",
            content="original",
            type=MemoryType.SEMANTIC,
            source=MemorySource.AGENT_INFERENCE,
            confidence_score=0.8,
        )
    )

    a = MemoryItem(
        memory_id="a",
        user_id="u1",
        content="a",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
    )
    b = MemoryItem(
        memory_id="b",
        user_id="u1",
        content="b",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.9,
    )
    update = MemoryItem(
        memory_id="u",
        user_id="u1",
        content="updated",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        confidence_score=0.95,
        metadata={"is_update": True, "updates_memory_id": "m_existing"},
    )

    # Consolidator says keep a and update existing; b is duplicate (dropped)
    consolidator = _FakeConsolidator(result=[a, update])
    store = DeduplicatingMemoryStore(base, consolidator=consolidator)

    ids = await store.add_batch([a, b, update])
    assert ids[0]  # a stored
    assert ids[1] == ""  # b skipped
    assert ids[2] == "m_existing"  # update applied


