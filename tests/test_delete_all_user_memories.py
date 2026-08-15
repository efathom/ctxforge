import pytest

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.factory import EngineFactory
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


@pytest.mark.asyncio
async def test_delete_all_user_memories_deletes_all_for_user():
    cfg = TESTING_CONFIG
    engine = await EngineFactory().create(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
    )

    u1 = "u1"
    u2 = "u2"
    await engine.add_memory(MemoryFactory.semantic_memory(u1, "A"))
    await engine.add_memory(MemoryFactory.semantic_memory(u1, "B"))
    await engine.add_memory(MemoryFactory.semantic_memory(u2, "C"))

    deleted = await engine.delete_all_user_memories(u1)
    assert deleted == 2
    assert await engine.memory_store.count(u1) == 0
    assert await engine.memory_store.count(u2) == 1


