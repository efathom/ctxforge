import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.context_engine import CtxForge
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


@pytest.mark.asyncio
async def test_fetch_memories_merges_user_and_global_without_retriever():
    cfg = DEFAULT_CONFIG.merge_with(
        {
            "scopes": {
                "enable_global": True,
                "global_scope_id": "global",
                "global_retrieval_limit": 10,
            }
        }
    )

    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()

    engine = CtxForge(
        config=cfg,
        session_store=session_store,
        memory_store=memory_store,
        retriever=None,  # force IMemoryStore.search path
    )

    user_id = "user-1"
    await engine.add_memory(MemoryFactory.semantic_memory(user_id=user_id, content="User likes coffee"))
    await engine.add_memory(MemoryFactory.semantic_memory(user_id="global", content="Coffee is a popular drink"))

    memories = await engine.search_memories(user_id=user_id, query="coffee", limit=10)
    contents = [m.content for m in memories]

    assert "User likes coffee" in contents
    assert "Coffee is a popular drink" in contents


@pytest.mark.asyncio
async def test_fetch_memories_does_not_recurse_when_user_is_global():
    cfg = DEFAULT_CONFIG.merge_with(
        {
            "scopes": {
                "enable_global": True,
                "global_scope_id": "global",
                "global_retrieval_limit": 10,
            }
        }
    )

    engine = CtxForge(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        retriever=None,
    )

    await engine.add_memory(MemoryFactory.semantic_memory(user_id="global", content="Shared memory"))

    # When requesting "global" as the user_id, we should not attempt to also query "global" again.
    memories = await engine.search_memories(user_id="global", query="Shared", limit=10)
    assert len(memories) == 1
    assert memories[0].user_id == "global"


