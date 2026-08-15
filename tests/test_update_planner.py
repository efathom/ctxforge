import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.memory_update_service import MemoryUpdateService
from ctxforge.extraction.update_planner import LLMMemoryUpdatePlanner
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.update_planner import MemoryOperationType
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


@pytest.mark.asyncio
async def test_update_planner_parses_ops_and_fills_missing_items_with_add():
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            '{"operations": [{"op": "UPDATE", "new_temp_id": "n1", "target_memory_id": "m_old", "content": "User prefers tea"}]}'
        ]
    )
    planner = LLMMemoryUpdatePlanner(llm_provider=llm, default_model="mock-model")

    n1 = MemoryFactory.semantic_memory(user_id="u", content="User likes tea")
    n2 = MemoryFactory.semantic_memory(user_id="u", content="User likes cookies")

    ops = await planner.plan(
        user_id="u",
        query="I like tea and cookies",
        new_items=[n1, n2],
        user_candidates={"n1": [], "n2": []},
        global_candidates={"n1": [], "n2": []},
        model="mock-model",
    )

    # One UPDATE for n1 from the model, plus an ADD fallback for n2.
    assert any(o.op == MemoryOperationType.UPDATE and o.new_temp_id == "n1" for o in ops)
    assert any(o.op == MemoryOperationType.ADD and o.new_temp_id == "n2" for o in ops)


@pytest.mark.asyncio
async def test_engine_applies_update_and_deactivate_from_plan():
    store = InMemoryMemoryStore()
    session_store = InMemorySessionStore()

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            '{"operations": [{"op": "UPDATE", "new_temp_id": "n1", "target_memory_id": "m1", "content": "User hates coffee"}, {"op": "DELETE", "target_memory_id": "m2"}]}'
        ]
    )
    planner = LLMMemoryUpdatePlanner(llm_provider=llm, default_model="mock-model")

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "extraction": {
                "update_planning_enabled": True,
                "update_planning_candidates_per_item": 5,
            }
        }
    )

    _engine = CtxForge(
        config=cfg,
        session_store=session_store,
        memory_store=store,
        update_planner=planner,
    )
    svc = MemoryUpdateService(config=cfg, memory_store=store, update_planner=planner, memory_indexer=None)

    m1 = MemoryFactory.semantic_memory(user_id="u", content="User likes coffee")
    m1.memory_id = "m1"
    m2 = MemoryFactory.semantic_memory(user_id="u", content="User likes soda")
    m2.memory_id = "m2"
    await store.add(m1)
    await store.add(m2)

    new_item = MemoryFactory.semantic_memory(user_id="u", content="User hates coffee")
    await svc.plan_and_apply(user_id="u", query="I hate coffee", new_items=[new_item])

    updated = await store.get("m1")
    assert updated is not None
    assert updated.content == "User hates coffee"

    deleted = await store.get("m2")
    assert deleted is not None
    assert deleted.is_active is False


@pytest.mark.asyncio
async def test_engine_does_not_update_global_scope_when_not_allowed():
    store = InMemoryMemoryStore()
    session_store = InMemorySessionStore()

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            '{"operations": [{"op": "UPDATE", "new_temp_id": "n1", "target_memory_id": "g1", "target_scope_id": "global", "content": "Global updated"}]}'
        ]
    )
    planner = LLMMemoryUpdatePlanner(llm_provider=llm, default_model="mock-model")

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "scopes": {"enable_global": True, "global_scope_id": "global", "allow_global_writes": False},
            "extraction": {"update_planning_enabled": True, "update_planning_candidates_per_item": 5},
        }
    )

    _engine = CtxForge(
        config=cfg,
        session_store=session_store,
        memory_store=store,
        update_planner=planner,
    )
    svc = MemoryUpdateService(config=cfg, memory_store=store, update_planner=planner, memory_indexer=None)

    g1 = MemoryFactory.semantic_memory(user_id="global", content="Global original")
    g1.memory_id = "g1"
    await store.add(g1)

    new_item = MemoryFactory.semantic_memory(user_id="u", content="Some new item")
    await svc.plan_and_apply(user_id="u", query="x", new_items=[new_item])

    after = await store.get("g1")
    assert after is not None
    assert after.content == "Global original"


