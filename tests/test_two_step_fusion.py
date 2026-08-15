import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.session import Session
from ctxforge.engine.context_engine import CtxForge
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.graph import GraphEdge
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


@pytest.mark.asyncio
async def test_answer_two_step_runs_three_llm_calls_when_graph_available():
    # LLM returns: kg_answer, memory_answer, final_answer
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["KG", "MEM", "FINAL"])

    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()

    graph_store = InMemoryGraphStore()
    await graph_store.upsert_edges(
        "u",
        [
            GraphEdge(
                edge_id="e1",
                scope_id="u",
                source_node_id="n1",
                target_node_id="n2",
                edge_type="reports",
                fact="Revenue increased by 10% in Q4",
            )
        ],
    )

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    # Force planner path so graph section is rendered even without BFS/semantic.
                    "planner_mode": "global",
                    "methods": ["keyword"],
                    "seed_k": 0,
                    "bfs_max_depth": 0,
                    "bfs_edges_per_node": 0,
                    "include_entities": False,
                },
            },
            "fusion": {"enabled": True, "max_tokens": 50},
        }
    )

    engine = CtxForge(
        config=cfg,
        session_store=session_store,
        memory_store=memory_store,
        graph_store=graph_store,
    )

    # Ensure session exists with minimal history
    sess = Session(session_id="s", user_id="u")
    await session_store.save(sess)

    out = await engine.answer_two_step(
        session_id="s",
        user_id="u",
        user_input="What happened to revenue?",
        llm=llm,
    )
    assert out == "FINAL"
    assert llm.call_count == 3


@pytest.mark.asyncio
async def test_prepare_two_step_inputs_returns_context_and_does_not_call_llm():
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["SHOULD_NOT_BE_USED"])

    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    graph_store = InMemoryGraphStore()
    await graph_store.upsert_edges(
        "u",
        [
            GraphEdge(
                edge_id="e1",
                scope_id="u",
                source_node_id="n1",
                target_node_id="n2",
                edge_type="reports",
                fact="Revenue increased by 10% in Q4",
            )
        ],
    )

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "global",
                    "methods": ["keyword"],
                    "seed_k": 0,
                    "bfs_max_depth": 0,
                    "bfs_edges_per_node": 0,
                    "include_entities": False,
                },
            },
            "fusion": {"enabled": True, "max_tokens": 50},
        }
    )

    engine = CtxForge(
        config=cfg,
        session_store=session_store,
        memory_store=memory_store,
        graph_store=graph_store,
    )
    await session_store.save(Session(session_id="s", user_id="u"))

    inputs = await engine.prepare_two_step_inputs(
        session_id="s",
        user_id="u",
        user_input="What happened to revenue?",
    )
    assert inputs.memory_context is not None
    assert isinstance(inputs.memory_messages, list)
    # prepare_two_step_inputs should not call llm at all
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_answer_two_step_falls_back_to_single_call_when_no_graph():
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["ONLY"])

    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    cfg = DEFAULT_CONFIG.merge_with({"llm": {"provider": "mock"}, "fusion": {"enabled": True}})

    engine = CtxForge(config=cfg, session_store=session_store, memory_store=memory_store)
    await session_store.save(Session(session_id="s", user_id="u"))

    out = await engine.answer_two_step(
        session_id="s",
        user_id="u",
        user_input="Hello?",
        llm=llm,
    )
    assert out == "ONLY"
    assert llm.call_count == 1


