from __future__ import annotations

import json

import pytest

from ctxforge.config.base import EngineConfig
from ctxforge.core.expertise import Expertise, ExpertiseSection
from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.factory import EngineFactory
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode
from ctxforge.storage import DeduplicatingMemoryStore, InMemoryMemoryStore, InMemorySessionStore
from ctxforge.storage.memory.expertise import InMemoryExpertiseStore


@pytest.mark.asyncio
async def test_answer_with_controller_uses_memory_graph_and_expertise_and_returns_final_answer():
    # Enable controller + graph; keep defaults simple and in-memory.
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 2,
                "max_llm_calls": 4,
                "memory_limit_per_iter": 5,
                "expertise_limit_per_iter": 5,
                "sources": {"memory": True, "graph": True, "expertise": True},
                "enable_query_planning": False,
            },
            "graph": {"enabled": True},
            "expertise": {"enabled": True},
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )

    session_store = InMemorySessionStore()
    memory_store = DeduplicatingMemoryStore(InMemoryMemoryStore())
    expertise_store = InMemoryExpertiseStore()
    factory = EngineFactory()
    engine = await factory.create(
        cfg,
        session_store=session_store,
        memory_store=memory_store,
        expertise_store=expertise_store,
    )

    user_id = "u1"
    session_id = "s1"

    # Seed memory
    await engine.add_memory(MemoryFactory.semantic_memory(user_id, "User likes spicy food"))

    # Seed expertise
    exp = Expertise(expertise_id="exp1", name="Demo expertise", description="")
    exp.add_item(section=ExpertiseSection.STRATEGIES, content="Avoid peanuts: user is allergic.", source="seed")
    await engine.save_expertise(exp)

    # Seed a minimal graph (node + edge + episode) so graph retrieval has something to find.
    graph_store = engine._graph_service.store  # type: ignore[union-attr]
    await graph_store.add_episodes(
        user_id,
        [
            GraphEpisode(
                episode_id="ep1",
                scope_id=user_id,
                content="User works for Acme.",
            )
        ],
    )
    await graph_store.upsert_nodes(
        user_id,
        [
            GraphNode(node_id="n1", scope_id=user_id, name="Acme", labels=["Organization"], summary="Employer"),
            GraphNode(node_id="n2", scope_id=user_id, name="User", labels=["Person"], summary="The user"),
        ],
    )
    await graph_store.upsert_edges(
        user_id,
        [
            GraphEdge(
                edge_id="e1",
                scope_id=user_id,
                source_node_id="n2",
                target_node_id="n1",
                edge_type="WORKS_FOR",
                fact="User WORKS_FOR Acme",
            )
        ],
    )

    # Router (JSON) then final answer.
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            json.dumps(
                {
                    "decision": "answer",
                    "evidence": ["User likes spicy food", "User works for Acme", "User is allergic to peanuts"],
                    "gaps": "None",
                    "draft_answer": "You work for Acme; you like spicy food; you are allergic to peanuts.",
                }
            ),
            "FINAL",
        ]
    )

    out = await engine.answer_with_controller(
        session_id=session_id,
        user_id=user_id,
        user_input="What do you know about my preferences and employer?",
        llm=llm,
        expertise_id="exp1",
    )

    assert out == "FINAL"
    assert llm.call_count == 2  # router + final


@pytest.mark.asyncio
async def test_prepare_context_with_controller_returns_context_for_agent_use():
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {"enabled": True, "max_iterations": 1, "max_llm_calls": 2, "enable_query_planning": False},
            "retrieval": {"strategy": "keyword", "rerank_enabled": False},
            "graph": {"enabled": True},
            "expertise": {"enabled": True},
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )
    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
        expertise_store=InMemoryExpertiseStore(),
    )
    await engine.add_memory(MemoryFactory.semantic_memory("u1", "User likes spicy food"))

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            json.dumps(
                {"decision": "answer", "evidence": ["User likes spicy food"], "gaps": "None", "draft_answer": "OK"}
            )
        ]
    )

    ctx = await engine.prepare_context_with_controller(
        session_id="s1",
        user_id="u1",
        user_input="What spicy food do I like?",
        llm=llm,
    )

    assert ctx is not None
    assert ctx.memories  # controller should have pulled the spicy memory
    assert "retrieval_controller" in ctx.metadata  # trace present


@pytest.mark.asyncio
async def test_answer_with_controller_can_iterate_and_changes_query():
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 2,
                "max_llm_calls": 4,
                "memory_limit_per_iter": 5,
                "expertise_limit_per_iter": 5,
                "sources": {"memory": True, "graph": True, "expertise": True},
                "enable_query_planning": False,
            },
            "graph": {"enabled": True},
            "expertise": {"enabled": True},
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )

    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
        expertise_store=InMemoryExpertiseStore(),
    )

    # Seed expertise for this test as well (controller should handle expertise source gracefully).
    exp = Expertise(expertise_id="exp1", name="Demo expertise", description="")
    exp.add_item(section=ExpertiseSection.STRATEGIES, content="Employer is Acme.", source="seed")
    await engine.save_expertise(exp)

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            json.dumps(
                {
                    "decision": "retrieve",
                    "evidence": [],
                    "gaps": "Need employer name",
                    "retrieval_query": "employer organization",
                }
            ),
            json.dumps(
                {
                    "decision": "answer",
                    "evidence": ["Employer is Acme"],
                    "gaps": "None",
                    "draft_answer": "Acme",
                }
            ),
            "FINAL",
        ]
    )

    out = await engine.answer_with_controller(
        session_id="s1",
        user_id="u1",
        user_input="Who is my employer?",
        llm=llm,
        expertise_id="exp1",
    )

    assert out == "FINAL"
    assert llm.call_count == 3  # router1 + router2 + final


@pytest.mark.asyncio
async def test_query_planning_decomposes_multi_entity_question():
    """When query planning is enabled, multi-entity questions produce sub-queries."""
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 1,
                "max_llm_calls": 4,
                "enable_query_planning": True,
                "max_sub_queries": 3,
            },
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )
    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
    )

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            # Planner response: decompose into sub-queries
            json.dumps({"sub_queries": ["Alice project timeline", "Bob project timeline"]}),
            # Router response: answer immediately
            json.dumps({"decision": "answer", "evidence": ["found info"], "gaps": "None", "draft_answer": "OK"}),
        ]
    )

    ctx = await engine.prepare_context(
        session_id="s1",
        user_id="u1",
        user_input="Compare what Alice and Bob said about the project timeline",
        use_controller=True,
        llm=llm,
    )

    assert ctx is not None
    trace = ctx.metadata.get("retrieval_controller", {})
    iters = trace.get("iterations", [])
    assert len(iters) >= 1
    # First iteration should have sub_queries from the planner
    assert len(iters[0].get("sub_queries", [])) == 2


@pytest.mark.asyncio
async def test_query_planning_single_entity_returns_original():
    """Simple questions should not be decomposed."""
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 1,
                "max_llm_calls": 4,
                "enable_query_planning": True,
            },
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )
    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
    )

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            # Planner returns single query (no decomposition)
            json.dumps({"sub_queries": ["What is my name?"]}),
            # Router
            json.dumps({"decision": "answer", "evidence": [], "gaps": "None", "draft_answer": "OK"}),
        ]
    )

    ctx = await engine.prepare_context(
        session_id="s1",
        user_id="u1",
        user_input="What is my name?",
        use_controller=True,
        llm=llm,
    )

    assert ctx is not None
    trace = ctx.metadata.get("retrieval_controller", {})
    iters = trace.get("iterations", [])
    # Single query => no sub_queries recorded (empty tuple serialized)
    sub_q = iters[0].get("sub_queries", ())
    assert len(sub_q) == 0


@pytest.mark.asyncio
async def test_reflection_with_gap_queries_triggers_retrieval():
    """When the router reflects with gap_queries, the controller uses them for next retrieval."""
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 3,
                "max_llm_calls": 6,
                "enable_query_planning": False,
            },
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )
    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
    )

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            # Iteration 1: reflect with gap queries
            json.dumps({
                "decision": "reflect",
                "evidence": ["partial info"],
                "gaps": "Missing employer details",
                "coverage_percentage": 0.3,
                "gap_queries": ["employer name", "employer location"],
                "reasoning": "Need more employer info",
            }),
            # Iteration 2: answer with sufficient coverage
            json.dumps({
                "decision": "answer",
                "evidence": ["employer is Acme", "located in SF"],
                "gaps": "None",
                "coverage_percentage": 0.9,
                "draft_answer": "Acme in SF",
            }),
            # Final answer
            "Acme in SF",
        ]
    )

    out = await engine.answer_with_controller(
        session_id="s1",
        user_id="u1",
        user_input="Tell me about my employer",
        llm=llm,
    )

    assert out == "Acme in SF"
    # 2 router calls + 1 final = 3
    assert llm.call_count == 3


@pytest.mark.asyncio
async def test_coverage_percentage_stops_retrieval():
    """When coverage meets threshold, the controller stops early."""
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 3,
                "max_llm_calls": 6,
                "enable_query_planning": False,
                "min_coverage_percentage": 0.7,
            },
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )
    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
    )

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            # Router reports high coverage with no gaps -> should stop
            json.dumps({
                "decision": "retrieve",
                "evidence": ["enough info"],
                "gaps": "None",
                "coverage_percentage": 0.85,
                "retrieval_query": "more stuff",
            }),
            # Final answer
            "Done",
        ]
    )

    out = await engine.answer_with_controller(
        session_id="s1",
        user_id="u1",
        user_input="Quick question",
        llm=llm,
    )

    assert out == "Done"
    # Only 1 router call (stopped due to coverage) + 1 final = 2
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_parallel_sub_queries_produce_same_results():
    """Parallel sub-query retrieval should produce the same results as sequential."""
    cfg = EngineConfig().merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval_controller": {
                "enabled": True,
                "max_iterations": 1,
                "max_llm_calls": 4,
                "enable_query_planning": True,
                "max_parallel_queries": 4,
            },
            "storage": {"session": {"backend": "memory"}, "memory": {"backend": "memory"}},
        }
    )
    engine = await EngineFactory().create(
        cfg,
        session_store=InMemorySessionStore(),
        memory_store=DeduplicatingMemoryStore(InMemoryMemoryStore()),
    )

    # Add several memories that should be found by different sub-queries
    await engine.add_memory(MemoryFactory.semantic_memory("u1", "User likes Python"))
    await engine.add_memory(MemoryFactory.semantic_memory("u1", "User lives in Seattle"))
    await engine.add_memory(MemoryFactory.semantic_memory("u1", "User works at Acme"))

    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(
        [
            # Planner decomposes into sub-queries
            json.dumps({
                "sub_queries": [
                    "What programming language does the user prefer?",
                    "Where does the user live?",
                    "Where does the user work?",
                ]
            }),
            # Router says answer
            json.dumps({
                "decision": "answer",
                "evidence": ["Python", "Seattle", "Acme"],
                "gaps": "None",
                "draft_answer": "OK",
            }),
            # Final answer
            "All good",
        ]
    )

    out = await engine.answer_with_controller(
        session_id="s1",
        user_id="u1",
        user_input="Tell me about the user's preferences, location, and job",
        llm=llm,
    )

    assert out == "All good"
    # 1 planner + 1 router + 1 final = 3
    assert llm.call_count == 3


