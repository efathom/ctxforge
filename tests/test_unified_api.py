"""
Tests for the unified CtxForge API (refactored prepare_context / record_turn).

These tests exercise the *new* keyword-argument-based API directly, as opposed
to the deprecated method variants which are covered by the existing test files.
"""

from __future__ import annotations

import json

import pytest

from ctxforge.config.base import EngineConfig
from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.expertise import (
    ExpertiseSection,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.core.memory import MemoryFactory
from ctxforge.core.session import Session
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.services.fusion_service import TwoStepInputs
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.graph import GraphEdge
from ctxforge.storage import DeduplicatingMemoryStore, InMemoryMemoryStore, InMemorySessionStore
from ctxforge.storage.memory.expertise import InMemoryExpertiseStore

# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_engine():
    """Minimal engine with in-memory stores (no factory overhead)."""
    return CtxForge(
        config=DEFAULT_CONFIG,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        expertise_store=InMemoryExpertiseStore(),
    )


# =========================================================================
# prepare_context — expertise_id param
# =========================================================================

class TestPrepareContextWithExpertiseId:

    @pytest.mark.asyncio
    async def test_expertise_items_attached_via_expertise_id_param(self, simple_engine):
        """prepare_context(expertise_id=...) retrieves and attaches expertise items."""
        engine = simple_engine

        await engine.create_expertise("exp-1", "Test")
        await engine.add_expertise_item(
            expertise_id="exp-1",
            section=ExpertiseSection.STRATEGIES,
            content="Always greet the user",
        )
        await engine.add_expertise_item(
            expertise_id="exp-1",
            section=ExpertiseSection.FORMULAS,
            content="Price = Cost * 1.2",
        )

        context = await engine.prepare_context(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            expertise_id="exp-1",
        )

        assert context.expertise_id == "exp-1"
        assert len(context.expertise_items) == 2
        assert context.metadata["expertise_id"] == "exp-1"
        assert context.metadata["expertise_item_count"] == 2
        assert len(context.expertise_items_used) == 2

    @pytest.mark.asyncio
    async def test_no_expertise_when_param_omitted(self, simple_engine):
        """Without expertise_id the context has no expertise data."""
        context = await simple_engine.prepare_context(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
        )

        assert context.expertise_id is None
        assert context.expertise_items == []

    @pytest.mark.asyncio
    async def test_max_expertise_items_respected(self, simple_engine):
        """max_expertise_items caps the number of retrieved items."""
        engine = simple_engine
        await engine.create_expertise("exp-1", "Test")
        for i in range(5):
            await engine.add_expertise_item(
                expertise_id="exp-1",
                section=ExpertiseSection.STRATEGIES,
                content=f"Strategy {i}",
            )

        context = await engine.prepare_context(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            expertise_id="exp-1",
            max_expertise_items=2,
        )

        assert len(context.expertise_items) == 2


# =========================================================================
# prepare_context — return_session param
# =========================================================================

class TestPrepareContextReturnSession:

    @pytest.mark.asyncio
    async def test_return_session_flag(self, simple_engine):
        """return_session=True returns (Context, Session) tuple."""
        result = await simple_engine.prepare_context(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            return_session=True,
        )

        assert isinstance(result, tuple)
        context, session = result
        assert context is not None
        assert session is not None
        assert session.session_id == "s1"
        assert context.session_id == "s1"

    @pytest.mark.asyncio
    async def test_return_session_with_expertise(self, simple_engine):
        """return_session and expertise_id compose together."""
        engine = simple_engine
        await engine.create_expertise("exp-1", "Test")
        await engine.add_expertise_item(
            expertise_id="exp-1",
            section=ExpertiseSection.STRATEGIES,
            content="Be nice",
        )

        context, session = await engine.prepare_context(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            expertise_id="exp-1",
            return_session=True,
        )

        assert context.expertise_id == "exp-1"
        assert len(context.expertise_items) == 1
        assert session.session_id == "s1"


# =========================================================================
# prepare_context — return_two_step_inputs param
# =========================================================================

class TestPrepareContextReturnTwoStepInputs:

    @pytest.mark.asyncio
    async def test_return_two_step_inputs_flag(self):
        """return_two_step_inputs=True returns TwoStepInputs."""
        session_store = InMemorySessionStore()
        await session_store.save(Session(session_id="s", user_id="u"))

        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=session_store,
            memory_store=InMemoryMemoryStore(),
        )

        result = await engine.prepare_context(
            session_id="s",
            user_id="u",
            user_input="Hello",
            return_two_step_inputs=True,
        )

        assert isinstance(result, TwoStepInputs)
        assert result.memory_context is not None
        assert isinstance(result.memory_messages, list)


# =========================================================================
# prepare_context — use_controller param
# =========================================================================

class TestPrepareContextUseController:

    @pytest.mark.asyncio
    async def test_use_controller_requires_llm(self, simple_engine):
        """use_controller=True without llm raises ValueError."""
        with pytest.raises(ValueError, match="requires llm"):
            await simple_engine.prepare_context(
                session_id="s1",
                user_id="u1",
                user_input="Hello",
                use_controller=True,
            )

    @pytest.mark.asyncio
    async def test_use_controller_returns_context_with_trace(self):
        """use_controller=True returns a Context with controller metadata."""
        cfg = EngineConfig().merge_with(
            {
                "llm": {"provider": "mock"},
                "retrieval_controller": {"enabled": True, "max_iterations": 1, "max_llm_calls": 2},
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

        ctx = await engine.prepare_context(
            session_id="s1",
            user_id="u1",
            user_input="What spicy food do I like?",
            use_controller=True,
            llm=llm,
        )

        assert ctx is not None
        assert ctx.memories
        assert "retrieval_controller" in ctx.metadata


# =========================================================================
# record_turn — unified with expertise feedback
# =========================================================================

class TestRecordTurnUnified:

    @pytest.mark.asyncio
    async def test_basic_record_turn_returns_none(self, simple_engine):
        """record_turn without feedback params returns None."""
        result = await simple_engine.record_turn(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            assistant_response="Hi!",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_record_turn_with_expertise_feedback(self):
        """record_turn with expertise_items_used + outcome runs reflection."""
        from ctxforge.expertise.reflector import MockReflector

        expertise_store = InMemoryExpertiseStore()
        mock_reflector = MockReflector(
            feedback_map={"strat-00001": UsageFeedback.HELPFUL},
        )

        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=expertise_store,
            reflector=mock_reflector,
        )

        await engine.create_expertise("exp-1", "Test")
        item = await engine.add_expertise_item(
            expertise_id="exp-1",
            section=ExpertiseSection.STRATEGIES,
            content="Strategy",
        )
        initial_helpful = item.helpful_count

        updated = await engine.record_turn(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            assistant_response="Hi!",
            expertise_items_used=[item.item_id],
            outcome=TurnOutcome.SUCCESS,
            expertise_id="exp-1",
        )

        assert updated is not None
        loaded = await engine.load_expertise("exp-1")
        updated_item = loaded.get_item(item.item_id)
        assert updated_item.helpful_count > initial_helpful

    @pytest.mark.asyncio
    async def test_record_turn_feedback_without_outcome_skips_reflection(self, simple_engine):
        """Providing expertise_items_used without outcome does NOT trigger reflection."""
        result = await simple_engine.record_turn(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            assistant_response="Hi!",
            expertise_items_used=["item-1"],
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_record_turn_feedback_without_items_skips_reflection(self, simple_engine):
        """Providing outcome without expertise_items_used does NOT trigger reflection."""
        result = await simple_engine.record_turn(
            session_id="s1",
            user_id="u1",
            user_input="Hello",
            assistant_response="Hi!",
            outcome=TurnOutcome.SUCCESS,
        )

        assert result is None


# =========================================================================
# Deprecation warnings
# =========================================================================

class TestDeprecationWarnings:

    @pytest.mark.asyncio
    async def test_prepare_context_with_session_emits_warning(self, simple_engine):
        with pytest.warns(DeprecationWarning, match="prepare_context_with_session"):
            await simple_engine.prepare_context_with_session(
                session_id="s1",
                user_id="u1",
                user_input="Hello",
            )

    @pytest.mark.asyncio
    async def test_prepare_context_with_expertise_emits_warning(self, simple_engine):
        await simple_engine.create_expertise("exp-1", "Test")

        with pytest.warns(DeprecationWarning, match="prepare_context_with_expertise"):
            await simple_engine.prepare_context_with_expertise(
                session_id="s1",
                user_id="u1",
                user_input="Hello",
                expertise_id="exp-1",
            )

    @pytest.mark.asyncio
    async def test_record_turn_with_feedback_emits_warning(self, simple_engine):
        with pytest.warns(DeprecationWarning, match="record_turn_with_feedback"):
            await simple_engine.record_turn_with_feedback(
                session_id="s1",
                user_id="u1",
                user_input="Hello",
                assistant_response="Hi!",
                expertise_items_used=[],
                outcome=TurnOutcome.SUCCESS,
            )

    @pytest.mark.asyncio
    async def test_prepare_two_step_inputs_emits_warning(self):
        session_store = InMemorySessionStore()
        await session_store.save(Session(session_id="s", user_id="u"))
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=session_store,
            memory_store=InMemoryMemoryStore(),
        )

        with pytest.warns(DeprecationWarning, match="prepare_two_step_inputs"):
            await engine.prepare_two_step_inputs(
                session_id="s",
                user_id="u",
                user_input="Hello",
            )

    @pytest.mark.asyncio
    async def test_answer_two_step_emits_warning(self):
        session_store = InMemorySessionStore()
        await session_store.save(Session(session_id="s", user_id="u"))
        cfg = DEFAULT_CONFIG.merge_with({"fusion": {"enabled": True}})
        engine = CtxForge(
            config=cfg,
            session_store=session_store,
            memory_store=InMemoryMemoryStore(),
        )
        llm = MockLLMProvider(latency_ms=0)
        llm.set_responses(["ANSWER"])

        with pytest.warns(DeprecationWarning, match="answer_two_step"):
            await engine.answer_two_step(
                session_id="s",
                user_id="u",
                user_input="Hello",
                llm=llm,
            )

    @pytest.mark.asyncio
    async def test_answer_with_controller_emits_warning(self):
        cfg = EngineConfig().merge_with(
            {
                "llm": {"provider": "mock"},
                "retrieval_controller": {"enabled": True, "max_iterations": 1, "max_llm_calls": 2},
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

        llm = MockLLMProvider(latency_ms=0)
        llm.set_responses(
            [
                json.dumps({"decision": "answer", "evidence": [], "gaps": "None", "draft_answer": "OK"}),
                "FINAL",
            ]
        )

        with pytest.warns(DeprecationWarning, match="answer_with_controller"):
            await engine.answer_with_controller(
                session_id="s1",
                user_id="u1",
                user_input="Hello",
                llm=llm,
            )


# =========================================================================
# helpers module
# =========================================================================

class TestHelpersModule:

    @pytest.mark.asyncio
    async def test_helpers_answer_two_step_no_graph(self):
        """helpers.answer_two_step falls back to single LLM call without graph."""
        from ctxforge.helpers import answer_two_step

        session_store = InMemorySessionStore()
        await session_store.save(Session(session_id="s", user_id="u"))
        cfg = DEFAULT_CONFIG.merge_with({"fusion": {"enabled": True}})
        engine = CtxForge(
            config=cfg,
            session_store=session_store,
            memory_store=InMemoryMemoryStore(),
        )

        llm = MockLLMProvider(latency_ms=0)
        llm.set_responses(["ONLY"])

        out = await answer_two_step(
            engine,
            llm,
            session_id="s",
            user_id="u",
            user_input="Hello?",
        )

        assert out == "ONLY"
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_helpers_answer_two_step_with_graph(self):
        """helpers.answer_two_step runs three LLM calls when graph is available."""
        from ctxforge.helpers import answer_two_step

        llm = MockLLMProvider(latency_ms=0)
        llm.set_responses(["KG", "MEM", "FINAL"])

        session_store = InMemorySessionStore()
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
            memory_store=InMemoryMemoryStore(),
            graph_store=graph_store,
        )
        await session_store.save(Session(session_id="s", user_id="u"))

        out = await answer_two_step(
            engine,
            llm,
            session_id="s",
            user_id="u",
            user_input="What happened to revenue?",
        )

        assert out == "FINAL"
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_helpers_answer_with_controller(self):
        """helpers.answer_with_controller runs controller + final LLM call."""
        from ctxforge.helpers import answer_with_controller

        cfg = EngineConfig().merge_with(
            {
                "llm": {"provider": "mock"},
                "retrieval_controller": {"enabled": True, "max_iterations": 1, "max_llm_calls": 2, "enable_query_planning": False},
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
                ),
                "FINAL",
            ]
        )

        out = await answer_with_controller(
            engine,
            llm,
            session_id="s1",
            user_id="u1",
            user_input="What spicy food do I like?",
        )

        assert out == "FINAL"
        assert llm.call_count == 2  # router + final
