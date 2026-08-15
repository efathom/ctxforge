"""
Integration tests for the compaction system.

Tests the interaction between:
- CompactionView and Session
- CondenserPipeline with multiple condensers
- StructuredSummarizingCondenser integration
- CompactionService with ICondenser
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from ctxforge.compaction.importance import ImportanceCondenser
from ctxforge.compaction.pipeline import CondenserPipeline
from ctxforge.compaction.sliding_window import SlidingWindowCondenser
from ctxforge.compaction.structured_summary import (
    StructuredSummarizingCondenser,
    StructuredSummary,
)
from ctxforge.compaction.summarizing import SummarizingCondenser
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.core.session import Session
from ctxforge.protocols.compactor import CompactionConfig

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_session():
    """Create a session with realistic conversation events."""
    session = Session(session_id="integration_test", user_id="test_user")

    now = datetime.now(timezone.utc)

    # System message
    session.add_event(Event(
        event_id="sys_1",
        type=EventType.SYSTEM,
        content="You are a helpful coding assistant.",
        timestamp=now - timedelta(minutes=30),
    ))

    # Conversation about building an API
    events_data = [
        (EventType.USER, "I want to build a REST API for managing tasks."),
        (EventType.AGENT, "Great! I can help you build a REST API. What language would you like?"),
        (EventType.USER, "Let's use Python with FastAPI."),
        (EventType.AGENT, "Excellent choice! FastAPI is modern and fast. Let's set up the project."),
        (EventType.TOOL_CALL, "create_directory(path='task_api')"),
        (EventType.TOOL_OUTPUT, "Directory created: task_api"),
        (EventType.TOOL_CALL, "create_file(path='task_api/main.py', content='...')"),
        (EventType.TOOL_OUTPUT, "File created: task_api/main.py"),
        (EventType.USER, "Can you add a Task model with title and description?"),
        (EventType.AGENT, "Sure! I'll add a Pydantic model for Task."),
        (EventType.TOOL_CALL, "edit_file(path='task_api/models.py', ...)"),
        (EventType.TOOL_OUTPUT, "File modified: task_api/models.py"),
        (EventType.USER, "Now add CRUD endpoints."),
        (EventType.AGENT, "I'll add create, read, update, and delete endpoints."),
        (EventType.TOOL_CALL, "edit_file(path='task_api/main.py', ...)"),
        (EventType.TOOL_OUTPUT, "File modified: task_api/main.py"),
        (EventType.USER, "Please add input validation."),
        (EventType.AGENT, "I'll add Pydantic validation to the endpoints."),
        (EventType.USER, "Also add authentication with JWT."),
        (EventType.AGENT, "Adding JWT authentication..."),
    ]

    for i, (event_type, content) in enumerate(events_data):
        session.add_event(Event(
            event_id=f"evt_{i+1}",
            type=event_type,
            content=content,
            timestamp=now - timedelta(minutes=29 - i),
        ))

    return session


@pytest.fixture
def large_session():
    """Create a session with many events for testing condensation."""
    session = Session(session_id="large_test", user_id="test_user")
    now = datetime.now(timezone.utc)

    # System message
    session.add_event(Event(
        event_id="sys_1",
        type=EventType.SYSTEM,
        content="You are a helpful assistant.",
        timestamp=now - timedelta(hours=2),
    ))

    # Create 100+ conversation events
    for i in range(150):
        event_type = EventType.USER if i % 2 == 0 else EventType.AGENT
        session.add_event(Event(
            event_id=f"evt_{i}",
            type=event_type,
            content=f"Message number {i} with some content for testing.",
            timestamp=now - timedelta(minutes=150 - i),
        ))

    return session


# =============================================================================
# Test View <-> Session Round-Trip
# =============================================================================


class TestViewSessionRoundTrip:
    """Test converting between Session and CompactionView."""

    def test_session_to_view(self, sample_session):
        """Session correctly converts to CompactionView."""
        view = CompactionView.from_session(sample_session)

        assert len(view) == len(sample_session.events)
        assert view.summary == sample_session.summary

        # Events should match
        for i, event in enumerate(view.events):
            assert event.event_id == sample_session.events[i].event_id

    def test_view_preserves_event_order(self, sample_session):
        """View preserves chronological event order."""
        view = CompactionView.from_session(sample_session)

        timestamps = [e.timestamp for e in view.events]
        assert timestamps == sorted(timestamps)

    def test_view_with_session_summary(self, sample_session):
        """View captures existing session summary."""
        sample_session.summary = "Previous conversation about API development."
        view = CompactionView.from_session(sample_session)

        assert view.summary == "Previous conversation about API development."

    def test_apply_condensation_to_session(self, sample_session):
        """Condensation result can be applied back to session."""
        view = CompactionView.from_session(sample_session)
        original_count = len(sample_session.events)

        # Condense: forget first half of events
        half = len(view) // 2
        events_to_forget = {e.event_id for e in view.events[:half]}
        new_view = view.with_forgotten(
            events_to_forget,
            summary="Summary of first half of conversation.",
        )

        # Apply back to session
        sample_session.events.clear()
        sample_session.events.extend(new_view.to_context_events())
        sample_session.summary = new_view.summary

        assert len(sample_session.events) == original_count - half
        assert sample_session.summary == "Summary of first half of conversation."


# =============================================================================
# Test Full Compaction Pipeline
# =============================================================================


class TestFullCompactionPipeline:
    """Test complete condensation pipeline scenarios."""

    @pytest.mark.asyncio
    async def test_sliding_window_condenser(self, sample_session):
        """SlidingWindowCondenser reduces event count."""
        view = CompactionView.from_session(sample_session)
        original_count = len(view)

        condenser = SlidingWindowCondenser()
        config = CompactionConfig(keep_recent=5)

        result = await condenser.condense(view, config)

        assert isinstance(result, CondensationResult)
        assert len(result.view) < original_count
        assert len(result.view) <= 6  # 5 recent + system

    @pytest.mark.asyncio
    async def test_summarizing_condenser(self, sample_session):
        """SummarizingCondenser creates summary."""
        view = CompactionView.from_session(sample_session)

        async def mock_summarizer(text: str, existing: Optional[str] = None) -> str:
            return "User asked to build a FastAPI REST API with CRUD and JWT auth."

        condenser = SummarizingCondenser(mock_summarizer)
        config = CompactionConfig(keep_recent=5)

        result = await condenser.condense(view, config)

        assert result.summary_generated is True
        assert "FastAPI" in result.view.summary

    @pytest.mark.asyncio
    async def test_importance_condenser(self, sample_session):
        """ImportanceCondenser keeps important events."""
        view = CompactionView.from_session(sample_session)

        condenser = ImportanceCondenser()
        config = CompactionConfig(keep_recent=8)

        result = await condenser.condense(view, config)

        # Should keep a mix of recent and important events
        assert len(result.view) <= len(view)

    @pytest.mark.asyncio
    async def test_structured_summarizing_condenser(self, sample_session):
        """StructuredSummarizingCondenser produces structured summary."""
        view = CompactionView.from_session(sample_session)

        async def mock_llm(messages, tools):
            return json.dumps({
                "user_context": "Building a REST API for task management",
                "completed_tasks": "Set up project, added Task model, created CRUD endpoints",
                "pending_tasks": "Add input validation, implement JWT authentication",
                "files_modified": "task_api/main.py, task_api/models.py",
            })

        condenser = StructuredSummarizingCondenser(
            llm_func=mock_llm,
            max_events=10,
            keep_first=1,
            keep_last=3,
        )

        result = await condenser.condense(view)

        assert result.summary_generated is True
        assert "REST API" in result.view.summary
        assert "Task model" in result.view.summary


# =============================================================================
# Test Pipeline with Multiple Condensers
# =============================================================================


class TestPipelineIntegration:
    """Test CondenserPipeline with multiple condensers."""

    @pytest.mark.asyncio
    async def test_summarize_then_window(self, large_session):
        """Pipeline: summarize old events, then apply sliding window."""
        view = CompactionView.from_session(large_session)
        original_count = len(view)

        async def mock_summarizer(text: str, existing: Optional[str] = None) -> str:
            return "Summary of earlier conversation."

        pipeline = CondenserPipeline(
            SummarizingCondenser(mock_summarizer),
            SlidingWindowCondenser(),
        )

        config = CompactionConfig(keep_recent=10)
        result = await pipeline.condense(view, config)

        # Should be significantly reduced
        assert len(result.view) < original_count
        assert len(result.view) <= 15  # ~10 recent + some system

    @pytest.mark.asyncio
    async def test_importance_then_sliding(self, large_session):
        """Pipeline: importance filter, then sliding window."""
        view = CompactionView.from_session(large_session)

        pipeline = CondenserPipeline(
            ImportanceCondenser(min_importance=0.5),
            SlidingWindowCondenser(),
        )

        config = CompactionConfig(keep_recent=20)
        result = await pipeline.condense(view, config)

        assert isinstance(result, CondensationResult)
        assert len(result.view) <= 25

    @pytest.mark.asyncio
    async def test_three_stage_pipeline(self, large_session):
        """Pipeline with three condensers."""
        view = CompactionView.from_session(large_session)

        async def mock_summarizer(text: str, existing: Optional[str] = None) -> str:
            return "Comprehensive conversation summary."

        pipeline = CondenserPipeline(
            ImportanceCondenser(min_importance=0.4),
            SummarizingCondenser(mock_summarizer),
            SlidingWindowCondenser(),
        )

        config = CompactionConfig(keep_recent=15)
        result = await pipeline.condense(view, config)

        # Pipeline metadata should track all stages
        assert "stages" in result.metadata
        assert result.tokens_saved > 0

    @pytest.mark.asyncio
    async def test_pipeline_accumulates_forgotten_events(self, sample_session):
        """Pipeline correctly tracks all forgotten events."""
        view = CompactionView.from_session(sample_session)
        original_ids = {e.event_id for e in view.events}

        pipeline = CondenserPipeline(
            SlidingWindowCondenser(),
            SlidingWindowCondenser(),  # Second pass
        )

        config = CompactionConfig(keep_recent=3)
        result = await pipeline.condense(view, config)

        # Remaining events should be subset of original
        remaining_ids = {e.event_id for e in result.view.events}
        assert remaining_ids.issubset(original_ids)

        # Forgotten should be tracked
        forgotten = result.view.forgotten_event_ids
        assert len(forgotten) > 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestCompactionEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_session(self):
        """Handles empty session gracefully."""
        session = Session(session_id="empty", user_id="test")
        view = CompactionView.from_session(session)

        condenser = SlidingWindowCondenser()
        config = CompactionConfig(keep_recent=5)

        result = await condenser.condense(view, config)

        assert isinstance(result, CondensationResult)
        assert len(result.view) == 0

    @pytest.mark.asyncio
    async def test_session_with_only_system_events(self):
        """Handles session with only system events."""
        session = Session(session_id="system_only", user_id="test")
        session.add_event(Event(
            event_id="sys_1",
            type=EventType.SYSTEM,
            content="System prompt",
        ))

        view = CompactionView.from_session(session)
        condenser = SlidingWindowCondenser()
        config = CompactionConfig(keep_recent=5, preserve_system_events=True)

        result = await condenser.condense(view, config)

        # System event should be preserved
        assert len(result.view) == 1
        assert result.view.events[0].type == EventType.SYSTEM

    @pytest.mark.asyncio
    async def test_pipeline_with_no_condensers(self, sample_session):
        """Empty pipeline returns view unchanged."""
        view = CompactionView.from_session(sample_session)
        original_count = len(view)

        pipeline = CondenserPipeline()
        result = await pipeline.condense(view)

        assert len(result.view) == original_count

    @pytest.mark.asyncio
    async def test_condense_idempotent(self, sample_session):
        """Multiple condensations don't cause errors."""
        view = CompactionView.from_session(sample_session)

        condenser = SlidingWindowCondenser()
        config = CompactionConfig(keep_recent=5)

        # First condensation
        result1 = await condenser.condense(view, config)

        # Second condensation on result
        result2 = await condenser.condense(result1.view, config)

        # Should not lose more events
        assert len(result2.view) <= len(result1.view)

    @pytest.mark.asyncio
    async def test_structured_summary_without_llm(self, sample_session):
        """StructuredSummarizingCondenser works without LLM."""
        view = CompactionView.from_session(sample_session)

        # No LLM provided - should use simple extraction
        condenser = StructuredSummarizingCondenser(
            max_events=10,
            keep_first=1,
            keep_last=3,
        )

        result = await condenser.condense(view)

        assert isinstance(result, CondensationResult)
        # Should still produce a summary (simple extraction)
        assert result.view.summary is not None


# =============================================================================
# Test StructuredSummary Integration
# =============================================================================


class TestStructuredSummaryIntegration:
    """Test StructuredSummary in realistic scenarios."""

    def test_tool_definition_valid_for_openai(self):
        """Tool definition is valid for OpenAI API."""
        tool = StructuredSummary.tool_definition()

        # Required structure for OpenAI
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"

    def test_summary_serialization_roundtrip(self):
        """Summary survives JSON serialization."""
        original = StructuredSummary(
            user_context="Building API",
            completed_tasks="Setup done",
            pending_tasks="Add auth",
            files_modified="main.py",
            key_decisions="Use FastAPI",
        )

        json_str = original.to_json()
        restored = StructuredSummary.from_json(json_str)

        assert restored.user_context == original.user_context
        assert restored.completed_tasks == original.completed_tasks
        assert restored.pending_tasks == original.pending_tasks
        assert restored.files_modified == original.files_modified
        assert restored.key_decisions == original.key_decisions

    def test_prompt_format_in_view_summary(self):
        """Structured summary prompt format works in view."""
        summary = StructuredSummary(
            user_context="Building REST API",
            completed_tasks="Project setup",
            pending_tasks="Authentication",
        )

        view = CompactionView(
            events=(),
            summary=summary.to_prompt_format(),
        )

        assert "# Conversation Summary" in view.summary
        assert "Building REST API" in view.summary


# =============================================================================
# Test Condenser Registration
# =============================================================================


class TestCondenserRegistry:
    """Test condenser registration in registry."""

    def test_condensers_are_registered(self):
        """All condensers are registered in registry."""
        from ctxforge.engine.registry import registry

        # Check standard condensers are registered
        sliding = registry.get_condenser("sliding_window")
        summarizing = registry.get_condenser("summarizing")
        importance = registry.get_condenser("importance")

        assert sliding is not None
        assert summarizing is not None
        assert importance is not None

    def test_new_condensers_are_registered(self):
        """New condensers (structured, pipeline) are registered."""
        from ctxforge.engine.registry import registry

        # Check new condensers are registered
        structured = registry.get_condenser("structured")
        pipeline = registry.get_condenser("pipeline")

        assert structured is not None
        assert pipeline is not None

    def test_can_instantiate_from_registry(self):
        """Can instantiate condensers from registry."""
        from ctxforge.engine.registry import registry

        sliding_cls = registry.get_condenser("sliding_window")
        condenser = sliding_cls()

        assert condenser.name == "sliding_window"

    def test_can_instantiate_structured_from_registry(self):
        """Can instantiate structured condenser from registry."""
        from ctxforge.engine.registry import registry

        structured_cls = registry.get_condenser("structured")
        condenser = structured_cls(max_events=50, keep_first=1, keep_last=5)

        assert "structured" in condenser.name

    def test_can_instantiate_pipeline_from_registry(self):
        """Can instantiate pipeline from registry."""
        from ctxforge.engine.registry import registry

        pipeline_cls = registry.get_condenser("pipeline")
        sliding_cls = registry.get_condenser("sliding_window")

        pipeline = pipeline_cls(sliding_cls())

        assert "pipeline" in pipeline.name


# =============================================================================
# Test Factory Configuration
# =============================================================================


class TestFactoryCondenserCreation:
    """Test factory creates condensers from configuration."""

    def test_factory_creates_sliding_window(self):
        """Factory creates sliding window condenser from config."""
        from ctxforge.config.base import (
            CompactionConfig,
            CompactionStrategyType,
            EngineConfig,
        )
        from ctxforge.engine.factory import EngineFactory

        factory = EngineFactory()
        config = EngineConfig(
            compaction=CompactionConfig(
                strategy=CompactionStrategyType.SLIDING_WINDOW,
            )
        )

        condenser = factory._create_condenser(config)

        assert condenser is not None
        assert condenser.name == "sliding_window"

    def test_factory_creates_structured_condenser(self):
        """Factory creates structured condenser from config."""
        from ctxforge.config.base import (
            CompactionConfig,
            CompactionStrategyType,
            EngineConfig,
        )
        from ctxforge.engine.factory import EngineFactory

        factory = EngineFactory()
        config = EngineConfig(
            compaction=CompactionConfig(
                strategy=CompactionStrategyType.STRUCTURED,
                structured_max_events=50,
                structured_keep_first=2,
                structured_keep_last=8,
            )
        )

        condenser = factory._create_condenser(config)

        assert condenser is not None
        assert "structured" in condenser.name
        assert condenser._max_events == 50
        assert condenser._keep_first == 2
        assert condenser._keep_last == 8

    def test_factory_creates_pipeline_condenser(self):
        """Factory creates pipeline condenser from config."""
        from ctxforge.config.base import (
            CompactionConfig,
            CompactionStrategyType,
            CondenserStepConfig,
            EngineConfig,
        )
        from ctxforge.engine.factory import EngineFactory

        factory = EngineFactory()
        config = EngineConfig(
            compaction=CompactionConfig(
                strategy=CompactionStrategyType.PIPELINE,
                pipeline=[
                    CondenserStepConfig(type="importance"),
                    CondenserStepConfig(type="sliding_window"),
                ],
            )
        )

        condenser = factory._create_condenser(config)

        assert condenser is not None
        assert "pipeline" in condenser.name
        assert len(condenser._condensers) == 2

    def test_factory_pipeline_with_step_config(self):
        """Factory passes step config to condensers."""
        from ctxforge.config.base import (
            CompactionConfig,
            CompactionStrategyType,
            CondenserStepConfig,
            EngineConfig,
        )
        from ctxforge.engine.factory import EngineFactory

        factory = EngineFactory()
        config = EngineConfig(
            compaction=CompactionConfig(
                strategy=CompactionStrategyType.PIPELINE,
                pipeline=[
                    CondenserStepConfig(
                        type="structured",
                        config={
                            "max_events": 75,
                            "keep_first": 3,
                            "keep_last": 12,
                        },
                    ),
                    CondenserStepConfig(type="sliding_window"),
                ],
            )
        )

        condenser = factory._create_condenser(config)

        assert condenser is not None
        assert len(condenser._condensers) == 2
        # First condenser should have custom config
        assert condenser._condensers[0]._max_events == 75

    def test_factory_empty_pipeline_returns_none(self):
        """Factory returns None for empty pipeline."""
        from ctxforge.config.base import (
            CompactionConfig,
            CompactionStrategyType,
            EngineConfig,
        )
        from ctxforge.engine.factory import EngineFactory

        factory = EngineFactory()
        config = EngineConfig(
            compaction=CompactionConfig(
                strategy=CompactionStrategyType.PIPELINE,
                pipeline=[],  # Empty pipeline
            )
        )

        condenser = factory._create_condenser(config)

        assert condenser is None
