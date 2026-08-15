"""
Tests for CondenserPipeline.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import pytest

from ctxforge.compaction.importance import ImportanceCondenser
from ctxforge.compaction.pipeline import CondenserPipeline
from ctxforge.compaction.sliding_window import SlidingWindowCondenser
from ctxforge.compaction.summarizing import SummarizingCondenser
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.core.session import Session
from ctxforge.protocols.compactor import CompactionConfig

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    now = datetime.now(timezone.utc)
    return [
        Event(
            event_id=f"evt_{i}",
            type=EventType.USER if i % 2 == 0 else EventType.AGENT,
            content=f"Message content {i}",
            timestamp=now - timedelta(minutes=10 - i),
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_view(sample_events):
    """Create a CompactionView from sample events."""
    return CompactionView(events=tuple(sample_events))


@pytest.fixture
def session_with_events(sample_events):
    """Create a session with sample events."""
    session = Session(session_id="test_session", user_id="test_user")
    for event in sample_events:
        session.add_event(event)
    return session


# =============================================================================
# Mock Condenser for Testing
# =============================================================================


class MockCondenser:
    """Mock condenser for testing pipeline behavior."""

    def __init__(
        self,
        name: str = "mock",
        should_condense_value: bool = True,
        tokens_to_save: int = 100,
        generate_summary: bool = False,
    ):
        self._name = name
        self._should_condense_value = should_condense_value
        self._tokens_to_save = tokens_to_save
        self._generate_summary = generate_summary
        self.condense_call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def should_condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> bool:
        return self._should_condense_value

    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        self.condense_call_count += 1

        # Remove 2 events if there are enough
        events = list(view.events)
        if len(events) > 2:
            removed = events[:2]
            kept = events[2:]
            forgotten_ids = {e.event_id for e in removed}

            new_view = view.with_forgotten(
                forgotten_ids,
                summary="Mock summary" if self._generate_summary else None,
            )
            new_view = new_view.with_events(kept)

            return CondensationResult(
                view=new_view,
                events_forgotten_start_id=removed[0].event_id,
                events_forgotten_end_id=removed[-1].event_id,
                summary_generated=self._generate_summary,
                tokens_saved=self._tokens_to_save,
                metadata={"mock_key": f"mock_value_{self._name}"},
            )

        return CondensationResult(
            view=view,
            summary_generated=False,
            tokens_saved=0,
            metadata={"mock_key": "no_change"},
        )


# =============================================================================
# Test Pipeline Creation
# =============================================================================


class TestPipelineCreation:
    """Tests for CondenserPipeline creation and properties."""

    def test_empty_pipeline(self):
        """Can create empty pipeline."""
        pipeline = CondenserPipeline()
        assert len(pipeline) == 0
        assert pipeline.name == "pipeline(empty)"

    def test_pipeline_with_condensers(self):
        """Can create pipeline with condensers."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("second")
        pipeline = CondenserPipeline(c1, c2)

        assert len(pipeline) == 2
        assert pipeline.condensers == [c1, c2]

    def test_pipeline_name_composition(self):
        """Pipeline name shows all condenser names."""
        c1 = MockCondenser("alpha")
        c2 = MockCondenser("beta")
        c3 = MockCondenser("gamma")
        pipeline = CondenserPipeline(c1, c2, c3)

        assert pipeline.name == "pipeline(alpha -> beta -> gamma)"

    def test_pipeline_single_condenser_name(self):
        """Pipeline with single condenser has correct name."""
        c1 = MockCondenser("only")
        pipeline = CondenserPipeline(c1)

        assert pipeline.name == "pipeline(only)"


# =============================================================================
# Test Pipeline Should Condense
# =============================================================================


class TestPipelineShouldCondense:
    """Tests for CondenserPipeline.should_condense."""

    def test_empty_pipeline_should_not_condense(self, sample_view):
        """Empty pipeline never needs condensation."""
        pipeline = CondenserPipeline()
        assert not pipeline.should_condense(sample_view)

    def test_should_condense_if_any_condenser_wants(self, sample_view):
        """Returns True if any condenser wants to condense."""
        c1 = MockCondenser("first", should_condense_value=False)
        c2 = MockCondenser("second", should_condense_value=True)
        pipeline = CondenserPipeline(c1, c2)

        assert pipeline.should_condense(sample_view)

    def test_should_not_condense_if_none_want(self, sample_view):
        """Returns False if no condensers want to condense."""
        c1 = MockCondenser("first", should_condense_value=False)
        c2 = MockCondenser("second", should_condense_value=False)
        pipeline = CondenserPipeline(c1, c2)

        assert not pipeline.should_condense(sample_view)

    def test_should_condense_uses_config(self, sample_view):
        """Config is passed to condensers."""
        condenser = SlidingWindowCondenser()
        pipeline = CondenserPipeline(condenser)

        # With low threshold, should condense
        config_low = CompactionConfig(event_threshold=5)
        assert pipeline.should_condense(sample_view, config_low)

        # With high threshold, should not
        config_high = CompactionConfig(event_threshold=100)
        assert not pipeline.should_condense(sample_view, config_high)


# =============================================================================
# Test Pipeline Condense
# =============================================================================


class TestPipelineCondense:
    """Tests for CondenserPipeline.condense."""

    @pytest.mark.asyncio
    async def test_empty_pipeline_returns_view(self, sample_view):
        """Empty pipeline returns unchanged view."""
        pipeline = CondenserPipeline()
        result = await pipeline.condense(sample_view)

        assert isinstance(result, CondensationResult)
        assert result.view == sample_view
        assert result.tokens_saved == 0
        assert result.metadata.get("action") == "no_condensers"

    @pytest.mark.asyncio
    async def test_single_condenser(self, sample_view):
        """Single condenser in pipeline works."""
        condenser = MockCondenser("single", tokens_to_save=50)
        pipeline = CondenserPipeline(condenser)

        result = await pipeline.condense(sample_view)

        assert isinstance(result, CondensationResult)
        assert result.tokens_saved == 50
        assert condenser.condense_call_count == 1
        assert "single" in result.metadata.get("stages", {})

    @pytest.mark.asyncio
    async def test_chain_multiple_condensers(self, sample_view):
        """Multiple condensers run in sequence."""
        c1 = MockCondenser("first", tokens_to_save=100)
        c2 = MockCondenser("second", tokens_to_save=50)
        pipeline = CondenserPipeline(c1, c2)

        result = await pipeline.condense(sample_view)

        assert isinstance(result, CondensationResult)
        # Both condensers should have been called
        assert c1.condense_call_count == 1
        assert c2.condense_call_count == 1
        # Tokens should be accumulated
        assert result.tokens_saved == 150
        # Both should be in metadata
        stages = result.metadata.get("stages", {})
        assert "first" in stages
        assert "second" in stages

    @pytest.mark.asyncio
    async def test_events_reduced_through_pipeline(self, sample_view):
        """Events are reduced as they pass through pipeline."""
        c1 = MockCondenser("first")  # Removes 2 events
        c2 = MockCondenser("second")  # Removes 2 more events
        pipeline = CondenserPipeline(c1, c2)

        original_count = len(sample_view)
        result = await pipeline.condense(sample_view)

        # Should have 4 fewer events (2 per condenser)
        assert len(result.view) == original_count - 4

    @pytest.mark.asyncio
    async def test_summary_generated_tracking(self, sample_view):
        """Tracks if any condenser generated a summary."""
        c1 = MockCondenser("first", generate_summary=False)
        c2 = MockCondenser("second", generate_summary=True)
        pipeline = CondenserPipeline(c1, c2)

        result = await pipeline.condense(sample_view)

        assert result.summary_generated is True

    @pytest.mark.asyncio
    async def test_forgotten_event_range_tracking(self, sample_view):
        """Tracks first and last forgotten events across pipeline."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("second")
        pipeline = CondenserPipeline(c1, c2)

        result = await pipeline.condense(sample_view)

        # First condenser removes evt_0, evt_1
        # Second condenser removes evt_2, evt_3
        assert result.events_forgotten_start_id == "evt_0"
        assert result.events_forgotten_end_id == "evt_3"


# =============================================================================
# Test Pipeline Manipulation
# =============================================================================


class TestPipelineManipulation:
    """Tests for adding/removing condensers."""

    def test_add_condenser(self):
        """Can add condensers to pipeline."""
        pipeline = CondenserPipeline()
        c1 = MockCondenser("first")

        result = pipeline.add_condenser(c1)

        assert len(pipeline) == 1
        assert result is pipeline  # Returns self for chaining

    def test_add_condenser_chaining(self):
        """Can chain add_condenser calls."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("second")
        c3 = MockCondenser("third")

        pipeline = (
            CondenserPipeline()
            .add_condenser(c1)
            .add_condenser(c2)
            .add_condenser(c3)
        )

        assert len(pipeline) == 3
        assert pipeline.condensers == [c1, c2, c3]

    def test_insert_condenser(self):
        """Can insert condenser at specific position."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("second")
        c3 = MockCondenser("middle")

        pipeline = CondenserPipeline(c1, c2)
        pipeline.insert_condenser(1, c3)

        assert pipeline.condensers == [c1, c3, c2]

    def test_insert_condenser_at_start(self):
        """Can insert condenser at start."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("new_first")

        pipeline = CondenserPipeline(c1)
        pipeline.insert_condenser(0, c2)

        assert pipeline.condensers == [c2, c1]

    def test_remove_condenser(self):
        """Can remove condenser by index."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("second")
        c3 = MockCondenser("third")

        pipeline = CondenserPipeline(c1, c2, c3)
        pipeline.remove_condenser(1)

        assert pipeline.condensers == [c1, c3]

    def test_remove_condenser_invalid_index(self):
        """Remove with invalid index does nothing."""
        c1 = MockCondenser("first")
        pipeline = CondenserPipeline(c1)

        pipeline.remove_condenser(99)  # Should not raise

        assert len(pipeline) == 1

    def test_clear_pipeline(self):
        """Can clear all condensers."""
        c1 = MockCondenser("first")
        c2 = MockCondenser("second")

        pipeline = CondenserPipeline(c1, c2)
        result = pipeline.clear()

        assert len(pipeline) == 0
        assert result is pipeline


# =============================================================================
# Test Pipeline with Real Condensers
# =============================================================================


class TestPipelineWithRealCondensers:
    """Integration tests with actual condenser implementations."""

    @pytest.mark.asyncio
    async def test_sliding_then_importance(self, session_with_events):
        """Pipeline with SlidingWindow then Importance."""
        view = CompactionView.from_session(session_with_events)

        pipeline = CondenserPipeline(
            SlidingWindowCondenser(),
            ImportanceCondenser(),
        )

        config = CompactionConfig(keep_recent=5)
        result = await pipeline.condense(view, config)

        assert isinstance(result, CondensationResult)
        assert len(result.view) <= 5

    @pytest.mark.asyncio
    async def test_summarizing_then_sliding(self, session_with_events):
        """Pipeline with Summarizing then SlidingWindow."""
        view = CompactionView.from_session(session_with_events)

        async def mock_summarize(text: str, existing=None) -> str:
            return "Test summary of conversation."

        pipeline = CondenserPipeline(
            SummarizingCondenser(mock_summarize),
            SlidingWindowCondenser(),
        )

        config = CompactionConfig(keep_recent=3)
        result = await pipeline.condense(view, config)

        assert isinstance(result, CondensationResult)
        assert result.view.summary is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestPipelineEdgeCases:
    """Edge case tests for CondenserPipeline."""

    @pytest.mark.asyncio
    async def test_condenser_returning_view_directly(self, sample_view):
        """Handles condenser that returns CompactionView instead of Result."""

        class ViewReturningCondenser:
            @property
            def name(self) -> str:
                return "view_returner"

            def should_condense(self, view, config=None) -> bool:
                return True

            async def condense(self, view, config=None) -> CompactionView:
                # Return view directly, not wrapped in CondensationResult
                return view

        pipeline = CondenserPipeline(ViewReturningCondenser())
        result = await pipeline.condense(sample_view)

        assert isinstance(result, CondensationResult)
        assert result.view == sample_view

    @pytest.mark.asyncio
    async def test_default_config_used(self, sample_view):
        """Uses default config when none provided."""
        condenser = MockCondenser("test")
        pipeline = CondenserPipeline(condenser)

        # Should not raise without config
        result = await pipeline.condense(sample_view)
        assert isinstance(result, CondensationResult)

    def test_pipeline_name_with_real_condensers(self):
        """Pipeline name works with real condenser implementations."""
        pipeline = CondenserPipeline(
            SlidingWindowCondenser(),
            SummarizingCondenser(),
            ImportanceCondenser(),
        )

        assert "sliding_window" in pipeline.name
        assert "summarizing" in pipeline.name
        assert "importance" in pipeline.name
