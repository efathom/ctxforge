"""
Tests for CompactionView and CondensationResult.

Sprint 1 tests for the view-based condensation abstraction.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.core.session import Session

# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_event(
    event_id: str,
    event_type: EventType = EventType.USER,
    content: str = "Test content",
    timestamp: datetime = None,
) -> Event:
    """Helper to create test events."""
    return Event(
        event_id=event_id,
        type=event_type,
        content=content,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_events():
    """Create a list of sample events for testing."""
    now = datetime.now(timezone.utc)
    return [
        create_test_event(
            f"event-{i}",
            EventType.USER if i % 2 == 0 else EventType.AGENT,
            f"Content for event {i}",
            now - timedelta(minutes=10 - i),
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_session(sample_events):
    """Create a sample session with events."""
    session = Session(
        session_id="test-session",
        user_id="test-user",
        summary="Previous conversation summary",
    )
    for event in sample_events:
        session.add_event(event)
    return session


# =============================================================================
# Test CompactionView - Creation
# =============================================================================


class TestCompactionViewCreation:
    """Tests for CompactionView factory methods."""

    def test_from_events_basic(self, sample_events):
        """Test creating view from events list."""
        view = CompactionView.from_events(sample_events)

        assert len(view) == 5
        assert len(view.forgotten_event_ids) == 0
        assert view.summary is None

    def test_from_events_with_existing_forgotten(self, sample_events):
        """Test creating view filters out forgotten events."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"event-0", "event-2"},
        )

        assert len(view) == 3
        assert view.is_forgotten("event-0")
        assert view.is_forgotten("event-2")
        assert not view.is_forgotten("event-1")

    def test_from_events_with_existing_summary(self, sample_events):
        """Test creating view with existing summary."""
        view = CompactionView.from_events(
            sample_events,
            existing_summary="Previous summary text",
        )

        assert view.summary == "Previous summary text"

    def test_from_session(self, sample_session):
        """Test creating view from Session object."""
        view = CompactionView.from_session(sample_session)

        assert len(view) == 5
        assert view.summary == "Previous conversation summary"

    def test_empty_factory(self):
        """Test creating empty view."""
        view = CompactionView.empty()

        assert len(view) == 0
        assert len(view.forgotten_event_ids) == 0
        assert not view  # bool should be False


# =============================================================================
# Test CompactionView - List Access
# =============================================================================


class TestCompactionViewListAccess:
    """Tests for CompactionView list-like access methods."""

    def test_len(self, sample_events):
        """Test __len__ returns correct count."""
        view = CompactionView.from_events(sample_events)
        assert len(view) == 5

    def test_iter(self, sample_events):
        """Test __iter__ returns events in order."""
        view = CompactionView.from_events(sample_events)
        events_list = list(view)

        assert len(events_list) == 5
        assert events_list[0].event_id == "event-0"

    def test_getitem_int(self, sample_events):
        """Test __getitem__ with integer index."""
        view = CompactionView.from_events(sample_events)

        assert view[0].event_id == "event-0"
        assert view[-1].event_id == "event-4"

    def test_getitem_slice(self, sample_events):
        """Test __getitem__ with slice."""
        view = CompactionView.from_events(sample_events)
        sliced = view[1:3]

        assert isinstance(sliced, list)
        assert len(sliced) == 2
        assert sliced[0].event_id == "event-1"
        assert sliced[1].event_id == "event-2"

    def test_bool_true(self, sample_events):
        """Test __bool__ returns True when has events."""
        view = CompactionView.from_events(sample_events)
        assert bool(view) is True

    def test_bool_false(self):
        """Test __bool__ returns False when empty."""
        view = CompactionView.empty()
        assert bool(view) is False


# =============================================================================
# Test CompactionView - Transformation
# =============================================================================


class TestCompactionViewTransformation:
    """Tests for CompactionView transformation methods."""

    def test_with_forgotten_creates_new_view(self, sample_events):
        """Test with_forgotten returns new immutable view."""
        view1 = CompactionView.from_events(sample_events)
        view2 = view1.with_forgotten({"event-0", "event-1"})

        # Original unchanged
        assert len(view1) == 5
        assert not view1.is_forgotten("event-0")

        # New view has fewer events
        assert len(view2) == 3
        assert view2.is_forgotten("event-0")
        assert view2.is_forgotten("event-1")

    def test_with_forgotten_updates_summary(self, sample_events):
        """Test with_forgotten can update summary."""
        view1 = CompactionView.from_events(sample_events)
        view2 = view1.with_forgotten(
            {"event-0"},
            summary="New summary after forgetting",
            summary_offset=1,
        )

        assert view2.summary == "New summary after forgetting"
        assert view2.summary_offset == 1

    def test_with_forgotten_preserves_summary_when_none_provided(self, sample_events):
        """Test with_forgotten preserves existing summary."""
        view1 = CompactionView.from_events(
            sample_events,
            existing_summary="Existing summary",
        )
        view2 = view1.with_forgotten({"event-0"})

        assert view2.summary == "Existing summary"

    def test_with_events_creates_new_view(self, sample_events):
        """Test with_events creates new view with different events."""
        view1 = CompactionView.from_events(
            sample_events,
            existing_forgotten={"old-event"},
            existing_summary="Summary",
        )

        new_events = sample_events[:2]
        view2 = view1.with_events(new_events)

        # New view has different events
        assert len(view2) == 2

        # Tracking preserved
        assert view2.is_forgotten("old-event")
        assert view2.summary == "Summary"

    def test_with_summary(self, sample_events):
        """Test with_summary creates new view with updated summary."""
        view1 = CompactionView.from_events(sample_events)
        view2 = view1.with_summary("New summary", offset=5)

        assert view1.summary is None
        assert view2.summary == "New summary"
        assert view2.summary_offset == 5

    def test_with_condensation_request(self, sample_events):
        """Test with_condensation_request sets flag."""
        view1 = CompactionView.from_events(sample_events)
        view2 = view1.with_condensation_request(True)

        assert not view1.unhandled_condensation_request
        assert view2.unhandled_condensation_request


# =============================================================================
# Test CompactionView - Utility Methods
# =============================================================================


class TestCompactionViewUtility:
    """Tests for CompactionView utility methods."""

    def test_is_forgotten_true(self, sample_events):
        """Test is_forgotten returns True for forgotten events."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"event-0"},
        )
        assert view.is_forgotten("event-0")

    def test_is_forgotten_false(self, sample_events):
        """Test is_forgotten returns False for non-forgotten events."""
        view = CompactionView.from_events(sample_events)
        assert not view.is_forgotten("event-0")

    def test_get_events_by_type(self, sample_events):
        """Test get_events_by_type filters correctly."""
        view = CompactionView.from_events(sample_events)

        user_events = view.get_events_by_type(EventType.USER)
        agent_events = view.get_events_by_type(EventType.AGENT)

        assert len(user_events) == 3  # events 0, 2, 4
        assert len(agent_events) == 2  # events 1, 3

    def test_get_user_events(self, sample_events):
        """Test get_user_events helper."""
        view = CompactionView.from_events(sample_events)
        user_events = view.get_user_events()

        assert len(user_events) == 3
        assert all(e.type == EventType.USER for e in user_events)

    def test_get_agent_events(self, sample_events):
        """Test get_agent_events helper."""
        view = CompactionView.from_events(sample_events)
        agent_events = view.get_agent_events()

        assert len(agent_events) == 2
        assert all(e.type == EventType.AGENT for e in agent_events)

    def test_get_conversation_events(self, sample_events):
        """Test get_conversation_events returns user and agent."""
        view = CompactionView.from_events(sample_events)
        conv_events = view.get_conversation_events()

        assert len(conv_events) == 5
        assert all(
            e.type in (EventType.USER, EventType.AGENT)
            for e in conv_events
        )

    def test_to_context_events(self, sample_events):
        """Test to_context_events returns list copy."""
        view = CompactionView.from_events(sample_events)
        events = view.to_context_events()

        assert isinstance(events, list)
        assert len(events) == 5
        # Should be a copy, not the internal tuple
        assert events is not view.events

    def test_forgotten_count(self, sample_events):
        """Test forgotten_count returns correct count."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"old-1", "old-2", "old-3"},
        )
        assert view.forgotten_count() == 3

    def test_total_events_processed(self, sample_events):
        """Test total_events_processed includes forgotten."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"old-1", "old-2"},
        )
        assert view.total_events_processed() == 7  # 5 current + 2 forgotten


# =============================================================================
# Test CompactionView - Immutability
# =============================================================================


class TestCompactionViewImmutability:
    """Tests for CompactionView immutability."""

    def test_events_tuple_immutable(self, sample_events):
        """Test events are stored as immutable tuple."""
        view = CompactionView.from_events(sample_events)

        assert isinstance(view.events, tuple)

    def test_forgotten_ids_frozenset_immutable(self, sample_events):
        """Test forgotten_event_ids is immutable frozenset."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"event-0"},
        )

        assert isinstance(view.forgotten_event_ids, frozenset)

    def test_frozen_dataclass(self, sample_events):
        """Test dataclass is frozen."""
        view = CompactionView.from_events(sample_events)

        with pytest.raises(FrozenInstanceError):
            view.summary = "Modified"


# =============================================================================
# Test CondensationResult
# =============================================================================


class TestCondensationResult:
    """Tests for CondensationResult class."""

    def test_creation_basic(self, sample_events):
        """Test basic result creation."""
        view = CompactionView.from_events(sample_events)
        result = CondensationResult(view=view)

        assert result.view is view
        assert result.summary_generated is False
        assert result.tokens_saved == 0
        assert result.events_forgotten_start_id is None
        assert result.events_forgotten_end_id is None

    def test_creation_with_all_fields(self, sample_events):
        """Test result creation with all fields."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"old-1", "old-2"},
        )
        result = CondensationResult(
            view=view,
            events_forgotten_start_id="old-1",
            events_forgotten_end_id="old-2",
            summary_generated=True,
            tokens_saved=100,
            metadata={"strategy": "summarizing"},
        )

        assert result.events_forgotten_start_id == "old-1"
        assert result.events_forgotten_end_id == "old-2"
        assert result.summary_generated is True
        assert result.tokens_saved == 100
        assert result.metadata["strategy"] == "summarizing"

    def test_events_forgotten_count(self, sample_events):
        """Test events_forgotten_count property."""
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten={"old-1", "old-2", "old-3"},
        )
        result = CondensationResult(view=view)

        assert result.events_forgotten_count == 3

    def test_success_property(self, sample_events):
        """Test success property."""
        view = CompactionView.from_events(sample_events)
        result = CondensationResult(view=view)

        assert result.success is True

    def test_with_metadata(self, sample_events):
        """Test with_metadata creates new result."""
        view = CompactionView.from_events(sample_events)
        result1 = CondensationResult(
            view=view,
            metadata={"key1": "value1"},
        )

        result2 = result1.with_metadata(key2="value2", key3="value3")

        # Original unchanged
        assert "key2" not in result1.metadata

        # New result has all metadata
        assert result2.metadata["key1"] == "value1"
        assert result2.metadata["key2"] == "value2"
        assert result2.metadata["key3"] == "value3"


# =============================================================================
# Test ICondenser Protocol (compile-time check)
# =============================================================================


class TestICondenserProtocol:
    """Tests for ICondenser protocol compliance."""

    def test_protocol_imported(self):
        """Test ICondenser protocol can be imported."""
        from ctxforge.compaction.view import ICondenser
        assert ICondenser is not None

    def test_protocol_is_runtime_checkable(self):
        """Test ICondenser is runtime checkable."""
        from ctxforge.compaction.view import ICondenser

        # Should be decorated with @runtime_checkable
        assert hasattr(ICondenser, "__protocol_attrs__") or hasattr(
            ICondenser, "_is_protocol"
        )

    def test_protocol_exported_from_compaction(self):
        """Test ICondenser is exported from compaction package."""
        from ctxforge.compaction import ICondenser
        assert ICondenser is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_events_list(self):
        """Test creating view from empty list."""
        view = CompactionView.from_events([])

        assert len(view) == 0
        assert view.forgotten_count() == 0
        assert view.total_events_processed() == 0

    def test_all_events_forgotten(self, sample_events):
        """Test view with all events forgotten."""
        all_ids = {f"event-{i}" for i in range(5)}
        view = CompactionView.from_events(
            sample_events,
            existing_forgotten=all_ids,
        )

        assert len(view) == 0
        assert view.forgotten_count() == 5

    def test_with_forgotten_no_matching_events(self, sample_events):
        """Test with_forgotten when IDs don't match current events."""
        view = CompactionView.from_events(sample_events)
        view2 = view.with_forgotten({"nonexistent-id"})

        # No events removed, but ID added to forgotten set
        assert len(view2) == 5
        assert view2.is_forgotten("nonexistent-id")

    def test_session_without_summary(self):
        """Test creating view from session without summary."""
        session = Session(
            session_id="test",
            user_id="user",
        )
        session.add_event(create_test_event("event-1"))

        view = CompactionView.from_session(session)

        assert view.summary is None

    def test_getitem_out_of_range(self, sample_events):
        """Test __getitem__ raises IndexError for out of range."""
        view = CompactionView.from_events(sample_events)

        with pytest.raises(IndexError):
            _ = view[100]

    def test_negative_index(self, sample_events):
        """Test negative indexing works correctly."""
        view = CompactionView.from_events(sample_events)

        assert view[-1].event_id == "event-4"
        assert view[-5].event_id == "event-0"
