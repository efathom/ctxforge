"""
Tests for timeline-based event retrieval.

Tests TimelineQuery, TimelineResult, TimelineFilter, TimelineService,
and Session timeline methods.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List

from ctxforge.config.base import TimelineConfig
from ctxforge.core.events import Event, EventType
from ctxforge.core.session import Session
from ctxforge.core.timeline import (
    TimelineFilter,
    TimelineQuery,
    TimelineResult,
    TimeRange,
)
from ctxforge.engine.services.timeline_service import TimelineService


def create_event(
    event_type: EventType,
    content: str,
    timestamp: datetime,
) -> Event:
    """Helper to create events with specific timestamps."""
    return Event(
        type=event_type,
        content=content,
        timestamp=timestamp,
    )


def create_test_events(num_turns: int = 3) -> List[Event]:
    """Create a series of test events with realistic timestamps."""
    events = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)

    for i in range(num_turns):
        # User message
        events.append(create_event(
            EventType.USER,
            f"User message {i + 1}",
            base_time + timedelta(minutes=i * 10),
        ))
        # Agent response
        events.append(create_event(
            EventType.AGENT,
            f"Agent response {i + 1}",
            base_time + timedelta(minutes=i * 10 + 1),
        ))
        # Tool call (optional)
        if i % 2 == 0:
            events.append(create_event(
                EventType.TOOL_CALL,
                f"Tool call {i + 1}",
                base_time + timedelta(minutes=i * 10 + 2),
            ))
            events.append(create_event(
                EventType.TOOL_OUTPUT,
                f"Tool output {i + 1}",
                base_time + timedelta(minutes=i * 10 + 3),
            ))

    return events


class TestTimeRange(unittest.TestCase):
    """Tests for TimeRange enum."""

    def test_enum_values(self):
        """TimeRange has expected values."""
        self.assertEqual(TimeRange.LAST_HOUR.value, "last_hour")
        self.assertEqual(TimeRange.LAST_24H.value, "last_24h")
        self.assertEqual(TimeRange.LAST_7D.value, "last_7d")
        self.assertEqual(TimeRange.TODAY.value, "today")
        self.assertEqual(TimeRange.THIS_SESSION.value, "this_session")

    def test_string_comparison(self):
        """TimeRange can be compared as strings."""
        self.assertEqual(TimeRange.LAST_HOUR, "last_hour")


class TestTimelineQuery(unittest.TestCase):
    """Tests for TimelineQuery."""

    def test_default_values(self):
        """Query has sensible defaults."""
        query = TimelineQuery()
        self.assertIsNone(query.start_time)
        self.assertIsNone(query.end_time)
        self.assertIsNone(query.time_range)
        self.assertIsNone(query.start_turn)
        self.assertIsNone(query.end_turn)
        self.assertIsNone(query.event_types)
        self.assertTrue(query.include_tool_events)
        self.assertEqual(query.max_events, 100)

    def test_resolve_time_bounds_last_hour(self):
        """LAST_HOUR resolves to correct bounds."""
        query = TimelineQuery(time_range=TimeRange.LAST_HOUR)
        ref_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        start, end = query.resolve_time_bounds(ref_time)

        self.assertEqual(end, ref_time)
        self.assertEqual(start, ref_time - timedelta(hours=1))

    def test_resolve_time_bounds_last_24h(self):
        """LAST_24H resolves to correct bounds."""
        query = TimelineQuery(time_range=TimeRange.LAST_24H)
        ref_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        start, end = query.resolve_time_bounds(ref_time)

        self.assertEqual(end, ref_time)
        self.assertEqual(start, ref_time - timedelta(days=1))

    def test_resolve_time_bounds_today(self):
        """TODAY resolves to start of day."""
        query = TimelineQuery(time_range=TimeRange.TODAY)
        ref_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        start, end = query.resolve_time_bounds(ref_time)

        self.assertEqual(end, ref_time)
        expected_start = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(start, expected_start)

    def test_resolve_time_bounds_this_session(self):
        """THIS_SESSION returns None bounds (no filtering)."""
        query = TimelineQuery(time_range=TimeRange.THIS_SESSION)
        start, end = query.resolve_time_bounds()

        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_resolve_time_bounds_explicit(self):
        """Explicit time bounds are returned as-is."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        query = TimelineQuery(start_time=start_time, end_time=end_time)

        start, end = query.resolve_time_bounds()
        self.assertEqual(start, start_time)
        self.assertEqual(end, end_time)

    def test_has_time_filter(self):
        """has_time_filter detects time filtering."""
        self.assertFalse(TimelineQuery().has_time_filter())
        self.assertTrue(TimelineQuery(
            time_range=TimeRange.LAST_HOUR
        ).has_time_filter())
        self.assertTrue(TimelineQuery(
            start_time=datetime.now(timezone.utc)
        ).has_time_filter())

    def test_has_turn_filter(self):
        """has_turn_filter detects turn filtering."""
        self.assertFalse(TimelineQuery().has_turn_filter())
        self.assertTrue(TimelineQuery(start_turn=0).has_turn_filter())
        self.assertTrue(TimelineQuery(end_turn=5).has_turn_filter())


class TestTimelineResult(unittest.TestCase):
    """Tests for TimelineResult."""

    def test_empty_result(self):
        """Empty result has correct defaults."""
        result = TimelineResult()
        self.assertEqual(result.events, [])
        self.assertEqual(result.total_matching, 0)
        self.assertIsNone(result.time_span)
        self.assertEqual(result.event_counts, {})
        self.assertTrue(result.is_empty())

    def test_to_summary_empty(self):
        """Empty result summary is informative."""
        result = TimelineResult()
        summary = result.to_summary()
        self.assertIn("No events found", summary)

    def test_to_summary_with_events(self):
        """Summary includes event counts."""
        events = create_test_events(2)
        result = TimelineResult(
            events=events,
            total_matching=len(events),
            event_counts={"user": 2, "agent": 2},
            turn_count=2,
        )
        summary = result.to_summary()
        self.assertIn("Events:", summary)
        self.assertIn("user: 2", summary)

    def test_get_events_by_type(self):
        """Can filter result events by type."""
        events = create_test_events(2)
        result = TimelineResult(events=events, total_matching=len(events))

        user_events = result.get_events_by_type(EventType.USER)
        self.assertEqual(len(user_events), 2)
        self.assertTrue(all(e.type == EventType.USER for e in user_events))


class TestTimelineFilter(unittest.TestCase):
    """Tests for TimelineFilter."""

    def setUp(self):
        """Create test events."""
        self.events = create_test_events(3)

    def test_filter_no_criteria(self):
        """No criteria returns all events."""
        query = TimelineQuery()
        result = TimelineFilter.filter_events(self.events, query)

        self.assertEqual(len(result.events), len(self.events))
        self.assertEqual(result.total_matching, len(self.events))

    def test_filter_by_event_type(self):
        """Can filter by event type."""
        query = TimelineQuery(event_types=[EventType.USER, EventType.AGENT])
        result = TimelineFilter.filter_events(self.events, query)

        self.assertTrue(all(
            e.type in (EventType.USER, EventType.AGENT)
            for e in result.events
        ))

    def test_filter_exclude_tool_events(self):
        """Can exclude tool events."""
        query = TimelineQuery(include_tool_events=False)
        result = TimelineFilter.filter_events(self.events, query)

        self.assertTrue(all(
            e.type not in (EventType.TOOL_CALL, EventType.TOOL_OUTPUT)
            for e in result.events
        ))

    def test_filter_max_events(self):
        """max_events limits result count."""
        query = TimelineQuery(max_events=3)
        result = TimelineFilter.filter_events(self.events, query)

        self.assertEqual(len(result.events), 3)
        # total_matching should be the count before limit
        self.assertGreaterEqual(result.total_matching, len(result.events))

    def test_filter_by_turn(self):
        """Can filter by conversation turn."""
        # Get only first turn
        query = TimelineQuery(start_turn=0, end_turn=1)
        result = TimelineFilter.filter_events(self.events, query)

        # Should have events from only the first turn
        user_events = [e for e in result.events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 1)
        self.assertIn("message 1", user_events[0].content)

    def test_filter_by_turn_last_n(self):
        """Can get last N turns."""
        # Get last turn only
        total_turns = TimelineFilter.count_turns(self.events)
        query = TimelineQuery(start_turn=total_turns - 1)
        result = TimelineFilter.filter_events(self.events, query)

        user_events = [e for e in result.events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 1)
        self.assertIn("message 3", user_events[0].content)

    def test_filter_event_counts(self):
        """Result includes event type counts."""
        query = TimelineQuery()
        result = TimelineFilter.filter_events(self.events, query)

        self.assertIn("user", result.event_counts)
        self.assertIn("agent", result.event_counts)
        self.assertEqual(result.event_counts["user"], 3)

    def test_filter_time_span(self):
        """Result includes time span."""
        query = TimelineQuery()
        result = TimelineFilter.filter_events(self.events, query)

        self.assertIsNotNone(result.time_span)
        start, end = result.time_span
        self.assertLess(start, end)

    def test_get_turns(self):
        """get_turns correctly groups events."""
        turns = TimelineFilter.get_turns(self.events)

        self.assertEqual(len(turns), 3)
        # Each turn starts with a USER event
        for turn in turns:
            self.assertEqual(turn[0].type, EventType.USER)

    def test_count_turns(self):
        """count_turns returns correct count."""
        count = TimelineFilter.count_turns(self.events)
        self.assertEqual(count, 3)


class TestTimelineService(unittest.TestCase):
    """Tests for TimelineService."""

    def setUp(self):
        """Create test session and service."""
        self.service = TimelineService()
        self.session = Session(user_id="test-user")
        for event in create_test_events(3):
            self.session.add_event(event)

    def test_query_session(self):
        """Can query session with timeline parameters."""
        query = TimelineQuery(max_events=5)
        result = self.service.query_session(self.session, query)

        self.assertEqual(len(result.events), 5)
        self.assertGreater(result.total_matching, 0)

    def test_get_recent_context(self):
        """get_recent_context returns recent events."""
        result = self.service.get_recent_context(
            self.session,
            hours=2.0,
            include_tools=True,
        )

        self.assertGreater(len(result.events), 0)

    def test_get_recent_context_no_tools(self):
        """get_recent_context can exclude tool events."""
        result = self.service.get_recent_context(
            self.session,
            hours=2.0,
            include_tools=False,
        )

        self.assertTrue(all(
            e.type not in (EventType.TOOL_CALL, EventType.TOOL_OUTPUT)
            for e in result.events
        ))

    def test_get_conversation_turns(self):
        """get_conversation_turns filters by turn number."""
        result = self.service.get_conversation_turns(
            self.session,
            start_turn=0,
            end_turn=2,
        )

        user_events = [e for e in result.events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 2)

    def test_get_last_n_turns(self):
        """get_last_n_turns returns last N turns."""
        result = self.service.get_last_n_turns(self.session, n=1)

        user_events = [e for e in result.events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 1)
        self.assertIn("message 3", user_events[0].content)

    def test_get_events_by_type(self):
        """get_events_by_type filters correctly."""
        result = self.service.get_events_by_type(
            self.session,
            event_types=[EventType.USER],
        )

        self.assertTrue(all(e.type == EventType.USER for e in result.events))
        self.assertEqual(len(result.events), 3)

    def test_get_user_assistant_exchanges(self):
        """get_user_assistant_exchanges excludes tool events."""
        result = self.service.get_user_assistant_exchanges(self.session)

        self.assertTrue(all(
            e.type in (EventType.USER, EventType.AGENT)
            for e in result.events
        ))

    def test_summarize_activity(self):
        """summarize_activity returns complete summary."""
        summary = self.service.summarize_activity(self.session)

        self.assertIn("event_counts", summary)
        self.assertIn("turn_count", summary)
        self.assertIn("total_events", summary)
        self.assertEqual(summary["turn_count"], 3)

    def test_format_timeline(self):
        """format_timeline produces readable output."""
        query = TimelineQuery(max_events=5)
        result = self.service.query_session(self.session, query)
        formatted = self.service.format_timeline(result)

        self.assertIn("Events:", formatted)
        self.assertIn("USER:", formatted)


class TestSessionTimelineMethods(unittest.TestCase):
    """Tests for timeline methods on Session class."""

    def setUp(self):
        """Create test session."""
        self.session = Session(user_id="test-user")
        for event in create_test_events(3):
            self.session.add_event(event)

    def test_query_timeline(self):
        """Session.query_timeline works correctly."""
        query = TimelineQuery(event_types=[EventType.USER])
        result = self.session.query_timeline(query)

        self.assertEqual(len(result.events), 3)
        self.assertTrue(all(e.type == EventType.USER for e in result.events))

    def test_get_events_in_range(self):
        """Session.get_events_in_range filters by time."""
        start = datetime.now(timezone.utc) - timedelta(hours=2)
        end = datetime.now(timezone.utc)
        events = self.session.get_events_in_range(start, end)

        self.assertGreater(len(events), 0)

    def test_get_events_by_range(self):
        """Session.get_events_by_range uses predefined ranges."""
        events = self.session.get_events_by_range(TimeRange.THIS_SESSION)

        self.assertEqual(len(events), len(self.session.events))

    def test_get_turns(self):
        """Session.get_turns returns turn groups."""
        turns = self.session.get_turns()

        self.assertEqual(len(turns), 3)
        self.assertTrue(all(t[0].type == EventType.USER for t in turns))

    def test_get_turn_events(self):
        """Session.get_turn_events filters by turn number."""
        events = self.session.get_turn_events(start_turn=1, end_turn=2)

        user_events = [e for e in events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 1)
        self.assertIn("message 2", user_events[0].content)

    def test_get_last_n_turn_events(self):
        """Session.get_last_n_turn_events returns recent turns."""
        events = self.session.get_last_n_turn_events(n=2)

        user_events = [e for e in events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 2)


class TestTimelineConfig(unittest.TestCase):
    """Tests for TimelineConfig."""

    def test_default_values(self):
        """TimelineConfig has sensible defaults."""
        config = TimelineConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.default_time_range, "this_session")
        self.assertFalse(config.include_timestamps_in_history)
        self.assertFalse(config.group_by_turns)
        self.assertEqual(config.max_events_default, 100)

    def test_custom_values(self):
        """TimelineConfig accepts custom values."""
        config = TimelineConfig(
            enabled=False,
            default_time_range="last_24h",
            include_timestamps_in_history=True,
            group_by_turns=True,
            max_events_default=50,
        )

        self.assertFalse(config.enabled)
        self.assertEqual(config.default_time_range, "last_24h")
        self.assertTrue(config.include_timestamps_in_history)
        self.assertTrue(config.group_by_turns)
        self.assertEqual(config.max_events_default, 50)


if __name__ == "__main__":
    unittest.main()
