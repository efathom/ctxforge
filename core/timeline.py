"""
Timeline-Based Event Query Support.

Provides temporal filtering and retrieval for session events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ctxforge.core.events import Event, EventType


class TimeRange(str, Enum):
    """Predefined time ranges for convenience."""
    LAST_HOUR = "last_hour"
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    TODAY = "today"
    THIS_SESSION = "this_session"


@dataclass
class TimelineQuery:
    """
    Query parameters for timeline-based event retrieval.

    Supports both absolute time bounds and predefined ranges,
    as well as conversation turn-based filtering.
    """
    # Time bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Convenience range (overrides start/end if set)
    time_range: Optional[TimeRange] = None

    # Turn-based bounds (0-indexed, based on USER events)
    start_turn: Optional[int] = None
    end_turn: Optional[int] = None

    # Event type filters
    event_types: Optional[List[EventType]] = None
    include_tool_events: bool = True

    # Limits
    max_events: int = 100

    def resolve_time_bounds(
        self,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Resolve time range to absolute bounds.

        Args:
            reference_time: Reference time for relative ranges.
                Defaults to current time.

        Returns:
            Tuple of (start_time, end_time), either may be None.
        """
        if self.time_range:
            now = reference_time or datetime.now(timezone.utc)
            if self.time_range == TimeRange.LAST_HOUR:
                return (now - timedelta(hours=1), now)
            elif self.time_range == TimeRange.LAST_24H:
                return (now - timedelta(days=1), now)
            elif self.time_range == TimeRange.LAST_7D:
                return (now - timedelta(days=7), now)
            elif self.time_range == TimeRange.TODAY:
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                return (start, now)
            else:  # THIS_SESSION - no time filtering
                return (None, None)
        return (self.start_time, self.end_time)

    def has_time_filter(self) -> bool:
        """Check if query has any time-based filtering."""
        return (
            self.time_range is not None
            or self.start_time is not None
            or self.end_time is not None
        )

    def has_turn_filter(self) -> bool:
        """Check if query has turn-based filtering."""
        return self.start_turn is not None or self.end_turn is not None


@dataclass
class TimelineResult:
    """
    Result of a timeline query.

    Contains filtered events along with statistics and metadata.
    """
    events: List[Event] = field(default_factory=list)
    total_matching: int = 0
    time_span: Optional[Tuple[datetime, datetime]] = None

    # Statistics
    event_counts: Dict[str, int] = field(default_factory=dict)
    turn_count: int = 0

    def to_summary(self) -> str:
        """Generate a textual summary of the timeline."""
        if not self.events:
            return "No events found in the specified time range."

        lines = []
        if self.time_span:
            start, end = self.time_span
            start_str = start.strftime("%Y-%m-%d %H:%M")
            end_str = end.strftime("%H:%M")
            # Same day? Only show time for end
            if start.date() == end.date():
                lines.append(f"Timeline: {start_str} - {end_str}")
            else:
                end_str = end.strftime("%Y-%m-%d %H:%M")
                lines.append(f"Timeline: {start_str} - {end_str}")

        lines.append(f"Events: {len(self.events)} (of {self.total_matching} total)")

        if self.turn_count > 0:
            lines.append(f"Conversation turns: {self.turn_count}")

        for event_type, count in sorted(self.event_counts.items()):
            lines.append(f"  - {event_type}: {count}")

        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Check if result has no events."""
        return len(self.events) == 0

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get events of a specific type."""
        return [e for e in self.events if e.type == event_type]


class TimelineFilter:
    """
    Filters events based on timeline query parameters.

    Provides static methods for filtering event lists.
    """

    @staticmethod
    def filter_events(
        events: List[Event],
        query: TimelineQuery,
    ) -> TimelineResult:
        """
        Filter events based on timeline query.

        Args:
            events: List of events to filter
            query: Timeline query parameters

        Returns:
            TimelineResult with filtered events and statistics
        """
        start_time, end_time = query.resolve_time_bounds()

        filtered: List[Event] = []
        event_counts: Dict[str, int] = {}

        for event in events:
            # Time filtering
            if start_time and event.timestamp:
                # Ensure both are timezone-aware for comparison
                event_ts = event.timestamp
                if event_ts.tzinfo is None:
                    event_ts = event_ts.replace(tzinfo=timezone.utc)
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if event_ts < start_time:
                    continue

            if end_time and event.timestamp:
                event_ts = event.timestamp
                if event_ts.tzinfo is None:
                    event_ts = event_ts.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                if event_ts > end_time:
                    continue

            # Type filtering
            if query.event_types and event.type not in query.event_types:
                continue

            # Tool event filtering
            if not query.include_tool_events:
                if event.type in (EventType.TOOL_CALL, EventType.TOOL_OUTPUT):
                    continue

            # Count by type
            type_name = event.type.value
            event_counts[type_name] = event_counts.get(type_name, 0) + 1

            filtered.append(event)

        # Apply turn-based bounds if specified
        if query.has_turn_filter():
            filtered = TimelineFilter._filter_by_turn(
                filtered,
                query.start_turn,
                query.end_turn,
            )
            # Recount after turn filtering
            event_counts = {}
            for event in filtered:
                type_name = event.type.value
                event_counts[type_name] = event_counts.get(type_name, 0) + 1

        # Calculate turn count (number of USER events)
        turn_count = sum(1 for e in filtered if e.type == EventType.USER)

        # Store total before applying limit
        total = len(filtered)

        # Apply limit
        filtered = filtered[:query.max_events]

        # Compute time span
        time_span = None
        if filtered:
            timestamps = [e.timestamp for e in filtered if e.timestamp]
            if timestamps:
                time_span = (min(timestamps), max(timestamps))

        return TimelineResult(
            events=filtered,
            total_matching=total,
            time_span=time_span,
            event_counts=event_counts,
            turn_count=turn_count,
        )

    @staticmethod
    def _filter_by_turn(
        events: List[Event],
        start_turn: Optional[int],
        end_turn: Optional[int],
    ) -> List[Event]:
        """
        Filter events by conversation turn number.

        Turns are counted by USER events (each USER event starts a new turn).
        """
        if not events:
            return []

        # Group events by turn (delimited by USER events)
        turn_events: List[List[Event]] = []
        current_turn: List[Event] = []

        for event in events:
            if event.type == EventType.USER:
                if current_turn:
                    turn_events.append(current_turn)
                current_turn = [event]
            else:
                current_turn.append(event)

        # Don't forget the last turn
        if current_turn:
            turn_events.append(current_turn)

        # Select turn range
        start = start_turn if start_turn is not None else 0
        end = end_turn if end_turn is not None else len(turn_events)

        # Ensure bounds are valid
        start = max(0, start)
        end = min(len(turn_events), end)

        selected: List[Event] = []
        for turn in turn_events[start:end]:
            selected.extend(turn)

        return selected

    @staticmethod
    def get_turns(events: List[Event]) -> List[List[Event]]:
        """
        Split events into conversation turns.

        Returns list of turns, where each turn is a list of events
        starting with a USER event.
        """
        turns: List[List[Event]] = []
        current_turn: List[Event] = []

        for event in events:
            if event.type == EventType.USER:
                if current_turn:
                    turns.append(current_turn)
                current_turn = [event]
            else:
                current_turn.append(event)

        if current_turn:
            turns.append(current_turn)

        return turns

    @staticmethod
    def count_turns(events: List[Event]) -> int:
        """Count the number of conversation turns (USER events)."""
        return sum(1 for e in events if e.type == EventType.USER)
