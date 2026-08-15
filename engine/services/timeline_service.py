"""
Timeline Service.

Provides timeline-based event retrieval and analysis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.events import EventType
from ctxforge.core.session import Session
from ctxforge.core.timeline import (
    TimelineFilter,
    TimelineQuery,
    TimelineResult,
    TimeRange,
)


class TimelineService:
    """
    Service for timeline-based event queries and analysis.

    Provides methods for querying session events by time range,
    conversation turns, and event types.
    """

    def query_session(
        self,
        session: Session,
        query: TimelineQuery,
    ) -> TimelineResult:
        """
        Query events from a session based on timeline parameters.

        Args:
            session: The session to query
            query: Timeline query parameters

        Returns:
            TimelineResult with filtered events
        """
        return TimelineFilter.filter_events(session.events, query)

    def get_recent_context(
        self,
        session: Session,
        hours: float = 1.0,
        include_tools: bool = False,
        max_events: int = 50,
    ) -> TimelineResult:
        """
        Get events from the recent time window.

        Convenience method for common use case of getting recent activity.

        Args:
            session: The session to query
            hours: Number of hours to look back
            include_tools: Whether to include tool call/output events
            max_events: Maximum events to return

        Returns:
            TimelineResult with recent events
        """
        query = TimelineQuery(
            start_time=datetime.now(timezone.utc) - timedelta(hours=hours),
            include_tool_events=include_tools,
            max_events=max_events,
        )
        return self.query_session(session, query)

    def get_conversation_turns(
        self,
        session: Session,
        start_turn: int = 0,
        end_turn: Optional[int] = None,
        include_tools: bool = True,
    ) -> TimelineResult:
        """
        Get events from specific conversation turns.

        Turns are counted by USER events (each USER event starts a new turn).

        Args:
            session: The session to query
            start_turn: Starting turn (0-indexed)
            end_turn: Ending turn (exclusive), None for all remaining
            include_tools: Whether to include tool events

        Returns:
            TimelineResult with events from specified turns
        """
        query = TimelineQuery(
            start_turn=start_turn,
            end_turn=end_turn,
            include_tool_events=include_tools,
        )
        return self.query_session(session, query)

    def get_last_n_turns(
        self,
        session: Session,
        n: int = 3,
        include_tools: bool = True,
    ) -> TimelineResult:
        """
        Get events from the last N conversation turns.

        Args:
            session: The session to query
            n: Number of turns to retrieve
            include_tools: Whether to include tool events

        Returns:
            TimelineResult with events from last N turns
        """
        total_turns = TimelineFilter.count_turns(session.events)
        start_turn = max(0, total_turns - n)
        return self.get_conversation_turns(
            session,
            start_turn=start_turn,
            include_tools=include_tools,
        )

    def get_events_by_type(
        self,
        session: Session,
        event_types: List[EventType],
        time_range: Optional[TimeRange] = None,
        max_events: int = 100,
    ) -> TimelineResult:
        """
        Get events of specific types.

        Args:
            session: The session to query
            event_types: List of event types to include
            time_range: Optional time range filter
            max_events: Maximum events to return

        Returns:
            TimelineResult with filtered events
        """
        query = TimelineQuery(
            event_types=event_types,
            time_range=time_range,
            max_events=max_events,
        )
        return self.query_session(session, query)

    def get_user_assistant_exchanges(
        self,
        session: Session,
        time_range: Optional[TimeRange] = None,
        max_events: int = 100,
    ) -> TimelineResult:
        """
        Get only user and assistant message events.

        Filters out tool calls and system events for cleaner conversation view.

        Args:
            session: The session to query
            time_range: Optional time range filter
            max_events: Maximum events to return

        Returns:
            TimelineResult with user/assistant events only
        """
        return self.get_events_by_type(
            session,
            event_types=[EventType.USER, EventType.AGENT],
            time_range=time_range,
            max_events=max_events,
        )

    def summarize_activity(
        self,
        session: Session,
        time_range: TimeRange = TimeRange.THIS_SESSION,
    ) -> Dict[str, Any]:
        """
        Generate activity summary for a time range.

        Args:
            session: The session to analyze
            time_range: Time range to summarize

        Returns:
            Dict with:
            - event_counts: Count by event type
            - time_span: (start, end) datetime tuple
            - tool_calls: List of tool names used
            - turn_count: Number of conversation turns
            - duration_seconds: Session duration in seconds
        """
        query = TimelineQuery(time_range=time_range)
        result = self.query_session(session, query)

        # Collect tool names
        tool_calls: List[str] = []
        for event in result.events:
            if event.type == EventType.TOOL_CALL:
                tool_name = getattr(event.metadata, 'tool_name', None)
                if tool_name and tool_name not in tool_calls:
                    tool_calls.append(tool_name)

        # Calculate duration
        duration_seconds = 0.0
        if result.time_span:
            start, end = result.time_span
            duration_seconds = (end - start).total_seconds()

        return {
            "event_counts": result.event_counts,
            "time_span": result.time_span,
            "tool_calls": tool_calls,
            "turn_count": result.turn_count,
            "total_events": result.total_matching,
            "duration_seconds": duration_seconds,
        }

    def format_timeline(
        self,
        result: TimelineResult,
        include_timestamps: bool = True,
        max_content_length: int = 200,
    ) -> str:
        """
        Format a timeline result as a human-readable string.

        Args:
            result: TimelineResult to format
            include_timestamps: Whether to include timestamps
            max_content_length: Maximum content length per event

        Returns:
            Formatted timeline string
        """
        if result.is_empty():
            return "No events in timeline."

        lines = [result.to_summary(), "", "Events:"]

        for event in result.events:
            # Format timestamp
            ts_str = ""
            if include_timestamps and event.timestamp:
                ts_str = f"[{event.timestamp.strftime('%H:%M:%S')}] "

            # Format content
            content = event.content
            if len(content) > max_content_length:
                content = content[:max_content_length - 3] + "..."

            # Format event type
            type_label = event.type.value.upper()

            lines.append(f"  {ts_str}{type_label}: {content}")

        return "\n".join(lines)
