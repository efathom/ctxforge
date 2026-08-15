"""
Session management - The 'Workbench' container for active conversations.

Sessions represent the current working state of a conversation, including:
- Event history (immutable log of what happened)
- Session state (mutable scratchpad for current task data)
- Summarization state (rolled-up history for context management)
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ctxforge.core.events import Event, EventFactory, EventType
from ctxforge.core.timeline import (
    TimelineFilter,
    TimelineQuery,
    TimelineResult,
    TimeRange,
)


class SessionState(BaseModel):
    """
    Mutable Working Memory.
    
    A scratchpad for current task data that can be read and modified
    during a conversation. Examples: shopping cart, user verification status,
    current task progress, etc.
    
    Attributes:
        data: Key-value store for arbitrary session data
    """
    
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the session state."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the session state."""
        self.data[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete a key from the session state. Returns True if key existed."""
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the session state."""
        return key in self.data
    
    def clear(self) -> None:
        """Clear all session state data."""
        self.data.clear()
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple values in the session state."""
        self.data.update(data)
    
    def keys(self) -> List[str]:
        """Get all keys in the session state."""
        return list(self.data.keys())


class Session(BaseModel):
    """
    The 'Workbench' Container for active conversations.
    
    Contains all context needed for a conversation:
    - Event history (immutable audit log)
    - Session state (mutable working memory)
    - Summarization (rolled-up history)
    - Version control (for optimistic locking)
    
    Attributes:
        session_id: Unique identifier for the session
        user_id: The user this session belongs to
        version: Version number for optimistic locking
        events: List of immutable events in the session
        state: Mutable working memory
        summary: Rolled-up summary of older events
        created_at: When the session was created
        updated_at: When the session was last updated
        metadata: Additional session metadata
    """
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    version: int = 0  # For optimistic locking
    events: List[Event] = Field(default_factory=list)
    state: SessionState = Field(default_factory=SessionState)
    summary: Optional[str] = None  # Rolled-up history
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_event(self, event: Event) -> None:
        """Add an event to the session history."""
        self.events.append(event)
        self.updated_at = datetime.datetime.now()
    
    def add_user_message(self, content: str) -> Event:
        """Convenience method to add a user message event."""
        event = EventFactory.user_message(content)
        self.add_event(event)
        return event
    
    def add_agent_message(
        self,
        content: str,
        model: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Event:
        """Convenience method to add an agent message event."""
        event = EventFactory.agent_message(content, model=model, latency_ms=latency_ms)
        self.add_event(event)
        return event
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]
    
    def get_recent_events(self, count: int) -> List[Event]:
        """Get the most recent N events."""
        return self.events[-count:] if count > 0 else []
    
    def get_conversation_history(self) -> List[Event]:
        """Get only user and agent events (the conversation)."""
        return [
            e for e in self.events 
            if e.type in (EventType.USER, EventType.AGENT)
        ]
    
    def get_last_user_message(self) -> Optional[Event]:
        """Get the last user message event."""
        user_events = self.get_events_by_type(EventType.USER)
        return user_events[-1] if user_events else None
    
    def get_last_agent_message(self) -> Optional[Event]:
        """Get the last agent message event."""
        agent_events = self.get_events_by_type(EventType.AGENT)
        return agent_events[-1] if agent_events else None
    
    def event_count(self) -> int:
        """Get the total number of events."""
        return len(self.events)
    
    def turn_count(self) -> int:
        """Get the number of conversation turns (user-agent pairs)."""
        return len(self.get_events_by_type(EventType.USER))
    
    def prune_events(self, keep_last: int) -> List[Event]:
        """
        Prune events, keeping only the last N events.
        Returns the pruned events.
        """
        if len(self.events) <= keep_last:
            return []
        
        pruned = self.events[:-keep_last]
        self.events = self.events[-keep_last:]
        self.updated_at = datetime.datetime.now()
        return pruned
    
    def set_summary(self, summary: str) -> None:
        """Set or update the session summary."""
        self.summary = summary
        self.updated_at = datetime.datetime.now()
    
    def append_summary(self, additional_summary: str, separator: str = " -> ") -> None:
        """Append to the existing summary."""
        if self.summary:
            self.summary = f"{self.summary}{separator}{additional_summary}"
        else:
            self.summary = additional_summary
        self.updated_at = datetime.datetime.now()
    
    def increment_version(self) -> int:
        """Increment the version number and return the new version."""
        self.version += 1
        self.updated_at = datetime.datetime.now()
        return self.version
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to a dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create a session from a dictionary."""
        return cls.model_validate(data)
    
    def copy_deep(self) -> "Session":
        """Create a deep copy of the session."""
        return self.model_copy(deep=True)

    # Timeline-based query methods

    def query_timeline(self, query: TimelineQuery) -> TimelineResult:
        """
        Query events using timeline parameters.

        Args:
            query: Timeline query with time bounds, turn range, event types, etc.

        Returns:
            TimelineResult with filtered events and statistics
        """
        return TimelineFilter.filter_events(self.events, query)

    def get_events_in_range(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> List[Event]:
        """
        Get events within a time range.

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            List of events within the range
        """
        query = TimelineQuery(start_time=start, end_time=end)
        result = TimelineFilter.filter_events(self.events, query)
        return result.events

    def get_events_by_range(self, time_range: TimeRange) -> List[Event]:
        """
        Get events using a predefined time range.

        Args:
            time_range: Predefined range (LAST_HOUR, LAST_24H, TODAY, etc.)

        Returns:
            List of events in the range
        """
        query = TimelineQuery(time_range=time_range)
        result = TimelineFilter.filter_events(self.events, query)
        return result.events

    def get_turns(self) -> List[List[Event]]:
        """
        Split events into conversation turns.

        Each turn starts with a USER event and contains all events
        until the next USER event.

        Returns:
            List of turns, where each turn is a list of events
        """
        return TimelineFilter.get_turns(self.events)

    def get_turn_events(
        self,
        start_turn: int,
        end_turn: Optional[int] = None,
    ) -> List[Event]:
        """
        Get events from specific conversation turns.

        Args:
            start_turn: Starting turn (0-indexed)
            end_turn: Ending turn (exclusive), None for all remaining

        Returns:
            List of events from specified turns
        """
        query = TimelineQuery(start_turn=start_turn, end_turn=end_turn)
        result = TimelineFilter.filter_events(self.events, query)
        return result.events

    def get_last_n_turn_events(self, n: int = 3) -> List[Event]:
        """
        Get events from the last N conversation turns.

        Args:
            n: Number of turns to retrieve

        Returns:
            List of events from last N turns
        """
        total_turns = TimelineFilter.count_turns(self.events)
        start_turn = max(0, total_turns - n)
        return self.get_turn_events(start_turn)


class SessionBuilder:
    """Builder pattern for creating sessions with fluent API."""
    
    def __init__(self, user_id: str):
        self._user_id = user_id
        self._session_id: Optional[str] = None
        self._state_data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._events: List[Event] = []
        self._summary: Optional[str] = None
    
    def with_session_id(self, session_id: str) -> "SessionBuilder":
        """Set a custom session ID."""
        self._session_id = session_id
        return self
    
    def with_state(self, key: str, value: Any) -> "SessionBuilder":
        """Add state data."""
        self._state_data[key] = value
        return self
    
    def with_metadata(self, key: str, value: Any) -> "SessionBuilder":
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def with_event(self, event: Event) -> "SessionBuilder":
        """Add an event."""
        self._events.append(event)
        return self
    
    def with_summary(self, summary: str) -> "SessionBuilder":
        """Set the summary."""
        self._summary = summary
        return self
    
    def build(self) -> Session:
        """Build and return the session."""
        session = Session(
            user_id=self._user_id,
            state=SessionState(data=self._state_data),
            metadata=self._metadata,
            summary=self._summary,
        )
        
        if self._session_id:
            session.session_id = self._session_id
        
        for event in self._events:
            session.add_event(event)
        
        return session

