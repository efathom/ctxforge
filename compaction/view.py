"""
CompactionView - Immutable filtered view of events for compaction.

Provides an immutable projection of session events that tracks
which events have been condensed/forgotten. This enables:
- Tracking what was removed during compaction
- Supporting condenser chaining
- Enabling condensation auditing

Also defines ICondenser protocol for view-based condensation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Union,
    overload,
    runtime_checkable,
)

from ctxforge.core.events import Event, EventType


@runtime_checkable
class SessionLike(Protocol):
    """Protocol for session-like objects that CompactionView can read from."""

    @property
    def events(self) -> List[Event]:
        """The list of events in the session."""
        ...

    @property
    def summary(self) -> Optional[str]:
        """The session summary, if any."""
        ...


@dataclass(frozen=True)
class CompactionView:
    """
    Immutable filtered view of events for LLM context.

    Tracks which events have been forgotten/condensed to:
    - Prevent re-insertion of old instructions
    - Support condenser chaining
    - Enable condensation auditing

    Example:
        >>> view = CompactionView.from_events(session.events)
        >>> condensed = await condenser.condense(view)
        >>> print(f"Forgotten: {condensed.forgotten_event_ids}")
    """

    events: tuple[Event, ...]
    """The filtered list of events to include in context."""

    forgotten_event_ids: frozenset[str] = field(default_factory=frozenset)
    """Set of event IDs that have been condensed/forgotten."""

    summary: Optional[str] = None
    """Optional summary of forgotten events."""

    summary_offset: int = 0
    """Position to insert summary in the event list."""

    unhandled_condensation_request: bool = False
    """Whether there's a pending condensation request."""

    # =========================================================================
    # LIST-LIKE ACCESS
    # =========================================================================

    def __len__(self) -> int:
        """Return the number of events in the view."""
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events in the view."""
        return iter(self.events)

    @overload
    def __getitem__(self, key: slice) -> List[Event]:
        ...

    @overload
    def __getitem__(self, key: int) -> Event:
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[Event, List[Event]]:
        """Get event(s) by index or slice."""
        if isinstance(key, slice):
            return list(self.events[key])
        return self.events[key]

    def __bool__(self) -> bool:
        """Return True if the view has events."""
        return len(self.events) > 0

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def from_events(
        cls,
        events: List[Event],
        existing_forgotten: Optional[Set[str]] = None,
        existing_summary: Optional[str] = None,
    ) -> CompactionView:
        """
        Create a view from a list of events.

        Args:
            events: List of events to filter
            existing_forgotten: Previously forgotten event IDs
            existing_summary: Existing summary from prior condensation

        Returns:
            CompactionView with events and tracking info
        """
        forgotten = set(existing_forgotten or set())

        # Filter out already-forgotten events
        kept_events = [e for e in events if e.event_id not in forgotten]

        return cls(
            events=tuple(kept_events),
            forgotten_event_ids=frozenset(forgotten),
            summary=existing_summary,
        )

    @classmethod
    def from_session(cls, session: SessionLike) -> CompactionView:
        """
        Create a view from a Session or session-like object.

        Args:
            session: Any object with `events` and `summary` attributes

        Returns:
            CompactionView with session's events and summary
        """
        return cls.from_events(
            events=list(session.events),
            existing_summary=session.summary,
        )

    @classmethod
    def empty(cls) -> CompactionView:
        """Create an empty view with no events."""
        return cls(events=tuple())

    # =========================================================================
    # TRANSFORMATION METHODS
    # =========================================================================

    def with_forgotten(
        self,
        event_ids: Set[str],
        summary: Optional[str] = None,
        summary_offset: int = 0,
    ) -> CompactionView:
        """
        Create a new view with additional forgotten events.

        Args:
            event_ids: Event IDs to mark as forgotten
            summary: New summary to include
            summary_offset: Where to insert summary

        Returns:
            New CompactionView with updated state
        """
        new_forgotten = self.forgotten_event_ids | frozenset(event_ids)
        new_events = tuple(
            e for e in self.events if e.event_id not in event_ids
        )

        return CompactionView(
            events=new_events,
            forgotten_event_ids=new_forgotten,
            summary=summary if summary is not None else self.summary,
            summary_offset=summary_offset,
        )

    def with_events(self, events: List[Event]) -> CompactionView:
        """
        Create a new view with different events but same tracking.

        Args:
            events: New list of events

        Returns:
            New CompactionView with updated events
        """
        return CompactionView(
            events=tuple(events),
            forgotten_event_ids=self.forgotten_event_ids,
            summary=self.summary,
            summary_offset=self.summary_offset,
        )

    def with_summary(self, summary: str, offset: int = 0) -> CompactionView:
        """
        Create a new view with an updated summary.

        Args:
            summary: The new summary text
            offset: Position to insert summary

        Returns:
            New CompactionView with updated summary
        """
        return CompactionView(
            events=self.events,
            forgotten_event_ids=self.forgotten_event_ids,
            summary=summary,
            summary_offset=offset,
        )

    def with_condensation_request(self, requested: bool = True) -> CompactionView:
        """
        Create a new view with condensation request flag set.

        Args:
            requested: Whether condensation is requested

        Returns:
            New CompactionView with updated flag
        """
        return CompactionView(
            events=self.events,
            forgotten_event_ids=self.forgotten_event_ids,
            summary=self.summary,
            summary_offset=self.summary_offset,
            unhandled_condensation_request=requested,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def is_forgotten(self, event_id: str) -> bool:
        """
        Check if an event has been forgotten.

        Args:
            event_id: The event ID to check

        Returns:
            True if the event has been forgotten
        """
        return event_id in self.forgotten_event_ids

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """
        Get all events of a specific type.

        Args:
            event_type: The type of events to get

        Returns:
            List of events matching the type
        """
        return [e for e in self.events if e.type == event_type]

    def get_user_events(self) -> List[Event]:
        """Get all user message events."""
        return self.get_events_by_type(EventType.USER)

    def get_agent_events(self) -> List[Event]:
        """Get all agent message events."""
        return self.get_events_by_type(EventType.AGENT)

    def get_conversation_events(self) -> List[Event]:
        """Get user and agent events (the conversation)."""
        return [
            e for e in self.events
            if e.type in (EventType.USER, EventType.AGENT)
        ]

    def to_context_events(self) -> List[Event]:
        """
        Convert to list of events for Context assembly.

        Returns:
            List of events (copy of internal tuple)
        """
        return list(self.events)

    def forgotten_count(self) -> int:
        """Return the number of forgotten events."""
        return len(self.forgotten_event_ids)

    def total_events_processed(self) -> int:
        """Return total events (current + forgotten)."""
        return len(self.events) + len(self.forgotten_event_ids)


@dataclass
class CondensationResult:
    """
    Result of a condensation operation.

    Returned by condensers to indicate what was done during condensation.

    Attributes:
        view: The resulting view after condensation
        events_forgotten_start_id: First event ID that was forgotten
        events_forgotten_end_id: Last event ID that was forgotten
        summary_generated: Whether a new summary was generated
        tokens_saved: Estimated tokens saved by this condensation
        metadata: Additional metadata about the condensation
    """

    view: CompactionView
    """The resulting view after condensation."""

    events_forgotten_start_id: Optional[str] = None
    """First event ID that was forgotten (for range tracking)."""

    events_forgotten_end_id: Optional[str] = None
    """Last event ID that was forgotten (for range tracking)."""

    summary_generated: bool = False
    """Whether a new summary was generated."""

    tokens_saved: int = 0
    """Estimated tokens saved by this condensation."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata about the condensation."""

    @property
    def events_forgotten_count(self) -> int:
        """Return the number of events forgotten in this operation."""
        return self.view.forgotten_count()

    @property
    def success(self) -> bool:
        """Return True if condensation was successful (has a view)."""
        return self.view is not None

    def with_metadata(self, **kwargs) -> CondensationResult:
        """
        Create a new result with additional metadata.

        Args:
            **kwargs: Metadata key-value pairs to add

        Returns:
            New CondensationResult with updated metadata
        """
        new_metadata = dict(self.metadata)
        new_metadata.update(kwargs)
        return CondensationResult(
            view=self.view,
            events_forgotten_start_id=self.events_forgotten_start_id,
            events_forgotten_end_id=self.events_forgotten_end_id,
            summary_generated=self.summary_generated,
            tokens_saved=self.tokens_saved,
            metadata=new_metadata,
        )


# =============================================================================
# ICondenser Protocol
# =============================================================================


@runtime_checkable
class ICondenser(Protocol):
    """
    Protocol for view-based condensation.

    Condensers take a CompactionView and return either:
    - A new CompactionView (filtering complete)
    - A CondensationResult (with metadata about what was done)

    Unlike ICompactor which mutates sessions, ICondenser is pure
    and works with immutable views. This enables:
    - Chaining multiple condensers
    - Tracking what was forgotten
    - Easier testing and debugging

    Example implementations:
    - SlidingWindowCondenser: Simple FIFO removal
    - SummarizingCondenser: LLM-based summarization
    - ImportanceCondenser: Score-based retention
    - ObservationMaskingCondenser: Hide sensitive data

    Note:
        The `config` parameter accepts any CompactionConfig-like object.
        See ctxforge.protocols.compactor.CompactionConfig for the expected shape.
    """

    @property
    def name(self) -> str:
        """The name of this condensation strategy."""
        ...

    def should_condense(
        self,
        view: CompactionView,
        config: Any = None,
    ) -> bool:
        """
        Check if condensation is needed.

        Args:
            view: The view to check
            config: Optional compaction configuration (CompactionConfig)

        Returns:
            True if condensation is needed
        """
        ...

    async def condense(
        self,
        view: CompactionView,
        config: Any = None,
    ) -> Union[CompactionView, CondensationResult]:
        """
        Condense the view.

        Returns either a filtered view or a result with metadata.
        The original view is never modified.

        Args:
            view: The view to condense
            config: Optional compaction configuration (CompactionConfig)

        Returns:
            A new CompactionView or CondensationResult
        """
        ...
