"""
Base condenser implementation.

Provides a base class with common functionality for all condensers.
Condensers work with immutable CompactionView objects.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from ctxforge.compaction.utils import estimate_event_tokens
from ctxforge.compaction.view import (
    CompactionView,
    CondensationResult,
)
from ctxforge.core.events import Event, EventType
from ctxforge.protocols.compactor import CompactionConfig


class BaseCondenser(ABC):
    """
    Base class for all condensers.

    Provides common functionality:
    - Token estimation
    - should_condense implementation
    - Event separation by type
    - Result creation helpers

    Condensers work with immutable CompactionView objects and return
    new views or CondensationResult objects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this condensation strategy."""
        ...

    @abstractmethod
    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        """Condense the view, returning a new view or result."""
        ...

    def should_condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> bool:
        """
        Check if the view needs condensation.

        Returns True if either:
        - Event count exceeds threshold
        - Token count exceeds threshold
        """
        config = config or CompactionConfig()

        # Check event count
        if len(view) > config.event_threshold:
            return True

        # Check token count
        total_tokens = self.estimate_tokens(view.to_context_events())
        if total_tokens > config.token_threshold:
            return True

        return False

    def estimate_tokens(self, events: List[Event]) -> int:
        """Estimate total tokens for a list of events."""
        return estimate_event_tokens(events)

    def _separate_events_by_type(
        self,
        events: List[Event],
        config: CompactionConfig,
    ) -> Tuple[List[Event], List[Event]]:
        """
        Separate events into system and non-system events.

        Args:
            events: All events
            config: Compaction config (for preserve_system_events setting)

        Returns:
            Tuple of (system_events, other_events)
        """
        system_events: List[Event] = []
        other_events: List[Event] = []

        for event in events:
            if config.preserve_system_events and event.type == EventType.SYSTEM:
                system_events.append(event)
            else:
                other_events.append(event)

        return system_events, other_events

    def _create_no_op_result(self, view: CompactionView) -> CondensationResult:
        """Create a result indicating no condensation was needed."""
        return CondensationResult(
            view=view,
            summary_generated=False,
            tokens_saved=0,
            metadata={"strategy": self.name, "action": "no_op"},
        )

    def _sort_by_timestamp(self, events: List[Event]) -> List[Event]:
        """Sort events by timestamp (oldest first)."""
        return sorted(events, key=lambda e: e.timestamp)
