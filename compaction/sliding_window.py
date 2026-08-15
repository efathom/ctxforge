"""
Sliding Window Condenser.

Simple FIFO-based condensation that removes oldest events.
"""

from typing import Optional, Union

from ctxforge.compaction.base import BaseCondenser
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.engine.registry import registry
from ctxforge.protocols.compactor import CompactionConfig


@registry.register_condenser("sliding_window")
class SlidingWindowCondenser(BaseCondenser):
    """
    Sliding window condenser using FIFO removal.

    The simplest condensation strategy - just removes the oldest
    events to stay within limits. No summarization.

    Best for:
    - Applications where old context is not important
    - Testing and development
    - When you want predictable, fast condensation

    Example:
        >>> condenser = SlidingWindowCondenser()
        >>> if condenser.should_condense(view):
        ...     result = await condenser.condense(view)
    """

    @property
    def name(self) -> str:
        return "sliding_window"

    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        """
        Condense by removing oldest events.

        Keeps the most recent events up to keep_recent limit.
        """
        config = config or CompactionConfig()

        events = list(view.events)
        original_count = len(events)
        original_tokens = self.estimate_tokens(events)

        if original_count <= config.keep_recent:
            return self._create_no_op_result(view)

        # Separate system events from others
        system_events, other_events = self._separate_events_by_type(events, config)

        # Sort non-system events by timestamp (oldest first)
        other_events = self._sort_by_timestamp(other_events)

        # Keep the most recent non-system events
        if len(other_events) > config.keep_recent:
            events_to_remove = other_events[:-config.keep_recent]
            events_to_keep = other_events[-config.keep_recent:]
        else:
            events_to_remove = []
            events_to_keep = other_events

        if not events_to_remove:
            return self._create_no_op_result(view)

        # Combine system events with kept events
        all_kept = system_events + events_to_keep
        all_kept = self._sort_by_timestamp(all_kept)

        # Calculate tokens saved
        new_tokens = self.estimate_tokens(all_kept)
        tokens_saved = original_tokens - new_tokens

        # Create new view with forgotten events tracked
        forgotten_ids = {e.event_id for e in events_to_remove}
        new_view = view.with_forgotten(forgotten_ids)
        # Update events in the new view
        new_view = new_view.with_events(all_kept)

        return CondensationResult(
            view=new_view,
            events_forgotten_start_id=events_to_remove[0].event_id if events_to_remove else None,
            events_forgotten_end_id=events_to_remove[-1].event_id if events_to_remove else None,
            summary_generated=False,
            tokens_saved=tokens_saved,
            metadata={
                "strategy": self.name,
                "original_count": original_count,
                "new_count": len(all_kept),
            },
        )
