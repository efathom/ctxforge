"""
Summarizing Condenser.

Condenses history by creating summaries of old events.
"""

from typing import List, Optional, Union

from ctxforge.compaction.base import BaseCondenser
from ctxforge.compaction.utils import SummarizeFunc, estimate_tokens_simple
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.engine.registry import registry
from ctxforge.protocols.compactor import CompactionConfig


@registry.register_condenser("summarizing")
class SummarizingCondenser(BaseCondenser):
    """
    Condenser that summarizes old events.

    Instead of simply dropping old events, this condenser creates
    a summary that preserves key information while reducing tokens.

    Requires an LLM function for summarization.

    Best for:
    - Maintaining context over long conversations
    - Applications where historical context matters
    - When you need to preserve key information

    Example:
        >>> async def summarize(text, existing=None):
        ...     return await llm.summarize(text, existing)
        >>>
        >>> condenser = SummarizingCondenser(summarize)
        >>> result = await condenser.condense(view)
    """

    def __init__(
        self,
        summarize_func: Optional[SummarizeFunc] = None,
    ):
        """
        Initialize with an optional summarization function.

        Args:
            summarize_func: Async function to summarize text.
                If not provided, uses a simple extraction method.
        """
        self._summarize = summarize_func

    @property
    def name(self) -> str:
        return "summarizing"

    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        """
        Condense by summarizing old events.
        """
        config = config or CompactionConfig()

        events = list(view.events)
        original_count = len(events)
        original_tokens = self.estimate_tokens(events)

        if original_count <= config.keep_recent:
            return self._create_no_op_result(view)

        # Separate events
        system_events, other_events = self._separate_events_by_type(events, config)
        other_events = self._sort_by_timestamp(other_events)

        if len(other_events) <= config.keep_recent:
            return self._create_no_op_result(view)

        # Split into events to summarize and events to keep
        events_to_summarize = other_events[:-config.keep_recent]
        events_to_keep = other_events[-config.keep_recent:]

        # Create summary
        new_summary = await self._summarize_events(
            events_to_summarize,
            view.summary,
        )

        # Combine kept events
        all_kept = system_events + events_to_keep
        all_kept = self._sort_by_timestamp(all_kept)

        # Calculate tokens saved
        new_tokens = self.estimate_tokens(all_kept)
        summary_tokens = estimate_tokens_simple(new_summary) if new_summary else 0
        tokens_saved = max(0, original_tokens - new_tokens - summary_tokens)

        # Create new view with forgotten events and summary
        forgotten_ids = {e.event_id for e in events_to_summarize}
        new_view = view.with_forgotten(
            forgotten_ids,
            summary=new_summary,
        )
        new_view = new_view.with_events(all_kept)

        return CondensationResult(
            view=new_view,
            events_forgotten_start_id=events_to_summarize[0].event_id if events_to_summarize else None,
            events_forgotten_end_id=events_to_summarize[-1].event_id if events_to_summarize else None,
            summary_generated=True,
            tokens_saved=tokens_saved,
            metadata={
                "strategy": self.name,
                "original_count": original_count,
                "new_count": len(all_kept),
                "summary_length": len(new_summary) if new_summary else 0,
            },
        )

    async def _summarize_events(
        self,
        events: List[Event],
        existing_summary: Optional[str] = None,
    ) -> str:
        """
        Create a summary of events.

        Uses the provided summarize function if available,
        otherwise uses a simple extraction method.
        """
        if not events:
            return existing_summary or ""

        # Format events for summarization
        event_text = self._format_events_for_summary(events)

        # Build the text to summarize
        if existing_summary:
            text_to_summarize = f"Previous summary:\n{existing_summary}\n\nNew events:\n{event_text}"
        else:
            text_to_summarize = event_text

        # Use LLM if available
        if self._summarize:
            return await self._summarize(text_to_summarize, existing_summary)

        # Otherwise, use simple extraction
        return self._simple_summarize(events, existing_summary)

    def _format_events_for_summary(self, events: List[Event]) -> str:
        """Format events as text for summarization."""
        lines = []
        for event in events:
            role = event.type.value
            content = event.content[:500]  # Truncate long content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _simple_summarize(
        self,
        events: List[Event],
        existing_summary: Optional[str] = None,
    ) -> str:
        """
        Simple summarization without LLM.

        Extracts key sentences and combines with existing summary.
        """
        parts = []

        if existing_summary:
            parts.append(existing_summary)

        # Extract key content from events
        for event in events:
            if event.type == EventType.USER:
                first_sentence = event.content.split('.')[0]
                if len(first_sentence) < 200:
                    parts.append(f"User asked about: {first_sentence}")
            elif event.type == EventType.AGENT:
                if len(event.content) > 100:
                    parts.append("Agent responded with information.")

        return " ".join(parts[-5:])  # Keep last 5 parts
