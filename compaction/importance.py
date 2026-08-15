"""
Importance-based Condenser.

Scores events and keeps the most important ones.
"""

import re
from typing import Dict, List, Optional, Union

from ctxforge.compaction.base import BaseCondenser
from ctxforge.compaction.utils import ScoringFunc
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.engine.registry import registry
from ctxforge.protocols.compactor import CompactionConfig


def default_importance_scorer(event: Event) -> float:
    """
    Default importance scoring for events.

    Scores are 0-1 based on:
    - Event type (system > tool > user > agent)
    - Content length (longer = potentially more important)
    - Keywords (questions, decisions, actions)
    """
    score = 0.5  # Base score

    # Event type weights
    type_weights = {
        EventType.SYSTEM: 0.3,
        EventType.TOOL_CALL: 0.2,
        EventType.TOOL_OUTPUT: 0.15,
        EventType.USER: 0.1,
        EventType.AGENT: 0.0,
    }
    score += type_weights.get(event.type, 0.0)

    content = event.content.lower()

    # Boost for questions
    if '?' in content:
        score += 0.1

    # Boost for decisions/actions
    decision_keywords = ['decided', 'chose', 'will', 'going to', 'plan to', 'want to']
    if any(kw in content for kw in decision_keywords):
        score += 0.1

    # Boost for factual information
    fact_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b\d+\s*(dollars?|usd|\$|euros?|pounds?)\b',  # Money
        r'\b\d+\s*(percent|%)\b',  # Percentages
    ]
    for pattern in fact_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            score += 0.05

    # Boost for named entities (simple heuristic)
    capitals = len(re.findall(r'\b[A-Z][a-z]+\b', event.content))
    score += min(0.1, capitals * 0.02)

    # Slight boost for longer content
    length = len(content)
    if 100 < length < 500:
        score += 0.05
    elif length >= 500:
        score += 0.02

    return min(1.0, max(0.0, score))


@registry.register_condenser("importance")
class ImportanceCondenser(BaseCondenser):
    """
    Condenser that keeps events based on importance scores.

    Instead of simply removing oldest events, this condenser
    scores each event and keeps the most important ones.

    Best for:
    - Conversations where key decisions/facts appear throughout
    - Applications needing to preserve critical context
    - When recency isn't the only factor

    Example:
        >>> def custom_scorer(event):
        ...     if "important" in event.content:
        ...         return 1.0
        ...     return 0.5
        >>>
        >>> condenser = ImportanceCondenser(scoring_func=custom_scorer)
        >>> result = await condenser.condense(view)
    """

    def __init__(
        self,
        scoring_func: Optional[ScoringFunc] = None,
        min_importance: float = 0.3,
    ):
        """
        Initialize with optional custom scoring function.

        Args:
            scoring_func: Custom function to score events (0-1)
            min_importance: Minimum importance to keep an event
        """
        self._score = scoring_func or default_importance_scorer
        self._min_importance = min_importance

    @property
    def name(self) -> str:
        return "importance"

    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        """
        Condense by keeping most important events.
        """
        config = config or CompactionConfig()

        events = list(view.events)
        original_count = len(events)
        original_tokens = self.estimate_tokens(events)

        if original_count <= config.keep_recent:
            return self._create_no_op_result(view)

        # Separate system events
        system_events, other_events = self._separate_events_by_type(events, config)

        # Score all non-system events
        scored_events = [(event, self._score(event)) for event in other_events]

        # Select events to keep using importance + recency balance
        events_to_keep = self._select_important_events(
            system_events,
            scored_events,
            config.keep_recent,
        )

        # Determine which events were removed
        kept_ids = {e.event_id for e in events_to_keep}
        events_removed = [e for e in events if e.event_id not in kept_ids]

        # Calculate tokens saved
        new_tokens = self.estimate_tokens(events_to_keep)
        tokens_saved = original_tokens - new_tokens

        # Create new view with forgotten events tracked
        forgotten_ids = {e.event_id for e in events_removed}
        new_view = view.with_forgotten(forgotten_ids)
        new_view = new_view.with_events(self._sort_by_timestamp(events_to_keep))

        return CondensationResult(
            view=new_view,
            events_forgotten_start_id=events_removed[0].event_id if events_removed else None,
            events_forgotten_end_id=events_removed[-1].event_id if events_removed else None,
            summary_generated=False,
            tokens_saved=tokens_saved,
            metadata={
                "strategy": self.name,
                "original_count": original_count,
                "new_count": len(events_to_keep),
            },
        )

    def _select_important_events(
        self,
        system_events: List[Event],
        scored_events: List[tuple],
        keep_recent: int,
    ) -> List[Event]:
        """
        Select events to keep based on importance and recency.

        Balances keeping recent events with keeping important ones.
        """
        events_to_keep: List[Event] = list(system_events)

        if not scored_events:
            return events_to_keep

        # Reserve slots: half for recent, half for important
        recent_slots = min(keep_recent // 2, len(scored_events))
        importance_slots = keep_recent - recent_slots

        # Get most recent events
        by_time = sorted(scored_events, key=lambda x: x[0].timestamp, reverse=True)
        recent_events = [e for e, _ in by_time[:recent_slots]]

        # Get most important events (excluding already selected)
        remaining = [(e, s) for e, s in scored_events if e not in recent_events]
        remaining.sort(key=lambda x: x[1], reverse=True)
        important_events = [
            e for e, s in remaining[:importance_slots]
            if s >= self._min_importance
        ]

        events_to_keep.extend(recent_events)
        events_to_keep.extend(important_events)

        # Remove duplicates
        return self._deduplicate_events(events_to_keep)

    def _deduplicate_events(self, events: List[Event]) -> List[Event]:
        """Remove duplicate events by id."""
        seen_ids = set()
        unique = []
        for event in events:
            if id(event) not in seen_ids:
                seen_ids.add(id(event))
                unique.append(event)
        return unique

    def get_event_scores(self, events: List[Event]) -> Dict[str, float]:
        """
        Get importance scores for a list of events.

        Useful for debugging/inspection.

        Returns:
            Dict mapping event content preview to score
        """
        scores = {}
        for event in events:
            preview = event.content[:50] + "..." if len(event.content) > 50 else event.content
            scores[preview] = self._score(event)
        return scores
