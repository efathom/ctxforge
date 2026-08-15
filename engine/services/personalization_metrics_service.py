"""
Personalization effectiveness metrics service.

Tracks lightweight signals during prepare_context() and record_turn() flows
to measure whether personalization is improving.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.extraction.integration_config import PersonalizationMetricsConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn."""

    turn_number: int
    memories_retrieved: int = 0
    memories_used: int = 0
    memory_hit: bool = False
    memory_avg_score: float = 0.0
    feedback_occurred: bool = False
    extraction_count: int = 0
    preference_changes: int = 0
    timestamp: datetime.datetime = field(
        default_factory=datetime.datetime.now,
    )


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""

    session_id: str
    user_id: str
    turn_metrics: List[TurnMetrics] = field(default_factory=list)

    @property
    def memory_hit_rate(self) -> float:
        """Percentage of turns where relevant memories were found."""
        if not self.turn_metrics:
            return 0.0
        return sum(1 for t in self.turn_metrics if t.memory_hit) / len(
            self.turn_metrics
        )

    @property
    def avg_memory_relevance(self) -> float:
        """Average relevance score across turns."""
        scores = [
            t.memory_avg_score
            for t in self.turn_metrics
            if t.memories_retrieved > 0
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def feedback_frequency(self) -> float:
        """Feedback frequency: % of turns requiring feedback (should decrease over time)."""
        if not self.turn_metrics:
            return 0.0
        return sum(
            1 for t in self.turn_metrics if t.feedback_occurred
        ) / len(self.turn_metrics)

    @property
    def cumulative_personalization_score(self) -> List[float]:
        """Cumulative personalization score: running average of memory hit rate."""
        scores: List[float] = []
        cumulative = 0.0
        for i, t in enumerate(self.turn_metrics):
            cumulative += 1.0 if t.memory_hit else 0.0
            scores.append(cumulative / (i + 1))
        return scores

    @property
    def memory_utilization(self) -> float:
        """% of retrieved memories that fit in context budget."""
        retrieved = sum(t.memories_retrieved for t in self.turn_metrics)
        used = sum(t.memories_used for t in self.turn_metrics)
        return used / retrieved if retrieved > 0 else 0.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PersonalizationMetricsService:
    """Tracks personalization effectiveness metrics."""

    def __init__(
        self,
        config: Optional[PersonalizationMetricsConfig] = None,
    ):
        self._config = config or PersonalizationMetricsConfig()
        self._sessions: Dict[str, SessionMetrics] = {}
        self._hit_threshold = self._config.memory_hit_threshold

    def _get_or_create_session(
        self,
        session_id: str,
        user_id: str = "",
    ) -> SessionMetrics:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMetrics(
                session_id=session_id,
                user_id=user_id,
            )
        return self._sessions[session_id]

    def record_retrieval(
        self,
        session_id: str,
        user_id: str,
        memories: List[MemoryItem],
        memories_in_context: int = 0,
    ) -> None:
        """Called during prepare_context after memory retrieval."""
        session = self._get_or_create_session(session_id, user_id)
        turn_number = len(session.turn_metrics) + 1

        # Compute hit: at least one memory with confidence above threshold
        has_hit = any(
            m.confidence_score >= self._hit_threshold for m in memories
        )
        avg_score = (
            sum(m.confidence_score for m in memories) / len(memories)
            if memories
            else 0.0
        )

        turn = TurnMetrics(
            turn_number=turn_number,
            memories_retrieved=len(memories),
            memories_used=memories_in_context if memories_in_context else len(memories),
            memory_hit=has_hit,
            memory_avg_score=avg_score,
        )
        session.turn_metrics.append(turn)

    def record_feedback(
        self,
        session_id: str,
        feedback_occurred: bool,
    ) -> None:
        """Called when user provides correction/feedback."""
        if session_id not in self._sessions:
            return
        session = self._sessions[session_id]
        if session.turn_metrics:
            session.turn_metrics[-1].feedback_occurred = feedback_occurred

    def record_extraction(
        self,
        session_id: str,
        extraction_count: int,
        preference_changes: int = 0,
    ) -> None:
        """Called after memory extraction in record_turn."""
        if session_id not in self._sessions:
            session = self._get_or_create_session(session_id)
        else:
            session = self._sessions[session_id]

        if session.turn_metrics:
            session.turn_metrics[-1].extraction_count = extraction_count
            session.turn_metrics[-1].preference_changes = preference_changes
        else:
            # No retrieval recorded yet for this turn
            turn = TurnMetrics(
                turn_number=1,
                extraction_count=extraction_count,
                preference_changes=preference_changes,
            )
            session.turn_metrics.append(turn)

    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        return self._sessions.get(session_id)

    def get_user_metrics(self, user_id: str) -> List[SessionMetrics]:
        return [
            s for s in self._sessions.values() if s.user_id == user_id
        ]

    def to_dict(self, session_id: str) -> Dict[str, Any]:
        """Export metrics as dict for logging/monitoring."""
        session = self._sessions.get(session_id)
        if session is None:
            return {}
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "turns": len(session.turn_metrics),
            "memory_hit_rate": session.memory_hit_rate,
            "avg_memory_relevance": session.avg_memory_relevance,
            "feedback_frequency": session.feedback_frequency,
            "memory_utilization": session.memory_utilization,
            "cumulative_personalization_score": session.cumulative_personalization_score,
        }
