"""
Context health service.

Wraps ContextWindowService to produce actionable health status reports
and exposes strategies for the agent to manage its own context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ctxforge.compaction.view import (
    CompactionView,
    CondensationResult,
    ICondenser,
)
from ctxforge.core.context import Context
from ctxforge.core.session import Session
from ctxforge.engine.services.context_window_service import (
    ContextWindowOverview,
    ContextWindowService,
)

logger = logging.getLogger(__name__)


class ContextStatusLevel(str, Enum):
    """Health status levels for the context window."""

    GOOD = "good"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ContextHealthReport:
    """Actionable health report for the context window."""

    status: ContextStatusLevel
    usage_percent: float
    total_budget: int
    used_tokens: int
    remaining_tokens: int
    section_breakdown: Dict[str, int]
    largest_section: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "usage_percent": round(self.usage_percent, 2),
            "total_budget": self.total_budget,
            "used_tokens": self.used_tokens,
            "remaining_tokens": self.remaining_tokens,
            "section_breakdown": dict(self.section_breakdown),
            "largest_section": self.largest_section,
            "recommendations": list(self.recommendations),
        }


@dataclass
class ContextHealthConfig:
    """Configuration for context health monitoring."""

    enabled: bool = True
    info_threshold: float = 0.50
    warning_threshold: float = 0.70
    critical_threshold: float = 0.85
    inject_warnings: bool = False
    inject_at_level: str = "warning"


@dataclass
class ManageContextResult:
    """Result of a manage_context operation."""

    strategy: str
    tokens_saved: int
    events_removed: int
    summary_generated: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "tokens_saved": self.tokens_saved,
            "events_removed": self.events_removed,
            "summary_generated": self.summary_generated,
            "detail": self.detail,
        }


class ContextHealthService:
    """Produces actionable health reports and executes context management
    strategies.

    Wraps ``ContextWindowService`` for status assessment and delegates
    to condensers for actual compaction.
    """

    def __init__(
        self,
        *,
        context_window_service: ContextWindowService,
        config: Optional[ContextHealthConfig] = None,
    ):
        self._cws = context_window_service
        self._config = config or ContextHealthConfig()

    @property
    def config(self) -> ContextHealthConfig:
        return self._config

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    def get_report(self, context: Context) -> ContextHealthReport:
        """Build a health report from an assembled context."""
        overview = self._cws.build_overview(context)
        return self._build_report(overview)

    def get_report_from_overview(
        self, overview: ContextWindowOverview,
    ) -> ContextHealthReport:
        """Build a health report from a pre-computed overview."""
        return self._build_report(overview)

    # ------------------------------------------------------------------
    # Context management strategies
    # ------------------------------------------------------------------

    async def apply_strategy(
        self,
        *,
        strategy: str,
        session: Session,
        condenser: Optional[ICondenser] = None,
        n: Optional[int] = None,
    ) -> ManageContextResult:
        """Execute a context management strategy on a session.

        Supported strategies:
        - ``keep_recent_turns``: keep last *n* turn-pairs (requires *n*).
        - ``summarize_history``: summarize older events (requires *condenser*).
        - ``drop_first_turns``: drop the first *n* turn-pairs (requires *n*).
        """
        if strategy == "keep_recent_turns":
            return self._keep_recent_turns(session, n or 5)
        if strategy == "summarize_history":
            return await self._summarize_history(session, condenser)
        if strategy == "drop_first_turns":
            return self._drop_first_turns(session, n or 1)

        return ManageContextResult(
            strategy=strategy,
            tokens_saved=0,
            events_removed=0,
            summary_generated=False,
            detail=f"Unknown strategy: {strategy}",
        )

    # ------------------------------------------------------------------
    # Warning injection helper
    # ------------------------------------------------------------------

    def build_warning_note(self, report: ContextHealthReport) -> Optional[str]:
        """Return a warning string to inject into the system prompt, or
        ``None`` if injection is not warranted."""
        if not self._config.inject_warnings:
            return None

        min_level = _LEVEL_ORDER.get(
            self._config.inject_at_level, _LEVEL_ORDER["warning"],
        )
        current_level = _LEVEL_ORDER.get(report.status.value, 0)
        if current_level < min_level:
            return None

        return (
            f"[Context Health: {report.status.value.upper()} "
            f"- {report.usage_percent:.0f}% used]\n"
            "Your context window is filling up. Call "
            "`check_context_status` for details or "
            "`manage_context` to free space."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(
        self, overview: ContextWindowOverview,
    ) -> ContextHealthReport:
        available = overview.available_input_budget
        used = overview.total_input_tokens

        if available > 0:
            usage_pct = (used / available) * 100.0
        else:
            usage_pct = 100.0 if used > 0 else 0.0

        status = self._classify(usage_pct)

        breakdown = self._collect_breakdown(overview)
        largest = max(breakdown, key=breakdown.get) if breakdown else ""

        recommendations = self._generate_recommendations(
            overview, usage_pct, status, breakdown, largest,
        )

        remaining = max(0, available - used)

        return ContextHealthReport(
            status=status,
            usage_percent=usage_pct,
            total_budget=overview.total_budget,
            used_tokens=used,
            remaining_tokens=remaining,
            section_breakdown=breakdown,
            largest_section=largest,
            recommendations=recommendations,
        )

    def _classify(self, usage_pct: float) -> ContextStatusLevel:
        if usage_pct >= self._config.critical_threshold * 100:
            return ContextStatusLevel.CRITICAL
        if usage_pct >= self._config.warning_threshold * 100:
            return ContextStatusLevel.WARNING
        if usage_pct >= self._config.info_threshold * 100:
            return ContextStatusLevel.INFO
        return ContextStatusLevel.GOOD

    @staticmethod
    def _collect_breakdown(
        overview: ContextWindowOverview,
    ) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        if overview.system_instructions_tokens:
            breakdown["system_instructions"] = overview.system_instructions_tokens
        for name, tokens in overview.section_tokens.items():
            if tokens:
                breakdown[name] = tokens
        if overview.memories_tokens:
            breakdown["memories"] = overview.memories_tokens
        if overview.expertise_tokens:
            breakdown["expertise"] = overview.expertise_tokens
        if overview.history_tokens:
            breakdown["history"] = overview.history_tokens
        if overview.current_query_tokens:
            breakdown["current_query"] = overview.current_query_tokens
        return breakdown

    def _generate_recommendations(
        self,
        overview: ContextWindowOverview,
        usage_pct: float,
        status: ContextStatusLevel,
        breakdown: Dict[str, int],
        largest: str,
    ) -> List[str]:
        recs: List[str] = []
        used = overview.total_input_tokens
        if used == 0:
            return recs

        # History dominates (> 60%)
        if overview.history_tokens > used * 0.6:
            recs.append(
                "Conversation history dominates context. Consider "
                "summarizing with "
                "`manage_context(strategy='summarize_history')`."
            )

        # Memories large (> 30%)
        if overview.memories_tokens > used * 0.3:
            recs.append(
                "Memory section is large. Consider scoping retrieval "
                "or reducing memory count."
            )

        # Single section > 40%
        for name, tokens in breakdown.items():
            if name in ("history", "current_query"):
                continue
            pct = (tokens / used) * 100
            if pct > 40:
                recs.append(
                    f"Section '{name}' uses {pct:.0f}% of context. "
                    "Consider compacting it."
                )

        # Always recommend action at CRITICAL
        if status == ContextStatusLevel.CRITICAL:
            recs.append(
                "Context is nearly full. Summarize or drop early "
                "turns immediately to avoid truncation."
            )

        return recs

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _keep_recent_turns(
        self, session: Session, n: int,
    ) -> ManageContextResult:
        events = list(session.events)
        original_count = len(events)
        if original_count <= n * 2:
            return ManageContextResult(
                strategy="keep_recent_turns",
                tokens_saved=0,
                events_removed=0,
                summary_generated=False,
                detail=f"Already at or below {n} turn-pairs.",
            )

        kept = events[-(n * 2):]
        removed_count = original_count - len(kept)

        tokens_before = self._estimate_event_tokens(events)
        tokens_after = self._estimate_event_tokens(kept)
        saved = max(0, tokens_before - tokens_after)

        session.events.clear()
        session.events.extend(kept)

        return ManageContextResult(
            strategy="keep_recent_turns",
            tokens_saved=saved,
            events_removed=removed_count,
            summary_generated=False,
            detail=f"Kept last {n} turn-pairs, removed {removed_count} events.",
        )

    async def _summarize_history(
        self,
        session: Session,
        condenser: Optional[ICondenser],
    ) -> ManageContextResult:
        if condenser is None:
            return ManageContextResult(
                strategy="summarize_history",
                tokens_saved=0,
                events_removed=0,
                summary_generated=False,
                detail="No condenser provided; cannot summarize.",
            )

        view = CompactionView.from_session(session)
        result = await condenser.condense(view)

        if isinstance(result, CondensationResult):
            result_view = result.view
            tokens_saved = result.tokens_saved
            summary_generated = result.summary_generated
        else:
            result_view = result
            tokens_saved = 0
            summary_generated = result_view.summary is not None

        new_events = result_view.to_context_events()
        removed_count = len(session.events) - len(new_events)

        session.events.clear()
        session.events.extend(new_events)
        if result_view.summary is not None:
            session.summary = result_view.summary

        return ManageContextResult(
            strategy="summarize_history",
            tokens_saved=tokens_saved,
            events_removed=max(0, removed_count),
            summary_generated=summary_generated,
            detail=(
                f"Summarized history, removed {max(0, removed_count)} events, "
                f"saved ~{tokens_saved} tokens."
            ),
        )

    def _drop_first_turns(
        self, session: Session, n: int,
    ) -> ManageContextResult:
        events = list(session.events)
        original_count = len(events)
        to_drop = min(n * 2, original_count)
        if to_drop == 0:
            return ManageContextResult(
                strategy="drop_first_turns",
                tokens_saved=0,
                events_removed=0,
                summary_generated=False,
                detail="No events to drop.",
            )

        kept = events[to_drop:]
        dropped = events[:to_drop]

        tokens_saved = self._estimate_event_tokens(dropped)

        session.events.clear()
        session.events.extend(kept)

        return ManageContextResult(
            strategy="drop_first_turns",
            tokens_saved=tokens_saved,
            events_removed=len(dropped),
            summary_generated=False,
            detail=f"Dropped first {len(dropped)} events.",
        )

    @staticmethod
    def _estimate_event_tokens(events: List) -> int:
        total = 0
        for e in events:
            total += len(e.content.split())
        return total


# Internal ordering for level comparison
_LEVEL_ORDER: Dict[str, int] = {
    "good": 0,
    "info": 1,
    "warning": 2,
    "critical": 3,
}
