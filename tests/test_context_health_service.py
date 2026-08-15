"""Tests for ContextHealthService."""

import pytest

from ctxforge.core.context import Context
from ctxforge.core.events import EventFactory
from ctxforge.core.session import Session
from ctxforge.engine.services.context_health_service import (
    ContextHealthConfig,
    ContextHealthService,
    ContextStatusLevel,
    ManageContextResult,
)
from ctxforge.engine.services.context_window_service import (
    ContextWindowOverview,
    ContextWindowService,
)
from ctxforge.protocols.llm import ChatMessage

# ---- helpers ---------------------------------------------------------------


class DummyTokenizer:
    """1 token per whitespace-separated word."""

    name = "dummy"

    def count_tokens(self, text: str, model=None) -> int:
        return len((text or "").split())

    def count_message_tokens(
        self, messages: list[ChatMessage], model=None,
    ) -> int:
        return sum(self.count_tokens(m.content, model=model) for m in messages)


def _make_service(
    *,
    info: float = 0.50,
    warning: float = 0.70,
    critical: float = 0.85,
    inject_warnings: bool = False,
    inject_at_level: str = "warning",
) -> ContextHealthService:
    cws = ContextWindowService(tokenizer=DummyTokenizer())
    cfg = ContextHealthConfig(
        info_threshold=info,
        warning_threshold=warning,
        critical_threshold=critical,
        inject_warnings=inject_warnings,
        inject_at_level=inject_at_level,
    )
    return ContextHealthService(context_window_service=cws, config=cfg)


def _context_with_usage(
    *,
    budget: int = 1000,
    reserved: int = 0,
    sys_words: int = 0,
    history_words: int = 0,
    query_words: int = 0,
    memory_words: int = 0,
) -> Context:
    """Build a Context whose token usage is roughly controllable by word count.

    The DummyTokenizer counts 1 token per word, so this lets us set up
    precise usage scenarios.
    """
    events = []
    if history_words > 0:
        half = history_words // 2
        events = [
            EventFactory.user_message("u " * max(1, half)),
            EventFactory.agent_message("a " * max(1, history_words - half)),
        ]

    memories = []
    if memory_words > 0:
        from ctxforge.core.memory import MemoryItem, MemoryType
        memories = [
            MemoryItem(
                content="m " * memory_words,
                type=MemoryType.SEMANTIC,
                user_id="u",
            ),
        ]

    return Context(
        session_id="s",
        user_id="u",
        system_instructions="s " * sys_words if sys_words else "",
        events=events,
        current_query="q " * query_words if query_words else "",
        total_token_budget=budget,
        reserved_output_tokens=reserved,
        memories=memories,
    )


def _session_with_events(n_pairs: int) -> Session:
    """Create a session with *n_pairs* user+agent turn-pairs."""
    session = Session(session_id="s", user_id="u")
    for i in range(n_pairs):
        session.events.append(EventFactory.user_message(f"user msg {i}"))
        session.events.append(EventFactory.agent_message(f"agent msg {i}"))
    return session


# ---- status level tests ----------------------------------------------------


class TestStatusLevels:

    def test_status_good_under_50_percent(self):
        svc = _make_service()
        ctx = _context_with_usage(budget=1000, sys_words=100)
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.GOOD
        assert report.usage_percent < 50

    def test_status_info_between_50_and_70(self):
        svc = _make_service()
        # ~600 words used out of 1000 budget → 60%
        ctx = _context_with_usage(budget=1000, sys_words=300, history_words=300)
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.INFO

    def test_status_warning_between_70_and_85(self):
        svc = _make_service()
        ctx = _context_with_usage(budget=1000, sys_words=400, history_words=350)
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.WARNING

    def test_status_critical_above_85(self):
        svc = _make_service()
        ctx = _context_with_usage(budget=1000, sys_words=500, history_words=400)
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.CRITICAL

    def test_custom_thresholds(self):
        svc = _make_service(info=0.10, warning=0.20, critical=0.30)
        # ~50 words used out of 1000 → 5%, should be GOOD even with tight thresholds
        ctx = _context_with_usage(budget=1000, sys_words=50)
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.GOOD

        # ~350 words → 35%, exceeds critical=30%
        ctx2 = _context_with_usage(budget=1000, sys_words=200, history_words=150)
        report2 = svc.get_report(ctx2)
        assert report2.status == ContextStatusLevel.CRITICAL


# ---- breakdown tests -------------------------------------------------------


class TestBreakdown:

    def test_section_breakdown_populated(self):
        svc = _make_service()
        ctx = _context_with_usage(
            budget=2000, sys_words=50, history_words=100, query_words=10,
        )
        report = svc.get_report(ctx)
        assert isinstance(report.section_breakdown, dict)
        assert len(report.section_breakdown) > 0

    def test_largest_section_identified(self):
        svc = _make_service()
        ctx = _context_with_usage(
            budget=2000, sys_words=10, history_words=200, query_words=5,
        )
        report = svc.get_report(ctx)
        assert report.largest_section == "history"


# ---- recommendation tests --------------------------------------------------


class TestRecommendations:

    def test_recommendation_history_dominates(self):
        svc = _make_service()
        # History ~80% of used tokens
        ctx = _context_with_usage(budget=1000, sys_words=10, history_words=400)
        report = svc.get_report(ctx)
        assert any("history" in r.lower() for r in report.recommendations)

    def test_recommendation_memories_large(self):
        svc = _make_service()
        # Memories > 30% of used
        ctx = _context_with_usage(
            budget=2000, sys_words=50, memory_words=200,
        )
        report = svc.get_report(ctx)
        assert any("memory" in r.lower() for r in report.recommendations)

    def test_recommendation_critical_always_present(self):
        svc = _make_service()
        ctx = _context_with_usage(budget=1000, sys_words=500, history_words=400)
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.CRITICAL
        assert any("nearly full" in r.lower() for r in report.recommendations)

    def test_recommendation_section_heavy(self):
        svc = _make_service()
        ctx = _context_with_usage(budget=2000, sys_words=400, history_words=10)
        report = svc.get_report(ctx)
        # system_instructions > 40% of used
        assert any("section" in r.lower() for r in report.recommendations)

    def test_no_recommendations_when_good(self):
        svc = _make_service()
        # Keep all parts balanced so no single section > 40%
        ctx = _context_with_usage(
            budget=10000, sys_words=30, history_words=30, query_words=30,
        )
        report = svc.get_report(ctx)
        assert report.status == ContextStatusLevel.GOOD
        assert report.recommendations == []


# ---- strategy tests --------------------------------------------------------


class TestKeepRecentTurns:

    @pytest.mark.asyncio
    async def test_apply_keep_recent_turns(self):
        svc = _make_service()
        session = _session_with_events(10)
        assert len(session.events) == 20

        result = await svc.apply_strategy(
            strategy="keep_recent_turns", session=session, n=3,
        )
        assert isinstance(result, ManageContextResult)
        assert result.strategy == "keep_recent_turns"
        assert result.events_removed == 14  # 20 - 6
        assert result.tokens_saved > 0
        assert len(session.events) == 6

    @pytest.mark.asyncio
    async def test_apply_strategy_idempotent(self):
        svc = _make_service()
        session = _session_with_events(2)  # 4 events

        result = await svc.apply_strategy(
            strategy="keep_recent_turns", session=session, n=5,
        )
        assert result.events_removed == 0
        assert result.tokens_saved == 0
        assert len(session.events) == 4


class TestDropFirstTurns:

    @pytest.mark.asyncio
    async def test_apply_drop_first_turns(self):
        svc = _make_service()
        session = _session_with_events(5)
        assert len(session.events) == 10

        result = await svc.apply_strategy(
            strategy="drop_first_turns", session=session, n=2,
        )
        assert result.events_removed == 4
        assert result.tokens_saved > 0
        assert len(session.events) == 6

    @pytest.mark.asyncio
    async def test_drop_first_turns_empty_session(self):
        svc = _make_service()
        session = Session(session_id="s", user_id="u")

        result = await svc.apply_strategy(
            strategy="drop_first_turns", session=session, n=1,
        )
        assert result.events_removed == 0
        assert len(session.events) == 0


class TestSummarizeHistory:

    @pytest.mark.asyncio
    async def test_apply_summarize_history(self):
        svc = _make_service()
        session = _session_with_events(10)

        from ctxforge.compaction.summarizing import SummarizingCondenser

        condenser = SummarizingCondenser(summarize_func=None)
        result = await svc.apply_strategy(
            strategy="summarize_history",
            session=session,
            condenser=condenser,
        )
        assert result.strategy == "summarize_history"
        assert result.summary_generated is True
        assert result.events_removed > 0
        # Session should have fewer events now
        assert len(session.events) < 20

    @pytest.mark.asyncio
    async def test_summarize_history_no_condenser(self):
        svc = _make_service()
        session = _session_with_events(5)

        result = await svc.apply_strategy(
            strategy="summarize_history", session=session,
        )
        assert result.tokens_saved == 0
        assert "No condenser" in result.detail


class TestUnknownStrategy:

    @pytest.mark.asyncio
    async def test_unknown_strategy(self):
        svc = _make_service()
        session = _session_with_events(3)

        result = await svc.apply_strategy(
            strategy="does_not_exist", session=session,
        )
        assert "Unknown strategy" in result.detail
        assert result.tokens_saved == 0


# ---- proactive injection tests ---------------------------------------------


class TestProactiveInjection:

    def test_proactive_injection_warning(self):
        svc = _make_service(inject_warnings=True, inject_at_level="warning")
        ctx = _context_with_usage(budget=1000, sys_words=400, history_words=350)
        report = svc.get_report(ctx)
        assert report.status in (
            ContextStatusLevel.WARNING, ContextStatusLevel.CRITICAL,
        )
        note = svc.build_warning_note(report)
        assert note is not None
        assert "context window" in note.lower()

    def test_proactive_injection_respects_level(self):
        svc = _make_service(inject_warnings=True, inject_at_level="critical")
        ctx = _context_with_usage(budget=1000, sys_words=400, history_words=350)
        report = svc.get_report(ctx)
        if report.status == ContextStatusLevel.WARNING:
            note = svc.build_warning_note(report)
            assert note is None

    def test_no_injection_when_disabled(self):
        svc = _make_service(inject_warnings=False)
        ctx = _context_with_usage(budget=1000, sys_words=500, history_words=400)
        report = svc.get_report(ctx)
        note = svc.build_warning_note(report)
        assert note is None


# ---- serialization tests ---------------------------------------------------


class TestSerialization:

    def test_report_to_dict(self):
        svc = _make_service()
        ctx = _context_with_usage(budget=1000, sys_words=100, history_words=50)
        report = svc.get_report(ctx)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["status"] in ("good", "info", "warning", "critical")
        assert isinstance(d["usage_percent"], float)
        assert isinstance(d["recommendations"], list)

    def test_manage_result_to_dict(self):
        result = ManageContextResult(
            strategy="keep_recent_turns",
            tokens_saved=100,
            events_removed=4,
            summary_generated=False,
            detail="Kept last 3 turn-pairs.",
        )
        d = result.to_dict()
        assert d["strategy"] == "keep_recent_turns"
        assert d["tokens_saved"] == 100


# ---- from overview tests ---------------------------------------------------


class TestFromOverview:

    def test_get_report_from_overview(self):
        svc = _make_service()
        overview = ContextWindowOverview(
            total_budget=1000,
            reserved_output_tokens=200,
            available_input_budget=800,
            total_input_tokens=600,  # 75% of 800
            system_tokens=400,
            history_tokens=150,
            current_query_tokens=50,
            system_instructions_tokens=300,
            section_tokens={"skills": 100},
            memories_tokens=0,
            expertise_tokens=0,
        )
        report = svc.get_report_from_overview(overview)
        assert report.status == ContextStatusLevel.WARNING
        assert report.used_tokens == 600
        assert report.remaining_tokens == 200
