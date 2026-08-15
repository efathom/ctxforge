"""
Tests for Query Rewriter Middleware.

Tests for middleware integration and context handling.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.config.base import QueryRewriteConfig
from ctxforge.core.events import Event, EventType
from ctxforge.core.session import Session
from ctxforge.engine.services.query_rewriter_service import QueryRewriterService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.query_rewriter import (
    QueryRewriterMiddleware,
    QueryRewriterMiddlewareFactory,
)

# =============================================================================
# Test Fixtures
# =============================================================================

def create_event(
    event_type: EventType,
    content: str,
    event_id: Optional[str] = None,
) -> Event:
    """Helper to create test events."""
    return Event(
        event_id=event_id or f"test-{event_type.value}-{len(content)}",
        type=event_type,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def create_session_with_history() -> Session:
    """Create a session with conversation history."""
    session = Session(session_id="test-session", user_id="test-user")
    session.events = [
        create_event(EventType.USER, "I'm planning a trip to Japan"),
        create_event(EventType.AGENT, "Japan is beautiful! What would you like to know?"),
        create_event(EventType.USER, "My friend John recommended Kyoto"),
        create_event(EventType.AGENT, "Kyoto is wonderful for temples."),
    ]
    session.summary = "User planning Japan trip, friend John recommended Kyoto"
    return session


def create_middleware_context(
    user_input: str,
    session: Optional[Session] = None,
    phase: str = "prepare",
) -> MiddlewareContext:
    """Create a middleware context for testing."""
    context = MiddlewareContext(user_input=user_input)
    context.session = session
    context.phase = phase
    return context


class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content


# =============================================================================
# Test QueryRewriterMiddleware
# =============================================================================

class TestQueryRewriterMiddleware:
    """Tests for QueryRewriterMiddleware."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = QueryRewriteConfig(enabled=True)
        self.rewriter = QueryRewriterService(self.mock_llm, self.config)

    def test_middleware_initialization(self):
        """Test middleware initializes correctly."""
        middleware = QueryRewriterMiddleware(
            rewriter_service=self.rewriter,
            max_history_turns=5,
            enabled=True,
        )

        assert middleware.name == "query_rewriter"
        assert middleware.enabled is True
        assert middleware._max_history_turns == 5

    def test_middleware_initialization_with_llm_provider(self):
        """Test middleware creates service from LLM provider."""
        middleware = QueryRewriterMiddleware(
            llm_provider=self.mock_llm,
            config=self.config,
        )

        assert middleware._rewriter is not None

    def test_middleware_initialization_without_service(self):
        """Test middleware handles missing service."""
        middleware = QueryRewriterMiddleware(
            rewriter_service=None,
            llm_provider=None,
        )

        assert middleware._rewriter is None

    @pytest.mark.asyncio
    async def test_middleware_skips_non_prepare_phase(self):
        """Test middleware skips non-prepare phases."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context("What about them?", phase="record")

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        next_fn.assert_called_once()
        # Should not have rewrite metadata
        assert context.get_metadata("query_rewriter.was_rewritten") is None

    @pytest.mark.asyncio
    async def test_middleware_skips_without_service(self):
        """Test middleware skips when service is unavailable."""
        middleware = QueryRewriterMiddleware(
            rewriter_service=None,
            llm_provider=None,
        )
        context = create_middleware_context("What about them?", phase="prepare")

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        next_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_skips_empty_input(self):
        """Test middleware skips empty user input."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context("", phase="prepare")

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        next_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_rewrites_query(self):
        """Test middleware rewrites ambiguous query."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>What food does John prefer for the Japan trip?</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.9</confidence>
              <resolved_entities>John, Japan trip</resolved_entities>
            </rewrite_response>
            """
        ))

        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        session = create_session_with_history()
        context = create_middleware_context(
            "What about their food preferences?",
            session=session,
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Check query was rewritten
        assert context.processed_input == "What food does John prefer for the Japan trip?"
        assert context.get_metadata("query_rewriter.was_rewritten") is True
        assert context.get_metadata("query_rewriter.reason") == "pronoun"

    @pytest.mark.asyncio
    async def test_middleware_preserves_clear_query(self):
        """Test middleware preserves clear queries."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context(
            "What are the best restaurants in Tokyo for sushi?",
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Query should not be rewritten - was_rewritten should be False
        assert context.get_metadata("query_rewriter.was_rewritten") is False
        # processed_input may be set to original query (unchanged)
        assert context.get_metadata("query_rewriter.reason") == "no_change"

    @pytest.mark.asyncio
    async def test_middleware_records_metadata(self):
        """Test middleware records all metadata."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>Rewritten query about John</rewritten_query>
              <reason>reference</reason>
              <confidence>0.85</confidence>
              <resolved_entities>John, Kyoto</resolved_entities>
            </rewrite_response>
            """
        ))

        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        session = create_session_with_history()
        context = create_middleware_context(
            "Tell me more about that",
            session=session,
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Check all metadata
        assert context.get_metadata("query_rewriter.was_rewritten") is True
        assert context.get_metadata("query_rewriter.original_query") == "Tell me more about that"
        assert "John" in context.get_metadata("query_rewriter.rewritten_query")
        assert context.get_metadata("query_rewriter.reason") == "reference"
        assert context.get_metadata("query_rewriter.confidence") == 0.85
        assert "John" in context.get_metadata("query_rewriter.resolved_entities")

    @pytest.mark.asyncio
    async def test_middleware_handles_rewriter_error(self):
        """Test middleware handles rewriter errors gracefully."""
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        session = create_session_with_history()
        context = create_middleware_context(
            "What about them?",
            session=session,
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Should continue with original query (unchanged)
        assert context.get_metadata("query_rewriter.was_rewritten") is False
        # Error was caught and logged, query unchanged
        assert context.get_metadata("query_rewriter.reason") == "no_change"

    @pytest.mark.asyncio
    async def test_middleware_uses_session_summary(self):
        """Test middleware passes session summary to rewriter."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>Query with context</rewritten_query>
              <reason>implicit</reason>
              <confidence>0.9</confidence>
              <resolved_entities></resolved_entities>
            </rewrite_response>
            """
        ))

        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        session = create_session_with_history()
        context = create_middleware_context(
            "And the temples?",
            session=session,
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Verify LLM was called (session summary should be passed)
        self.mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_respects_max_history_turns(self):
        """Test middleware limits history turns."""
        middleware = QueryRewriterMiddleware(
            rewriter_service=self.rewriter,
            max_history_turns=2,
        )

        # Create session with many events
        session = Session(session_id="test", user_id="test")
        session.events = [
            create_event(EventType.USER, f"Message {i}")
            for i in range(10)
        ]

        context = create_middleware_context(
            "What about them?",
            session=session,
            phase="prepare",
        )

        # The middleware should only pass last 2 events to rewriter
        assert middleware._max_history_turns == 2

        # Verify context was created (no actual rewrite test here)
        assert context.user_input == "What about them?"

    def test_middleware_stats(self):
        """Test middleware statistics tracking."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)

        # Initial stats
        assert middleware.total_queries == 0
        assert middleware.total_rewrites == 0
        assert middleware.rewrite_rate == 0.0

        # Simulate some queries
        middleware._total_queries = 10
        middleware._total_rewrites = 3

        assert middleware.total_queries == 10
        assert middleware.total_rewrites == 3
        assert middleware.rewrite_rate == 30.0

    def test_middleware_reset_stats(self):
        """Test statistics reset."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        middleware._total_queries = 10
        middleware._total_rewrites = 3

        middleware.reset_stats()

        assert middleware.total_queries == 0
        assert middleware.total_rewrites == 0

    def test_middleware_get_stats(self):
        """Test get_stats returns correct data."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        middleware._total_queries = 20
        middleware._total_rewrites = 5

        stats = middleware.get_stats()

        assert stats["total_queries"] == 20
        assert stats["total_rewrites"] == 5
        assert stats["rewrite_rate"] == 25.0

    @pytest.mark.asyncio
    async def test_middleware_disabled(self):
        """Test middleware can be disabled."""
        middleware = QueryRewriterMiddleware(
            rewriter_service=self.rewriter,
            enabled=False,
        )
        context = create_middleware_context("What about them?", phase="prepare")

        next_fn = AsyncMock(return_value=context)
        await middleware.process(context, next_fn)

        # Should skip processing when disabled
        next_fn.assert_called_once()


# =============================================================================
# Test QueryRewriterMiddlewareFactory
# =============================================================================

class TestQueryRewriterMiddlewareFactory:
    """Tests for QueryRewriterMiddlewareFactory."""

    def test_create_with_provider(self):
        """Test factory creates middleware with provider."""
        mock_llm = MagicMock()

        middleware = QueryRewriterMiddlewareFactory.create(
            llm_provider=mock_llm,
        )

        assert middleware is not None
        assert isinstance(middleware, QueryRewriterMiddleware)

    def test_create_without_provider(self):
        """Test factory returns None without provider."""
        middleware = QueryRewriterMiddlewareFactory.create(
            llm_provider=None,
        )

        assert middleware is None

    def test_create_with_config(self):
        """Test factory applies custom config."""
        mock_llm = MagicMock()
        config = QueryRewriteConfig(
            enabled=True,
            max_history_turns=5,
        )

        middleware = QueryRewriterMiddlewareFactory.create(
            llm_provider=mock_llm,
            config=config,
            max_history_turns=8,
        )

        assert middleware._max_history_turns == 8


# =============================================================================
# Test Phase Handling
# =============================================================================

class TestPhaseHandling:
    """Tests for middleware phase handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = QueryRewriteConfig(enabled=True)
        self.rewriter = QueryRewriterService(self.mock_llm, self.config)

    @pytest.mark.asyncio
    async def test_handles_prepare_phase(self):
        """Test middleware handles 'prepare' phase."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context(
            "What are the best sushi restaurants in Tokyo?",
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Should process (even if no rewrite needed)
        assert context.get_metadata("query_rewriter.was_rewritten") is False

    @pytest.mark.asyncio
    async def test_handles_prepare_input_phase(self):
        """Test middleware handles 'prepare_input' phase."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context(
            "What are the best sushi restaurants?",
            phase="prepare_input",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Should process
        assert context.get_metadata("query_rewriter.was_rewritten") is False

    @pytest.mark.asyncio
    async def test_handles_prepare_context_phase(self):
        """Test middleware handles 'prepare_context' phase."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context(
            "What are the best sushi restaurants?",
            phase="prepare_context",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Should process
        assert context.get_metadata("query_rewriter.was_rewritten") is False

    @pytest.mark.asyncio
    async def test_skips_record_phase(self):
        """Test middleware skips 'record' phase."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context("What about them?", phase="record")

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Should not process
        assert context.get_metadata("query_rewriter.was_rewritten") is None

    @pytest.mark.asyncio
    async def test_skips_unknown_phase(self):
        """Test middleware skips unknown phases."""
        middleware = QueryRewriterMiddleware(rewriter_service=self.rewriter)
        context = create_middleware_context("What about them?", phase="unknown")

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Should not process
        assert context.get_metadata("query_rewriter.was_rewritten") is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestMiddlewareIntegration:
    """Integration tests for query rewriter middleware."""

    @pytest.mark.asyncio
    async def test_full_rewrite_flow(self):
        """Test complete middleware flow with rewriting."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>What food does John prefer for the Japan trip?</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.92</confidence>
              <resolved_entities>John, Japan trip</resolved_entities>
            </rewrite_response>
            """
        ))

        config = QueryRewriteConfig(enabled=True)
        middleware = QueryRewriterMiddleware(
            llm_provider=mock_llm,
            config=config,
        )

        session = create_session_with_history()
        context = create_middleware_context(
            "What about their food preferences?",
            session=session,
            phase="prepare",
        )

        next_fn = AsyncMock(return_value=context)
        await middleware._do_process(context, next_fn)

        # Verify complete flow
        assert context.processed_input is not None
        assert "John" in context.processed_input
        assert context.get_metadata("query_rewriter.was_rewritten") is True
        assert middleware.total_rewrites == 1
        assert middleware.total_queries == 1

    @pytest.mark.asyncio
    async def test_multiple_queries_stats(self):
        """Test statistics across multiple queries."""
        mock_llm = MagicMock()

        # First query: rewritten
        mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>Rewritten query</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.9</confidence>
              <resolved_entities></resolved_entities>
            </rewrite_response>
            """
        ))

        config = QueryRewriteConfig(enabled=True)
        middleware = QueryRewriterMiddleware(
            llm_provider=mock_llm,
            config=config,
        )

        session = create_session_with_history()
        next_fn = AsyncMock(side_effect=lambda ctx: ctx)

        # Query 1: needs rewriting
        ctx1 = create_middleware_context("What about them?", session, "prepare")
        await middleware._do_process(ctx1, next_fn)

        # Query 2: clear query (no rewrite)
        ctx2 = create_middleware_context(
            "What are the best sushi restaurants in Tokyo?",
            session,
            "prepare",
        )
        await middleware._do_process(ctx2, next_fn)

        # Query 3: needs rewriting
        ctx3 = create_middleware_context("And their preferences?", session, "prepare")
        await middleware._do_process(ctx3, next_fn)

        # Check stats
        assert middleware.total_queries == 3
        assert middleware.total_rewrites == 2
        assert middleware.rewrite_rate == pytest.approx(66.67, rel=0.1)
