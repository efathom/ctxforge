"""
Query Rewriter Middleware.

Rewrites ambiguous queries using conversation context
during the prepare phase.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ctxforge.config.base import QueryRewriteConfig
from ctxforge.core.events import Event
from ctxforge.core.query import QueryRewriteResult
from ctxforge.engine.services.query_rewriter_service import (
    QueryRewriterService,
    QueryRewriterServiceFactory,
)
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction
from ctxforge.protocols.llm import ILLMProvider

logger = logging.getLogger(__name__)


class QueryRewriterMiddleware(BaseMiddleware):
    """
    Middleware that rewrites ambiguous queries.

    This middleware runs in the 'prepare' phase and rewrites queries
    containing pronouns, references, or implicit context by resolving
    them using conversation history.

    Example:
        >>> middleware = QueryRewriterMiddleware(
        ...     rewriter_service=rewriter,
        ...     max_history_turns=10,
        ... )
        >>> # Add to middleware chain
        >>> engine.add_middleware(middleware)

    The middleware records the rewrite result in context metadata:
        - query_rewriter.was_rewritten: bool
        - query_rewriter.original_query: str
        - query_rewriter.rewritten_query: str
        - query_rewriter.reason: str
        - query_rewriter.resolved_entities: List[str]
    """

    def __init__(
        self,
        rewriter_service: Optional[QueryRewriterService] = None,
        llm_provider: Optional[ILLMProvider] = None,
        config: Optional[QueryRewriteConfig] = None,
        max_history_turns: int = 10,
        enabled: bool = True,
    ):
        """
        Initialize the query rewriter middleware.

        Args:
            rewriter_service: Pre-configured rewriter service (optional)
            llm_provider: LLM provider for creating service (if no service given)
            config: Configuration for query rewriting
            max_history_turns: Maximum history turns to use for context
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled=enabled)

        # Use provided service or create one
        if rewriter_service is not None:
            self._rewriter = rewriter_service
        else:
            self._rewriter = QueryRewriterServiceFactory.create(
                llm_provider=llm_provider,
                config=config or QueryRewriteConfig(enabled=True),
            )

        self._max_history_turns = max_history_turns
        self._total_rewrites = 0
        self._total_queries = 0

    @property
    def name(self) -> str:
        return "query_rewriter"

    @property
    def total_rewrites(self) -> int:
        """Total number of queries rewritten."""
        return self._total_rewrites

    @property
    def total_queries(self) -> int:
        """Total number of queries processed."""
        return self._total_queries

    @property
    def rewrite_rate(self) -> float:
        """Percentage of queries that were rewritten."""
        if self._total_queries == 0:
            return 0.0
        return (self._total_rewrites / self._total_queries) * 100

    async def _do_process(
        self,
        context: MiddlewareContext,
        next_fn: NextFunction,
    ) -> MiddlewareContext:
        """Process and potentially rewrite the query."""
        # Only act during prepare phase
        phase = context.get_metadata("phase") or context.phase
        if phase not in ("prepare", "prepare_input", "prepare_context"):
            return await next_fn(context)

        # Check if rewriter is available
        if self._rewriter is None:
            logger.debug("Query rewriter service not available, skipping")
            return await next_fn(context)

        # Get user input
        user_input = context.user_input
        if not user_input:
            return await next_fn(context)

        self._total_queries += 1

        # Get conversation history from session
        history: List[Event] = []
        if context.session and context.session.events:
            history = context.session.events[-self._max_history_turns:]

        # Get session summary for additional context
        session_context: Optional[str] = None
        if context.session and context.session.summary:
            session_context = context.session.summary

        # Rewrite the query
        try:
            result = await self._rewriter.rewrite(
                query=user_input,
                conversation_history=history,
                session_context=session_context,
            )
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            result = QueryRewriteResult.unchanged(user_input)

        # Record the result in metadata
        self._record_result(context, result)

        # Update processed input if rewritten
        if result.was_rewritten:
            self._total_rewrites += 1
            context.processed_input = result.rewritten_query
            logger.info(
                f"Query rewritten: '{result.original_query[:50]}...' → "
                f"'{result.rewritten_query[:50]}...' (reason: {result.reason.value})"
            )

        return await next_fn(context)

    def _record_result(
        self,
        context: MiddlewareContext,
        result: QueryRewriteResult,
    ) -> None:
        """Record rewrite result in context metadata."""
        context.set_metadata("query_rewriter.was_rewritten", result.was_rewritten)
        context.set_metadata("query_rewriter.original_query", result.original_query)
        context.set_metadata("query_rewriter.rewritten_query", result.rewritten_query)
        context.set_metadata("query_rewriter.reason", result.reason.value)
        context.set_metadata("query_rewriter.confidence", result.confidence)
        context.set_metadata(
            "query_rewriter.resolved_entities",
            result.resolved_entities
        )

    def reset_stats(self) -> None:
        """Reset query rewriting statistics."""
        self._total_rewrites = 0
        self._total_queries = 0

    def get_stats(self) -> dict:
        """Get query rewriting statistics."""
        return {
            "total_queries": self._total_queries,
            "total_rewrites": self._total_rewrites,
            "rewrite_rate": self.rewrite_rate,
        }


class QueryRewriterMiddlewareFactory:
    """Factory for creating QueryRewriterMiddleware instances."""

    @staticmethod
    def create(
        llm_provider: Optional[ILLMProvider] = None,
        config: Optional[QueryRewriteConfig] = None,
        max_history_turns: int = 10,
        enabled: bool = True,
    ) -> Optional[QueryRewriterMiddleware]:
        """
        Create a QueryRewriterMiddleware if LLM provider is available.

        Args:
            llm_provider: LLM provider for rewriting
            config: Configuration for rewriting
            max_history_turns: Maximum history to consider
            enabled: Whether middleware is enabled

        Returns:
            QueryRewriterMiddleware or None if no LLM provider
        """
        if llm_provider is None:
            return None

        return QueryRewriterMiddleware(
            llm_provider=llm_provider,
            config=config,
            max_history_turns=max_history_turns,
            enabled=enabled,
        )
