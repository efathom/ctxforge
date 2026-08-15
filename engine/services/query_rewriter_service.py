"""
Query Rewriter Service.

Transforms ambiguous queries into explicit, self-contained queries
by resolving references using conversation context.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from ctxforge.config.base import QueryRewriteConfig
from ctxforge.core.events import Event, EventType
from ctxforge.core.query import QueryRewriteResult, RewriteReason
from ctxforge.prompts.query_rewrite import QUERY_REWRITE_PROMPT
from ctxforge.protocols.llm import ILLMProvider

logger = logging.getLogger(__name__)

# Heuristic patterns for quick detection
PRONOUN_PATTERN = re.compile(
    r'\b(they|them|their|theirs|it|its|he|him|his|she|her|hers|we|us|our|ours)\b',
    re.IGNORECASE
)
REFERENCE_PATTERN = re.compile(
    r'\b(that|those|this|these|the same|such|said|mentioned|above|previous)\b',
    re.IGNORECASE
)
IMPLICIT_PATTERN = re.compile(
    r'^(and|also|what about|how about|same for|likewise|plus|too)\b',
    re.IGNORECASE
)
ELLIPSIS_PATTERN = re.compile(
    r'^(yes|no|maybe|sure|ok|okay|right|exactly|correct)\s*[,.]?\s*',
    re.IGNORECASE
)


class QueryRewriterService:
    """
    Service for rewriting ambiguous queries.

    Uses heuristics for quick detection and LLM for actual rewriting.
    Supports caching for efficiency.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        config: Optional[QueryRewriteConfig] = None,
    ):
        self._llm = llm_provider
        self._config = config or QueryRewriteConfig()
        self._cache: Dict[str, Tuple[QueryRewriteResult, float]] = {}

    @property
    def config(self) -> QueryRewriteConfig:
        """Get current configuration."""
        return self._config

    async def rewrite(
        self,
        query: str,
        conversation_history: List[Event],
        session_context: Optional[str] = None,
    ) -> QueryRewriteResult:
        """
        Rewrite a query to be self-contained.

        Args:
            query: The user's original query
            conversation_history: Recent conversation events
            session_context: Optional additional context (e.g., session summary)

        Returns:
            QueryRewriteResult with original and rewritten query
        """
        if not self._config.enabled:
            return QueryRewriteResult.unchanged(query)

        # Quick heuristic check
        detection = self._detect_ambiguity(query)
        if detection is None:
            logger.debug(f"Query appears clear, skipping rewrite: {query[:50]}...")
            return QueryRewriteResult.unchanged(query)

        # Check cache
        cache_key = self._build_cache_key(query, conversation_history)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for query rewrite: {query[:50]}...")
            return cached

        # No history means we can't resolve references
        if not conversation_history:
            logger.debug("No conversation history available for rewriting")
            return QueryRewriteResult.unchanged(query)

        # Format history for LLM
        history_text = self._format_history(conversation_history)

        # Call LLM
        try:
            result = await self._llm_rewrite(
                query=query,
                history_text=history_text,
                session_context=session_context,
                detected_reason=detection,
            )
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}, returning original query")
            return QueryRewriteResult.unchanged(query)

        # Cache result
        self._add_to_cache(cache_key, result)

        return result

    def _detect_ambiguity(self, query: str) -> Optional[RewriteReason]:
        """
        Quick heuristic check to detect if query needs rewriting.

        Returns the detected reason, or None if query appears clear.
        """
        query_stripped = query.strip()

        # Check for pronouns
        if self._config.check_pronouns and PRONOUN_PATTERN.search(query_stripped):
            return RewriteReason.PRONOUN

        # Check for references
        if self._config.check_references and REFERENCE_PATTERN.search(query_stripped):
            return RewriteReason.REFERENCE

        # Check for implicit context
        if self._config.check_implicit and IMPLICIT_PATTERN.match(query_stripped):
            return RewriteReason.IMPLICIT

        # Check for ellipsis (very short follow-up)
        if ELLIPSIS_PATTERN.match(query_stripped):
            return RewriteReason.ELLIPSIS

        # Very short queries are likely incomplete
        word_count = len(query_stripped.split())
        if word_count <= 3:
            return RewriteReason.IMPLICIT

        return None

    def _format_history(
        self,
        events: List[Event],
        max_turns: Optional[int] = None,
    ) -> str:
        """Format conversation history for LLM prompt."""
        max_turns = max_turns or self._config.max_history_turns
        recent = events[-max_turns:] if len(events) > max_turns else events

        formatted_lines = []
        for event in recent:
            role = "User" if event.type == EventType.USER else "Assistant"
            content = event.content[:500]  # Truncate long content
            formatted_lines.append(f"{role}: {content}")

        return "\n".join(formatted_lines) if formatted_lines else "No history"

    async def _llm_rewrite(
        self,
        query: str,
        history_text: str,
        session_context: Optional[str],
        detected_reason: RewriteReason,
    ) -> QueryRewriteResult:
        """Use LLM to rewrite the query."""
        prompt = QUERY_REWRITE_PROMPT.format(
            conversation_history=history_text,
            session_context=session_context or "No additional context",
            query=query,
        )

        response = await self._llm.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.2,  # Low temperature for consistency
        )

        return self._parse_response(query, response.content, detected_reason)

    def _parse_response(
        self,
        original_query: str,
        response: str,
        fallback_reason: RewriteReason,
    ) -> QueryRewriteResult:
        """Parse LLM response XML to extract rewrite result."""
        # Extract fields from XML
        rewritten_match = re.search(
            r'<rewritten_query>(.*?)</rewritten_query>',
            response,
            re.DOTALL
        )
        reason_match = re.search(
            r'<reason>(.*?)</reason>',
            response,
            re.DOTALL
        )
        confidence_match = re.search(
            r'<confidence>(.*?)</confidence>',
            response,
            re.DOTALL
        )
        entities_match = re.search(
            r'<resolved_entities>(.*?)</resolved_entities>',
            response,
            re.DOTALL
        )

        # Get rewritten query
        rewritten = rewritten_match.group(1).strip() if rewritten_match else None
        if not rewritten:
            logger.warning("No rewritten query in LLM response, using original")
            return QueryRewriteResult.unchanged(original_query)

        # Parse reason
        reason_str = reason_match.group(1).strip().lower() if reason_match else ""
        reason = self._parse_reason(reason_str, fallback_reason)

        # Parse confidence
        confidence = 1.0
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1).strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        # Parse entities
        entities: List[str] = []
        if entities_match:
            entities_str = entities_match.group(1).strip()
            if entities_str and entities_str.lower() not in ("none", "empty", ""):
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]

        # Check if actually rewritten
        if rewritten.lower() == original_query.lower():
            return QueryRewriteResult.unchanged(original_query)

        # Check confidence threshold
        if confidence < self._config.min_confidence:
            logger.debug(
                f"Confidence {confidence} below threshold "
                f"{self._config.min_confidence}, keeping original"
            )
            return QueryRewriteResult.unchanged(original_query)

        return QueryRewriteResult.rewritten(
            original=original_query,
            rewritten=rewritten,
            reason=reason,
            confidence=confidence,
            entities=entities,
        )

    def _parse_reason(
        self,
        reason_str: str,
        fallback: RewriteReason,
    ) -> RewriteReason:
        """Parse reason string to enum."""
        reason_map = {
            "pronoun": RewriteReason.PRONOUN,
            "reference": RewriteReason.REFERENCE,
            "implicit": RewriteReason.IMPLICIT,
            "ellipsis": RewriteReason.ELLIPSIS,
            "no_change": RewriteReason.NO_CHANGE,
        }
        return reason_map.get(reason_str, fallback)

    def _build_cache_key(
        self,
        query: str,
        history: List[Event],
    ) -> str:
        """Build cache key from query and recent history."""
        # Use last 3 events for cache key
        history_summary = "|".join(
            f"{e.type.value}:{e.content[:50]}"
            for e in history[-3:]
        )
        combined = f"{query}|{history_summary}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[QueryRewriteResult]:
        """Get result from cache if valid."""
        if not self._config.cache_enabled:
            return None

        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self._config.cache_ttl_seconds:
            del self._cache[key]
            return None

        return result

    def _add_to_cache(self, key: str, result: QueryRewriteResult) -> None:
        """Add result to cache."""
        if not self._config.cache_enabled:
            return

        self._cache[key] = (result, time.time())

        # Simple cache size management (keep last 100)
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def clear_cache(self) -> None:
        """Clear the rewrite cache."""
        self._cache.clear()

    def needs_rewriting(self, query: str) -> bool:
        """
        Quick check if query likely needs rewriting.

        Useful for deciding whether to call rewrite().
        """
        return self._detect_ambiguity(query) is not None


class QueryRewriterServiceFactory:
    """Factory for creating QueryRewriterService instances."""

    @staticmethod
    def create(
        llm_provider: Optional[ILLMProvider] = None,
        config: Optional[QueryRewriteConfig] = None,
    ) -> Optional[QueryRewriterService]:
        """
        Create a QueryRewriterService if LLM provider is available.

        Returns None if no LLM provider is given.
        """
        if llm_provider is None:
            return None

        return QueryRewriterService(
            llm_provider=llm_provider,
            config=config or QueryRewriteConfig(),
        )
