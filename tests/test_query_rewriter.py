"""
Tests for Query Rewriter Service.

Tests for query rewriting models, heuristics, and service.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.config.base import QueryRewriteConfig
from ctxforge.core.events import Event, EventType
from ctxforge.core.query import QueryRewriteResult, RewriteReason
from ctxforge.engine.services.query_rewriter_service import (
    ELLIPSIS_PATTERN,
    IMPLICIT_PATTERN,
    PRONOUN_PATTERN,
    REFERENCE_PATTERN,
    QueryRewriterService,
    QueryRewriterServiceFactory,
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


def create_conversation_history() -> List[Event]:
    """Create a sample conversation history."""
    return [
        create_event(EventType.USER, "I'm planning a trip to Japan next month"),
        create_event(
            EventType.AGENT,
            "That sounds exciting! Japan has beautiful cherry blossoms in spring."
        ),
        create_event(EventType.USER, "My friend John recommended visiting Kyoto"),
        create_event(
            EventType.AGENT,
            "Kyoto is wonderful for temples and traditional culture."
        ),
    ]


class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content


# =============================================================================
# Test QueryRewriteResult Model
# =============================================================================

class TestQueryRewriteResult:
    """Tests for QueryRewriteResult model."""

    def test_unchanged_creates_no_change_result(self):
        """Test unchanged() factory method."""
        result = QueryRewriteResult.unchanged("What is the weather?")

        assert result.original_query == "What is the weather?"
        assert result.rewritten_query == "What is the weather?"
        assert result.was_rewritten is False
        assert result.reason == RewriteReason.NO_CHANGE
        assert result.confidence == 1.0
        assert result.resolved_entities == []

    def test_rewritten_creates_changed_result(self):
        """Test rewritten() factory method."""
        result = QueryRewriteResult.rewritten(
            original="What about their food?",
            rewritten="What food does John prefer for the Japan trip?",
            reason=RewriteReason.PRONOUN,
            confidence=0.9,
            entities=["John", "Japan trip"],
        )

        assert result.original_query == "What about their food?"
        assert result.rewritten_query == "What food does John prefer for the Japan trip?"
        assert result.was_rewritten is True
        assert result.reason == RewriteReason.PRONOUN
        assert result.confidence == 0.9
        assert result.resolved_entities == ["John", "Japan trip"]

    def test_rewritten_default_values(self):
        """Test rewritten() with default values."""
        result = QueryRewriteResult.rewritten(
            original="Tell me more",
            rewritten="Tell me more about Kyoto temples",
            reason=RewriteReason.IMPLICIT,
        )

        assert result.confidence == 1.0
        assert result.resolved_entities == []

    def test_confidence_validation(self):
        """Test confidence is clamped to valid range."""
        result = QueryRewriteResult(
            original_query="test",
            rewritten_query="test",
            was_rewritten=False,
            reason=RewriteReason.NO_CHANGE,
            confidence=0.5,
        )
        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Test RewriteReason Enum
# =============================================================================

class TestRewriteReason:
    """Tests for RewriteReason enum."""

    def test_all_reasons_have_values(self):
        """Test all reasons have string values."""
        assert RewriteReason.PRONOUN.value == "pronoun"
        assert RewriteReason.REFERENCE.value == "reference"
        assert RewriteReason.IMPLICIT.value == "implicit"
        assert RewriteReason.ELLIPSIS.value == "ellipsis"
        assert RewriteReason.NO_CHANGE.value == "no_change"

    def test_reason_count(self):
        """Test all expected reasons exist."""
        assert len(RewriteReason) == 5


# =============================================================================
# Test Heuristic Patterns
# =============================================================================

class TestHeuristicPatterns:
    """Tests for ambiguity detection patterns."""

    def test_pronoun_pattern_matches(self):
        """Test pronoun pattern matches various pronouns."""
        matches = [
            "What did they say?",
            "Tell me about it",
            "Their preferences are...",
            "Give him the details",
            "She mentioned it",
            "We discussed this",
            "Our project needs...",
        ]
        for text in matches:
            assert PRONOUN_PATTERN.search(text), f"Should match: {text}"

    def test_pronoun_pattern_no_false_positives(self):
        """Test pronoun pattern doesn't match common words."""
        non_matches = [
            "What is Python?",
            "How do I code?",
            "The weather is nice",
        ]
        for text in non_matches:
            # These should not have pronoun matches
            match = PRONOUN_PATTERN.search(text)
            if match:
                # Only fail if it's a false positive
                assert match.group() in text.lower()

    def test_reference_pattern_matches(self):
        """Test reference pattern matches various references."""
        matches = [
            "Tell me about that",
            "What about those options?",
            "This needs more work",
            "The same thing happened",
            "As mentioned above",
            "The previous result",
        ]
        for text in matches:
            assert REFERENCE_PATTERN.search(text), f"Should match: {text}"

    def test_implicit_pattern_matches(self):
        """Test implicit pattern matches at start of query."""
        matches = [
            "And the other option?",
            "Also show me the code",
            "What about Python?",
            "How about tomorrow?",
            "Same for JavaScript",
            "Likewise for tests",
        ]
        for text in matches:
            assert IMPLICIT_PATTERN.match(text), f"Should match: {text}"

    def test_implicit_pattern_position(self):
        """Test implicit pattern only matches at start."""
        # Should not match in middle
        assert not IMPLICIT_PATTERN.match("Show me and the other one")
        assert not IMPLICIT_PATTERN.match("Tell me also about this")

    def test_ellipsis_pattern_matches(self):
        """Test ellipsis pattern matches short responses."""
        matches = [
            "Yes, please",
            "No thanks",
            "Sure, go ahead",
            "Ok",
            "Right, that's it",
        ]
        for text in matches:
            assert ELLIPSIS_PATTERN.match(text), f"Should match: {text}"


# =============================================================================
# Test QueryRewriteConfig
# =============================================================================

class TestQueryRewriteConfig:
    """Tests for QueryRewriteConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QueryRewriteConfig()

        assert config.enabled is False
        assert config.max_history_turns == 10
        assert config.min_confidence == 0.7
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300
        assert config.check_pronouns is True
        assert config.check_references is True
        assert config.check_implicit is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = QueryRewriteConfig(
            enabled=True,
            max_history_turns=5,
            min_confidence=0.8,
            cache_enabled=False,
        )

        assert config.enabled is True
        assert config.max_history_turns == 5
        assert config.min_confidence == 0.8
        assert config.cache_enabled is False


# =============================================================================
# Test QueryRewriterService
# =============================================================================

class TestQueryRewriterService:
    """Tests for QueryRewriterService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = QueryRewriteConfig(enabled=True)

    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = QueryRewriterService(
            llm_provider=self.mock_llm,
            config=self.config,
        )

        assert service._llm == self.mock_llm
        assert service.config == self.config

    def test_needs_rewriting_with_pronoun(self):
        """Test needs_rewriting detects pronouns."""
        service = QueryRewriterService(self.mock_llm, self.config)

        assert service.needs_rewriting("What did they say?") is True
        assert service.needs_rewriting("Tell me about it") is True

    def test_needs_rewriting_with_reference(self):
        """Test needs_rewriting detects references."""
        service = QueryRewriterService(self.mock_llm, self.config)

        assert service.needs_rewriting("What about that?") is True
        assert service.needs_rewriting("The same thing please") is True

    def test_needs_rewriting_with_implicit(self):
        """Test needs_rewriting detects implicit context."""
        service = QueryRewriterService(self.mock_llm, self.config)

        assert service.needs_rewriting("And the other one?") is True
        assert service.needs_rewriting("Also Python?") is True

    def test_needs_rewriting_clear_query(self):
        """Test needs_rewriting returns false for clear queries."""
        service = QueryRewriterService(self.mock_llm, self.config)

        # Clear queries don't need rewriting
        assert service.needs_rewriting(
            "What are the best restaurants in Tokyo?"
        ) is False
        assert service.needs_rewriting(
            "How do I configure pytest for my project?"
        ) is False

    def test_needs_rewriting_short_query(self):
        """Test needs_rewriting flags short queries."""
        service = QueryRewriterService(self.mock_llm, self.config)

        # Very short queries likely need context
        assert service.needs_rewriting("Why?") is True
        assert service.needs_rewriting("Show me") is True

    @pytest.mark.asyncio
    async def test_rewrite_disabled(self):
        """Test rewrite returns unchanged when disabled."""
        config = QueryRewriteConfig(enabled=False)
        service = QueryRewriterService(self.mock_llm, config)

        result = await service.rewrite(
            query="What about them?",
            conversation_history=[],
        )

        assert result.was_rewritten is False
        assert result.rewritten_query == "What about them?"

    @pytest.mark.asyncio
    async def test_rewrite_clear_query(self):
        """Test rewrite skips clear queries."""
        service = QueryRewriterService(self.mock_llm, self.config)

        result = await service.rewrite(
            query="What are the best Python testing frameworks?",
            conversation_history=[],
        )

        assert result.was_rewritten is False
        self.mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_rewrite_no_history(self):
        """Test rewrite returns unchanged with no history."""
        service = QueryRewriterService(self.mock_llm, self.config)

        result = await service.rewrite(
            query="What about them?",
            conversation_history=[],
        )

        # Can't resolve without history
        assert result.was_rewritten is False

    @pytest.mark.asyncio
    async def test_rewrite_with_llm(self):
        """Test rewrite calls LLM and parses response."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <analysis>Query has pronoun 'their' referring to John.</analysis>
              <rewritten_query>What food does John prefer for the Japan trip?</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.9</confidence>
              <resolved_entities>John, Japan trip</resolved_entities>
            </rewrite_response>
            """
        ))

        service = QueryRewriterService(self.mock_llm, self.config)
        history = create_conversation_history()

        result = await service.rewrite(
            query="What about their food preferences?",
            conversation_history=history,
        )

        assert result.was_rewritten is True
        assert "John" in result.rewritten_query
        assert result.reason == RewriteReason.PRONOUN
        assert result.confidence == 0.9
        assert "John" in result.resolved_entities

    @pytest.mark.asyncio
    async def test_rewrite_caching(self):
        """Test rewrite caches results."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>Rewritten query</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.9</confidence>
              <resolved_entities></resolved_entities>
            </rewrite_response>
            """
        ))

        config = QueryRewriteConfig(enabled=True, cache_enabled=True)
        service = QueryRewriterService(self.mock_llm, config)
        history = create_conversation_history()

        # First call
        await service.rewrite("What about them?", history)
        # Second call (should use cache)
        await service.rewrite("What about them?", history)

        # LLM should only be called once
        assert self.mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_rewrite_cache_disabled(self):
        """Test caching can be disabled."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>Rewritten</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.9</confidence>
              <resolved_entities></resolved_entities>
            </rewrite_response>
            """
        ))

        config = QueryRewriteConfig(enabled=True, cache_enabled=False)
        service = QueryRewriterService(self.mock_llm, config)
        history = create_conversation_history()

        await service.rewrite("What about them?", history)
        await service.rewrite("What about them?", history)

        # LLM should be called twice
        assert self.mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_rewrite_confidence_threshold(self):
        """Test low confidence results are rejected."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <rewritten_query>Some rewritten query</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.3</confidence>
              <resolved_entities></resolved_entities>
            </rewrite_response>
            """
        ))

        config = QueryRewriteConfig(enabled=True, min_confidence=0.7)
        service = QueryRewriterService(self.mock_llm, config)
        history = create_conversation_history()

        result = await service.rewrite("What about them?", history)

        # Low confidence should return unchanged
        assert result.was_rewritten is False

    @pytest.mark.asyncio
    async def test_rewrite_llm_error_fallback(self):
        """Test LLM error returns unchanged query."""
        self.mock_llm.generate = AsyncMock(
            side_effect=Exception("LLM error")
        )

        service = QueryRewriterService(self.mock_llm, self.config)
        history = create_conversation_history()

        result = await service.rewrite("What about them?", history)

        assert result.was_rewritten is False
        assert result.rewritten_query == "What about them?"

    def test_format_history(self):
        """Test conversation history formatting."""
        service = QueryRewriterService(self.mock_llm, self.config)
        history = create_conversation_history()

        formatted = service._format_history(history)

        assert "User:" in formatted
        assert "Assistant:" in formatted
        assert "Japan" in formatted
        assert "Kyoto" in formatted

    def test_format_history_truncation(self):
        """Test history is truncated to max turns."""
        config = QueryRewriteConfig(enabled=True, max_history_turns=2)
        service = QueryRewriterService(self.mock_llm, config)
        history = create_conversation_history()

        formatted = service._format_history(history, max_turns=2)

        # Should only have 2 most recent events
        lines = [line for line in formatted.split("\n") if line.strip()]
        assert len(lines) == 2

    def test_clear_cache(self):
        """Test cache clearing."""
        service = QueryRewriterService(self.mock_llm, self.config)
        service._cache["test"] = (QueryRewriteResult.unchanged("test"), 0)

        service.clear_cache()

        assert len(service._cache) == 0

    def test_detect_ambiguity_pronoun(self):
        """Test ambiguity detection for pronouns."""
        service = QueryRewriterService(self.mock_llm, self.config)

        reason = service._detect_ambiguity("What did they say?")
        assert reason == RewriteReason.PRONOUN

    def test_detect_ambiguity_reference(self):
        """Test ambiguity detection for references."""
        service = QueryRewriterService(self.mock_llm, self.config)

        reason = service._detect_ambiguity("Tell me more about that")
        assert reason == RewriteReason.REFERENCE

    def test_detect_ambiguity_implicit(self):
        """Test ambiguity detection for implicit context."""
        service = QueryRewriterService(self.mock_llm, self.config)

        reason = service._detect_ambiguity("And Python?")
        assert reason == RewriteReason.IMPLICIT

    def test_detect_ambiguity_none(self):
        """Test ambiguity detection returns None for clear queries."""
        service = QueryRewriterService(self.mock_llm, self.config)

        reason = service._detect_ambiguity(
            "What are the best restaurants in Tokyo for sushi?"
        )
        assert reason is None


# =============================================================================
# Test QueryRewriterServiceFactory
# =============================================================================

class TestQueryRewriterServiceFactory:
    """Tests for QueryRewriterServiceFactory."""

    def test_create_with_provider(self):
        """Test factory creates service with provider."""
        mock_llm = MagicMock()

        service = QueryRewriterServiceFactory.create(
            llm_provider=mock_llm,
        )

        assert service is not None
        assert isinstance(service, QueryRewriterService)

    def test_create_without_provider(self):
        """Test factory returns None without provider."""
        service = QueryRewriterServiceFactory.create(
            llm_provider=None,
        )

        assert service is None

    def test_create_with_config(self):
        """Test factory applies custom config."""
        mock_llm = MagicMock()
        config = QueryRewriteConfig(
            enabled=True,
            max_history_turns=5,
        )

        service = QueryRewriterServiceFactory.create(
            llm_provider=mock_llm,
            config=config,
        )

        assert service.config.max_history_turns == 5


# =============================================================================
# Test Response Parsing
# =============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = QueryRewriteConfig(enabled=True)
        self.service = QueryRewriterService(self.mock_llm, self.config)

    def test_parse_complete_response(self):
        """Test parsing a complete XML response."""
        response = """
        <rewrite_response>
          <analysis>Found pronoun 'their' referring to John.</analysis>
          <rewritten_query>What are John's preferences?</rewritten_query>
          <reason>pronoun</reason>
          <confidence>0.95</confidence>
          <resolved_entities>John</resolved_entities>
        </rewrite_response>
        """

        result = self.service._parse_response(
            "What are their preferences?",
            response,
            RewriteReason.PRONOUN,
        )

        assert result.was_rewritten is True
        assert result.rewritten_query == "What are John's preferences?"
        assert result.reason == RewriteReason.PRONOUN
        assert result.confidence == 0.95
        assert "John" in result.resolved_entities

    def test_parse_no_change_response(self):
        """Test parsing response indicating no change."""
        response = """
        <rewrite_response>
          <rewritten_query>What are their preferences?</rewritten_query>
          <reason>no_change</reason>
          <confidence>0.8</confidence>
          <resolved_entities></resolved_entities>
        </rewrite_response>
        """

        result = self.service._parse_response(
            "What are their preferences?",
            response,
            RewriteReason.PRONOUN,
        )

        # Same query means no change
        assert result.was_rewritten is False

    def test_parse_missing_rewritten_query(self):
        """Test parsing handles missing rewritten_query."""
        response = """
        <rewrite_response>
          <reason>pronoun</reason>
        </rewrite_response>
        """

        result = self.service._parse_response(
            "Original query",
            response,
            RewriteReason.PRONOUN,
        )

        # Should return unchanged
        assert result.was_rewritten is False

    def test_parse_invalid_confidence(self):
        """Test parsing handles invalid confidence."""
        response = """
        <rewrite_response>
          <rewritten_query>New query</rewritten_query>
          <reason>pronoun</reason>
          <confidence>invalid</confidence>
          <resolved_entities></resolved_entities>
        </rewrite_response>
        """

        result = self.service._parse_response(
            "Original",
            response,
            RewriteReason.PRONOUN,
        )

        # Should use default confidence
        assert result.confidence == 1.0

    def test_parse_multiple_entities(self):
        """Test parsing multiple resolved entities."""
        response = """
        <rewrite_response>
          <rewritten_query>About John and Tokyo trip</rewritten_query>
          <reason>pronoun</reason>
          <confidence>0.9</confidence>
          <resolved_entities>John, Tokyo, trip planning</resolved_entities>
        </rewrite_response>
        """

        result = self.service._parse_response(
            "About them",
            response,
            RewriteReason.PRONOUN,
        )

        assert len(result.resolved_entities) == 3
        assert "John" in result.resolved_entities
        assert "Tokyo" in result.resolved_entities

    def test_parse_reason_mapping(self):
        """Test reason string to enum mapping."""
        reasons = [
            ("pronoun", RewriteReason.PRONOUN),
            ("reference", RewriteReason.REFERENCE),
            ("implicit", RewriteReason.IMPLICIT),
            ("ellipsis", RewriteReason.ELLIPSIS),
            ("no_change", RewriteReason.NO_CHANGE),
        ]

        for reason_str, expected in reasons:
            result = self.service._parse_reason(reason_str, RewriteReason.PRONOUN)
            assert result == expected

    def test_parse_unknown_reason_uses_fallback(self):
        """Test unknown reason uses fallback."""
        result = self.service._parse_reason("unknown", RewriteReason.IMPLICIT)
        assert result == RewriteReason.IMPLICIT


# =============================================================================
# Integration Tests
# =============================================================================

class TestQueryRewriterIntegration:
    """Integration tests for query rewriting."""

    @pytest.mark.asyncio
    async def test_full_rewrite_flow(self):
        """Test complete rewrite flow with mock LLM."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <rewrite_response>
              <analysis>Resolved 'their' to John's.</analysis>
              <rewritten_query>What food does John prefer?</rewritten_query>
              <reason>pronoun</reason>
              <confidence>0.92</confidence>
              <resolved_entities>John</resolved_entities>
            </rewrite_response>
            """
        ))

        config = QueryRewriteConfig(enabled=True)
        service = QueryRewriterService(mock_llm, config)

        history = [
            create_event(EventType.USER, "Tell me about John"),
            create_event(EventType.AGENT, "John is a software developer."),
        ]

        result = await service.rewrite(
            query="What about their food preferences?",
            conversation_history=history,
            session_context="Discussing John's profile",
        )

        assert result.was_rewritten is True
        assert "John" in result.rewritten_query
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_multiple_rewrites_different_queries(self):
        """Test multiple different queries are rewritten independently."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MockLLMResponse(
                f"""
                <rewrite_response>
                  <rewritten_query>Rewritten query {call_count}</rewritten_query>
                  <reason>pronoun</reason>
                  <confidence>0.9</confidence>
                  <resolved_entities></resolved_entities>
                </rewrite_response>
                """
            )

        mock_llm = MagicMock()
        mock_llm.generate = mock_generate

        config = QueryRewriteConfig(enabled=True, cache_enabled=True)
        service = QueryRewriterService(mock_llm, config)
        history = create_conversation_history()

        # Different queries should each call LLM
        result1 = await service.rewrite("What about them?", history)
        result2 = await service.rewrite("And their other stuff?", history)

        assert result1.rewritten_query != result2.rewritten_query
        assert call_count == 2
