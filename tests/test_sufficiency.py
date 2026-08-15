"""
Tests for Sufficiency Judging Service.

Tests for sufficiency models, service, and progressive retrieval.
"""
from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.config.base import SufficiencyConfig as ConfigSufficiencyConfig
from ctxforge.core.sufficiency import (
    ProgressiveRetrievalStats,
    SufficiencyResult,
    SufficiencyVerdict,
)
from ctxforge.engine.services.sufficiency_service import (
    SufficiencyConfig,
    SufficiencyService,
    SufficiencyServiceFactory,
)

# =============================================================================
# Test Fixtures
# =============================================================================

class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content


class MockMemoryItem:
    """Mock memory item for testing."""

    def __init__(self, content: str):
        self.content = content


# =============================================================================
# Test SufficiencyVerdict Enum
# =============================================================================

class TestSufficiencyVerdict:
    """Tests for SufficiencyVerdict enum."""

    def test_all_verdicts_have_values(self):
        """Test all verdicts have string values."""
        assert SufficiencyVerdict.ENOUGH.value == "enough"
        assert SufficiencyVerdict.MORE.value == "more"
        assert SufficiencyVerdict.UNCERTAIN.value == "uncertain"

    def test_verdict_count(self):
        """Test all expected verdicts exist."""
        assert len(SufficiencyVerdict) == 3


# =============================================================================
# Test SufficiencyResult Model
# =============================================================================

class TestSufficiencyResult:
    """Tests for SufficiencyResult model."""

    def test_enough_factory(self):
        """Test enough() factory method."""
        result = SufficiencyResult.enough(
            reasoning="Content is comprehensive",
            confidence=0.95,
        )

        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert result.reasoning == "Content is comprehensive"
        assert result.confidence == 0.95
        assert result.is_sufficient is True
        assert result.needs_more is False

    def test_need_more_factory(self):
        """Test need_more() factory method."""
        result = SufficiencyResult.need_more(
            reasoning="Missing user preferences",
            missing_aspects=["preferences", "history"],
            suggested_sources=["memories", "graph"],
        )

        assert result.verdict == SufficiencyVerdict.MORE
        assert result.reasoning == "Missing user preferences"
        assert "preferences" in result.missing_aspects
        assert "memories" in result.suggested_sources
        assert result.is_sufficient is False
        assert result.needs_more is True

    def test_uncertain_factory(self):
        """Test uncertain() factory method."""
        result = SufficiencyResult.uncertain(
            reasoning="Cannot determine",
            confidence=0.4,
        )

        assert result.verdict == SufficiencyVerdict.UNCERTAIN
        assert result.confidence == 0.4
        assert result.is_sufficient is False
        assert result.needs_more is False

    def test_default_values(self):
        """Test default values."""
        result = SufficiencyResult(
            verdict=SufficiencyVerdict.ENOUGH,
            reasoning="Test",
        )

        assert result.confidence == 1.0
        assert result.suggested_sources == []
        assert result.missing_aspects == []
        assert result.iteration == 1

    def test_iteration_tracking(self):
        """Test iteration tracking."""
        result = SufficiencyResult(
            verdict=SufficiencyVerdict.MORE,
            reasoning="Need more",
            iteration=3,
        )

        assert result.iteration == 3


# =============================================================================
# Test ProgressiveRetrievalStats
# =============================================================================

class TestProgressiveRetrievalStats:
    """Tests for ProgressiveRetrievalStats model."""

    def test_default_values(self):
        """Test default values."""
        stats = ProgressiveRetrievalStats()

        assert stats.total_iterations == 0
        assert stats.initial_results == 0
        assert stats.final_results == 0
        assert stats.sources_tried == []
        assert stats.final_verdict is None

    def test_results_added(self):
        """Test results_added calculation."""
        stats = ProgressiveRetrievalStats(
            initial_results=5,
            final_results=15,
        )

        assert stats.results_added == 10

    def test_full_stats(self):
        """Test full stats tracking."""
        stats = ProgressiveRetrievalStats(
            total_iterations=3,
            initial_results=5,
            final_results=20,
            sources_tried=["memories", "graph"],
            final_verdict=SufficiencyVerdict.ENOUGH,
        )

        assert stats.total_iterations == 3
        assert stats.results_added == 15
        assert "memories" in stats.sources_tried


# =============================================================================
# Test SufficiencyConfig
# =============================================================================

class TestSufficiencyConfig:
    """Tests for SufficiencyConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SufficiencyConfig()

        assert config.enabled is True
        assert config.max_iterations == 3
        assert config.min_confidence == 0.7
        assert "memories" in config.fallback_sources

    def test_custom_values(self):
        """Test custom configuration."""
        config = SufficiencyConfig(
            enabled=False,
            max_iterations=5,
            fallback_sources=["custom_source"],
        )

        assert config.enabled is False
        assert config.max_iterations == 5
        assert config.fallback_sources == ["custom_source"]

    def test_pydantic_config(self):
        """Test Pydantic config from base.py."""
        config = ConfigSufficiencyConfig()

        assert config.enabled is False  # Default is False in Pydantic model
        assert config.max_iterations == 3
        assert "memories" in config.fallback_sources


# =============================================================================
# Test SufficiencyService
# =============================================================================

class TestSufficiencyService:
    """Tests for SufficiencyService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = SufficiencyConfig(enabled=True)

    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = SufficiencyService(
            llm_provider=self.mock_llm,
            config=self.config,
        )

        assert service._llm == self.mock_llm
        assert service.config == self.config

    @pytest.mark.asyncio
    async def test_judge_disabled(self):
        """Test judge returns enough when disabled."""
        config = SufficiencyConfig(enabled=False)
        service = SufficiencyService(self.mock_llm, config)

        result = await service.judge(
            query="test query",
            retrieved_content="some content",
        )

        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert "disabled" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_judge_empty_content(self):
        """Test judge returns MORE for empty content."""
        service = SufficiencyService(self.mock_llm, self.config)

        result = await service.judge(
            query="test query",
            retrieved_content="",
        )

        assert result.verdict == SufficiencyVerdict.MORE
        assert "No content" in result.reasoning

    @pytest.mark.asyncio
    async def test_judge_whitespace_content(self):
        """Test judge returns MORE for whitespace-only content."""
        service = SufficiencyService(self.mock_llm, self.config)

        result = await service.judge(
            query="test query",
            retrieved_content="   \n\t  ",
        )

        assert result.verdict == SufficiencyVerdict.MORE

    @pytest.mark.asyncio
    async def test_judge_with_llm_enough(self):
        """Test judge with LLM returning ENOUGH."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <consideration>The content fully addresses the query.</consideration>
              <verdict>ENOUGH</verdict>
              <confidence>0.95</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        result = await service.judge(
            query="What is Python?",
            retrieved_content="Python is a programming language...",
        )

        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_judge_with_llm_more(self):
        """Test judge with LLM returning MORE."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <consideration>Missing specific details about user preferences.</consideration>
              <verdict>MORE</verdict>
              <confidence>0.8</confidence>
              <missing_aspects>user preferences, history</missing_aspects>
              <suggested_sources>memories, graph</suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        result = await service.judge(
            query="What does the user prefer?",
            retrieved_content="Some general info...",
        )

        assert result.verdict == SufficiencyVerdict.MORE
        assert "user preferences" in result.missing_aspects
        assert "memories" in result.suggested_sources

    @pytest.mark.asyncio
    async def test_judge_with_llm_uncertain(self):
        """Test judge with LLM returning UNCERTAIN."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <consideration>Cannot determine relevance.</consideration>
              <verdict>UNCERTAIN</verdict>
              <confidence>0.5</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        result = await service.judge(
            query="Complex query",
            retrieved_content="Ambiguous content",
        )

        assert result.verdict == SufficiencyVerdict.UNCERTAIN
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_judge_llm_error(self):
        """Test judge handles LLM errors gracefully."""
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        service = SufficiencyService(self.mock_llm, self.config)

        result = await service.judge(
            query="test",
            retrieved_content="content",
        )

        assert result.verdict == SufficiencyVerdict.UNCERTAIN
        assert "failed" in result.reasoning.lower()

    def test_is_content_empty(self):
        """Test content emptiness detection."""
        service = SufficiencyService(self.mock_llm, self.config)

        assert service.is_content_empty("") is True
        assert service.is_content_empty("   ") is True
        assert service.is_content_empty("none") is True
        assert service.is_content_empty("empty") is True
        assert service.is_content_empty("no results") is True
        assert service.is_content_empty("actual content") is False

    def test_default_format(self):
        """Test default result formatting."""
        service = SufficiencyService(self.mock_llm, self.config)

        # Test with objects having content attribute
        items = [
            MockMemoryItem("First item"),
            MockMemoryItem("Second item"),
        ]
        formatted = service._default_format(items)

        assert "1. First item" in formatted
        assert "2. Second item" in formatted

    def test_default_format_dict(self):
        """Test default formatting with dicts."""
        service = SufficiencyService(self.mock_llm, self.config)

        items = [
            {"content": "Dict item 1"},
            {"content": "Dict item 2"},
        ]
        formatted = service._default_format(items)

        assert "Dict item 1" in formatted
        assert "Dict item 2" in formatted

    def test_default_format_empty(self):
        """Test default formatting with empty list."""
        service = SufficiencyService(self.mock_llm, self.config)

        formatted = service._default_format([])
        assert formatted == ""


# =============================================================================
# Test Progressive Retrieval
# =============================================================================

class TestProgressiveRetrieval:
    """Tests for progressive retrieval functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = SufficiencyConfig(enabled=True, max_iterations=3)

    @pytest.mark.asyncio
    async def test_progressive_enough_first_iteration(self):
        """Test progressive retrieval when first iteration is enough."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>ENOUGH</verdict>
              <consideration>Sufficient</consideration>
              <confidence>0.9</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        async def retriever(limit: int) -> List[MockMemoryItem]:
            return [MockMemoryItem(f"Item {i}") for i in range(limit)]

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
            initial_limit=5,
        )

        assert len(results) == 5
        assert judgment.verdict == SufficiencyVerdict.ENOUGH
        assert stats.total_iterations == 1
        assert stats.final_verdict == SufficiencyVerdict.ENOUGH

    @pytest.mark.asyncio
    async def test_progressive_needs_more_iterations(self):
        """Test progressive retrieval with multiple iterations."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return MockLLMResponse(
                    """
                    <sufficiency_response>
                      <verdict>MORE</verdict>
                      <consideration>Need more</consideration>
                      <confidence>0.8</confidence>
                      <missing_aspects>details</missing_aspects>
                      <suggested_sources>memories</suggested_sources>
                    </sufficiency_response>
                    """
                )
            return MockLLMResponse(
                """
                <sufficiency_response>
                  <verdict>ENOUGH</verdict>
                  <consideration>Now sufficient</consideration>
                  <confidence>0.9</confidence>
                  <missing_aspects></missing_aspects>
                  <suggested_sources></suggested_sources>
                </sufficiency_response>
                """
            )

        self.mock_llm.generate = mock_generate

        service = SufficiencyService(self.mock_llm, self.config)

        async def retriever(limit: int) -> List[MockMemoryItem]:
            return [MockMemoryItem(f"Item {i}") for i in range(limit)]

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
            initial_limit=5,
            max_limit=20,
        )

        # With initial=5, max=20: iter1=5, iter2=10 (enough on iter2)
        assert stats.total_iterations == 2
        assert judgment.verdict == SufficiencyVerdict.ENOUGH
        assert stats.final_results == 10  # 5 -> 10

    @pytest.mark.asyncio
    async def test_progressive_max_iterations(self):
        """Test progressive retrieval hits max iterations."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>MORE</verdict>
              <consideration>Still need more</consideration>
              <confidence>0.7</confidence>
              <missing_aspects>everything</missing_aspects>
              <suggested_sources>all</suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        async def retriever(limit: int) -> List[MockMemoryItem]:
            return [MockMemoryItem(f"Item {i}") for i in range(limit)]

        # With initial=5, max=20: iter1=5, iter2=10, then 20>=20 breaks
        # So we get 2 iterations before max_limit is reached
        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
            initial_limit=5,
            max_limit=20,
        )

        assert stats.total_iterations == 2  # Hits max_limit after 2 iterations
        assert judgment.verdict == SufficiencyVerdict.UNCERTAIN
        assert "Max" in judgment.reasoning

    @pytest.mark.asyncio
    async def test_progressive_uncertain_stops(self):
        """Test progressive retrieval stops on UNCERTAIN."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>UNCERTAIN</verdict>
              <consideration>Cannot determine</consideration>
              <confidence>0.4</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        async def retriever(limit: int) -> List[MockMemoryItem]:
            return [MockMemoryItem(f"Item {i}") for i in range(limit)]

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
        )

        assert stats.total_iterations == 1  # Stopped after first
        assert judgment.verdict == SufficiencyVerdict.UNCERTAIN

    @pytest.mark.asyncio
    async def test_progressive_custom_formatter(self):
        """Test progressive retrieval with custom formatter."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>ENOUGH</verdict>
              <consideration>Good</consideration>
              <confidence>0.9</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        service = SufficiencyService(self.mock_llm, self.config)

        async def retriever(limit: int) -> List[str]:
            return [f"Result {i}" for i in range(limit)]

        def formatter(items: List[str]) -> str:
            return " | ".join(items)

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
            formatter_func=formatter,
        )

        assert judgment.verdict == SufficiencyVerdict.ENOUGH


# =============================================================================
# Test Response Parsing
# =============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.config = SufficiencyConfig(enabled=True)
        self.service = SufficiencyService(self.mock_llm, self.config)

    def test_parse_complete_response(self):
        """Test parsing a complete XML response."""
        response = """
        <sufficiency_response>
          <consideration>The content addresses all aspects.</consideration>
          <verdict>ENOUGH</verdict>
          <confidence>0.92</confidence>
          <missing_aspects></missing_aspects>
          <suggested_sources></suggested_sources>
        </sufficiency_response>
        """

        result = self.service._parse_response(response)

        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert result.confidence == 0.92
        assert "all aspects" in result.reasoning

    def test_parse_more_response(self):
        """Test parsing a MORE response."""
        response = """
        <sufficiency_response>
          <consideration>Missing key information.</consideration>
          <verdict>MORE</verdict>
          <confidence>0.75</confidence>
          <missing_aspects>user history, preferences</missing_aspects>
          <suggested_sources>memories, graph</suggested_sources>
        </sufficiency_response>
        """

        result = self.service._parse_response(response)

        assert result.verdict == SufficiencyVerdict.MORE
        assert len(result.missing_aspects) == 2
        assert len(result.suggested_sources) == 2

    def test_parse_invalid_verdict_defaults_uncertain(self):
        """Test invalid verdict defaults to UNCERTAIN."""
        response = """
        <sufficiency_response>
          <consideration>Test</consideration>
          <verdict>INVALID</verdict>
          <confidence>0.5</confidence>
          <missing_aspects></missing_aspects>
          <suggested_sources></suggested_sources>
        </sufficiency_response>
        """

        result = self.service._parse_response(response)

        assert result.verdict == SufficiencyVerdict.UNCERTAIN

    def test_parse_missing_fields(self):
        """Test parsing with missing fields."""
        response = """
        <sufficiency_response>
          <verdict>ENOUGH</verdict>
        </sufficiency_response>
        """

        result = self.service._parse_response(response)

        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert result.reasoning == "No reasoning"
        assert result.confidence == 1.0

    def test_parse_invalid_confidence(self):
        """Test parsing handles invalid confidence."""
        response = """
        <sufficiency_response>
          <consideration>Test</consideration>
          <verdict>ENOUGH</verdict>
          <confidence>invalid</confidence>
          <missing_aspects></missing_aspects>
          <suggested_sources></suggested_sources>
        </sufficiency_response>
        """

        result = self.service._parse_response(response)

        assert result.confidence == 1.0  # Default

    def test_parse_case_insensitive_verdict(self):
        """Test verdict parsing is case-insensitive."""
        response = """
        <sufficiency_response>
          <consideration>Test</consideration>
          <verdict>enough</verdict>
          <confidence>0.9</confidence>
          <missing_aspects></missing_aspects>
          <suggested_sources></suggested_sources>
        </sufficiency_response>
        """

        result = self.service._parse_response(response)

        assert result.verdict == SufficiencyVerdict.ENOUGH


# =============================================================================
# Test SufficiencyServiceFactory
# =============================================================================

class TestSufficiencyServiceFactory:
    """Tests for SufficiencyServiceFactory."""

    def test_create_with_provider(self):
        """Test factory creates service with provider."""
        mock_llm = MagicMock()

        service = SufficiencyServiceFactory.create(
            llm_provider=mock_llm,
        )

        assert service is not None
        assert isinstance(service, SufficiencyService)

    def test_create_without_provider(self):
        """Test factory returns None without provider."""
        service = SufficiencyServiceFactory.create(
            llm_provider=None,
        )

        assert service is None

    def test_create_with_config(self):
        """Test factory applies custom config."""
        mock_llm = MagicMock()
        config = SufficiencyConfig(
            max_iterations=5,
        )

        service = SufficiencyServiceFactory.create(
            llm_provider=mock_llm,
            config=config,
        )

        assert service.config.max_iterations == 5
