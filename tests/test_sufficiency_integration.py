"""
Integration Tests for Sufficiency Judging.

Tests for sufficiency integration with MemoryService and other components.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.config.base import SufficiencyConfig as PydanticSufficiencyConfig
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.sufficiency import SufficiencyVerdict
from ctxforge.engine.services.memory_service import MemoryService
from ctxforge.engine.services.sufficiency_service import (
    SufficiencyConfig,
    SufficiencyService,
)

# =============================================================================
# Test Fixtures
# =============================================================================

class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content


def create_memory(content: str, memory_id: str = None) -> MemoryItem:
    """Create a test memory item."""
    return MemoryItem(
        memory_id=memory_id or f"mem-{len(content)}",
        user_id="test-user",
        content=content,
        type=MemoryType.SEMANTIC,
        created_at=datetime.now(timezone.utc),
    )


class MockMemoryStore:
    """Mock memory store for testing."""

    def __init__(self, memories: List[MemoryItem] = None):
        self._memories = memories or []

    async def search(self, query) -> List[MemoryItem]:
        """Return memories up to query limit."""
        return self._memories[:query.limit]

    async def add(self, memory: MemoryItem) -> str:
        self._memories.append(memory)
        return memory.memory_id

    async def get(self, memory_id: str) -> MemoryItem:
        for m in self._memories:
            if m.memory_id == memory_id:
                return m
        return None

    async def update(self, memory: MemoryItem) -> bool:
        return True

    async def delete(self, memory_id: str) -> bool:
        return True


# =============================================================================
# Test MemoryService with Sufficiency
# =============================================================================

class TestMemoryServiceSufficiencyIntegration:
    """Tests for MemoryService with sufficiency checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.sufficiency_config = SufficiencyConfig(enabled=True, max_iterations=3)
        self.sufficiency_service = SufficiencyService(
            self.mock_llm, self.sufficiency_config
        )

        # Create test memories
        self.test_memories = [
            create_memory(f"Memory content {i}", f"mem-{i}")
            for i in range(10)
        ]
        self.mock_store = MockMemoryStore(self.test_memories)

    @pytest.mark.asyncio
    async def test_search_with_sufficiency_enough(self):
        """Test search_with_sufficiency when content is sufficient."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <consideration>Content addresses the query fully.</consideration>
              <verdict>ENOUGH</verdict>
              <confidence>0.95</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        memory_service = MemoryService(
            memory_store=self.mock_store,
            sufficiency_service=self.sufficiency_service,
        )

        memories, result, stats = await memory_service.search_with_sufficiency(
            user_id="test-user",
            query="What do I know?",
            initial_limit=5,
        )

        assert len(memories) == 5
        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert stats.total_iterations == 1

    @pytest.mark.asyncio
    async def test_search_with_sufficiency_needs_more(self):
        """Test search_with_sufficiency when more content is needed."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockLLMResponse(
                    """
                    <sufficiency_response>
                      <verdict>MORE</verdict>
                      <consideration>Need more details</consideration>
                      <confidence>0.7</confidence>
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

        memory_service = MemoryService(
            memory_store=self.mock_store,
            sufficiency_service=self.sufficiency_service,
        )

        memories, result, stats = await memory_service.search_with_sufficiency(
            user_id="test-user",
            query="What do I know?",
            initial_limit=3,
            max_limit=10,
        )

        assert len(memories) == 6  # 3 -> 6
        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert stats.total_iterations == 2

    @pytest.mark.asyncio
    async def test_search_with_sufficiency_no_service(self):
        """Test search_with_sufficiency falls back without service."""
        memory_service = MemoryService(
            memory_store=self.mock_store,
            sufficiency_service=None,  # No sufficiency service
        )

        memories, result, stats = await memory_service.search_with_sufficiency(
            user_id="test-user",
            query="What do I know?",
            initial_limit=5,
        )

        assert len(memories) == 5
        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert "not configured" in result.reasoning.lower()
        assert stats.total_iterations == 1

    @pytest.mark.asyncio
    async def test_search_with_sufficiency_max_limit(self):
        """Test search_with_sufficiency respects max_limit."""
        self.mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>MORE</verdict>
              <consideration>Always need more</consideration>
              <confidence>0.8</confidence>
              <missing_aspects>everything</missing_aspects>
              <suggested_sources>all</suggested_sources>
            </sufficiency_response>
            """
        ))

        memory_service = MemoryService(
            memory_store=self.mock_store,
            sufficiency_service=self.sufficiency_service,
        )

        memories, result, stats = await memory_service.search_with_sufficiency(
            user_id="test-user",
            query="What do I know?",
            initial_limit=3,
            max_limit=12,
        )

        # With initial=3, max=12: iter1=3, iter2=6 (then 12 >= max_limit breaks)
        # Always returns MORE, so hits max iterations (3)
        assert len(memories) == 6  # 3 -> 6, then max hit
        assert stats.total_iterations == 2


# =============================================================================
# Test Configuration
# =============================================================================

class TestSufficiencyConfiguration:
    """Tests for sufficiency configuration."""

    def test_pydantic_config_defaults(self):
        """Test Pydantic config has correct defaults."""
        config = PydanticSufficiencyConfig()

        assert config.enabled is False
        assert config.max_iterations == 3
        assert config.min_confidence == 0.7
        assert "memories" in config.fallback_sources

    def test_pydantic_config_custom(self):
        """Test Pydantic config accepts custom values."""
        config = PydanticSufficiencyConfig(
            enabled=True,
            max_iterations=5,
            initial_limit=10,
            max_limit=50,
        )

        assert config.enabled is True
        assert config.max_iterations == 5
        assert config.initial_limit == 10
        assert config.max_limit == 50


# =============================================================================
# Test Progressive Retrieval Pattern
# =============================================================================

class TestProgressiveRetrievalPattern:
    """Tests for the progressive retrieval pattern."""

    @pytest.mark.asyncio
    async def test_custom_retriever_integration(self):
        """Test progressive retrieval with custom retriever."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
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

        config = SufficiencyConfig(enabled=True)
        service = SufficiencyService(mock_llm, config)

        # Custom retriever that returns strings
        async def custom_retriever(limit: int) -> List[str]:
            return [f"Result {i}" for i in range(limit)]

        def custom_formatter(items: List[str]) -> str:
            return " | ".join(items)

        results, judgment, stats = await service.progressive_retrieve(
            query="test query",
            retriever_func=custom_retriever,
            formatter_func=custom_formatter,
            initial_limit=5,
        )

        assert len(results) == 5
        assert judgment.verdict == SufficiencyVerdict.ENOUGH

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics are tracked correctly across iterations."""
        mock_llm = MagicMock()
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
                      <confidence>0.7</confidence>
                      <missing_aspects>details</missing_aspects>
                      <suggested_sources>source1, source2</suggested_sources>
                    </sufficiency_response>
                    """
                )
            return MockLLMResponse(
                """
                <sufficiency_response>
                  <verdict>ENOUGH</verdict>
                  <consideration>Good</consideration>
                  <confidence>0.9</confidence>
                  <missing_aspects></missing_aspects>
                  <suggested_sources></suggested_sources>
                </sufficiency_response>
                """
            )

        mock_llm.generate = mock_generate

        config = SufficiencyConfig(enabled=True, max_iterations=5)
        service = SufficiencyService(mock_llm, config)

        async def retriever(limit: int) -> List[str]:
            return [f"Item {i}" for i in range(limit)]

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
            initial_limit=5,
            max_limit=20,
        )

        assert stats.total_iterations == 2
        assert stats.initial_results == 5
        assert stats.final_results == 10
        assert stats.results_added == 5
        assert "source1" in stats.sources_tried
        assert stats.final_verdict == SufficiencyVerdict.ENOUGH


# =============================================================================
# Test Error Handling
# =============================================================================

class TestSufficiencyErrorHandling:
    """Tests for error handling in sufficiency integration."""

    @pytest.mark.asyncio
    async def test_llm_error_during_progressive_retrieval(self):
        """Test graceful handling of LLM errors."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM unavailable"))

        config = SufficiencyConfig(enabled=True)
        service = SufficiencyService(mock_llm, config)

        async def retriever(limit: int) -> List[str]:
            return [f"Item {i}" for i in range(limit)]

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=retriever,
            initial_limit=5,
        )

        # Should return results with UNCERTAIN verdict
        assert len(results) == 5
        assert judgment.verdict == SufficiencyVerdict.UNCERTAIN

    @pytest.mark.asyncio
    async def test_empty_retrieval_results(self):
        """Test handling of empty retrieval results."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>MORE</verdict>
              <consideration>No content</consideration>
              <confidence>0.8</confidence>
              <missing_aspects>everything</missing_aspects>
              <suggested_sources>memories</suggested_sources>
            </sufficiency_response>
            """
        ))

        config = SufficiencyConfig(enabled=True, max_iterations=2)
        service = SufficiencyService(mock_llm, config)

        # Retriever that always returns empty
        async def empty_retriever(limit: int) -> List[str]:
            return []

        results, judgment, stats = await service.progressive_retrieve(
            query="test",
            retriever_func=empty_retriever,
        )

        assert len(results) == 0
        # With empty content, the heuristic returns MORE immediately
        # But after max iterations, it returns UNCERTAIN


# =============================================================================
# Test Query Rewriting + Sufficiency Combined
# =============================================================================

class TestQueryRewritingSufficiencyCombined:
    """Tests for query rewriting and sufficiency working together."""

    @pytest.mark.asyncio
    async def test_rewritten_query_with_sufficiency(self):
        """Test that a rewritten query can be used with sufficiency."""
        # This test verifies the two features can work together
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=MockLLMResponse(
            """
            <sufficiency_response>
              <verdict>ENOUGH</verdict>
              <consideration>Query was rewritten and results are sufficient</consideration>
              <confidence>0.9</confidence>
              <missing_aspects></missing_aspects>
              <suggested_sources></suggested_sources>
            </sufficiency_response>
            """
        ))

        config = SufficiencyConfig(enabled=True)
        service = SufficiencyService(mock_llm, config)

        # Simulate rewritten query (original was "What about them?")
        rewritten_query = "What are John's food preferences?"

        result = await service.judge(
            query=rewritten_query,  # Use rewritten query
            retrieved_content="John likes Italian food and sushi.",
        )

        assert result.verdict == SufficiencyVerdict.ENOUGH
        assert result.confidence >= 0.9
