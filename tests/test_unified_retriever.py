"""
Tests for Unified Cross-Store Retriever.

Tests unified search across multiple knowledge stores.
"""

from unittest.mock import AsyncMock

import pytest

from ctxforge.retrieval.unified_retriever import (
    ResultSource,
    RetrievalQuery,
    RetrievalResult,
    SimpleSearchAdapter,
    UnifiedRetriever,
)


class TestResultSource:
    """Tests for ResultSource enum."""
    
    def test_source_values(self):
        """Test all source values exist."""
        assert ResultSource.EXPERTISE == "expertise"
        assert ResultSource.MEMORY == "memory"
        assert ResultSource.SEMANTIC_MODEL == "semantic_model"
        assert ResultSource.GRAPH == "graph"
        assert ResultSource.EVENTS_INTENT == "events_intent"
        assert ResultSource.EXTERNAL == "external"


class TestRetrievalResult:
    """Tests for RetrievalResult model."""
    
    def test_create_result(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            content="Always validate user input",
            score=0.95,
            source=ResultSource.EXPERTISE,
            source_id="item-123",
            knowledge_type="rule",
            tags=["validation", "security"],
        )
        
        assert result.content == "Always validate user input"
        assert result.score == 0.95
        assert result.source == ResultSource.EXPERTISE
        assert "validation" in result.tags
    
    def test_to_context_string_with_type(self):
        """Test context string with knowledge type."""
        result = RetrievalResult(
            content="Check null values",
            score=0.8,
            source=ResultSource.EXPERTISE,
            knowledge_type="gotcha",
        )
        
        context = result.to_context_string()
        
        assert "⚠️" in context
        assert "[GOTCHA]" in context
        assert "Check null values" in context
    
    def test_to_context_string_without_type(self):
        """Test context string without knowledge type."""
        result = RetrievalResult(
            content="Simple fact",
            score=0.7,
            source=ResultSource.MEMORY,
        )
        
        context = result.to_context_string()
        
        assert context == "Simple fact"


class TestRetrievalQuery:
    """Tests for RetrievalQuery model."""
    
    def test_create_simple_query(self):
        """Test creating a simple query."""
        query = RetrievalQuery(query="How do I validate input?")
        
        assert query.query == "How do I validate input?"
        assert query.max_results == 10
        assert query.sources is None
    
    def test_create_filtered_query(self):
        """Test creating a filtered query."""
        query = RetrievalQuery(
            query="Find user preferences",
            user_id="user-123",
            sources=[ResultSource.MEMORY],
            knowledge_types=["preference"],
            max_results=5,
        )
        
        assert query.user_id == "user-123"
        assert ResultSource.MEMORY in query.sources
        assert query.max_results == 5


class MockStoreAdapter:
    """Mock store adapter for testing."""
    
    def __init__(self, results: list, delay: float = 0):
        self._results = results
        self._delay = delay
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> list:
        if self._delay > 0:
            import asyncio
            await asyncio.sleep(self._delay)
        return self._results[:limit]


class TestUnifiedRetriever:
    """Tests for UnifiedRetriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create a fresh retriever."""
        return UnifiedRetriever()
    
    @pytest.fixture
    def expertise_results(self):
        """Sample expertise results."""
        return [
            RetrievalResult(
                content="Always validate input",
                score=0.9,
                source=ResultSource.EXPERTISE,
                knowledge_type="rule",
            ),
            RetrievalResult(
                content="Use parameterized queries",
                score=0.85,
                source=ResultSource.EXPERTISE,
                knowledge_type="pattern",
            ),
        ]
    
    @pytest.fixture
    def memory_results(self):
        """Sample memory results."""
        return [
            RetrievalResult(
                content="User prefers dark mode",
                score=0.95,
                source=ResultSource.MEMORY,
                knowledge_type="preference",
            ),
            RetrievalResult(
                content="User asked about SQL yesterday",
                score=0.7,
                source=ResultSource.MEMORY,
            ),
        ]
    
    def test_register_store(self, retriever):
        """Test registering a store."""
        adapter = MockStoreAdapter([])
        
        retriever.register_store(
            name="test-store",
            source=ResultSource.EXPERTISE,
            adapter=adapter,
            priority=8,
        )
        
        assert "test-store" in retriever._stores
        assert retriever._stores["test-store"].priority == 8
    
    def test_unregister_store(self, retriever):
        """Test unregistering a store."""
        adapter = MockStoreAdapter([])
        retriever.register_store("test", ResultSource.EXPERTISE, adapter)
        
        result = retriever.unregister_store("test")
        
        assert result is True
        assert "test" not in retriever._stores
    
    def test_unregister_nonexistent(self, retriever):
        """Test unregistering non-existent store."""
        result = retriever.unregister_store("nonexistent")
        assert result is False
    
    def test_enable_store(self, retriever):
        """Test enabling/disabling a store."""
        adapter = MockStoreAdapter([])
        retriever.register_store("test", ResultSource.EXPERTISE, adapter)
        
        retriever.enable_store("test", enabled=False)
        assert retriever._stores["test"].enabled is False
        
        retriever.enable_store("test", enabled=True)
        assert retriever._stores["test"].enabled is True
    
    @pytest.mark.asyncio
    async def test_search_single_store(self, retriever, expertise_results):
        """Test searching a single store."""
        adapter = MockStoreAdapter(expertise_results)
        retriever.register_store("expertise", ResultSource.EXPERTISE, adapter)
        
        results = await retriever.search("validate input")
        
        assert len(results) == 2
        assert results[0].source == ResultSource.EXPERTISE
    
    @pytest.mark.asyncio
    async def test_search_multiple_stores(
        self, retriever, expertise_results, memory_results
    ):
        """Test searching multiple stores."""
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(expertise_results),
        )
        retriever.register_store(
            "memory",
            ResultSource.MEMORY,
            MockStoreAdapter(memory_results),
        )
        
        results = await retriever.search("user")
        
        # Should have results from both stores
        sources = {r.source for r in results}
        assert ResultSource.EXPERTISE in sources
        assert ResultSource.MEMORY in sources
    
    @pytest.mark.asyncio
    async def test_search_with_source_filter(
        self, retriever, expertise_results, memory_results
    ):
        """Test filtering by source."""
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(expertise_results),
        )
        retriever.register_store(
            "memory",
            ResultSource.MEMORY,
            MockStoreAdapter(memory_results),
        )
        
        results = await retriever.search(
            "user",
            sources=[ResultSource.MEMORY],
        )
        
        # Should only have memory results
        assert all(r.source == ResultSource.MEMORY for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_knowledge_type_filter(self, retriever, expertise_results):
        """Test filtering by knowledge type."""
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(expertise_results),
        )
        
        results = await retriever.search(
            "validate",
            knowledge_types=["rule"],
        )
        
        # Should only have rule type
        assert all(r.knowledge_type == "rule" for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_min_score(self, retriever, expertise_results):
        """Test filtering by minimum score."""
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(expertise_results),
        )
        
        results = await retriever.search("validate", min_score=0.88)
        
        # Should only have high-scoring results
        assert all(r.score >= 0.88 for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_score_weights(self, expertise_results, memory_results):
        """Test score weighting by source."""
        retriever = UnifiedRetriever(
            score_weights={ResultSource.EXPERTISE: 1.2, ResultSource.MEMORY: 0.8}
        )
        
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(expertise_results),
        )
        retriever.register_store(
            "memory",
            ResultSource.MEMORY,
            MockStoreAdapter(memory_results),
        )
        
        results = await retriever.search("user")
        
        # Expertise results should have boosted scores
        expertise_result = next(r for r in results if r.source == ResultSource.EXPERTISE)
        assert expertise_result.score > 0.9  # Original 0.9 * 1.2
    
    @pytest.mark.asyncio
    async def test_search_disabled_store(self, retriever, expertise_results):
        """Test that disabled stores are not searched."""
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(expertise_results),
            enabled=False,
        )
        
        results = await retriever.search("validate")
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_no_stores(self, retriever):
        """Test searching with no registered stores."""
        results = await retriever.search("anything")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_store_error_handled(self, retriever):
        """Test that store errors are handled gracefully."""
        # Create an adapter that raises an exception
        failing_adapter = AsyncMock(side_effect=Exception("Store error"))
        
        retriever.register_store(
            "failing",
            ResultSource.EXPERTISE,
            failing_adapter,
        )
        
        # Should not raise
        results = await retriever.search("query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_max_results(self, retriever):
        """Test max_results limiting."""
        many_results = [
            RetrievalResult(content=f"Result {i}", score=0.9 - i*0.1, source=ResultSource.EXPERTISE)
            for i in range(10)
        ]
        
        retriever.register_store(
            "expertise",
            ResultSource.EXPERTISE,
            MockStoreAdapter(many_results),
        )
        
        results = await retriever.search("query", max_results=3)
        
        assert len(results) == 3
    
    def test_format_results_empty(self, retriever):
        """Test formatting empty results."""
        formatted = retriever.format_results([])
        assert "No relevant knowledge found" in formatted
    
    def test_format_results(self, retriever, expertise_results):
        """Test formatting results."""
        formatted = retriever.format_results(expertise_results)
        
        assert "Retrieved Knowledge" in formatted
        assert "Always validate input" in formatted
        assert "[expertise]" in formatted
    
    def test_format_results_with_score(self, retriever, expertise_results):
        """Test formatting with scores."""
        formatted = retriever.format_results(
            expertise_results,
            include_score=True,
        )
        
        assert "score:" in formatted


class TestMergeStrategies:
    """Tests for different merge strategies."""
    
    @pytest.fixture
    def expertise_results(self):
        return [
            RetrievalResult(content="E1", score=0.9, source=ResultSource.EXPERTISE),
            RetrievalResult(content="E2", score=0.8, source=ResultSource.EXPERTISE),
        ]
    
    @pytest.fixture
    def memory_results(self):
        return [
            RetrievalResult(content="M1", score=0.95, source=ResultSource.MEMORY),
            RetrievalResult(content="M2", score=0.75, source=ResultSource.MEMORY),
        ]
    
    @pytest.mark.asyncio
    async def test_score_only_merge(self, expertise_results, memory_results):
        """Test score-only merge strategy."""
        retriever = UnifiedRetriever(merge_strategy="score_only")
        
        retriever.register_store("e", ResultSource.EXPERTISE, MockStoreAdapter(expertise_results))
        retriever.register_store("m", ResultSource.MEMORY, MockStoreAdapter(memory_results))
        
        results = await retriever.search("query")
        
        # Should be sorted purely by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_interleave_merge(self, expertise_results, memory_results):
        """Test interleave merge strategy."""
        retriever = UnifiedRetriever(merge_strategy="interleave")
        
        retriever.register_store("e", ResultSource.EXPERTISE, MockStoreAdapter(expertise_results))
        retriever.register_store("m", ResultSource.MEMORY, MockStoreAdapter(memory_results))
        
        results = await retriever.search("query")
        
        # Should alternate between sources
        # First two should be different sources (highest from each)
        if len(results) >= 2:
            assert results[0].source != results[1].source or len(results) == 2


class TestSimpleSearchAdapter:
    """Tests for SimpleSearchAdapter."""
    
    @pytest.mark.asyncio
    async def test_adapter(self):
        """Test simple search adapter."""
        # Mock search function
        async def mock_search(query: str, limit: int = 10, **kwargs):
            return [
                {"content": f"Result for {query}", "score": 0.9},
            ]
        
        # Mapper function
        def mapper(item):
            return RetrievalResult(
                content=item["content"],
                score=item["score"],
                source=ResultSource.EXTERNAL,
            )
        
        adapter = SimpleSearchAdapter(
            search_fn=mock_search,
            result_mapper=mapper,
        )
        
        results = await adapter.search("test query")
        
        assert len(results) == 1
        assert "test query" in results[0].content
        assert results[0].source == ResultSource.EXTERNAL
