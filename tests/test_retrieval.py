"""
Tests for retrieval implementations.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.protocols.retriever import RetrievalConfig, RetrievalResult
from ctxforge.retrieval.rerankers import (
    DiversityReranker,
    RecencyReranker,
    ScoreThresholdReranker,
)
from ctxforge.retrieval.retrievers import (
    HybridRetriever,
    SemanticRetriever,
    SimpleRetriever,
    TemporalRetriever,
)
from ctxforge.retrieval.retrievers.hybrid import keyword_match_score
from ctxforge.retrieval.retrievers.temporal import compute_recency_weight
from ctxforge.storage.memory.memory import InMemoryMemoryStore

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def memory_store():
    """Create an in-memory store with test data."""
    return InMemoryMemoryStore()


@pytest.fixture
async def populated_store(memory_store):
    """Memory store with test memories."""
    now = datetime.now(timezone.utc)
    memories = [
        MemoryItem(
            memory_id="mem_1",
            user_id="user_1",
            content="User is vegetarian and prefers spicy food",
            type=MemoryType.SEMANTIC,
            confidence_score=0.9,
            embedding=[1.0, 0.0, 0.0],  # Mock embedding
            tags=["food", "diet"],
            created_at=now,
        ),
        MemoryItem(
            memory_id="mem_2",
            user_id="user_1",
            content="User works as a software engineer at Google",
            type=MemoryType.SEMANTIC,
            confidence_score=0.95,
            embedding=[0.0, 1.0, 0.0],
            tags=["work", "career"],
            created_at=now,
        ),
        MemoryItem(
            memory_id="mem_3",
            user_id="user_1",
            content="User asked about Python programming yesterday",
            type=MemoryType.EPISODIC,
            confidence_score=0.8,
            embedding=[0.0, 0.0, 1.0],
            tags=["programming"],
            created_at=now - timedelta(days=1),
        ),
        MemoryItem(
            memory_id="mem_4",
            user_id="user_1",
            content="User lives in San Francisco, California",
            type=MemoryType.SEMANTIC,
            confidence_score=0.85,
            embedding=[0.5, 0.5, 0.0],
            tags=["location"],
            created_at=now,
        ),
        MemoryItem(
            memory_id="mem_5",
            user_id="user_2",
            content="Different user's memory",
            type=MemoryType.SEMANTIC,
            confidence_score=0.9,
            embedding=[1.0, 1.0, 1.0],
            created_at=now,
        ),
    ]
    
    for memory in memories:
        await memory_store.add(memory)
    
    return memory_store


async def mock_embedding(text: str) -> List[float]:
    """Mock embedding function for testing."""
    # Create a simple embedding based on content
    if "vegetarian" in text.lower() or "food" in text.lower():
        return [0.9, 0.1, 0.0]
    elif "software" in text.lower() or "engineer" in text.lower():
        return [0.1, 0.9, 0.0]
    elif "python" in text.lower() or "programming" in text.lower():
        return [0.0, 0.1, 0.9]
    else:
        return [0.33, 0.33, 0.34]


# =============================================================================
# Test Helper Functions
# =============================================================================

# Note: cosine_similarity tests are in test_utils.py


class TestKeywordMatchScore:
    """Tests for keyword match score function."""
    
    def test_exact_match(self):
        """Exact match returns 1."""
        result = keyword_match_score("hello world", "hello world")
        assert result == 1.0
    
    def test_partial_match(self):
        """Partial match returns fraction."""
        result = keyword_match_score("hello", "hello world goodbye")
        assert result == 1.0  # All query words found
        
        result = keyword_match_score("hello goodbye", "hello world")
        assert result == 0.5  # Half of query words found
    
    def test_no_match(self):
        """No match returns 0."""
        result = keyword_match_score("xyz", "abc def")
        assert result == 0.0
    
    def test_case_insensitive(self):
        """Matching is case insensitive."""
        result = keyword_match_score("Hello", "hello")
        assert result == 1.0
    
    def test_empty_query(self):
        """Empty query returns 0."""
        result = keyword_match_score("", "some content")
        assert result == 0.0


class TestRecencyWeight:
    """Tests for recency weight function."""
    
    def test_current_time(self):
        """Recent time has high weight."""
        now = datetime.now(timezone.utc)
        weight = compute_recency_weight(now)
        assert weight > 0.99
    
    def test_half_life(self):
        """At half-life, weight is ~0.5."""
        half_life = 7.0
        past = datetime.now(timezone.utc) - timedelta(days=half_life)
        weight = compute_recency_weight(past, half_life)
        assert abs(weight - 0.5) < 0.01
    
    def test_old_time(self):
        """Old time has low weight."""
        old = datetime.now(timezone.utc) - timedelta(days=100)
        weight = compute_recency_weight(old, half_life_days=7.0)
        assert weight < 0.01
    
    def test_naive_datetime(self):
        """Handles naive datetime."""
        naive = datetime.now()
        weight = compute_recency_weight(naive)
        assert weight > 0.9


# =============================================================================
# Test SimpleRetriever
# =============================================================================

class TestSimpleRetriever:
    """Tests for SimpleRetriever."""
    
    @pytest.mark.asyncio
    async def test_retrieve_basic(self, populated_store):
        """Basic retrieval works."""
        retriever = SimpleRetriever(populated_store)
        results = await retriever.retrieve("food", "user_1")
        
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.retrieval_method == "simple" for r in results)
    
    @pytest.mark.asyncio
    async def test_retrieve_user_isolation(self, populated_store):
        """Only returns memories for the specified user."""
        retriever = SimpleRetriever(populated_store)
        results = await retriever.retrieve("memory", "user_1")
        
        for result in results:
            assert result.memory.user_id == "user_1"
    
    @pytest.mark.asyncio
    async def test_retrieve_with_config(self, populated_store):
        """Respects retrieval config."""
        retriever = SimpleRetriever(populated_store)
        config = RetrievalConfig(
            limit=2,
            min_confidence=0.85,
        )
        results = await retriever.retrieve("", "user_1", config)
        
        assert len(results) <= 2
        for result in results:
            assert result.memory.confidence_score >= 0.85
    
    @pytest.mark.asyncio
    async def test_retrieve_by_tags(self, populated_store):
        """Filters by tags."""
        retriever = SimpleRetriever(populated_store)
        config = RetrievalConfig(tags=["food"])
        results = await retriever.retrieve("", "user_1", config)
        
        for result in results:
            assert "food" in result.memory.tags or "diet" in result.memory.tags
    
    @pytest.mark.asyncio
    async def test_retrieve_by_memory_type(self, populated_store):
        """Filters by memory type."""
        retriever = SimpleRetriever(populated_store)
        config = RetrievalConfig(memory_types=[MemoryType.SEMANTIC])
        results = await retriever.retrieve("", "user_1", config)
        
        for result in results:
            assert result.memory.type == MemoryType.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_retrieve_related(self, populated_store):
        """Retrieves related memories."""
        retriever = SimpleRetriever(populated_store)
        results = await retriever.retrieve_related("mem_1", "user_1", limit=3)
        
        # Should not include the reference memory
        assert all(r.memory.memory_id != "mem_1" for r in results)
    
    @pytest.mark.asyncio
    async def test_name_property(self, memory_store):
        """Has correct name."""
        retriever = SimpleRetriever(memory_store)
        assert retriever.name == "simple"


# =============================================================================
# Test SemanticRetriever
# =============================================================================

class TestSemanticRetriever:
    """Tests for SemanticRetriever."""
    
    @pytest.mark.asyncio
    async def test_retrieve_by_similarity(self, populated_store):
        """Retrieves by semantic similarity."""
        retriever = SemanticRetriever(populated_store, mock_embedding)
        results = await retriever.retrieve("vegetarian food preferences", "user_1")
        
        assert len(results) > 0
        # Food-related memory should be ranked high
        assert any("vegetarian" in r.memory.content.lower() for r in results[:2])
    
    @pytest.mark.asyncio
    async def test_scores_are_computed(self, populated_store):
        """Results have proper similarity scores."""
        retriever = SemanticRetriever(populated_store, mock_embedding)
        results = await retriever.retrieve("programming", "user_1")
        
        for result in results:
            assert 0.0 <= result.score <= 1.0
            assert result.retrieval_method == "semantic"
    
    @pytest.mark.asyncio
    async def test_retrieve_by_embedding(self, populated_store):
        """Retrieves by pre-computed embedding."""
        retriever = SemanticRetriever(populated_store, mock_embedding)
        
        # Embedding similar to food-related memories
        embedding = [0.9, 0.1, 0.0]
        results = await retriever.retrieve_by_embedding(embedding, "user_1")
        
        assert len(results) > 0
        # First result should be food-related
        assert "vegetarian" in results[0].memory.content.lower()
    
    @pytest.mark.asyncio
    async def test_min_score_filter(self, populated_store):
        """Filters by minimum score."""
        retriever = SemanticRetriever(populated_store, mock_embedding)
        config = RetrievalConfig(min_score=0.8)
        results = await retriever.retrieve("food", "user_1", config)
        
        for result in results:
            assert result.score >= 0.8


# =============================================================================
# Test HybridRetriever
# =============================================================================

class TestHybridRetriever:
    """Tests for HybridRetriever."""
    
    @pytest.mark.asyncio
    async def test_combines_semantic_and_keyword(self, populated_store):
        """Combines semantic and keyword scores."""
        retriever = HybridRetriever(
            populated_store,
            mock_embedding,
            semantic_weight=0.5,
            keyword_weight=0.5,
        )
        results = await retriever.retrieve("vegetarian", "user_1")
        
        assert len(results) > 0
        for result in results:
            assert "semantic_score" in result.metadata
            assert "keyword_score" in result.metadata
    
    @pytest.mark.asyncio
    async def test_weight_configuration(self, populated_store):
        """Respects weight configuration."""
        # Heavy semantic weight
        retriever = HybridRetriever(
            populated_store,
            mock_embedding,
            semantic_weight=0.9,
            keyword_weight=0.1,
        )
        results = await retriever.retrieve("food", "user_1")
        
        assert results[0].metadata["semantic_weight"] == 0.9
        assert results[0].metadata["keyword_weight"] == 0.1
    
    @pytest.mark.asyncio
    async def test_name_property(self, memory_store):
        """Has correct name."""
        retriever = HybridRetriever(memory_store, mock_embedding)
        assert retriever.name == "hybrid"


# =============================================================================
# Test TemporalRetriever
# =============================================================================

class TestTemporalRetriever:
    """Tests for TemporalRetriever."""
    
    @pytest.mark.asyncio
    async def test_recency_weighting(self, populated_store):
        """Applies recency weighting."""
        retriever = TemporalRetriever(
            populated_store,
            mock_embedding,
            recency_weight=0.5,
            semantic_weight=0.5,
        )
        results = await retriever.retrieve("test", "user_1")
        
        assert len(results) > 0
        for result in results:
            assert "recency_score" in result.metadata
            assert "semantic_score" in result.metadata
    
    @pytest.mark.asyncio
    async def test_half_life_configuration(self, populated_store):
        """Respects half-life configuration."""
        retriever = TemporalRetriever(
            populated_store,
            mock_embedding,
            half_life_days=14.0,
        )
        results = await retriever.retrieve("test", "user_1")
        
        assert results[0].metadata["half_life_days"] == 14.0


# =============================================================================
# Test Rerankers
# =============================================================================

class TestRecencyReranker:
    """Tests for RecencyReranker."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample retrieval results."""
        now = datetime.now(timezone.utc)
        return [
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="old",
                    user_id="user_1",
                    content="Old memory",
                    type=MemoryType.SEMANTIC,
                    created_at=now - timedelta(days=30),
                ),
                score=0.9,
                retrieval_method="test",
            ),
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="new",
                    user_id="user_1",
                    content="New memory",
                    type=MemoryType.SEMANTIC,
                    created_at=now - timedelta(hours=1),
                ),
                score=0.7,
                retrieval_method="test",
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_boosts_recent(self, sample_results):
        """Boosts recent memories."""
        reranker = RecencyReranker(recency_boost=0.3)
        results = await reranker.rerank("query", sample_results)
        
        # New memory should be boosted more
        new_result = next(r for r in results if r.memory.memory_id == "new")
        old_result = next(r for r in results if r.memory.memory_id == "old")
        
        assert new_result.metadata["recency_boost"] > old_result.metadata["recency_boost"]
    
    @pytest.mark.asyncio
    async def test_preserves_original_score(self, sample_results):
        """Preserves original score in metadata."""
        reranker = RecencyReranker()
        results = await reranker.rerank("query", sample_results)
        
        for result in results:
            assert "original_score" in result.metadata


class TestScoreThresholdReranker:
    """Tests for ScoreThresholdReranker."""
    
    @pytest.mark.asyncio
    async def test_filters_by_threshold(self):
        """Filters results below threshold."""
        results = [
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="high",
                    user_id="user_1",
                    content="High score",
                    type=MemoryType.SEMANTIC,
                ),
                score=0.9,
                retrieval_method="test",
            ),
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="low",
                    user_id="user_1",
                    content="Low score",
                    type=MemoryType.SEMANTIC,
                ),
                score=0.3,
                retrieval_method="test",
            ),
        ]
        
        reranker = ScoreThresholdReranker(min_score=0.5)
        filtered = await reranker.rerank("query", results)
        
        assert len(filtered) == 1
        assert filtered[0].memory.memory_id == "high"


class TestDiversityReranker:
    """Tests for DiversityReranker."""
    
    @pytest.mark.asyncio
    async def test_promotes_diversity(self):
        """Promotes diverse results."""
        results = [
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="1",
                    user_id="user_1",
                    content="Python programming language",
                    type=MemoryType.SEMANTIC,
                ),
                score=0.9,
                retrieval_method="test",
            ),
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="2",
                    user_id="user_1",
                    content="Python programming tutorial",  # Similar to #1
                    type=MemoryType.SEMANTIC,
                ),
                score=0.85,
                retrieval_method="test",
            ),
            RetrievalResult(
                memory=MemoryItem(
                    memory_id="3",
                    user_id="user_1",
                    content="Machine learning algorithms",  # Different topic
                    type=MemoryType.SEMANTIC,
                ),
                score=0.7,
                retrieval_method="test",
            ),
        ]
        
        reranker = DiversityReranker(diversity_weight=0.5)
        diverse = await reranker.rerank("query", results, top_k=2)
        
        # Should include diverse results
        ids = [r.memory.memory_id for r in diverse]
        assert len(ids) == 2


# =============================================================================
# Test Registry Integration
# =============================================================================

class TestRetrieverRegistry:
    """Tests for retriever registry integration."""
    
    def test_simple_retriever_registered(self):
        """SimpleRetriever is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_retriever("simple") is not None
    
    def test_semantic_retriever_registered(self):
        """SemanticRetriever is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_retriever("semantic") is not None
    
    def test_hybrid_retriever_registered(self):
        """HybridRetriever is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_retriever("hybrid") is not None
    
    def test_temporal_retriever_registered(self):
        """TemporalRetriever is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_retriever("temporal") is not None

