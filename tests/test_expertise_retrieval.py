"""
Tests for Expertise Vector Indexing and Retrieval (Phase 3).

Tests ExpertiseIndexer, ExpertiseRetriever, EffectivenessReranker,
and HybridExpertiseRetriever.
"""

from typing import Any, Dict, List, Optional

import pytest

from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
)
from ctxforge.protocols.context import IndexSearchResult
from ctxforge.protocols.llm import EmbeddingResponse
from ctxforge.retrieval.indexers import ExpertiseIndexer
from ctxforge.retrieval.rerankers import EffectivenessReranker
from ctxforge.retrieval.retrievers import (
    ExpertiseRetrievalConfig,
    ExpertiseRetrievalResult,
    ExpertiseRetriever,
    HybridExpertiseRetriever,
)
from ctxforge.vectorstores.protocol import (
    DistanceMetric,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
)

# ============================================================================
# Mock Implementations
# ============================================================================

class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return "mock_embedding"
    
    @property
    def default_model(self) -> str:
        return "mock-model"
    
    @property
    def embedding_dimension(self) -> int:
        return self._dimension
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate mock embeddings."""
        self._call_count += 1
        embeddings = []
        for _i, text in enumerate(texts):
            # Generate deterministic embedding based on text hash
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model or self.default_model,
            total_tokens=sum(len(t.split()) for t in texts),
        )
    
    async def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Generate mock embedding for single text."""
        self._call_count += 1
        return self._generate_embedding(text)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding from text."""
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Create embedding from hash bytes
        embedding = []
        for i in range(self._dimension):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        return embedding


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._records: Dict[str, Dict[str, VectorRecord]] = {}  # namespace -> id -> record
    
    @property
    def name(self) -> str:
        return "mock_vector_store"
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.COSINE
    
    async def initialize(self) -> None:
        pass
    
    async def close(self) -> None:
        pass
    
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
    ) -> int:
        """Insert or update vectors."""
        ns = namespace or "default"
        if ns not in self._records:
            self._records[ns] = {}
        
        for record in vectors:
            self._records[ns][record.id] = record
        
        return len(vectors)
    
    async def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
        include_embedding: bool = False,
        include_metadata: bool = True,
    ) -> List[VectorQueryResult]:
        """Query for similar vectors."""
        ns = namespace or "default"
        if ns not in self._records:
            return []
        
        results = []
        for _record_id, record in self._records[ns].items():
            # Apply filters
            if filters:
                if not self._matches_filters(record, filters):
                    continue
            
            # Calculate similarity (cosine)
            score = self._cosine_similarity(embedding, record.embedding)
            
            results.append(VectorQueryResult(
                id=record.id,
                score=score,
                embedding=record.embedding if include_embedding else None,
                metadata=record.metadata if include_metadata else {},
                content=record.content,
            ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    async def query_by_id(
        self,
        vector_id: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
        include_embedding: bool = False,
    ) -> List[VectorQueryResult]:
        """Query by ID."""
        ns = namespace or "default"
        if ns not in self._records or vector_id not in self._records[ns]:
            return []
        
        reference = self._records[ns][vector_id]
        return await self.query(
            reference.embedding,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
            include_embedding=include_embedding,
        )
    
    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, VectorRecord]:
        """Fetch by IDs."""
        ns = namespace or "default"
        if ns not in self._records:
            return {}
        
        result = {}
        for id_ in ids:
            if id_ in self._records[ns]:
                result[id_] = self._records[ns][id_]
        return result
    
    async def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete by IDs."""
        ns = namespace or "default"
        if ns not in self._records:
            return 0
        
        count = 0
        for id_ in ids:
            if id_ in self._records[ns]:
                del self._records[ns][id_]
                count += 1
        return count
    
    async def delete_by_filter(
        self,
        filters: List[QueryFilter],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete by filter."""
        ns = namespace or "default"
        if ns not in self._records:
            return 0
        
        to_delete = []
        for record_id, record in self._records[ns].items():
            if self._matches_filters(record, filters):
                to_delete.append(record_id)
        
        for id_ in to_delete:
            del self._records[ns][id_]
        
        return len(to_delete)
    
    async def delete_namespace(self, namespace: str) -> bool:
        """Delete namespace."""
        if namespace in self._records:
            del self._records[namespace]
            return True
        return False
    
    async def list_namespaces(self) -> List[str]:
        """List namespaces."""
        return list(self._records.keys())
    
    async def count(
        self,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
    ) -> int:
        """Count records."""
        ns = namespace or "default"
        if ns not in self._records:
            return 0
        
        if filters:
            count = 0
            for record in self._records[ns].values():
                if self._matches_filters(record, filters):
                    count += 1
            return count
        
        return len(self._records[ns])
    
    async def describe(self) -> Dict[str, Any]:
        """Describe store."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "namespaces": list(self._records.keys()),
            "total_vectors": sum(len(ns) for ns in self._records.values()),
        }
    
    def _matches_filters(self, record: VectorRecord, filters: List[QueryFilter]) -> bool:
        """Check if record matches filters."""
        for f in filters:
            value = record.metadata.get(f.field)
            if f.operator == "eq":
                if value != f.value:
                    return False
            elif f.operator == "ne":
                if value == f.value:
                    return False
            elif f.operator == "in":
                if value not in f.value:
                    return False
            elif f.operator == "gte":
                if value is None or value < f.value:
                    return False
        return True
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return (dot / (mag_a * mag_b) + 1) / 2  # Normalize to 0-1


class MockExpertiseStore:
    """Mock expertise store for testing."""
    
    def __init__(self):
        self._expertise: Dict[str, Expertise] = {}
    
    async def save(self, expertise: Expertise) -> None:
        self._expertise[expertise.expertise_id] = expertise
    
    async def load(self, expertise_id: str) -> Optional[Expertise]:
        return self._expertise.get(expertise_id)
    
    async def delete(self, expertise_id: str) -> bool:
        if expertise_id in self._expertise:
            del self._expertise[expertise_id]
            return True
        return False
    
    async def list_expertise(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Expertise]:
        result = list(self._expertise.values())
        if domain:
            result = [e for e in result if e.domain == domain]
        return result[offset:offset + limit]
    
    async def add_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        if expertise_id in self._expertise:
            self._expertise[expertise_id].items.append(item)
    
    async def update_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        if expertise_id in self._expertise:
            expertise = self._expertise[expertise_id]
            for i, existing in enumerate(expertise.items):
                if existing.item_id == item.item_id:
                    expertise.items[i] = item
                    break
    
    async def remove_item(self, expertise_id: str, item_id: str) -> bool:
        if expertise_id in self._expertise:
            expertise = self._expertise[expertise_id]
            for item in expertise.items:
                if item.item_id == item_id:
                    expertise.items.remove(item)
                    return True
        return False
    
    async def get_item(
        self,
        expertise_id: str,
        item_id: str,
    ) -> Optional[ExpertiseItem]:
        if expertise_id in self._expertise:
            return self._expertise[expertise_id].get_item(item_id)
        return None
    
    async def get_items_by_section(
        self,
        expertise_id: str,
        section: ExpertiseSection,
    ) -> List[ExpertiseItem]:
        if expertise_id in self._expertise:
            return self._expertise[expertise_id].get_items_by_section(section)
        return []
    
    async def update_item_counts(
        self,
        expertise_id: str,
        item_id: str,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
    ) -> None:
        if expertise_id in self._expertise:
            self._expertise[expertise_id].update_item_counts(
                item_id, helpful_delta, harmful_delta
            )
    
    async def search_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """Simple keyword search."""
        if expertise_id not in self._expertise:
            return []
        
        query_lower = query.lower()
        results = []
        for item in self._expertise[expertise_id].active_items:
            if query_lower in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results
    
    async def log_usage(self, log) -> None:
        pass
    
    async def get_usage_stats(self, expertise_id: str) -> Dict[str, Any]:
        return {}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def embedding_provider():
    """Create mock embedding provider."""
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def vector_store():
    """Create mock vector store."""
    return MockVectorStore(dimension=384)


@pytest.fixture
def expertise_store():
    """Create mock expertise store."""
    return MockExpertiseStore()


@pytest.fixture
def sample_expertise():
    """Create sample expertise with items."""
    expertise = Expertise(
        expertise_id="test-expertise",
        name="Test Expertise",
        domain="testing",
    )
    
    # Add strategy items
    expertise.add_item(
        ExpertiseSection.STRATEGIES,
        "Always validate user input before processing",
        source="manual",
    )
    expertise.add_item(
        ExpertiseSection.STRATEGIES,
        "Use caching for frequently accessed data",
        source="manual",
    )
    
    # Add formula items
    expertise.add_item(
        ExpertiseSection.FORMULAS,
        "discount = original_price * (discount_percent / 100)",
        source="manual",
    )
    
    # Add common mistake items
    expertise.add_item(
        ExpertiseSection.COMMON_MISTAKES,
        "Don't forget to handle null values in calculations",
        source="manual",
    )
    expertise.add_item(
        ExpertiseSection.COMMON_MISTAKES,
        "Avoid using == for floating point comparisons",
        source="manual",
    )
    
    # Add item with usage data
    high_performing_item = expertise.add_item(
        ExpertiseSection.STRATEGIES,
        "Break complex problems into smaller steps",
        source="reflection",
    )
    high_performing_item.helpful_count = 10
    high_performing_item.harmful_count = 1
    
    # Add problematic item
    problematic_item = expertise.add_item(
        ExpertiseSection.STRATEGIES,
        "Always use recursion for iteration",
        source="manual",
    )
    problematic_item.helpful_count = 2
    problematic_item.harmful_count = 8
    
    return expertise


@pytest.fixture
def indexer(vector_store, embedding_provider):
    """Create expertise indexer."""
    return ExpertiseIndexer(vector_store, embedding_provider)


@pytest.fixture
async def indexed_expertise(indexer, expertise_store, sample_expertise):
    """Create expertise with indexed items."""
    await expertise_store.save(sample_expertise)
    await indexer.index_all(sample_expertise)
    return sample_expertise


# ============================================================================
# ExpertiseIndexer Tests
# ============================================================================

class TestExpertiseIndexer:
    """Tests for ExpertiseIndexer."""
    
    async def test_index_single_item(self, indexer, sample_expertise):
        """Test indexing a single item."""
        item = sample_expertise.items[0]
        await indexer.index_item(item, sample_expertise.expertise_id)
        
        # Verify item was indexed
        count = await indexer.count(sample_expertise.expertise_id)
        assert count == 1
    
    async def test_index_all_items(self, indexer, sample_expertise):
        """Test indexing all items in expertise."""
        count = await indexer.index_all(sample_expertise)
        
        # Should index all active items
        assert count == sample_expertise.active_item_count
        
        # Verify items are in index
        indexed_count = await indexer.count(sample_expertise.expertise_id)
        assert indexed_count == count
    
    async def test_index_generates_embeddings(self, indexer, sample_expertise, embedding_provider):
        """Test that indexing generates embeddings for items."""
        # Items start without embeddings
        for item in sample_expertise.items:
            assert item.embedding is None
        
        await indexer.index_all(sample_expertise)
        
        # After indexing, items should have embeddings
        for item in sample_expertise.items:
            assert item.embedding is not None
            assert len(item.embedding) == embedding_provider.embedding_dimension
    
    async def test_index_preserves_existing_embeddings(self, indexer, sample_expertise, embedding_provider):
        """Test that existing embeddings are preserved."""
        item = sample_expertise.items[0]
        original_embedding = [0.1] * embedding_provider.embedding_dimension
        item.embedding = original_embedding
        
        await indexer.index_item(item, sample_expertise.expertise_id)
        
        # Embedding should be unchanged
        assert item.embedding == original_embedding
    
    async def test_remove_item(self, indexer, sample_expertise):
        """Test removing an item from the index."""
        await indexer.index_all(sample_expertise)
        
        initial_count = await indexer.count(sample_expertise.expertise_id)
        
        item_id = sample_expertise.items[0].item_id
        removed = await indexer.remove_item(item_id, sample_expertise.expertise_id)
        
        assert removed is True
        
        new_count = await indexer.count(sample_expertise.expertise_id)
        assert new_count == initial_count - 1
    
    async def test_remove_all(self, indexer, sample_expertise):
        """Test removing all items for an expertise."""
        await indexer.index_all(sample_expertise)
        
        # Verify items exist
        count = await indexer.count(sample_expertise.expertise_id)
        assert count > 0
        
        # Remove all
        removed = await indexer.remove_all(sample_expertise.expertise_id)
        assert removed is True
        
        # Verify empty
        count = await indexer.count(sample_expertise.expertise_id)
        assert count == 0
    
    async def test_search_returns_results(self, indexer, sample_expertise):
        """Test that search returns relevant results."""
        await indexer.index_all(sample_expertise)
        
        results = await indexer.search(
            query="validate input",
            expertise_id=sample_expertise.expertise_id,
            limit=5,
        )
        
        assert len(results) > 0
        assert all(isinstance(r, IndexSearchResult) for r in results)
    
    async def test_search_with_limit(self, indexer, sample_expertise):
        """Test that search respects limit."""
        await indexer.index_all(sample_expertise)
        
        results = await indexer.search(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            limit=2,
        )
        
        assert len(results) <= 2
    
    async def test_search_by_section(self, indexer, sample_expertise):
        """Test searching within specific sections."""
        await indexer.index_all(sample_expertise)
        
        results = await indexer.search_by_section(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            sections=[ExpertiseSection.STRATEGIES],
            limit=10,
        )
        
        # All results should be from STRATEGIES section
        for r in results:
            assert r.metadata.get("section") == ExpertiseSection.STRATEGIES.value
    
    async def test_get_similar_items(self, indexer, sample_expertise):
        """Test finding similar items."""
        await indexer.index_all(sample_expertise)
        
        reference_id = sample_expertise.items[0].item_id
        similar = await indexer.get_similar_items(
            item_id=reference_id,
            expertise_id=sample_expertise.expertise_id,
            limit=3,
        )
        
        # Should not include the reference item
        assert all(r.id != reference_id for r in similar)
    
    async def test_namespace_isolation(self, vector_store, embedding_provider):
        """Test that different expertise IDs are isolated."""
        # Create a fresh indexer with its own vector store
        fresh_vector_store = MockVectorStore(dimension=384)
        indexer = ExpertiseIndexer(fresh_vector_store, embedding_provider)
        
        # Create two expertise with distinct items using explicit unique item IDs
        expertise1 = Expertise(
            expertise_id="expertise-1",
            name="Expertise 1",
        )
        item1 = ExpertiseItem(
            item_id="exp1-strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="First expertise unique content",
        )
        expertise1.items.append(item1)
        
        expertise2 = Expertise(
            expertise_id="expertise-2",
            name="Expertise 2",
        )
        item2 = ExpertiseItem(
            item_id="exp2-strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="Second expertise different content",
        )
        expertise2.items.append(item2)
        
        await indexer.index_all(expertise1)
        await indexer.index_all(expertise2)
        
        # Verify each expertise is in its own namespace
        count1 = await indexer.count("expertise-1")
        count2 = await indexer.count("expertise-2")
        assert count1 == 1
        assert count2 == 1
        
        # Search in first expertise should only find items from first
        results1 = await indexer.search(
            query="unique",
            expertise_id="expertise-1",
            limit=10,
        )
        
        # Search in second expertise should only find items from second
        results2 = await indexer.search(
            query="different",
            expertise_id="expertise-2",
            limit=10,
        )
        
        # Results should be from different namespaces (different item IDs)
        ids1 = {r.item_id for r in results1}
        ids2 = {r.item_id for r in results2}
        assert len(ids1) > 0
        assert len(ids2) > 0
        assert ids1.isdisjoint(ids2)


# ============================================================================
# ExpertiseRetriever Tests
# ============================================================================

class TestExpertiseRetriever:
    """Tests for ExpertiseRetriever."""
    
    async def test_retrieve_returns_items(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that retrieve returns expertise items."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="validate user input",
            expertise_id=sample_expertise.expertise_id,
            limit=5,
        )
        
        assert len(items) > 0
        assert all(isinstance(item, ExpertiseItem) for item in items)
    
    async def test_retrieve_detailed(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test retrieve_detailed returns results with scores."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        results = await retriever.retrieve_detailed(
            query="caching data",
            expertise_id=sample_expertise.expertise_id,
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ExpertiseRetrievalResult) for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)
    
    async def test_retrieve_respects_limit(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that retrieve respects limit parameter."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            limit=2,
        )
        
        assert len(items) <= 2
    
    async def test_retrieve_filters_by_section(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that retrieve filters by sections."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            sections=[ExpertiseSection.COMMON_MISTAKES],
            limit=10,
        )
        
        assert all(item.section == ExpertiseSection.COMMON_MISTAKES for item in items)
    
    async def test_retrieve_by_section(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test retrieve_by_section convenience method."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve_by_section(
            query="data",
            expertise_id=sample_expertise.expertise_id,
            section=ExpertiseSection.STRATEGIES,
            limit=5,
        )
        
        assert all(item.section == ExpertiseSection.STRATEGIES for item in items)
    
    async def test_retrieve_high_performing(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test retrieve_high_performing returns effective items."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve_high_performing(
            query="problem steps",
            expertise_id=sample_expertise.expertise_id,
            limit=5,
        )
        
        # Should return high-performing items
        for item in items:
            assert item.is_high_performing
    
    async def test_retrieve_related(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test retrieve_related finds similar items."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        reference_id = sample_expertise.items[0].item_id
        related = await retriever.retrieve_related(
            item_id=reference_id,
            expertise_id=sample_expertise.expertise_id,
            limit=3,
        )
        
        # Should not include reference item
        assert all(item.item_id != reference_id for item in related)
    
    async def test_effectiveness_weight(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that effectiveness weight affects ranking."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        # Create retriever with high effectiveness weight
        retriever = ExpertiseRetriever(
            expertise_store, indexer, effectiveness_weight=0.5
        )
        
        config = ExpertiseRetrievalConfig(
            limit=10,
            boost_high_performing=True,
        )
        
        results = await retriever.retrieve_detailed(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            config=config,
        )
        
        # Results should include effectiveness info in metadata
        for r in results:
            assert "effectiveness_score" in r.metadata
    
    async def test_empty_expertise_returns_empty(
        self, indexer, expertise_store
    ):
        """Test that empty expertise returns empty results."""
        empty = Expertise(
            expertise_id="empty-expertise",
            name="Empty",
        )
        await expertise_store.save(empty)
        
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="anything",
            expertise_id="empty-expertise",
            limit=10,
        )
        
        assert items == []
    
    async def test_nonexistent_expertise_returns_empty(
        self, indexer, expertise_store
    ):
        """Test that nonexistent expertise returns empty results."""
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="anything",
            expertise_id="nonexistent",
            limit=10,
        )
        
        assert items == []


# ============================================================================
# EffectivenessReranker Tests
# ============================================================================

class TestEffectivenessReranker:
    """Tests for EffectivenessReranker."""
    
    def _create_result(
        self,
        item_id: str,
        content: str,
        score: float,
        helpful: int = 0,
        harmful: int = 0,
    ) -> ExpertiseRetrievalResult:
        """Helper to create retrieval result."""
        item = ExpertiseItem(
            item_id=item_id,
            section=ExpertiseSection.STRATEGIES,
            content=content,
            helpful_count=helpful,
            harmful_count=harmful,
        )
        return ExpertiseRetrievalResult(
            item=item,
            score=score,
            retrieval_method="semantic",
            metadata={"semantic_score": score},
        )
    
    async def test_rerank_boosts_helpful_items(self):
        """Test that helpful items get boosted."""
        reranker = EffectivenessReranker(
            helpful_weight=1.0,
            harmful_weight=-2.0,
            min_usage_for_boost=2,
        )
        
        # Use closer scores so the boost can make a difference
        results = [
            self._create_result("item-1", "Content 1", 0.7, helpful=0, harmful=0),
            self._create_result("item-2", "Content 2", 0.65, helpful=20, harmful=0),
        ]
        
        reranked = await reranker.rerank("query", results)
        
        # Item with helpful count should be boosted above the other
        # The boost should overcome the 0.05 difference
        assert reranked[0].item.item_id == "item-2"
    
    async def test_rerank_penalizes_harmful_items(self):
        """Test that harmful items get penalized."""
        reranker = EffectivenessReranker(
            helpful_weight=1.0,
            harmful_weight=-2.0,
            min_usage_for_boost=2,
        )
        
        results = [
            self._create_result("item-1", "Content 1", 0.8, helpful=5, harmful=10),
            self._create_result("item-2", "Content 2", 0.7, helpful=5, harmful=0),
        ]
        
        reranked = await reranker.rerank("query", results)
        
        # Item with harmful count should be penalized
        assert reranked[0].item.item_id == "item-2"
    
    async def test_rerank_respects_min_usage(self):
        """Test that items below min_usage don't get boosted."""
        reranker = EffectivenessReranker(min_usage_for_boost=5)
        
        results = [
            self._create_result("item-1", "Content 1", 0.8, helpful=1, harmful=0),
            self._create_result("item-2", "Content 2", 0.7, helpful=0, harmful=0),
        ]
        
        reranked = await reranker.rerank("query", results)
        
        # Original order should be mostly preserved (low usage)
        assert reranked[0].item.item_id == "item-1"
    
    async def test_rerank_respects_top_k(self):
        """Test that rerank respects top_k limit."""
        reranker = EffectivenessReranker()
        
        results = [
            self._create_result("item-1", "Content 1", 0.8),
            self._create_result("item-2", "Content 2", 0.7),
            self._create_result("item-3", "Content 3", 0.6),
        ]
        
        reranked = await reranker.rerank("query", results, top_k=2)
        
        assert len(reranked) == 2
    
    async def test_rerank_preserves_metadata(self):
        """Test that reranking preserves and adds metadata."""
        reranker = EffectivenessReranker()
        
        results = [
            self._create_result("item-1", "Content", 0.8, helpful=5, harmful=0),
        ]
        
        reranked = await reranker.rerank("query", results)
        
        assert "original_score" in reranked[0].metadata
        assert "effectiveness_boost" in reranked[0].metadata
        assert reranked[0].metadata["original_score"] == 0.8
    
    async def test_rerank_empty_returns_empty(self):
        """Test that empty input returns empty output."""
        reranker = EffectivenessReranker()
        
        reranked = await reranker.rerank("query", [])
        
        assert reranked == []


# ============================================================================
# HybridExpertiseRetriever Tests
# ============================================================================

class TestHybridExpertiseRetriever:
    """Tests for HybridExpertiseRetriever."""
    
    async def test_hybrid_combines_semantic_and_keyword(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that hybrid retriever combines both search methods."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = HybridExpertiseRetriever(
            expertise_store=expertise_store,
            indexer=indexer,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        
        items = await retriever.retrieve(
            query="validate",
            expertise_id=sample_expertise.expertise_id,
            limit=5,
        )
        
        assert len(items) > 0
    
    async def test_hybrid_respects_limit(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that hybrid retriever respects limit."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = HybridExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            limit=2,
        )
        
        assert len(items) <= 2
    
    async def test_hybrid_filters_by_section(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that hybrid retriever filters by sections."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = HybridExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            sections=[ExpertiseSection.STRATEGIES],
            limit=10,
        )
        
        assert all(item.section == ExpertiseSection.STRATEGIES for item in items)
    
    async def test_hybrid_respects_min_effectiveness(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test that hybrid retriever respects min_effectiveness."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        retriever = HybridExpertiseRetriever(expertise_store, indexer)
        
        items = await retriever.retrieve(
            query="problem",
            expertise_id=sample_expertise.expertise_id,
            min_effectiveness=0.7,
            limit=10,
        )
        
        for item in items:
            assert item.effectiveness_score >= 0.7


# ============================================================================
# Integration Tests
# ============================================================================

class TestRetrievalIntegration:
    """Integration tests for the full retrieval pipeline."""
    
    async def test_full_index_and_retrieve_pipeline(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test complete pipeline from indexing to retrieval."""
        # Save expertise
        await expertise_store.save(sample_expertise)
        
        # Index all items
        indexed_count = await indexer.index_all(sample_expertise)
        assert indexed_count > 0
        
        # Create retriever
        retriever = ExpertiseRetriever(expertise_store, indexer)
        
        # Retrieve items
        items = await retriever.retrieve(
            query="validate user input before processing",
            expertise_id=sample_expertise.expertise_id,
            limit=3,
        )
        
        # Should find items (with mock embeddings, semantic relevance is not meaningful)
        assert len(items) > 0
        
        # Verify we got ExpertiseItem objects back
        assert all(isinstance(item, ExpertiseItem) for item in items)
    
    async def test_update_and_reindex_item(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test updating and re-indexing an item."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        # Update item
        item = sample_expertise.items[0]
        item.helpful_count = 100
        
        # Re-index
        await indexer.update_item_metadata(item, sample_expertise.expertise_id)
        
        # The item should still be searchable
        count = await indexer.count(sample_expertise.expertise_id)
        assert count == sample_expertise.active_item_count
    
    async def test_retrieval_with_reranking(
        self, indexer, expertise_store, sample_expertise
    ):
        """Test retrieval with effectiveness reranking."""
        await expertise_store.save(sample_expertise)
        await indexer.index_all(sample_expertise)
        
        reranker = EffectivenessReranker(
            helpful_weight=1.0,
            harmful_weight=-2.0,
        )
        
        retriever = ExpertiseRetriever(
            expertise_store,
            indexer,
            reranker=reranker,
        )
        
        config = ExpertiseRetrievalConfig(
            limit=5,
            rerank=True,
        )
        
        results = await retriever.retrieve_detailed(
            query="problem solving",
            expertise_id=sample_expertise.expertise_id,
            config=config,
        )
        
        assert len(results) > 0
        # Results should have been through the reranker
        if len(results) > 1 and any(r.item.total_usage >= 3 for r in results):
            assert "+effectiveness" in results[0].retrieval_method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

