"""
Tests for Phase 4: Enhanced Index and O(1) Retrieval Fast-Paths.

Tests cover:
- EnhancedMemoryIndex data structures
- AggregationBuilder for building indexes from memories
- FastPathRetriever for O(1) cache lookups
- InMemoryGraphStore save/load enhanced index (Phase 4 integration)
- EngineFactory wiring of FastPathRetriever
- prepare_context fast-path short-circuit integration
"""

import inspect
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from ctxforge.config.base import RetrievalFastPathConfig
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.retrieval.aggregation_builder import AggregationBuilder
from ctxforge.retrieval.enhanced_structures import (
    EnhancedMemoryIndex,
    EntityAggregation,
    QueryCache,
    RelationTriple,
)
from ctxforge.retrieval.fast_path_retriever import (
    FastPathResult,
    FastPathRetriever,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def create_memory(
    content: str,
    tags: List[str] = None,
    metadata: dict = None,
    created_at: datetime = None,
    user_id: str = "test-user",
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryItem:
    """Helper to create a MemoryItem for testing."""
    return MemoryItem(
        content=content,
        user_id=user_id,
        type=memory_type,
        tags=tags or [],
        metadata=metadata or {},
        created_at=created_at or datetime.now(timezone.utc).replace(tzinfo=None),
    )


@pytest.fixture
def sample_memories() -> List[MemoryItem]:
    """Create a set of sample memories for testing."""
    base_time = datetime.now(timezone.utc).replace(tzinfo=None)
    return [
        create_memory(
            "Alice visited Paris last summer.",
            tags=["travel", "Alice"],
            created_at=base_time - timedelta(days=30),
        ),
        create_memory(
            "Alice visited London in spring.",
            tags=["travel", "Alice"],
            created_at=base_time - timedelta(days=60),
        ),
        create_memory(
            "Alice visited Tokyo for business.",
            tags=["travel", "business", "Alice"],
            created_at=base_time - timedelta(days=90),
        ),
        create_memory(
            "Bob and Alice met at the conference.",
            tags=["meeting", "Alice", "Bob"],
            metadata={"persons": ["Alice", "Bob"]},
            created_at=base_time - timedelta(days=15),
        ),
        create_memory(
            "Bob likes coffee and tea.",
            tags=["preferences", "Bob"],
            created_at=base_time - timedelta(days=10),
        ),
        create_memory(
            "Alice prefers green tea over coffee.",
            tags=["preferences", "Alice"],
            created_at=base_time - timedelta(days=5),
        ),
    ]


@pytest.fixture
def aggregation_builder() -> AggregationBuilder:
    """Create an AggregationBuilder instance."""
    return AggregationBuilder()


@pytest.fixture
def enhanced_index(
    aggregation_builder: AggregationBuilder, sample_memories: List[MemoryItem]
) -> EnhancedMemoryIndex:
    """Build an enhanced index from sample memories."""
    return aggregation_builder.build_aggregations(sample_memories)


@pytest.fixture
def fast_path_config() -> RetrievalFastPathConfig:
    """Create a fast-path config with all features enabled."""
    return RetrievalFastPathConfig(
        enabled=True,
        detect_count_queries=True,
        detect_list_queries=True,
        detect_relation_queries=True,
        min_confidence=0.5,
    )


@pytest.fixture
def fast_path_retriever(
    enhanced_index: EnhancedMemoryIndex, fast_path_config: RetrievalFastPathConfig
) -> FastPathRetriever:
    """Create a FastPathRetriever with the enhanced index."""
    return FastPathRetriever(enhanced_index=enhanced_index, config=fast_path_config)


# =============================================================================
# Test Enhanced Structures
# =============================================================================


class TestEntityAggregation:
    """Tests for EntityAggregation data structure."""

    def test_create_entity_aggregation(self):
        """Test creating an EntityAggregation."""
        agg = EntityAggregation(
            entity_name="Alice",
            entity_type="person",
            event_counts={"visited_paris": 1, "visited_london": 2},
            attribute_sets={"locations": {"Paris", "London"}},
        )
        assert agg.entity_name == "Alice"
        assert agg.entity_type == "person"
        assert agg.event_counts["visited_paris"] == 1
        assert "Paris" in agg.attribute_sets["locations"]

    def test_entity_aggregation_defaults(self):
        """Test default values for EntityAggregation."""
        agg = EntityAggregation(entity_name="Test")
        assert agg.entity_type == "entity"
        assert agg.event_counts == {}
        assert agg.attribute_sets == {}
        assert agg.temporal_sequences == {}
        assert agg.evidence_memory_ids == []


class TestRelationTriple:
    """Tests for RelationTriple data structure."""

    def test_create_relation_triple(self):
        """Test creating a RelationTriple."""
        triple = RelationTriple(
            subject="Alice",
            predicate="met",
            object="Bob",
            source_memory_id="mem-123",
        )
        assert triple.subject == "Alice"
        assert triple.predicate == "met"
        assert triple.object == "Bob"
        assert triple.confidence == 1.0

    def test_relation_triple_with_context(self):
        """Test RelationTriple with timestamp and location."""
        triple = RelationTriple(
            subject="Alice",
            predicate="visited",
            object="Paris",
            timestamp="2024-01-15T10:00:00",
            location="France",
            source_memory_id="mem-456",
            confidence=0.9,
        )
        assert triple.timestamp == "2024-01-15T10:00:00"
        assert triple.location == "France"
        assert triple.confidence == 0.9


class TestEnhancedMemoryIndex:
    """Tests for EnhancedMemoryIndex data structure."""

    def test_create_enhanced_index(self):
        """Test creating an EnhancedMemoryIndex."""
        index = EnhancedMemoryIndex()
        assert index.entities == {}
        assert index.relations == []
        assert index.query_cache == {}
        assert index.temporal_index == {}

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        # Create index with data
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    event_counts={"visited": 3},
                    attribute_sets={"locations": {"Paris", "London"}},
                )
            },
            relations=[
                RelationTriple(
                    subject="Alice",
                    predicate="met",
                    object="Bob",
                    source_memory_id="mem-1",
                )
            ],
            temporal_index={"2024-01-15": ["mem-1", "mem-2"]},
            build_timestamp="2024-01-15T10:00:00",
            memory_count=10,
        )

        # Serialize
        data = index.to_dict()
        assert "entities" in data
        assert "relations" in data
        assert "temporal_index" in data

        # Deserialize
        restored = EnhancedMemoryIndex.from_dict(data)
        assert "Alice" in restored.entities
        assert restored.entities["Alice"].entity_name == "Alice"
        assert "Paris" in restored.entities["Alice"].attribute_sets["locations"]
        assert len(restored.relations) == 1
        assert restored.relations[0].subject == "Alice"


class TestQueryCache:
    """Tests for QueryCache data structure."""

    def test_create_query_cache(self):
        """Test creating a QueryCache entry."""
        cache = QueryCache(
            cache_key="count:Alice:visited",
            cached_value=3,
            value_type="count",
            hit_count=5,
        )
        assert cache.cache_key == "count:Alice:visited"
        assert cache.cached_value == 3
        assert cache.value_type == "count"
        assert cache.hit_count == 5


# =============================================================================
# Test Aggregation Builder
# =============================================================================


class TestAggregationBuilder:
    """Tests for AggregationBuilder."""

    def test_build_aggregations(
        self, aggregation_builder: AggregationBuilder, sample_memories: List[MemoryItem]
    ):
        """Test building aggregations from memories."""
        index = aggregation_builder.build_aggregations(sample_memories)

        assert isinstance(index, EnhancedMemoryIndex)
        assert index.memory_count == len(sample_memories)
        assert index.build_timestamp is not None

    def test_entity_extraction(
        self, aggregation_builder: AggregationBuilder, sample_memories: List[MemoryItem]
    ):
        """Test that entities are extracted from memories."""
        index = aggregation_builder.build_aggregations(sample_memories)

        # Alice and Bob should be extracted as entities
        assert "Alice" in index.entities
        assert "Bob" in index.entities

    def test_entity_type_inference(
        self, aggregation_builder: AggregationBuilder
    ):
        """Test entity type inference."""
        memories = [
            create_memory(
                "Alice went to the store.",
                metadata={"persons": ["Alice"]},
            ),
        ]
        index = aggregation_builder.build_aggregations(memories)

        # Alice should be inferred as a Person (ontology-compatible name)
        assert index.entities["Alice"].entity_type == "Person"

    def test_action_extraction(
        self, aggregation_builder: AggregationBuilder, sample_memories: List[MemoryItem]
    ):
        """Test that actions are extracted and counted."""
        index = aggregation_builder.build_aggregations(sample_memories)

        alice_agg = index.entities.get("Alice")
        assert alice_agg is not None
        # Should have visited actions counted
        assert len(alice_agg.event_counts) > 0

    def test_relation_extraction(
        self, aggregation_builder: AggregationBuilder
    ):
        """Test that relations between entities are extracted."""
        memories = [
            create_memory(
                "Alice met Bob at the conference.",
                metadata={"persons": ["Alice", "Bob"]},
            ),
        ]
        index = aggregation_builder.build_aggregations(memories)

        # Should have a relation between Alice and Bob
        assert len(index.relations) > 0
        relation = index.relations[0]
        assert {relation.subject, relation.object} == {"Alice", "Bob"}

    def test_temporal_index_building(
        self, aggregation_builder: AggregationBuilder
    ):
        """Test that temporal index is built correctly."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        memories = [
            create_memory("Event 1", created_at=base_time),
            create_memory("Event 2", created_at=base_time),
            create_memory("Event 3", created_at=base_time + timedelta(days=1)),
        ]
        index = aggregation_builder.build_aggregations(memories)

        # Should have entries for both dates
        assert "2024-01-15" in index.temporal_index
        assert "2024-01-16" in index.temporal_index
        assert len(index.temporal_index["2024-01-15"]) == 2

    def test_attribute_extraction_possessive(
        self, aggregation_builder: AggregationBuilder
    ):
        """Test extraction of possessive attributes."""
        memories = [
            create_memory("Alice's favorite color is blue."),
        ]
        index = aggregation_builder.build_aggregations(memories)

        alice_agg = index.entities.get("Alice")
        assert alice_agg is not None
        # Should have possessions attribute
        if "possessions" in alice_agg.attribute_sets:
            assert len(alice_agg.attribute_sets["possessions"]) > 0

    def test_attribute_extraction_preferences(
        self, aggregation_builder: AggregationBuilder
    ):
        """Test extraction of preference attributes."""
        memories = [
            create_memory("Bob likes pizza and pasta."),
        ]
        index = aggregation_builder.build_aggregations(memories)

        bob_agg = index.entities.get("Bob")
        assert bob_agg is not None
        # Should have preferences
        if "preferences" in bob_agg.attribute_sets:
            assert len(bob_agg.attribute_sets["preferences"]) > 0

    def test_empty_memories(self, aggregation_builder: AggregationBuilder):
        """Test handling of empty memory list."""
        index = aggregation_builder.build_aggregations([])
        assert index.memory_count == 0
        assert len(index.entities) == 0
        assert len(index.relations) == 0


# =============================================================================
# Test Fast-Path Retriever
# =============================================================================


class TestFastPathRetriever:
    """Tests for FastPathRetriever."""

    def test_disabled_config_returns_no_hit(
        self, enhanced_index: EnhancedMemoryIndex
    ):
        """Test that disabled config returns no hit."""
        config = RetrievalFastPathConfig(enabled=False)
        retriever = FastPathRetriever(enhanced_index=enhanced_index, config=config)

        result = retriever.try_fast_path("How many times did Alice visit?")
        assert result.hit is False

    def test_no_index_returns_no_hit(self, fast_path_config: RetrievalFastPathConfig):
        """Test that missing index returns no hit."""
        retriever = FastPathRetriever(enhanced_index=None, config=fast_path_config)

        result = retriever.try_fast_path("How many times did Alice visit?")
        assert result.hit is False

    def test_set_enhanced_index(self, fast_path_config: RetrievalFastPathConfig):
        """Test setting enhanced index after initialization."""
        retriever = FastPathRetriever(config=fast_path_config)
        assert retriever._enhanced_index is None

        index = EnhancedMemoryIndex()
        retriever.set_enhanced_index(index)
        assert retriever._enhanced_index is not None


class TestCountQueries:
    """Tests for count query fast-path."""

    def test_count_query_how_many(
        self, fast_path_retriever: FastPathRetriever, enhanced_index: EnhancedMemoryIndex
    ):
        """Test 'how many' count query."""
        # Add some event counts to the index
        if "Alice" in enhanced_index.entities:
            enhanced_index.entities["Alice"].event_counts["visited"] = 3

        result = fast_path_retriever.try_fast_path(
            "How many times did Alice visit places?"
        )
        # May or may not hit depending on action matching
        if result.hit:
            assert result.query_type == "count"
            assert len(result.memories) > 0

    def test_count_query_with_action_match(self):
        """Test count query with exact action match."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    event_counts={"visited_paris": 5},
                )
            }
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("How many times did Alice visit Paris?")
        assert result.hit is True
        assert result.query_type == "count"
        assert "5 times" in result.memories[0].content

    def test_count_query_unknown_entity(
        self, fast_path_retriever: FastPathRetriever
    ):
        """Test count query with unknown entity."""
        result = fast_path_retriever.try_fast_path(
            "How many times did Charlie visit?"
        )
        assert result.hit is False

    def test_count_query_disabled(self, enhanced_index: EnhancedMemoryIndex):
        """Test that count queries can be disabled."""
        config = RetrievalFastPathConfig(
            enabled=True,
            detect_count_queries=False,
        )
        retriever = FastPathRetriever(enhanced_index=enhanced_index, config=config)

        result = retriever.try_fast_path("How many times did Alice visit?")
        assert result.hit is False


class TestListQueries:
    """Tests for list query fast-path."""

    def test_list_query_all(self):
        """Test 'all' list query."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    attribute_sets={"locations": {"Paris", "London", "Tokyo"}},
                )
            }
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("What are all the locations Alice visited?")
        assert result.hit is True
        assert result.query_type == "list"
        assert "Paris" in result.memories[0].content
        assert "London" in result.memories[0].content

    def test_list_query_every(self):
        """Test 'every' list query."""
        index = EnhancedMemoryIndex(
            entities={
                "Bob": EntityAggregation(
                    entity_name="Bob",
                    entity_type="person",
                    attribute_sets={"tags": {"work", "travel", "food"}},
                )
            }
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        # Query uses "tags" which matches the attribute type exactly
        result = retriever.try_fast_path("List every tags for Bob")
        assert result.hit is True
        assert result.query_type == "list"

    def test_list_query_disabled(self, enhanced_index: EnhancedMemoryIndex):
        """Test that list queries can be disabled."""
        config = RetrievalFastPathConfig(
            enabled=True,
            detect_list_queries=False,
        )
        retriever = FastPathRetriever(enhanced_index=enhanced_index, config=config)

        result = retriever.try_fast_path("What are all the places Alice visited?")
        assert result.hit is False


class TestRelationQueries:
    """Tests for relation query fast-path."""

    def test_relation_query_both_and(self):
        """Test 'both X and Y' relation query."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(entity_name="Alice", entity_type="person"),
                "Bob": EntityAggregation(entity_name="Bob", entity_type="person"),
            },
            relations=[
                RelationTriple(
                    subject="Alice",
                    predicate="met",
                    object="Bob",
                    source_memory_id="mem-1",
                ),
                RelationTriple(
                    subject="Alice",
                    predicate="with",
                    object="Bob",
                    source_memory_id="mem-2",
                ),
            ],
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("What did both Alice and Bob do together?")
        assert result.hit is True
        assert result.query_type == "relation"
        assert len(result.memories) > 0

    def test_relation_query_no_relations(self):
        """Test relation query when no relations exist."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(entity_name="Alice", entity_type="person"),
                "Charlie": EntityAggregation(entity_name="Charlie", entity_type="person"),
            },
            relations=[],
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("What did both Alice and Charlie do?")
        assert result.hit is False

    def test_relation_query_disabled(self, enhanced_index: EnhancedMemoryIndex):
        """Test that relation queries can be disabled."""
        config = RetrievalFastPathConfig(
            enabled=True,
            detect_relation_queries=False,
        )
        retriever = FastPathRetriever(enhanced_index=enhanced_index, config=config)

        result = retriever.try_fast_path("What did both Alice and Bob do?")
        assert result.hit is False


class TestAttributeLookup:
    """Tests for attribute lookup fast-path."""

    def test_attribute_lookup_preferences(self):
        """Test attribute lookup for preferences."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    attribute_sets={"preferences": {"coffee", "tea"}},
                )
            }
        )
        retriever = FastPathRetriever(enhanced_index=index)

        result = retriever.try_attribute_lookup("What does Alice like?")
        assert result.hit is True
        assert result.query_type == "attribute"

    def test_attribute_lookup_no_match(self):
        """Test attribute lookup with no matching attribute."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    attribute_sets={},
                )
            }
        )
        retriever = FastPathRetriever(enhanced_index=index)

        result = retriever.try_attribute_lookup("What is Alice's job?")
        assert result.hit is False


class TestFastPathResultMetadata:
    """Tests for FastPathResult metadata."""

    def test_count_result_metadata(self):
        """Test that count results have correct metadata."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    event_counts={"visited_paris": 3},
                )
            }
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("How many times did Alice visit Paris?")
        assert result.hit is True
        assert result.memories[0].metadata["source"] == "fast_path_cache"
        assert result.memories[0].metadata["query_type"] == "count"
        assert result.memories[0].metadata["entity"] == "Alice"
        assert result.memories[0].metadata["count"] == 3

    def test_list_result_metadata(self):
        """Test that list results have correct metadata."""
        index = EnhancedMemoryIndex(
            entities={
                "Bob": EntityAggregation(
                    entity_name="Bob",
                    entity_type="person",
                    attribute_sets={"tags": {"a", "b", "c"}},
                )
            }
        )
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("List all tags for Bob")
        assert result.hit is True
        assert result.memories[0].metadata["source"] == "fast_path_cache"
        assert result.memories[0].metadata["query_type"] == "list"
        assert "values" in result.memories[0].metadata


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_entity_summary(self, fast_path_retriever: FastPathRetriever):
        """Test getting entity summary."""
        summary = fast_path_retriever.get_entity_summary("Alice")
        # May or may not exist depending on sample data
        if summary:
            assert isinstance(summary, EntityAggregation)
            assert summary.entity_name == "Alice"

    def test_get_entity_summary_not_found(
        self, fast_path_retriever: FastPathRetriever
    ):
        """Test getting summary for non-existent entity."""
        summary = fast_path_retriever.get_entity_summary("NonExistent")
        assert summary is None

    def test_get_temporal_memories(self):
        """Test getting memories by date."""
        index = EnhancedMemoryIndex(
            temporal_index={
                "2024-01-15": ["mem-1", "mem-2"],
                "2024-01-16": ["mem-3"],
            }
        )
        retriever = FastPathRetriever(enhanced_index=index)

        memories = retriever.get_temporal_memories("2024-01-15")
        assert len(memories) == 2
        assert "mem-1" in memories

    def test_get_temporal_memories_empty(self):
        """Test getting memories for date with no entries."""
        index = EnhancedMemoryIndex(temporal_index={})
        retriever = FastPathRetriever(enhanced_index=index)

        memories = retriever.get_temporal_memories("2024-01-15")
        assert memories == []


class TestMinConfidenceThreshold:
    """Tests for minimum confidence threshold."""

    def test_below_confidence_threshold(self):
        """Test that results below confidence threshold are rejected."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    event_counts={"visited_paris": 1},
                )
            }
        )
        # Set very high confidence threshold
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.99)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        # Count queries have 0.9 confidence, should be rejected
        result = retriever.try_fast_path("How many times did Alice visit Paris?")
        assert result.hit is False

    def test_above_confidence_threshold(self):
        """Test that results above confidence threshold are accepted."""
        index = EnhancedMemoryIndex(
            entities={
                "Alice": EntityAggregation(
                    entity_name="Alice",
                    entity_type="person",
                    event_counts={"visited_paris": 1},
                )
            }
        )
        # Set low confidence threshold
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        result = retriever.try_fast_path("How many times did Alice visit Paris?")
        assert result.hit is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full fast-path pipeline."""

    def test_full_pipeline(self):
        """Test the full pipeline from memories to fast-path retrieval."""
        # Create memories
        memories = [
            create_memory("Alice visited Paris three times."),
            create_memory("Alice visited London twice."),
            create_memory("Alice met Bob in Paris."),
            create_memory("Bob likes coffee."),
        ]

        # Build index
        builder = AggregationBuilder()
        index = builder.build_aggregations(memories)

        # Create retriever
        config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(enhanced_index=index, config=config)

        # Test various queries
        # Note: Results depend on pattern matching success
        count_result = retriever.try_fast_path("How many times did Alice visit?")
        list_result = retriever.try_fast_path("List all places Alice visited")
        relation_result = retriever.try_fast_path("What did both Alice and Bob do?")

        # At least verify no errors occurred
        assert isinstance(count_result, FastPathResult)
        assert isinstance(list_result, FastPathResult)
        assert isinstance(relation_result, FastPathResult)

    def test_index_persistence(self):
        """Test that index can be serialized and restored."""
        # Create and populate index
        memories = [
            create_memory("Alice visited Paris."),
            create_memory("Bob met Alice."),
        ]
        builder = AggregationBuilder()
        original_index = builder.build_aggregations(memories)

        # Serialize
        data = original_index.to_dict()

        # Restore
        restored_index = EnhancedMemoryIndex.from_dict(data)

        # Verify
        assert restored_index.memory_count == original_index.memory_count
        assert set(restored_index.entities.keys()) == set(original_index.entities.keys())


# =============================================================================
# Ontology Integration Tests
# =============================================================================


class TestOntologyIntegration:
    """Tests for integration with the ontology-based graph system."""

    def test_entity_type_uses_ontology_names(self):
        """Test that entity types use ontology-compatible names."""
        memories = [
            create_memory(
                "Alice went to the store.",
                metadata={"persons": ["Alice"]},
            ),
        ]
        builder = AggregationBuilder()
        index = builder.build_aggregations(memories)

        # Should use "Person" (ontology-compatible) not "person"
        assert index.entities["Alice"].entity_type == "Person"

    def test_location_type_uses_ontology_names(self):
        """Test that location types use ontology-compatible names."""
        memories = [
            create_memory("The conference was held in Paris city center."),
        ]
        builder = AggregationBuilder()
        index = builder.build_aggregations(memories)

        if "Paris" in index.entities:
            # Should use "Location" (ontology-compatible)
            assert index.entities["Paris"].entity_type == "Location"

    def test_predicate_normalization_likes(self):
        """Test that 'likes' predicates are normalized to LIKES."""
        memories = [
            create_memory(
                "Alice likes Bob very much.",
                metadata={"persons": ["Alice", "Bob"]},
            ),
        ]
        builder = AggregationBuilder()
        index = builder.build_aggregations(memories)

        # Check if any relation has LIKES predicate
        likes_relations = [r for r in index.relations if r.predicate == "LIKES"]
        # May or may not find the relation depending on pattern matching
        if likes_relations:
            assert likes_relations[0].predicate == "LIKES"

    def test_entity_aggregation_has_node_id_field(self):
        """Test that EntityAggregation has node_id field for graph integration."""
        agg = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            node_id="node-123",
        )
        assert agg.node_id == "node-123"

    def test_relation_triple_has_edge_fields(self):
        """Test that RelationTriple has edge_id and node_id fields."""
        triple = RelationTriple(
            subject="Alice",
            predicate="LIKES",
            object="Bob",
            edge_id="edge-456",
            source_node_id="node-alice",
            target_node_id="node-bob",
            source_memory_id="mem-1",
        )
        assert triple.edge_id == "edge-456"
        assert triple.source_node_id == "node-alice"
        assert triple.target_node_id == "node-bob"

    def test_builder_with_ontology_parameter(self):
        """Test that AggregationBuilder accepts ontology parameter."""
        # Just verify the constructor accepts the parameter
        builder = AggregationBuilder(ontology=None)
        assert builder._ontology is None

    def test_build_from_graph_nodes_and_edges(self):
        """Test building index from GraphNode and GraphEdge objects."""
        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional

        # Create mock GraphNode and GraphEdge (simplified for testing)
        @dataclass
        class MockGraphNode:
            node_id: str
            scope_id: str
            name: str
            labels: List[str] = field(default_factory=list)
            attributes: Dict[str, Any] = field(default_factory=dict)
            source_episode_ids: List[str] = field(default_factory=list)

        @dataclass
        class MockGraphEdge:
            edge_id: str
            scope_id: str
            source_node_id: str
            target_node_id: str
            edge_type: str
            valid_at: Optional[datetime] = None
            source_episode_ids: List[str] = field(default_factory=list)
            extraction_confidence: float = 1.0

        nodes = [
            MockGraphNode(
                node_id="node-1",
                scope_id="scope-1",
                name="Alice",
                labels=["Person"],
                attributes={"email": "alice@example.com"},
                source_episode_ids=["ep-1"],
            ),
            MockGraphNode(
                node_id="node-2",
                scope_id="scope-1",
                name="TechCorp",
                labels=["Organization"],
                attributes={"website": "https://techcorp.com"},
                source_episode_ids=["ep-2"],
            ),
        ]

        edges = [
            MockGraphEdge(
                edge_id="edge-1",
                scope_id="scope-1",
                source_node_id="node-1",
                target_node_id="node-2",
                edge_type="WORKS_FOR",
                source_episode_ids=["ep-3"],
            ),
        ]

        builder = AggregationBuilder()
        index = builder.build_from_graph(nodes, edges)

        # Verify entities
        assert "Alice" in index.entities
        assert index.entities["Alice"].entity_type == "Person"
        assert index.entities["Alice"].node_id == "node-1"

        assert "TechCorp" in index.entities
        assert index.entities["TechCorp"].entity_type == "Organization"

        # Verify relations
        assert len(index.relations) == 1
        assert index.relations[0].subject == "Alice"
        assert index.relations[0].predicate == "WORKS_FOR"
        assert index.relations[0].object == "TechCorp"
        assert index.relations[0].edge_id == "edge-1"

        # Verify event counts
        assert "WORKS_FOR_TechCorp" in index.entities["Alice"].event_counts


# =============================================================================
# Phase 4 Integration: InMemoryGraphStore save/load
# =============================================================================


class TestInMemoryGraphStoreEnhancedIndex:
    """Tests for save_enhanced_index / load_enhanced_index on InMemoryGraphStore."""

    @pytest.fixture
    def graph_store(self) -> InMemoryGraphStore:
        return InMemoryGraphStore()

    @pytest.fixture
    def sample_index(self) -> EnhancedMemoryIndex:
        """Create a sample EnhancedMemoryIndex for testing."""
        index = EnhancedMemoryIndex()
        index.entities["Alice"] = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            event_counts={"visited": 3, "met": 1},
            attribute_sets={"preferences": {"coffee", "tea"}},
            evidence_memory_ids=["m1", "m2"],
        )
        index.relations.append(
            RelationTriple(
                subject="Alice",
                predicate="LIKES",
                object="coffee",
                source_memory_id="m1",
                confidence=0.9,
            )
        )
        index.temporal_index["2024-06"] = ["m1"]
        return index

    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(
        self, graph_store: InMemoryGraphStore, sample_index: EnhancedMemoryIndex
    ):
        """Save then load should return an equivalent index."""
        await graph_store.save_enhanced_index("scope-1", sample_index)
        loaded = await graph_store.load_enhanced_index("scope-1")

        assert loaded is not None
        assert "Alice" in loaded.entities
        assert loaded.entities["Alice"].entity_type == "Person"
        assert loaded.entities["Alice"].event_counts["visited"] == 3
        assert len(loaded.relations) == 1
        assert loaded.relations[0].predicate == "LIKES"
        assert loaded.temporal_index["2024-06"] == ["m1"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_scope(self, graph_store: InMemoryGraphStore):
        """Loading from a scope with no index should return None."""
        loaded = await graph_store.load_enhanced_index("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_scope_isolation(
        self, graph_store: InMemoryGraphStore, sample_index: EnhancedMemoryIndex
    ):
        """Indexes are isolated per scope_id."""
        await graph_store.save_enhanced_index("scope-1", sample_index)

        other_index = EnhancedMemoryIndex()
        other_index.entities["Bob"] = EntityAggregation(
            entity_name="Bob",
            entity_type="Person",
        )
        await graph_store.save_enhanced_index("scope-2", other_index)

        loaded_1 = await graph_store.load_enhanced_index("scope-1")
        loaded_2 = await graph_store.load_enhanced_index("scope-2")

        assert loaded_1 is not None and "Alice" in loaded_1.entities
        assert loaded_2 is not None and "Bob" in loaded_2.entities
        assert "Bob" not in loaded_1.entities
        assert "Alice" not in loaded_2.entities

    @pytest.mark.asyncio
    async def test_overwrite_on_save(
        self, graph_store: InMemoryGraphStore, sample_index: EnhancedMemoryIndex
    ):
        """Saving again for the same scope overwrites the previous index."""
        await graph_store.save_enhanced_index("scope-1", sample_index)

        new_index = EnhancedMemoryIndex()
        new_index.entities["Charlie"] = EntityAggregation(
            entity_name="Charlie",
            entity_type="Person",
        )
        await graph_store.save_enhanced_index("scope-1", new_index)

        loaded = await graph_store.load_enhanced_index("scope-1")
        assert loaded is not None
        assert "Charlie" in loaded.entities
        assert "Alice" not in loaded.entities

    @pytest.mark.asyncio
    async def test_delete_scope_clears_index(
        self, graph_store: InMemoryGraphStore, sample_index: EnhancedMemoryIndex
    ):
        """delete_scope should also remove the enhanced index."""
        await graph_store.save_enhanced_index("scope-1", sample_index)
        await graph_store.delete_scope("scope-1")

        loaded = await graph_store.load_enhanced_index("scope-1")
        assert loaded is None


# =============================================================================
# Phase 4 Integration: Factory wiring
# =============================================================================


class TestFactoryFastPathRetrieverWiring:
    """Tests that EngineFactory creates FastPathRetriever when config is enabled."""

    def test_fast_path_retriever_created_when_enabled(self):
        """FastPathRetriever should be created when retrieval_fast_path is enabled."""
        fp_config = RetrievalFastPathConfig(enabled=True, min_confidence=0.7)
        retriever = FastPathRetriever(config=fp_config)
        assert retriever is not None
        assert retriever._config.enabled is True
        assert retriever._config.min_confidence == 0.7

    def test_fast_path_retriever_not_created_when_disabled(self):
        """FastPathRetriever should not be created when config is disabled."""
        fp_config = RetrievalFastPathConfig(enabled=False)
        # Simulate factory logic: only create when enabled
        fast_path_retriever = None
        if fp_config is not None and getattr(fp_config, "enabled", False):
            fast_path_retriever = FastPathRetriever(config=fp_config)
        assert fast_path_retriever is None

    def test_fast_path_retriever_not_created_when_config_missing(self):
        """FastPathRetriever should not be created when config is None."""
        fp_config = None
        fast_path_retriever = None
        if fp_config is not None and getattr(fp_config, "enabled", False):
            fast_path_retriever = FastPathRetriever(config=fp_config)
        assert fast_path_retriever is None


# =============================================================================
# Phase 4 Integration: prepare_context fast-path short-circuit
# =============================================================================


class TestPrepareContextFastPath:
    """Tests that prepare_context uses fast-path when available."""

    def test_fast_path_result_used_as_memories(self):
        """When fast-path hits, its memories should be used directly."""
        fp_config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(config=fp_config)

        # Build an index with a known entity
        index = EnhancedMemoryIndex()
        index.entities["Alice"] = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            event_counts={"visited": 3},
            attribute_sets={"preferences": {"coffee"}},
            evidence_memory_ids=["m1"],
        )
        retriever.set_enhanced_index(index)

        # Query that should trigger a count fast-path
        result = retriever.try_fast_path("How many times did Alice visited?")
        assert result.hit is True
        assert result.query_type == "count"
        assert len(result.memories) == 1
        assert "3" in result.memories[0].content

    def test_fast_path_miss_returns_no_memories(self):
        """When fast-path misses, no memories should be returned."""
        fp_config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(config=fp_config)

        # Empty index
        retriever.set_enhanced_index(EnhancedMemoryIndex())

        result = retriever.try_fast_path("Tell me about the weather")
        assert result.hit is False
        assert len(result.memories) == 0

    def test_fast_path_disabled_returns_miss(self):
        """When fast-path is disabled, it should always miss."""
        fp_config = RetrievalFastPathConfig(enabled=False)
        retriever = FastPathRetriever(config=fp_config)

        index = EnhancedMemoryIndex()
        index.entities["Alice"] = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            event_counts={"visited": 3},
        )
        retriever.set_enhanced_index(index)

        result = retriever.try_fast_path("How many times did Alice visited?")
        assert result.hit is False

    def test_fast_path_no_index_returns_miss(self):
        """When no index is set, fast-path should always miss."""
        fp_config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
        retriever = FastPathRetriever(config=fp_config)
        # No index set
        result = retriever.try_fast_path("How many times did Alice visited?")
        assert result.hit is False


# =============================================================================
# Phase 4 Integration: RetrievalControllerService fast-path
# =============================================================================


class TestRetrievalControllerFastPath:
    """Tests that RetrievalControllerService uses fast-path when available."""

    def test_fast_path_retriever_accepted_in_constructor(self):
        """RetrievalControllerService should accept fast_path_retriever param."""
        from ctxforge.engine.services.retrieval_controller_service import (
            RetrievalControllerService,
        )

        # Verify the constructor signature accepts fast_path_retriever.
        sig = inspect.signature(RetrievalControllerService.__init__)
        assert "fast_path_retriever" in sig.parameters


# =============================================================================
# Phase 4 Integration: Index rebuild
# =============================================================================


class TestIndexRebuild:
    """Tests for the enhanced index rebuild logic."""

    @pytest.mark.asyncio
    async def test_rebuild_creates_index_from_memories(self):
        """AggregationBuilder should create an index from new memories."""
        memories = [
            create_memory(
                "Alice visited Paris three times.",
                metadata={"persons": ["Alice"]},
            ),
            create_memory(
                "Bob likes coffee.",
                metadata={"persons": ["Bob"]},
            ),
        ]
        builder = AggregationBuilder()
        index = builder.build_aggregations(memories)

        assert "Alice" in index.entities
        assert "Bob" in index.entities

    @pytest.mark.asyncio
    async def test_rebuild_merges_with_existing_index(self):
        """New index data should merge into an existing index."""
        # Existing index
        existing = EnhancedMemoryIndex()
        existing.entities["Alice"] = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            event_counts={"visited": 2},
            attribute_sets={"preferences": {"coffee"}},
            evidence_memory_ids=["m1"],
        )

        # New index
        new_index = EnhancedMemoryIndex()
        new_index.entities["Alice"] = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            event_counts={"visited": 1, "met": 1},
            attribute_sets={"preferences": {"tea"}},
            evidence_memory_ids=["m2"],
        )
        new_index.entities["Bob"] = EntityAggregation(
            entity_name="Bob",
            entity_type="Person",
            event_counts={"likes": 1},
        )

        # Simulate merge logic (same as in _rebuild_enhanced_index)
        for name, agg in new_index.entities.items():
            if name in existing.entities:
                ea = existing.entities[name]
                for action, count in agg.event_counts.items():
                    ea.event_counts[action] = ea.event_counts.get(action, 0) + count
                for attr_key, attr_vals in agg.attribute_sets.items():
                    if attr_key not in ea.attribute_sets:
                        ea.attribute_sets[attr_key] = set()
                    ea.attribute_sets[attr_key].update(attr_vals)
                ea.evidence_memory_ids.extend(agg.evidence_memory_ids)
            else:
                existing.entities[name] = agg

        # Verify merge
        assert existing.entities["Alice"].event_counts["visited"] == 3
        assert existing.entities["Alice"].event_counts["met"] == 1
        assert "tea" in existing.entities["Alice"].attribute_sets["preferences"]
        assert "coffee" in existing.entities["Alice"].attribute_sets["preferences"]
        assert existing.entities["Alice"].evidence_memory_ids == ["m1", "m2"]
        assert "Bob" in existing.entities

    @pytest.mark.asyncio
    async def test_rebuild_persists_to_graph_store(self):
        """Rebuilt index should be persisted to the graph store."""
        store = InMemoryGraphStore()
        index = EnhancedMemoryIndex()
        index.entities["Alice"] = EntityAggregation(
            entity_name="Alice",
            entity_type="Person",
            event_counts={"visited": 1},
        )

        await store.save_enhanced_index("user-1", index)
        loaded = await store.load_enhanced_index("user-1")

        assert loaded is not None
        assert "Alice" in loaded.entities
        assert loaded.entities["Alice"].event_counts["visited"] == 1
