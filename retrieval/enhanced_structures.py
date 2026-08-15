"""
Enhanced Data Structures for Fast-Path Retrieval.

Provides general-purpose caching and indexing without hard-coded domain logic.
Based on AriadneMem's Section 2.3 - Fast Paths and Enhanced Caching.

These structures are designed to work alongside the existing ontology-based
graph system (GraphNode, GraphEdge, GraphOntology) by providing lightweight
aggregation views for O(1) fast-path queries.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class EntityAggregation(BaseModel):
    """
    Lightweight entity-level aggregation for fast-path queries.

    This is a complementary structure to GraphNode that provides pre-computed
    aggregates for common query patterns (counts, lists, temporal sequences).

    Alignment with existing ontology system:
    - entity_name: Corresponds to GraphNode.name
    - entity_type: Should match GraphOntology.entity_types keys (e.g., "Person", "Location")
    - node_id: Optional reference to the corresponding GraphNode.node_id

    Attributes:
        entity_name: Name of the entity (e.g., "Alice", "Paris")
        entity_type: Type from ontology (e.g., "Person", "Location", "Organization")
        node_id: Optional reference to corresponding GraphNode
        event_counts: Action/event type -> count (e.g., {"visited_beach": 3})
        attribute_sets: Attribute type -> set of values (e.g., {"locations": {"Paris", "London"}})
        temporal_sequences: Event type -> (first_time, last_time, count)
        evidence_memory_ids: IDs of memories supporting this aggregation
    """

    entity_name: str
    entity_type: str = "entity"  # Should align with GraphOntology.entity_types
    node_id: Optional[str] = None  # Reference to GraphNode.node_id if available

    # Count-based aggregations (learned from data)
    event_counts: Dict[str, int] = Field(default_factory=dict)

    # Set-based aggregations (complete lists)
    attribute_sets: Dict[str, Set[str]] = Field(default_factory=dict)

    # Temporal aggregations (first/last occurrences)
    # Key: event type, Value: (first_time, last_time, count)
    temporal_sequences: Dict[str, Tuple[str, str, int]] = Field(default_factory=dict)

    # Supporting memory IDs
    evidence_memory_ids: List[str] = Field(default_factory=list)

    class Config:
        # Allow sets to be JSON serializable
        json_encoders = {set: list}


class RelationTriple(BaseModel):
    """
    Lightweight relation representation for fast-path queries.

    This is a complementary structure to GraphEdge that provides a simplified
    (subject, predicate, object) view for quick relationship lookups.

    Alignment with existing ontology system:
    - subject/object: Correspond to GraphNode.name values
    - predicate: Should align with GraphOntology.edge_types keys when possible
    - edge_id: Optional reference to the corresponding GraphEdge.edge_id
    - source_node_id/target_node_id: Optional references to GraphNode.node_id

    Example: ("Alice", "LIKES", "Paris")
    """

    subject: str
    predicate: str  # Should align with GraphOntology.edge_types when possible
    object: str

    # References to graph system (optional)
    edge_id: Optional[str] = None  # Reference to GraphEdge.edge_id if available
    source_node_id: Optional[str] = None  # Reference to source GraphNode.node_id
    target_node_id: Optional[str] = None  # Reference to target GraphNode.node_id

    # Context
    timestamp: Optional[str] = None
    location: Optional[str] = None

    # Evidence
    source_memory_id: str
    confidence: float = 1.0


class QueryCache(BaseModel):
    """
    Cache for frequently accessed query patterns.
    Learns common access patterns without hard-coding.
    """

    cache_key: str  # Hash of query pattern

    # Cached results (flexible structure)
    cached_value: Any
    value_type: str  # "count", "list", "entity", "relation"

    # Metadata
    hit_count: int = 0
    last_accessed: Optional[str] = None

    # Supporting evidence
    source_memory_ids: List[str] = Field(default_factory=list)


class EnhancedMemoryIndex(BaseModel):
    """
    Container for all enhanced indexing structures.
    All components are optional and built on-demand.

    This index enables O(1) fast-path lookups for:
    - Count queries ("how many times did X...")
    - List queries ("what are all the...")
    - Relation queries ("what do X and Y have in common")
    """

    # Entity-level aggregations
    entities: Dict[str, EntityAggregation] = Field(default_factory=dict)

    # Relation triples (for relationship queries)
    relations: List[RelationTriple] = Field(default_factory=list)

    # Query cache (for repeated patterns)
    query_cache: Dict[str, QueryCache] = Field(default_factory=dict)

    # Temporal index (for time-range queries)
    # date (YYYY-MM-DD) -> list of memory IDs
    temporal_index: Dict[str, List[str]] = Field(default_factory=dict)

    # Statistics
    build_timestamp: Optional[str] = None
    memory_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "entities": {k: v.model_dump() for k, v in self.entities.items()},
            "relations": [r.model_dump() for r in self.relations],
            "query_cache": {k: v.model_dump() for k, v in self.query_cache.items()},
            "temporal_index": self.temporal_index,
            "build_timestamp": self.build_timestamp,
            "memory_count": self.memory_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EnhancedMemoryIndex":
        """Load from dictionary."""
        # Handle sets in attribute_sets
        entities = {}
        for k, v in data.get("entities", {}).items():
            # Convert lists back to sets for attribute_sets
            if "attribute_sets" in v:
                v["attribute_sets"] = {
                    attr_k: set(attr_v) if isinstance(attr_v, list) else attr_v
                    for attr_k, attr_v in v["attribute_sets"].items()
                }
            entities[k] = EntityAggregation(**v)

        return cls(
            entities=entities,
            relations=[RelationTriple(**r) for r in data.get("relations", [])],
            query_cache={
                k: QueryCache(**v) for k, v in data.get("query_cache", {}).items()
            },
            temporal_index=data.get("temporal_index", {}),
            build_timestamp=data.get("build_timestamp"),
            memory_count=data.get("memory_count", 0),
        )
