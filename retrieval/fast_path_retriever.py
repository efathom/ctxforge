"""
Fast-Path Retriever - O(1) cache lookup for common query patterns.

Based on AriadneMem's ariadne_graph_retriever.py Section 2.3 - Fast Paths.
Handles count queries, list queries, and relationship queries without
requiring full semantic search.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ctxforge.config.base import RetrievalFastPathConfig
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.retrieval.enhanced_structures import (
    EnhancedMemoryIndex,
    EntityAggregation,
    RelationTriple,
)

logger = logging.getLogger(__name__)


@dataclass
class FastPathResult:
    """
    Result from fast-path lookup.

    Attributes:
        hit: Whether the fast-path was successful
        query_type: Type of query matched ("count", "list", "relation", None)
        memories: Synthetic memory items with the answer
        confidence: Confidence score for the match
        cache_key: Key used for cache lookup
    """

    hit: bool = False
    query_type: Optional[str] = None
    memories: List[MemoryItem] = field(default_factory=list)
    confidence: float = 0.0
    cache_key: Optional[str] = None


class FastPathRetriever:
    """
    Fast-path retriever for O(1) cache lookup of common query patterns.

    Handles three types of queries without full semantic search:
    1. Count queries: "how many times did X...", "count of X"
    2. List queries: "what are all the...", "list all X"
    3. Relation queries: "what do X and Y have in common", "both X and Y"

    Example:
        retriever = FastPathRetriever(enhanced_index, config)
        result = retriever.try_fast_path("How many times did Alice visit Paris?")
        if result.hit:
            print(f"Fast-path hit: {result.memories[0].content}")
    """

    def __init__(
        self,
        enhanced_index: Optional[EnhancedMemoryIndex] = None,
        config: Optional[RetrievalFastPathConfig] = None,
    ):
        """
        Initialize the fast-path retriever.

        Args:
            enhanced_index: Pre-built enhanced memory index for O(1) lookups
            config: Configuration for fast-path behavior
        """
        self._enhanced_index = enhanced_index
        self._config = config or RetrievalFastPathConfig()

    def set_enhanced_index(self, enhanced_index: EnhancedMemoryIndex) -> None:
        """
        Set or update the enhanced index.

        Args:
            enhanced_index: The enhanced memory index to use
        """
        self._enhanced_index = enhanced_index
        entity_count = len(enhanced_index.entities) if enhanced_index else 0
        logger.info(f"Enhanced index loaded with {entity_count} entities")

    def try_fast_path(self, query: str) -> FastPathResult:
        """
        Attempt fast-path lookup for the query.

        This is the main entry point. It tries each fast-path pattern
        in order and returns the first successful match.

        Args:
            query: The user query to process

        Returns:
            FastPathResult with hit=True if successful, hit=False otherwise
        """
        if not self._config.enabled or not self._enhanced_index:
            return FastPathResult(hit=False)

        query_lower = query.lower()

        # Pattern 1: Count queries ("how many times", "how many X")
        if self._config.detect_count_queries:
            if "how many" in query_lower or "count" in query_lower or "times" in query_lower:
                result = self._check_count_cache(query, query_lower)
                if result.hit and result.confidence >= self._config.min_confidence:
                    logger.info(f"Fast-path cache hit (count query): {result.cache_key}")
                    return result

        # Pattern 2: "All X" queries ("what are all the books")
        if self._config.detect_list_queries:
            if "all " in query_lower or "every " in query_lower or "list " in query_lower:
                result = self._check_list_cache(query, query_lower)
                if result.hit and result.confidence >= self._config.min_confidence:
                    logger.info(f"Fast-path cache hit (list query): {result.cache_key}")
                    return result

        # Pattern 3: Relationship queries ("both X and Y", "X and Y both")
        if self._config.detect_relation_queries:
            if "both" in query_lower and " and " in query_lower:
                result = self._check_relation_cache(query, query_lower)
                if result.hit and result.confidence >= self._config.min_confidence:
                    logger.info(f"Fast-path cache hit (relation query): {result.cache_key}")
                    return result

        return FastPathResult(hit=False)

    def _check_count_cache(
        self, query: str, query_lower: str
    ) -> FastPathResult:
        """
        Check if count query can be answered from cache.

        Handles queries like:
        - "How many times did Alice visit Paris?"
        - "Count of books read by Bob"
        """
        # Extract entity name
        entity_name = self._extract_target_entity(query)
        if not entity_name or entity_name not in self._enhanced_index.entities:
            return FastPathResult(hit=False)

        entity_agg = self._enhanced_index.entities[entity_name]

        # Look for action in query
        for action, count in entity_agg.event_counts.items():
            # Match action keywords in query
            action_words = action.replace("_", " ").split()
            if any(word in query_lower for word in action_words):
                # Create synthetic memory with the count
                synthetic_fact = f"{entity_name} {action.replace('_', ' ')} {count} times"
                memory = MemoryItem(
                    content=synthetic_fact,
                    user_id="_fast_path_cache",
                    type=MemoryType.SEMANTIC,
                    source=MemorySource.SYSTEM,
                    tags=[entity_name, action, str(count)],
                    metadata={
                        "source": "fast_path_cache",
                        "query_type": "count",
                        "entity": entity_name,
                        "action": action,
                        "count": count,
                    },
                )
                return FastPathResult(
                    hit=True,
                    query_type="count",
                    memories=[memory],
                    confidence=0.9,
                    cache_key=f"count:{entity_name}:{action}",
                )

        return FastPathResult(hit=False)

    def _check_list_cache(
        self, query: str, query_lower: str
    ) -> FastPathResult:
        """
        Check if list query can be answered from cache.

        Handles queries like:
        - "What are all the books Alice read?"
        - "List all places Bob visited"
        """
        entity_name = self._extract_target_entity(query)
        if not entity_name or entity_name not in self._enhanced_index.entities:
            return FastPathResult(hit=False)

        entity_agg = self._enhanced_index.entities[entity_name]

        # Check attribute sets
        for attr_type, values in entity_agg.attribute_sets.items():
            if attr_type in query_lower or any(
                word in query_lower for word in attr_type.split("_")
            ):
                if values:
                    values_str = ", ".join(sorted(values))
                    synthetic_fact = f"{entity_name}'s {attr_type}: {values_str}"
                    memory = MemoryItem(
                        content=synthetic_fact,
                        user_id="_fast_path_cache",
                        type=MemoryType.SEMANTIC,
                        source=MemorySource.SYSTEM,
                        tags=[entity_name, attr_type] + list(values)[:5],
                        metadata={
                            "source": "fast_path_cache",
                            "query_type": "list",
                            "entity": entity_name,
                            "attribute_type": attr_type,
                            "values": list(values),
                        },
                    )
                    return FastPathResult(
                        hit=True,
                        query_type="list",
                        memories=[memory],
                        confidence=0.85,
                        cache_key=f"list:{entity_name}:{attr_type}",
                    )

        return FastPathResult(hit=False)

    def _check_relation_cache(
        self, query: str, query_lower: str
    ) -> FastPathResult:
        """
        Check if relationship query can be answered from relation triples.

        Handles queries like:
        - "What do Alice and Bob both like?"
        - "Both Alice and Bob visited..."
        """
        # Extract entity names that exist in the index
        all_matches = re.findall(r"\b([A-Z][a-z]+)\b", query)
        entities = [e for e in all_matches if e in self._enhanced_index.entities]
        if len(entities) < 2:
            return FastPathResult(hit=False)

        entity1, entity2 = entities[0], entities[1]

        # Find relations between these entities
        matching_relations: List[RelationTriple] = []
        for relation in self._enhanced_index.relations:
            if (relation.subject == entity1 and relation.object == entity2) or (
                relation.subject == entity2 and relation.object == entity1
            ):
                matching_relations.append(relation)

        if matching_relations:
            # Create memories from relation triples
            memories = []
            for rel in matching_relations[:3]:  # Limit to 3 relations
                synthetic_fact = f"{rel.subject} {rel.predicate} {rel.object}"
                memory = MemoryItem(
                    content=synthetic_fact,
                    user_id="_fast_path_cache",
                    type=MemoryType.SEMANTIC,
                    source=MemorySource.SYSTEM,
                    tags=[rel.subject, rel.predicate, rel.object],
                    metadata={
                        "source": "fast_path_cache",
                        "query_type": "relation",
                        "subject": rel.subject,
                        "predicate": rel.predicate,
                        "object": rel.object,
                        "timestamp": rel.timestamp,
                    },
                )
                memories.append(memory)

            return FastPathResult(
                hit=True,
                query_type="relation",
                memories=memories,
                confidence=0.8,
                cache_key=f"relation:{entity1}:{entity2}",
            )

        return FastPathResult(hit=False)

    def _extract_target_entity(self, query: str) -> Optional[str]:
        """
        Extract the target entity name from the query.

        Uses capitalized word patterns to identify entity names.
        Returns the first entity name found that exists in the index.
        """
        if not self._enhanced_index:
            return None

        # Find all capitalized words (potential entity names)
        matches = re.findall(r"\b([A-Z][a-z]+)\b", query)
        for entity_name in matches:
            # Return first entity that exists in index
            if entity_name in self._enhanced_index.entities:
                return entity_name
        return None

    def try_attribute_lookup(self, query: str) -> FastPathResult:
        """
        Fast-path: Regex-based attribute lookup.

        Handles simple "X's attribute" queries via direct metadata lookup.
        No LLM call required.

        Args:
            query: The user query

        Returns:
            FastPathResult with the attribute value if found
        """
        if not self._enhanced_index:
            return FastPathResult(hit=False)

        query_lower = query.lower()

        # Extract person name from query that exists in index
        all_matches = re.findall(r"\b([A-Z][a-z]+)\b", query)
        person = None
        for match in all_matches:
            if match in self._enhanced_index.entities:
                person = match
                break

        if not person or person not in self._enhanced_index.entities:
            return FastPathResult(hit=False)

        entity_agg = self._enhanced_index.entities[person]

        # Check for attribute keywords in query
        attr_keywords = {
            "status": ["status", "relationship", "married", "single"],
            "job": ["job", "work", "occupation", "profession", "career"],
            "location": ["live", "from", "location", "city", "country"],
            "preference": ["like", "prefer", "favorite", "enjoy"],
        }

        for attr_type, keywords in attr_keywords.items():
            if any(kw in query_lower for kw in keywords):
                # Check if we have this attribute
                for stored_attr, values in entity_agg.attribute_sets.items():
                    if attr_type in stored_attr.lower() or any(
                        kw in stored_attr.lower() for kw in keywords
                    ):
                        if values:
                            values_str = ", ".join(sorted(values))
                            synthetic_fact = f"{person}'s {stored_attr}: {values_str}"
                            memory = MemoryItem(
                                content=synthetic_fact,
                                user_id="_fast_path_cache",
                                type=MemoryType.SEMANTIC,
                                source=MemorySource.SYSTEM,
                                tags=[person, stored_attr],
                                metadata={
                                    "source": "fast_path_attribute",
                                    "entity": person,
                                    "attribute": stored_attr,
                                    "values": list(values),
                                },
                            )
                            return FastPathResult(
                                hit=True,
                                query_type="attribute",
                                memories=[memory],
                                confidence=0.75,
                                cache_key=f"attr:{person}:{stored_attr}",
                            )

        return FastPathResult(hit=False)

    def get_entity_summary(self, entity_name: str) -> Optional[EntityAggregation]:
        """
        Get the aggregated summary for an entity.

        Args:
            entity_name: Name of the entity

        Returns:
            EntityAggregation if found, None otherwise
        """
        if not self._enhanced_index:
            return None
        return self._enhanced_index.entities.get(entity_name)

    def get_temporal_memories(self, date: str) -> List[str]:
        """
        Get memory IDs for a specific date.

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            List of memory IDs for that date
        """
        if not self._enhanced_index:
            return []
        return self._enhanced_index.temporal_index.get(date, [])
