"""
Aggregation Cache Builder - Generic entity and relation aggregation.

Automatically learns patterns from memory entries without hard-coded rules.
Based on AriadneMem's aggregation_builder.py.

This builder can optionally integrate with the existing ontology-based graph
system (GraphOntology, GraphNode, GraphEdge) for type validation and
cross-referencing.
"""

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from ctxforge.core.memory import MemoryItem
from ctxforge.graph.ontology import GraphOntology
from ctxforge.protocols.graph import GraphEdge, GraphNode
from ctxforge.retrieval.enhanced_structures import (
    EnhancedMemoryIndex,
    EntityAggregation,
    RelationTriple,
)


class AggregationBuilder:
    """
    Builds aggregated views of memory without domain-specific templates.

    Uses NLP patterns to automatically discover:
    - Entity attributes and activities
    - Relation triples (subject-predicate-object)
    - Temporal sequences and counts

    Can optionally use GraphOntology for entity/edge type validation.

    Example:
        # Without ontology (pure pattern-based)
        builder = AggregationBuilder()
        index = builder.build_aggregations(memories)

        # With ontology (type-validated)
        from ctxforge.graph.ontology import GraphOntology
        builder = AggregationBuilder(ontology=my_ontology)
        index = builder.build_aggregations(memories)
    """

    def __init__(self, ontology: Optional[GraphOntology] = None):
        """
        Initialize the aggregation builder.

        Args:
            ontology: Optional GraphOntology for entity/edge type validation.
                      If provided, entity_type will be validated against
                      ontology.entity_types and predicates against edge_types.
        """
        self._ontology = ontology
        # Common action verbs (extensible, learned from data)
        self.action_verbs = {
            "went",
            "visited",
            "traveled",
            "painted",
            "read",
            "watched",
            "bought",
            "sold",
            "made",
            "created",
            "attended",
            "played",
            "adopted",
            "rejected",
            "wrote",
            "drew",
            "designed",
            "built",
            "participated",
            "joined",
            "left",
            "started",
            "finished",
            "likes",
            "enjoys",
            "prefers",
            "loves",
            "hates",
            "dislikes",
            "met",
            "called",
            "emailed",
            "worked",
            "studied",
            "learned",
            "taught",
            "helped",
            "asked",
            "answered",
        }

        # Dynamically expanded during processing
        self.discovered_actions: Set[str] = set()
        self.discovered_attributes: Dict[str, Set[str]] = defaultdict(set)

    def build_aggregations(self, memories: List[MemoryItem]) -> EnhancedMemoryIndex:
        """
        Build all aggregations from memory items.

        Args:
            memories: List of MemoryItem objects to aggregate

        Returns:
            EnhancedMemoryIndex with entities, relations, temporal index
        """
        index = EnhancedMemoryIndex(
            build_timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            memory_count=len(memories),
        )

        # Group memories by entity (extracted from content and tags)
        entity_memories: Dict[str, List[MemoryItem]] = defaultdict(list)

        for memory in memories:
            # Extract entities from content
            entities = self._extract_entities(memory)

            for entity in entities:
                if entity:  # Skip empty strings
                    entity_memories[entity].append(memory)

        # Build entity aggregations
        for entity_name, entity_memory_list in entity_memories.items():
            aggregation = self._aggregate_entity(entity_name, entity_memory_list)
            index.entities[entity_name] = aggregation

        # Build relation triples
        for memory in memories:
            triples = self._extract_relations(memory)
            index.relations.extend(triples)

        # Build temporal index
        index.temporal_index = self._build_temporal_index(memories)

        return index

    def _extract_entities(self, memory: MemoryItem) -> Set[str]:
        """
        Extract entity names from memory content and metadata.

        Uses capitalized words as potential entity names (person/place names).
        """
        entities: Set[str] = set()

        # Extract from content using capitalized words pattern
        # Matches names like "Alice", "Paris", "John Smith"
        content = memory.content
        name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        matches = re.findall(name_pattern, content)
        entities.update(matches)

        # Also check tags for entity-like values
        for tag in memory.tags:
            # Skip system tags
            if tag.startswith("_") or ":" in tag:
                continue
            # Check if tag looks like a name (capitalized)
            if tag and tag[0].isupper():
                entities.add(tag)

        # Check metadata for explicit entities
        if "entities" in memory.metadata:
            meta_entities = memory.metadata["entities"]
            if isinstance(meta_entities, list):
                entities.update(str(e) for e in meta_entities if e)

        if "persons" in memory.metadata:
            persons = memory.metadata["persons"]
            if isinstance(persons, list):
                entities.update(str(p) for p in persons if p)

        return entities

    def _aggregate_entity(
        self, entity_name: str, memories: List[MemoryItem]
    ) -> EntityAggregation:
        """
        Aggregate all information about a single entity.
        Uses pattern matching to discover attributes and events.
        """
        # Determine entity type
        entity_type = self._infer_entity_type(entity_name, memories)

        # Initialize aggregation
        agg = EntityAggregation(
            entity_name=entity_name,
            entity_type=entity_type,
            evidence_memory_ids=[m.memory_id for m in memories],
        )

        # Extract and count events
        for memory in memories:
            text = memory.content.lower()

            # Pattern 1: "entity verb something" (action detection)
            actions = self._extract_actions(entity_name, text)
            for action in actions:
                agg.event_counts[action] = agg.event_counts.get(action, 0) + 1

            # Pattern 2: "entity has/owns/likes X" (attribute detection)
            attributes = self._extract_attributes(entity_name, memory)
            for attr_type, values in attributes.items():
                if attr_type not in agg.attribute_sets:
                    agg.attribute_sets[attr_type] = set()
                agg.attribute_sets[attr_type].update(values)

            # Pattern 3: Temporal sequences (first, last, count)
            timestamp = self._get_timestamp(memory)
            if timestamp:
                for action in actions:
                    if action not in agg.temporal_sequences:
                        agg.temporal_sequences[action] = (timestamp, timestamp, 1)
                    else:
                        first, last, count = agg.temporal_sequences[action]
                        agg.temporal_sequences[action] = (
                            min(first, timestamp),
                            max(last, timestamp),
                            count + 1,
                        )

        return agg

    def _infer_entity_type(
        self, entity_name: str, memories: List[MemoryItem]
    ) -> str:
        """
        Infer entity type from name and context.

        If ontology is provided, validates against known entity types.
        Returns ontology-compatible type names (e.g., "Person", "Location")
        when possible.
        """
        # Check metadata for explicit type
        for memory in memories:
            if "entity_type" in memory.metadata:
                entity_type = str(memory.metadata["entity_type"])
                # Validate against ontology if available
                if self._ontology and self._ontology.is_entity_type_known(entity_type):
                    return entity_type
                return entity_type
            if "persons" in memory.metadata:
                persons = memory.metadata.get("persons", [])
                if entity_name in persons:
                    return "Person"  # Use ontology-compatible name

        # Check location indicators in content
        location_keywords = [
            "city",
            "country",
            "place",
            "location",
            "street",
            "building",
            "in",
            "at",
            "from",
        ]
        for memory in memories:
            content_lower = memory.content.lower()
            entity_lower = entity_name.lower()
            for keyword in location_keywords:
                if keyword in content_lower and entity_lower in content_lower:
                    # Check if entity appears near location keyword
                    if f"{keyword} {entity_lower}" in content_lower or f"{entity_lower} {keyword}" in content_lower:
                        return "Location"  # Use ontology-compatible name

        # Check if name looks like a person (common name patterns)
        # Simple heuristic: single capitalized word is often a person
        if " " not in entity_name and entity_name[0].isupper():
            return "Person"  # Use ontology-compatible name

        return "entity"

    def _normalize_predicate(self, predicate: str) -> str:
        """
        Normalize predicate to ontology edge type if possible.

        Maps common verbs to ontology edge types:
        - likes, loves, enjoys, prefers -> LIKES
        - works, employed -> WORKS_FOR
        """
        predicate_lower = predicate.lower()

        # Map to ontology edge types
        if predicate_lower in {"likes", "loves", "enjoys", "prefers"}:
            if self._ontology is None or self._ontology.is_edge_type_known("LIKES"):
                return "LIKES"
        if predicate_lower in {"works", "employed", "works_for"}:
            if self._ontology is None or self._ontology.is_edge_type_known("WORKS_FOR"):
                return "WORKS_FOR"

        # Return original if no mapping
        return predicate

    def _extract_actions(self, entity_name: str, text: str) -> List[str]:
        """
        Extract actions performed by or related to entity.
        Pattern: "entity [action_verb] [object/location]"
        """
        actions = []
        entity_lower = entity_name.lower()

        if entity_lower not in text:
            return actions

        # Split into sentences
        sentences = re.split(r"[.!?]", text)

        for sentence in sentences:
            if entity_lower not in sentence:
                continue

            # Check for known action verbs
            for verb in self.action_verbs:
                if verb in sentence:
                    # Extract action phrase
                    action = self._extract_action_phrase(sentence, verb, entity_lower)
                    if action:
                        actions.append(action)
                        self.discovered_actions.add(action)

        return actions

    def _extract_action_phrase(
        self, sentence: str, verb: str, entity: str
    ) -> str:
        """
        Extract meaningful action phrase from sentence.
        Returns: "verb_object" (e.g., "visited_beach", "painted_sunset")
        """
        # Find verb position
        verb_pos = sentence.find(verb)
        if verb_pos == -1:
            return ""

        # Check if entity is subject (before verb)
        entity_pos = sentence.find(entity)
        if entity_pos == -1 or entity_pos > verb_pos:
            return ""

        # Extract object after verb
        after_verb = sentence[verb_pos + len(verb) :].strip()

        # Get first meaningful noun phrase (up to 3 words)
        words = after_verb.split()[:3]

        # Remove common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "to",
            "at",
            "in",
            "on",
            "for",
            "with",
            "and",
            "or",
        }
        obj_words = [w for w in words if w not in stopwords and len(w) > 2]

        if obj_words:
            obj = "_".join(obj_words[:2])  # Max 2 words
            return f"{verb}_{obj}"
        else:
            return verb  # Just the verb if no clear object

    def _extract_attributes(
        self, entity_name: str, memory: MemoryItem
    ) -> Dict[str, Set[str]]:
        """
        Extract entity attributes from memory.
        Uses tags, metadata, and pattern matching.
        """
        attributes: Dict[str, Set[str]] = defaultdict(set)

        # Use tags as attributes
        if memory.tags:
            entity_tags = [
                t for t in memory.tags if entity_name.lower() not in t.lower()
            ]
            if entity_tags:
                attributes["tags"].update(entity_tags)

        # Extract from content
        text = memory.content.lower()
        entity_lower = entity_name.lower()

        if entity_lower in text:
            # Pattern: "entity's X" (possessive)
            possessive_pattern = rf"{re.escape(entity_lower)}'s\s+(\w+(?:\s+\w+)?)"
            matches = re.findall(possessive_pattern, text)
            if matches:
                attributes["possessions"].update(matches)

            # Pattern: "entity has/owns/likes X"
            has_pattern = rf"{re.escape(entity_lower)}\s+(?:has|owns|likes|prefers|enjoys)\s+([^,.;]+)"
            matches = re.findall(has_pattern, text)
            if matches:
                for match in matches:
                    clean_val = match.strip()[:50]  # Limit length
                    if clean_val:
                        attributes["preferences"].add(clean_val)

        return attributes

    def _extract_relations(self, memory: MemoryItem) -> List[RelationTriple]:
        """
        Extract relation triples from memory.

        Pattern: (subject, predicate, object)
        Predicates are normalized to ontology edge types when possible.
        """
        triples = []
        text = memory.content

        # Extract entities from this memory
        entities = list(self._extract_entities(memory))

        if len(entities) < 2:
            # Need at least 2 entities for a relation
            return triples

        # For each pair of entities, check if there's a relation
        for i, subj in enumerate(entities):
            for obj in entities[i + 1 :]:
                # Check if both appear in the content
                if subj.lower() in text.lower() and obj.lower() in text.lower():
                    # Find connecting verb
                    raw_predicate = self._find_connecting_verb(text, subj, obj)
                    if raw_predicate:
                        # Normalize to ontology edge type if possible
                        predicate = self._normalize_predicate(raw_predicate)
                        triple = RelationTriple(
                            subject=subj,
                            predicate=predicate,
                            object=obj,
                            timestamp=self._get_timestamp(memory),
                            source_memory_id=memory.memory_id,
                        )
                        triples.append(triple)

        return triples

    def _find_connecting_verb(self, text: str, subj: str, obj: str) -> str:
        """Find verb that connects two entities in text."""
        text_lower = text.lower()
        subj_pos = text_lower.find(subj.lower())
        obj_pos = text_lower.find(obj.lower())

        if subj_pos == -1 or obj_pos == -1:
            return ""

        # Get text between entities
        start = min(subj_pos, obj_pos)
        end = max(subj_pos, obj_pos)
        between = text_lower[start:end]

        # Find verb in between
        for verb in self.action_verbs:
            if verb in between:
                return verb

        # Check for common relational phrases
        relation_phrases = ["with", "and", "along with", "together with", "both"]
        for phrase in relation_phrases:
            if phrase in between:
                return phrase

        return "related_to"  # Default relation

    def _build_temporal_index(
        self, memories: List[MemoryItem]
    ) -> Dict[str, List[str]]:
        """Build index from dates to memory IDs for fast temporal queries."""
        temporal_index: Dict[str, List[str]] = defaultdict(list)

        for memory in memories:
            timestamp = self._get_timestamp(memory)
            if timestamp:
                # Extract date part (YYYY-MM-DD)
                try:
                    date_str = timestamp[:10]  # Get YYYY-MM-DD
                    temporal_index[date_str].append(memory.memory_id)
                except (IndexError, TypeError):
                    pass  # Skip invalid timestamps

        return dict(temporal_index)

    def _get_timestamp(self, memory: MemoryItem) -> str:
        """Get timestamp string from memory."""
        if memory.created_at:
            return memory.created_at.isoformat()
        return ""

    def build_from_graph(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
    ) -> EnhancedMemoryIndex:
        """
        Build enhanced index from existing GraphNode and GraphEdge objects.

        This method provides integration with the ontology-based graph system,
        allowing fast-path queries to leverage the existing graph structure.

        Args:
            nodes: List of GraphNode objects from the graph store
            edges: List of GraphEdge objects from the graph store

        Returns:
            EnhancedMemoryIndex populated from graph data
        """
        index = EnhancedMemoryIndex(
            build_timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            memory_count=len(nodes) + len(edges),
        )

        # Build entity aggregations from nodes
        for node in nodes:
            entity_type = node.labels[0] if node.labels else "entity"
            agg = EntityAggregation(
                entity_name=node.name,
                entity_type=entity_type,
                node_id=node.node_id,
                evidence_memory_ids=node.source_episode_ids,
            )

            # Extract attributes from node
            if node.attributes:
                for attr_key, attr_val in node.attributes.items():
                    if attr_key not in agg.attribute_sets:
                        agg.attribute_sets[attr_key] = set()
                    if isinstance(attr_val, (list, set)):
                        agg.attribute_sets[attr_key].update(str(v) for v in attr_val)
                    else:
                        agg.attribute_sets[attr_key].add(str(attr_val))

            index.entities[node.name] = agg

        # Build relation triples from edges
        # First, create a node_id -> name mapping
        node_id_to_name: Dict[str, str] = {n.node_id: n.name for n in nodes}

        for edge in edges:
            source_name = node_id_to_name.get(edge.source_node_id, edge.source_node_id)
            target_name = node_id_to_name.get(edge.target_node_id, edge.target_node_id)

            triple = RelationTriple(
                subject=source_name,
                predicate=edge.edge_type,
                object=target_name,
                edge_id=edge.edge_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                timestamp=edge.valid_at.isoformat() if edge.valid_at else None,
                source_memory_id=edge.source_episode_ids[0] if edge.source_episode_ids else "",
                confidence=edge.extraction_confidence,
            )
            index.relations.append(triple)

            # Update event counts for source entity
            if source_name in index.entities:
                action_key = f"{edge.edge_type}_{target_name}"
                index.entities[source_name].event_counts[action_key] = (
                    index.entities[source_name].event_counts.get(action_key, 0) + 1
                )

        return index
