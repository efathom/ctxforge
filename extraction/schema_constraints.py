"""
JSON Schema generation for structured LLM outputs.

Generates schemas from ontologies and extraction configurations
to constrain LLM responses to valid structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ctxforge.core.memory import MemoryType
from ctxforge.graph.ontology import GraphOntology


@dataclass
class SchemaConfig:
    """Configuration for schema generation."""
    
    strict: bool = True
    additional_properties: bool = False
    include_descriptions: bool = True


def generate_memory_extraction_schema(
    allowed_types: Optional[List[MemoryType]] = None,
    config: Optional[SchemaConfig] = None,
) -> Dict[str, Any]:
    """
    Generate JSON schema for memory extraction.
    
    Args:
        allowed_types: List of allowed memory types. Defaults to all.
        config: Schema configuration options
        
    Returns:
        JSON Schema dict for memory extraction output
    """
    config = config or SchemaConfig()
    
    if allowed_types is None:
        allowed_types = list(MemoryType)
    
    type_enum = [t.value for t in allowed_types]
    
    extraction_item: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Clear, concise statement of the fact",
                "minLength": 5,
            },
            "type": {
                "type": "string",
                "enum": type_enum,
                "description": "Memory type classification",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score (0.0 to 1.0)",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant categorization tags",
            },
        },
        "required": ["content", "type", "confidence"],
        "additionalProperties": config.additional_properties,
    }
    
    schema: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": extraction_item,
        "description": "List of extracted memories",
    }
    
    return schema


def generate_graph_extraction_schema(
    ontology: "GraphOntology",
    config: Optional[SchemaConfig] = None,
) -> Dict[str, Any]:
    """
    Generate JSON schema for graph entity/edge extraction.
    
    Args:
        ontology: Graph ontology defining allowed types
        config: Schema configuration options
        
    Returns:
        JSON Schema dict for graph extraction output
    """
    config = config or SchemaConfig()
    
    entity_types = list(ontology.entity_types.keys())
    edge_types = list(ontology.edge_types.keys())
    
    entity_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Entity name/identifier",
                "minLength": 1,
            },
            "entity_type": {
                "type": "string",
                "enum": entity_types if entity_types else ["*"],
                "description": "Type of entity from ontology",
            },
            "attributes": {
                "type": "object",
                "description": "Additional entity attributes",
            },
            "summary": {
                "type": ["string", "null"],
                "description": "Optional entity summary",
            },
        },
        "required": ["name", "entity_type"],
        "additionalProperties": config.additional_properties,
    }
    
    edge_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "source_name": {
                "type": "string",
                "description": "Name of source entity",
            },
            "source_type": {
                "type": "string",
                "enum": entity_types if entity_types else ["*"],
                "description": "Type of source entity",
            },
            "edge_type": {
                "type": "string",
                "enum": edge_types if edge_types else ["*"],
                "description": "Type of relationship",
            },
            "target_name": {
                "type": "string",
                "description": "Name of target entity",
            },
            "target_type": {
                "type": "string",
                "enum": entity_types if entity_types else ["*"],
                "description": "Type of target entity",
            },
            "fact": {
                "type": ["string", "null"],
                "description": "Natural language fact statement",
            },
            "attributes": {
                "type": "object",
                "description": "Additional edge attributes",
            },
            "valid_at": {
                "type": ["string", "null"],
                "format": "date-time",
                "description": "When this relationship became valid",
            },
            "invalid_at": {
                "type": ["string", "null"],
                "format": "date-time",
                "description": "When this relationship becomes invalid",
            },
        },
        "required": ["source_name", "source_type", "edge_type", "target_name", "target_type"],
        "additionalProperties": config.additional_properties,
    }
    
    schema: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": entity_schema,
                "description": "Extracted entities",
            },
            "edges": {
                "type": "array",
                "items": edge_schema,
                "description": "Extracted relationships",
            },
        },
        "required": ["entities", "edges"],
        "additionalProperties": config.additional_properties,
    }
    
    return schema


def generate_reflection_schema(
    item_ids: List[str],
    config: Optional[SchemaConfig] = None,
) -> Dict[str, Any]:
    """
    Generate JSON schema for expertise reflection output.
    
    Args:
        item_ids: Valid item IDs that can be referenced
        config: Schema configuration options
        
    Returns:
        JSON Schema dict for reflection output
    """
    config = config or SchemaConfig()
    
    schema: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "bullet_tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "enum": item_ids if item_ids else ["*"],
                        },
                        "tag": {
                            "type": "string",
                            "enum": ["helpful", "harmful", "neutral"],
                        },
                    },
                    "required": ["id", "tag"],
                },
            },
            "insights": {
                "type": "string",
                "description": "Analysis insights",
            },
            "suggested_additions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "suggested_removals": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["bullet_tags"],
        "additionalProperties": config.additional_properties,
    }
    
    return schema

