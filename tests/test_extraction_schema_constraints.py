"""Tests for schema constraint generation module."""


from ctxforge.core.memory import MemoryType
from ctxforge.extraction.schema_constraints import (
    SchemaConfig,
    generate_graph_extraction_schema,
    generate_memory_extraction_schema,
    generate_reflection_schema,
)
from ctxforge.graph.ontology import GraphOntology


class TestSchemaConfig:
    """Tests for SchemaConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SchemaConfig()
        
        assert config.strict is True
        assert config.additional_properties is False
        assert config.include_descriptions is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = SchemaConfig(
            strict=False,
            additional_properties=True,
            include_descriptions=False,
        )
        
        assert config.strict is False
        assert config.additional_properties is True
        assert config.include_descriptions is False


class TestGenerateMemoryExtractionSchema:
    """Tests for generate_memory_extraction_schema function."""
    
    def test_generates_valid_schema(self):
        """Test that generated schema has required structure."""
        schema = generate_memory_extraction_schema()
        
        assert "$schema" in schema
        assert schema["type"] == "array"
        assert "items" in schema
    
    def test_default_includes_all_types(self):
        """Test that all memory types are included by default."""
        schema = generate_memory_extraction_schema()
        
        type_enum = schema["items"]["properties"]["type"]["enum"]
        assert "semantic" in type_enum
        assert "episodic" in type_enum
        assert "procedural" in type_enum
    
    def test_restricts_to_allowed_types(self):
        """Test restricting to specific memory types."""
        schema = generate_memory_extraction_schema(
            allowed_types=[MemoryType.SEMANTIC]
        )
        
        type_enum = schema["items"]["properties"]["type"]["enum"]
        assert type_enum == ["semantic"]
    
    def test_required_fields(self):
        """Test that required fields are specified."""
        schema = generate_memory_extraction_schema()
        
        required = schema["items"]["required"]
        assert "content" in required
        assert "type" in required
        assert "confidence" in required
    
    def test_content_properties(self):
        """Test content field properties."""
        schema = generate_memory_extraction_schema()
        
        content = schema["items"]["properties"]["content"]
        assert content["type"] == "string"
        assert "minLength" in content
    
    def test_confidence_constraints(self):
        """Test confidence field constraints."""
        schema = generate_memory_extraction_schema()
        
        confidence = schema["items"]["properties"]["confidence"]
        assert confidence["type"] == "number"
        assert confidence["minimum"] == 0.0
        assert confidence["maximum"] == 1.0
    
    def test_additional_properties_config(self):
        """Test additional properties configuration."""
        config = SchemaConfig(additional_properties=True)
        schema = generate_memory_extraction_schema(config=config)
        
        assert schema["items"]["additionalProperties"] is True
    
    def test_tags_field(self):
        """Test tags field is present."""
        schema = generate_memory_extraction_schema()
        
        tags = schema["items"]["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"]["type"] == "string"


class TestGenerateGraphExtractionSchema:
    """Tests for generate_graph_extraction_schema function."""
    
    def test_generates_valid_schema(self):
        """Test that generated schema has required structure."""
        ontology = GraphOntology(
            entity_types={"Person": None, "Organization": None},
            edge_types={"works_at": None},
        )
        schema = generate_graph_extraction_schema(ontology)
        
        assert "$schema" in schema
        assert schema["type"] == "object"
        assert "entities" in schema["properties"]
        assert "edges" in schema["properties"]
    
    def test_entity_types_enum(self):
        """Test entity types are included in enum."""
        ontology = GraphOntology(
            entity_types={"Person": None, "Organization": None},
            edge_types={},
        )
        schema = generate_graph_extraction_schema(ontology)
        
        entity_type_enum = schema["properties"]["entities"]["items"]["properties"]["entity_type"]["enum"]
        assert "Person" in entity_type_enum
        assert "Organization" in entity_type_enum
    
    def test_edge_types_enum(self):
        """Test edge types are included in enum."""
        ontology = GraphOntology(
            entity_types={},
            edge_types={"works_at": None, "knows": None},
        )
        schema = generate_graph_extraction_schema(ontology)
        
        edge_type_enum = schema["properties"]["edges"]["items"]["properties"]["edge_type"]["enum"]
        assert "works_at" in edge_type_enum
        assert "knows" in edge_type_enum
    
    def test_required_entity_fields(self):
        """Test required fields for entities."""
        ontology = GraphOntology()
        schema = generate_graph_extraction_schema(ontology)
        
        required = schema["properties"]["entities"]["items"]["required"]
        assert "name" in required
        assert "entity_type" in required
    
    def test_required_edge_fields(self):
        """Test required fields for edges."""
        ontology = GraphOntology()
        schema = generate_graph_extraction_schema(ontology)
        
        required = schema["properties"]["edges"]["items"]["required"]
        assert "source_name" in required
        assert "source_type" in required
        assert "edge_type" in required
        assert "target_name" in required
        assert "target_type" in required
    
    def test_temporal_fields(self):
        """Test temporal fields are present."""
        ontology = GraphOntology()
        schema = generate_graph_extraction_schema(ontology)
        
        edge_props = schema["properties"]["edges"]["items"]["properties"]
        assert "valid_at" in edge_props
        assert "invalid_at" in edge_props
    
    def test_empty_ontology_defaults(self):
        """Test handling of empty ontology."""
        ontology = GraphOntology()
        schema = generate_graph_extraction_schema(ontology)
        
        # Should use wildcard for empty types
        entity_type_enum = schema["properties"]["entities"]["items"]["properties"]["entity_type"]["enum"]
        assert "*" in entity_type_enum


class TestGenerateReflectionSchema:
    """Tests for generate_reflection_schema function."""
    
    def test_generates_valid_schema(self):
        """Test that generated schema has required structure."""
        schema = generate_reflection_schema(item_ids=["item1", "item2"])
        
        assert "$schema" in schema
        assert schema["type"] == "object"
        assert "bullet_tags" in schema["properties"]
    
    def test_item_ids_enum(self):
        """Test item IDs are included in enum."""
        schema = generate_reflection_schema(item_ids=["a", "b", "c"])
        
        id_enum = schema["properties"]["bullet_tags"]["items"]["properties"]["id"]["enum"]
        assert "a" in id_enum
        assert "b" in id_enum
        assert "c" in id_enum
    
    def test_tag_values(self):
        """Test tag values are constrained."""
        schema = generate_reflection_schema(item_ids=["item1"])
        
        tag_enum = schema["properties"]["bullet_tags"]["items"]["properties"]["tag"]["enum"]
        assert "helpful" in tag_enum
        assert "harmful" in tag_enum
        assert "neutral" in tag_enum
    
    def test_required_fields(self):
        """Test required fields."""
        schema = generate_reflection_schema(item_ids=["item1"])
        
        assert "bullet_tags" in schema["required"]
    
    def test_insights_field(self):
        """Test insights field is present."""
        schema = generate_reflection_schema(item_ids=["item1"])
        
        assert "insights" in schema["properties"]
        assert schema["properties"]["insights"]["type"] == "string"
    
    def test_suggested_fields(self):
        """Test suggested additions/removals fields."""
        schema = generate_reflection_schema(item_ids=["item1"])
        
        assert "suggested_additions" in schema["properties"]
        assert "suggested_removals" in schema["properties"]
        
        additions = schema["properties"]["suggested_additions"]
        assert additions["type"] == "array"
        assert additions["items"]["type"] == "string"
    
    def test_confidence_field(self):
        """Test confidence field."""
        schema = generate_reflection_schema(item_ids=["item1"])
        
        confidence = schema["properties"]["confidence"]
        assert confidence["type"] == "number"
        assert confidence["minimum"] == 0.0
        assert confidence["maximum"] == 1.0
    
    def test_empty_item_ids(self):
        """Test handling of empty item IDs."""
        schema = generate_reflection_schema(item_ids=[])
        
        id_enum = schema["properties"]["bullet_tags"]["items"]["properties"]["id"]["enum"]
        assert "*" in id_enum

