"""
Tests for Semantic Model.

Tests semantic model creation, context injection, and storage.
"""

import json
import os
import tempfile

import pytest

from ctxforge.core.context import Context
from ctxforge.core.semantic_model import (
    EntityDefinition,
    FileBasedSemanticModelStore,
    InMemorySemanticModelStore,
    RelationshipDefinition,
    SemanticModel,
)
from ctxforge.engine.services.semantic_model_service import SemanticModelService


class TestEntityDefinition:
    """Tests for EntityDefinition model."""
    
    def test_create_entity(self):
        """Test creating an entity definition."""
        entity = EntityDefinition(
            name="users",
            description="User accounts and profiles",
            use_cases=["Lookup user by ID", "Search users by email"],
            attributes=[
                {"name": "id", "type": "integer"},
                {"name": "email", "type": "string"},
            ],
            related_entities=["orders", "preferences"],
            retrieval_hint="Use user_id for exact match",
        )
        
        assert entity.name == "users"
        assert entity.description == "User accounts and profiles"
        assert len(entity.use_cases) == 2
        assert len(entity.attributes) == 2
        assert "orders" in entity.related_entities


class TestRelationshipDefinition:
    """Tests for RelationshipDefinition model."""
    
    def test_create_relationship(self):
        """Test creating a relationship definition."""
        rel = RelationshipDefinition(
            name="user_orders",
            from_entity="users",
            to_entity="orders",
            description="User's purchase history",
            cardinality="one_to_many",
        )
        
        assert rel.name == "user_orders"
        assert rel.from_entity == "users"
        assert rel.to_entity == "orders"
        assert rel.cardinality == "one_to_many"


class TestSemanticModel:
    """Tests for SemanticModel."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample semantic model."""
        return SemanticModel(
            name="Customer Service",
            description="Knowledge base for customer service agents",
            version="1.0",
            entities=[
                EntityDefinition(
                    name="customers",
                    description="Customer profiles and information",
                    use_cases=["Find customer", "Update preferences"],
                ),
                EntityDefinition(
                    name="orders",
                    description="Order history and status",
                    use_cases=["Check order status", "Process refunds"],
                ),
            ],
            relationships=[
                RelationshipDefinition(
                    name="customer_orders",
                    from_entity="customers",
                    to_entity="orders",
                    description="Customer's order history",
                ),
            ],
            global_rules=[
                "Always verify customer identity",
                "Escalate refunds over $500",
            ],
            common_gotchas=[
                "Check both email and phone for customer lookup",
            ],
        )
    
    def test_create_model(self, sample_model):
        """Test creating a semantic model."""
        assert sample_model.name == "Customer Service"
        assert len(sample_model.entities) == 2
        assert len(sample_model.relationships) == 1
        assert len(sample_model.global_rules) == 2
    
    def test_get_entity(self, sample_model):
        """Test getting an entity by name."""
        entity = sample_model.get_entity("customers")
        assert entity is not None
        assert entity.name == "customers"
        
        missing = sample_model.get_entity("nonexistent")
        assert missing is None
    
    def test_add_entity(self):
        """Test adding an entity."""
        model = SemanticModel(name="Test")
        entity = EntityDefinition(name="test_entity", description="Test")
        
        model.add_entity(entity)
        
        assert len(model.entities) == 1
        assert model.get_entity("test_entity") is not None
    
    def test_add_relationship(self):
        """Test adding a relationship."""
        model = SemanticModel(name="Test")
        rel = RelationshipDefinition(
            name="test_rel",
            from_entity="a",
            to_entity="b",
            description="Test relationship",
        )
        
        model.add_relationship(rel)
        
        assert len(model.relationships) == 1
    
    def test_add_rule(self):
        """Test adding a rule."""
        model = SemanticModel(name="Test")
        
        model.add_rule("Rule 1")
        model.add_rule("Rule 2")
        model.add_rule("Rule 1")  # Duplicate
        
        assert len(model.global_rules) == 2
    
    def test_add_gotcha(self):
        """Test adding a gotcha."""
        model = SemanticModel(name="Test")
        
        model.add_gotcha("Gotcha 1")
        model.add_gotcha("Gotcha 2")
        model.add_gotcha("Gotcha 1")  # Duplicate
        
        assert len(model.common_gotchas) == 2
    
    def test_to_context_string_compact(self, sample_model):
        """Test compact context string generation."""
        context_str = sample_model.to_context_string(compact=True)
        
        assert "## Domain: Customer Service" in context_str
        assert "### Available Knowledge Areas" in context_str
        assert "**customers**" in context_str
        assert "### Rules" in context_str
        assert "Always verify customer identity" in context_str
        # Use cases should not be expanded in compact mode
        assert "- Find customer" not in context_str
    
    def test_to_context_string_verbose(self, sample_model):
        """Test verbose context string generation."""
        context_str = sample_model.to_context_string(compact=False)
        
        assert "## Domain: Customer Service" in context_str
        # Use cases should be expanded in verbose mode
        assert "Find customer" in context_str or "Update preferences" in context_str
        # Gotchas should be shown in verbose mode
        assert "### Common Gotchas" in context_str


class TestInMemorySemanticModelStore:
    """Tests for InMemorySemanticModelStore."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store."""
        return InMemorySemanticModelStore()
    
    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        """Test saving and loading a model."""
        model = SemanticModel(name="Test Model")
        
        await store.save("test-model", model)
        loaded = await store.load("test-model")
        
        assert loaded is not None
        assert loaded.name == "Test Model"
    
    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        """Test loading a non-existent model."""
        loaded = await store.load("nonexistent")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_list_models(self, store):
        """Test listing models."""
        await store.save("model-1", SemanticModel(name="Model 1"))
        await store.save("model-2", SemanticModel(name="Model 2"))
        
        models = await store.list_models()
        
        assert len(models) == 2
        assert "model-1" in models
        assert "model-2" in models
    
    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting a model."""
        await store.save("test", SemanticModel(name="Test"))
        
        result = await store.delete("test")
        assert result is True
        
        loaded = await store.load("test")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a non-existent model."""
        result = await store.delete("nonexistent")
        assert result is False


class TestSemanticModelService:
    """Tests for SemanticModelService."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store."""
        return InMemorySemanticModelStore()
    
    @pytest.fixture
    def default_model(self):
        """Create a default model."""
        return SemanticModel(
            name="Default Model",
            description="The default semantic model",
        )
    
    @pytest.fixture
    def service(self, store, default_model):
        """Create a service with store and default model."""
        return SemanticModelService(
            store=store,
            default_model=default_model,
        )
    
    @pytest.mark.asyncio
    async def test_get_default_model(self, service, default_model):
        """Test getting the default model."""
        model = await service.get_model()
        
        assert model is not None
        assert model.name == "Default Model"
    
    @pytest.mark.asyncio
    async def test_get_model_by_id(self, service, store):
        """Test getting a model by ID."""
        custom_model = SemanticModel(name="Custom Model")
        await store.save("custom", custom_model)
        
        model = await service.get_model("custom")
        
        assert model is not None
        assert model.name == "Custom Model"
    
    @pytest.mark.asyncio
    async def test_get_model_falls_back_to_default(self, service):
        """Test that get_model falls back to default when model not found."""
        model = await service.get_model("nonexistent")
        
        assert model is not None
        assert model.name == "Default Model"
    
    @pytest.mark.asyncio
    async def test_inject_into_context(self, service):
        """Test injecting semantic model into context."""
        context = Context(
            session_id="session-123",
            user_id="user-123",
            user_input="Hello",
        )
        
        result = await service.inject_into_context(context)
        
        assert result.metadata.get("semantic_model_name") == "Default Model"
        assert result.metadata.get("semantic_model_version") == "1.0"
        
        # Check that section was added
        section = result.get_section("semantic_model")
        assert section is not None
        assert "Default Model" in section.content
    
    @pytest.mark.asyncio
    async def test_inject_no_model_available(self):
        """Test injection when no model is available."""
        service = SemanticModelService()  # No store, no default
        context = Context(
            session_id="session-123",
            user_id="user-123",
            user_input="Hello",
        )
        
        result = await service.inject_into_context(context)
        
        # Context should be unchanged
        assert result.metadata.get("semantic_model_name") is None
    
    @pytest.mark.asyncio
    async def test_save_model(self, service, store):
        """Test saving a model via service."""
        model = SemanticModel(name="New Model")
        
        await service.save_model("new-model", model)
        
        loaded = await store.load("new-model")
        assert loaded is not None
        assert loaded.name == "New Model"
    
    @pytest.mark.asyncio
    async def test_list_models(self, service, store):
        """Test listing models via service."""
        await store.save("model-1", SemanticModel(name="M1"))
        await store.save("model-2", SemanticModel(name="M2"))
        
        models = await service.list_models()
        
        assert len(models) == 2
    
    def test_build_from_expertise(self, service):
        """Test building a semantic model from expertise items."""
        from ctxforge.core.expertise import ExpertiseItem, ExpertiseSection
        
        items = [
            ExpertiseItem(
                section=ExpertiseSection.STRATEGIES,
                content="Always validate input",
            ),
            ExpertiseItem(
                section=ExpertiseSection.COMMON_MISTAKES,
                content="Forgetting null checks",
            ),
            ExpertiseItem(
                section=ExpertiseSection.HEURISTICS,
                content="Start with simple solution",
            ),
        ]
        
        model = service.build_from_expertise(
            name="Test Expertise",
            expertise_items=items,
            description="Built from expertise",
        )
        
        assert model.name == "Test Expertise"
        assert len(model.common_gotchas) == 1
        assert "Forgetting null checks" in model.common_gotchas[0]
        # Strategies and Heuristics should be entities
        assert len(model.entities) >= 1


class TestFileBasedSemanticModelStore:
    """Tests for file-based semantic model store."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temp directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_model(self) -> SemanticModel:
        """Create a sample model for testing."""
        return SemanticModel(
            name="Test Model",
            description="A test semantic model",
            version="1.0",
            entities=[
                EntityDefinition(
                    name="users",
                    description="User accounts",
                    use_cases=["Find user by ID"],
                ),
            ],
            global_rules=["Always validate input"],
        )
    
    @pytest.mark.asyncio
    async def test_save_and_load_json(self, temp_dir, sample_model):
        """Test saving and loading a model in JSON format."""
        store = FileBasedSemanticModelStore(temp_dir)
        
        await store.save("test-model", sample_model)
        
        # Verify file exists
        assert os.path.exists(os.path.join(temp_dir, "test-model.json"))
        
        # Load it back
        loaded = await store.load("test-model")
        
        assert loaded is not None
        assert loaded.name == "Test Model"
        assert loaded.version == "1.0"
        assert len(loaded.entities) == 1
        assert loaded.entities[0].name == "users"
    
    @pytest.mark.asyncio
    async def test_load_nonexistent(self, temp_dir):
        """Test loading a nonexistent model returns None."""
        store = FileBasedSemanticModelStore(temp_dir)
        
        result = await store.load("does-not-exist")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_models(self, temp_dir, sample_model):
        """Test listing models in directory."""
        store = FileBasedSemanticModelStore(temp_dir)
        
        await store.save("model-a", sample_model)
        await store.save("model-b", sample_model)
        
        models = await store.list_models()
        
        assert len(models) == 2
        assert "model-a" in models
        assert "model-b" in models
    
    @pytest.mark.asyncio
    async def test_delete(self, temp_dir, sample_model):
        """Test deleting a model file."""
        store = FileBasedSemanticModelStore(temp_dir)
        
        await store.save("to-delete", sample_model)
        assert os.path.exists(os.path.join(temp_dir, "to-delete.json"))
        
        result = await store.delete("to-delete")
        
        assert result is True
        assert not os.path.exists(os.path.join(temp_dir, "to-delete.json"))
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_dir):
        """Test deleting a nonexistent model returns False."""
        store = FileBasedSemanticModelStore(temp_dir)
        
        result = await store.delete("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_load_from_existing_json(self, temp_dir):
        """Test loading from an existing JSON file."""
        # Create a JSON file directly
        model_data = {
            "name": "Preexisting Model",
            "description": "Created externally",
            "version": "2.0",
            "entities": [
                {
                    "name": "products",
                    "description": "Product catalog",
                    "use_cases": ["Search products"],
                    "attributes": [],
                    "related_entities": [],
                }
            ],
            "relationships": [],
            "global_rules": ["Check stock"],
            "common_gotchas": ["Out of stock items"],
        }
        
        with open(os.path.join(temp_dir, "external.json"), 'w') as f:
            json.dump(model_data, f)
        
        store = FileBasedSemanticModelStore(temp_dir)
        loaded = await store.load("external")
        
        assert loaded is not None
        assert loaded.name == "Preexisting Model"
        assert loaded.version == "2.0"
        assert len(loaded.entities) == 1
        assert loaded.entities[0].name == "products"
        assert "Check stock" in loaded.global_rules
    
    @pytest.mark.asyncio
    async def test_caching(self, temp_dir, sample_model):
        """Test that models are cached after loading."""
        store = FileBasedSemanticModelStore(temp_dir, auto_reload=False)
        
        await store.save("cached", sample_model)
        
        # First load
        loaded1 = await store.load("cached")
        # Second load should return cached version
        loaded2 = await store.load("cached")
        
        # Should be the same object (cached)
        assert loaded1 is loaded2
    
    @pytest.mark.asyncio
    async def test_auto_reload(self, temp_dir, sample_model):
        """Test auto_reload bypasses cache."""
        store = FileBasedSemanticModelStore(temp_dir, auto_reload=True)
        
        await store.save("reloaded", sample_model)
        
        loaded1 = await store.load("reloaded")
        assert loaded1.name == "Test Model"  # Original name

        # Modify file externally
        model_data = {
            "name": "Modified Model",
            "version": "3.0",
            "entities": [],
        }
        with open(os.path.join(temp_dir, "reloaded.json"), 'w') as f:
            json.dump(model_data, f)
        
        loaded2 = await store.load("reloaded")
        
        # Should be different (reloaded from disk)
        assert loaded2.name == "Modified Model"
        assert loaded2.version == "3.0"
    
    def test_clear_cache(self, temp_dir):
        """Test clearing the cache."""
        store = FileBasedSemanticModelStore(temp_dir)
        store._cache["test"] = SemanticModel(name="Cached")
        
        store.clear_cache()
        
        assert len(store._cache) == 0
