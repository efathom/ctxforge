"""
Tests for ValidatedKnowledgeService.

Tests the direct save path for user-validated knowledge items.
"""

from unittest.mock import AsyncMock

import pytest

from ctxforge.core.expertise import Expertise, ExpertiseSection
from ctxforge.core.memory import MemorySource, MemoryType
from ctxforge.engine.services.validated_knowledge_service import (
    ValidatedKnowledgeEntry,
    ValidatedKnowledgeService,
)


class TestValidatedKnowledgeService:
    """Tests for ValidatedKnowledgeService."""
    
    @pytest.fixture
    def mock_expertise_store(self):
        """Create a mock expertise store."""
        store = AsyncMock()
        expertise = Expertise(
            expertise_id="test-expertise",
            name="Test Expertise",
            domain="testing",
        )
        store.load.return_value = expertise
        store.save.return_value = None
        return store
    
    @pytest.fixture
    def mock_memory_store(self):
        """Create a mock memory store."""
        store = AsyncMock()
        store.add.return_value = "mem-123"
        return store
    
    @pytest.fixture
    def mock_expertise_indexer(self):
        """Create a mock expertise indexer."""
        indexer = AsyncMock()
        indexer.index_item.return_value = None
        return indexer
    
    @pytest.fixture
    def mock_memory_indexer(self):
        """Create a mock memory indexer."""
        indexer = AsyncMock()
        indexer.index.return_value = None
        return indexer
    
    @pytest.fixture
    def service(
        self,
        mock_expertise_store,
        mock_memory_store,
        mock_expertise_indexer,
        mock_memory_indexer,
    ):
        """Create a ValidatedKnowledgeService with mocks."""
        return ValidatedKnowledgeService(
            expertise_store=mock_expertise_store,
            expertise_indexer=mock_expertise_indexer,
            memory_store=mock_memory_store,
            memory_indexer=mock_memory_indexer,
        )
    
    @pytest.mark.asyncio
    async def test_save_expertise_item_success(self, service, mock_expertise_store):
        """Test saving a validated expertise item."""
        item = await service.save_expertise_item(
            expertise_id="test-expertise",
            section=ExpertiseSection.STRATEGIES,
            content="Always validate user input before processing",
            name="Input Validation Rule",
            source_question="What's a good practice for handling user input?",
            summary="Validate before processing",
            notes="Applies to all user-facing endpoints",
            validated_by="user-123",
            tags=["security", "validation"],
        )
        
        assert item is not None
        assert item.content == "Always validate user input before processing"
        assert item.section == ExpertiseSection.STRATEGIES
        assert item.source == "user_validated"
        assert item.metadata["human_validated"] is True
        assert item.metadata["name"] == "Input Validation Rule"
        assert item.metadata["validated_by"] == "user-123"
        assert "validated_at" in item.metadata
        
        # Verify store was called
        mock_expertise_store.load.assert_called_once_with("test-expertise")
        mock_expertise_store.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_expertise_item_not_found(self, service, mock_expertise_store):
        """Test saving to non-existent expertise."""
        mock_expertise_store.load.return_value = None
        
        item = await service.save_expertise_item(
            expertise_id="nonexistent",
            section=ExpertiseSection.STRATEGIES,
            content="Some content",
        )
        
        assert item is None
    
    @pytest.mark.asyncio
    async def test_save_expertise_item_no_store(self):
        """Test saving without expertise store configured."""
        service = ValidatedKnowledgeService()
        
        item = await service.save_expertise_item(
            expertise_id="test",
            section=ExpertiseSection.STRATEGIES,
            content="Content",
        )
        
        assert item is None
    
    @pytest.mark.asyncio
    async def test_save_expertise_item_indexes(
        self, service, mock_expertise_indexer
    ):
        """Test that saved expertise item is indexed."""
        await service.save_expertise_item(
            expertise_id="test-expertise",
            section=ExpertiseSection.STRATEGIES,
            content="Indexed content",
        )
        
        mock_expertise_indexer.index_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_validated_memory_success(self, service, mock_memory_store):
        """Test saving a validated memory item."""
        memory = await service.save_validated_memory(
            user_id="user-123",
            content="User prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            name="UI Preference",
            source_question="What theme do you prefer?",
            summary="Dark mode preference",
            notes="Mentioned multiple times",
            tags=["preferences", "ui"],
        )
        
        assert memory is not None
        assert memory.content == "User prefers dark mode"
        assert memory.type == MemoryType.SEMANTIC
        assert memory.source == MemorySource.USER_EXPLICIT
        assert memory.confidence_score == 1.0
        assert memory.metadata["human_validated"] is True
        assert memory.metadata["name"] == "UI Preference"
        assert "validated_at" in memory.metadata
        assert "preferences" in memory.tags
        
        # Verify store was called
        mock_memory_store.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_validated_memory_default_type(self, service):
        """Test that memory defaults to PROCEDURAL type."""
        memory = await service.save_validated_memory(
            user_id="user-123",
            content="Step 1: Do this. Step 2: Do that.",
        )
        
        assert memory is not None
        assert memory.type == MemoryType.PROCEDURAL
    
    @pytest.mark.asyncio
    async def test_save_validated_memory_no_store(self):
        """Test saving without memory store configured."""
        service = ValidatedKnowledgeService()
        
        memory = await service.save_validated_memory(
            user_id="user-123",
            content="Content",
        )
        
        assert memory is None
    
    @pytest.mark.asyncio
    async def test_save_validated_memory_indexes(
        self, service, mock_memory_indexer
    ):
        """Test that saved memory is indexed."""
        await service.save_validated_memory(
            user_id="user-123",
            content="Indexed memory",
        )
        
        mock_memory_indexer.index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_validated_entry_to_expertise(self, service):
        """Test saving entry via convenience method to expertise."""
        entry = ValidatedKnowledgeEntry(
            name="Test Entry",
            question="How do I test?",
            content="Use pytest for testing",
            summary="Testing guidance",
            notes="Applies to Python projects",
            validated_by="user-123",
            tags=["testing"],
        )
        
        item_id = await service.save_validated_entry(
            entry=entry,
            expertise_id="test-expertise",
            section=ExpertiseSection.HEURISTICS,
        )
        
        assert item_id is not None
    
    @pytest.mark.asyncio
    async def test_save_validated_entry_to_memory(self, service):
        """Test saving entry via convenience method to memory."""
        entry = ValidatedKnowledgeEntry(
            name="Test Entry",
            question="What's your preference?",
            content="I prefer concise answers",
            tags=["preferences"],
        )
        
        item_id = await service.save_validated_entry(
            entry=entry,
            user_id="user-123",
        )
        
        assert item_id is not None
    
    @pytest.mark.asyncio
    async def test_save_validated_entry_no_target(self, service):
        """Test saving entry without expertise_id or user_id."""
        entry = ValidatedKnowledgeEntry(
            name="Test Entry",
            question="Question",
            content="Content",
        )
        
        item_id = await service.save_validated_entry(entry=entry)
        
        assert item_id is None


class TestValidatedKnowledgeEntry:
    """Tests for ValidatedKnowledgeEntry model."""
    
    def test_create_entry(self):
        """Test creating a validated knowledge entry."""
        entry = ValidatedKnowledgeEntry(
            name="Test Entry",
            question="How do I do X?",
            content="Do X by doing Y",
            summary="Summary of X",
            notes="Note about X",
            knowledge_type="pattern",
            tags=["tag1", "tag2"],
            validated_by="user-123",
        )
        
        assert entry.name == "Test Entry"
        assert entry.question == "How do I do X?"
        assert entry.content == "Do X by doing Y"
        assert entry.knowledge_type == "pattern"
        assert len(entry.tags) == 2
        assert entry.validated_by == "user-123"
        assert entry.created_at is not None
    
    def test_entry_defaults(self):
        """Test default values for entry."""
        entry = ValidatedKnowledgeEntry(
            name="Test",
            question="Q",
            content="C",
        )
        
        assert entry.knowledge_type == "validated_pattern"
        assert entry.tags == []
        assert entry.metadata == {}
        assert entry.summary is None
        assert entry.notes is None
