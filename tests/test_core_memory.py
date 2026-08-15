"""
Tests for core memory data structures.
"""

from datetime import datetime, timedelta

import pytest

from ctxforge.core.memory import (
    MemoryFactory,
    MemoryItem,
    MemoryQuery,
    MemorySource,
    MemoryType,
)


class TestMemoryType:
    """Tests for MemoryType enum."""
    
    def test_memory_types_exist(self):
        """Verify all expected memory types exist."""
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.PROCEDURAL == "procedural"


class TestMemorySource:
    """Tests for MemorySource enum."""
    
    def test_memory_sources_exist(self):
        """Verify all expected memory sources exist."""
        assert MemorySource.USER_EXPLICIT == "user_explicit"
        assert MemorySource.USER_IMPLICIT == "user_implicit"
        assert MemorySource.AGENT_INFERENCE == "agent_inference"
        assert MemorySource.SYSTEM == "system"
        assert MemorySource.EXTERNAL == "external"


class TestMemoryItem:
    """Tests for MemoryItem."""
    
    def test_create_memory(self):
        """Test creating a memory item."""
        memory = MemoryItem(
            user_id="user_123",
            content="User is vegetarian",
            type=MemoryType.SEMANTIC,
        )
        
        assert memory.user_id == "user_123"
        assert memory.content == "User is vegetarian"
        assert memory.type == MemoryType.SEMANTIC
        assert memory.source == MemorySource.AGENT_INFERENCE
        assert memory.confidence_score == 1.0
        assert memory.is_active is True
        assert memory.memory_id is not None
    
    def test_content_validation(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError):
            MemoryItem(
                user_id="user_123",
                content="",
                type=MemoryType.SEMANTIC,
            )
        
        with pytest.raises(ValueError):
            MemoryItem(
                user_id="user_123",
                content="   ",
                type=MemoryType.SEMANTIC,
            )
    
    def test_confidence_score_bounds(self):
        """Test confidence score validation."""
        # Valid scores
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
            confidence_score=0.5,
        )
        assert memory.confidence_score == 0.5
        
        # Invalid scores should be rejected
        with pytest.raises(ValueError):
            MemoryItem(
                user_id="u1",
                content="Test",
                type=MemoryType.SEMANTIC,
                confidence_score=1.5,
            )
    
    def test_record_access(self):
        """Test recording access."""
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        assert memory.access_count == 0
        assert memory.accessed_at is None
        
        memory.record_access()
        
        assert memory.access_count == 1
        assert memory.accessed_at is not None
    
    def test_update_content(self):
        """Test updating content."""
        memory = MemoryItem(
            user_id="u1",
            content="Original",
            type=MemoryType.SEMANTIC,
            source=MemorySource.AGENT_INFERENCE,
        )
        
        memory.update_content("Updated", source=MemorySource.USER_EXPLICIT)
        
        assert memory.content == "Updated"
        assert memory.source == MemorySource.USER_EXPLICIT
    
    def test_update_confidence(self):
        """Test updating confidence score."""
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
            confidence_score=0.5,
        )
        
        memory.update_confidence(0.8)
        assert memory.confidence_score == 0.8
        
        # Test clamping
        memory.update_confidence(1.5)
        assert memory.confidence_score == 1.0
        
        memory.update_confidence(-0.5)
        assert memory.confidence_score == 0.0
    
    def test_tags(self):
        """Test tag operations."""
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        memory.add_tag("important")
        assert "important" in memory.tags
        
        # Adding same tag again should not duplicate
        memory.add_tag("important")
        assert memory.tags.count("important") == 1
        
        assert memory.remove_tag("important") is True
        assert "important" not in memory.tags
        
        assert memory.remove_tag("nonexistent") is False
    
    def test_related_memories(self):
        """Test related memory operations."""
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        memory.add_related_memory("mem_123")
        assert "mem_123" in memory.related_memory_ids
        
        # Adding same ID again should not duplicate
        memory.add_related_memory("mem_123")
        assert memory.related_memory_ids.count("mem_123") == 1
    
    def test_expiration(self):
        """Test expiration checking."""
        # Non-expiring memory
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        assert memory.is_expired() is False
        
        # Expired memory
        expired_memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
            expires_at=datetime.now() - timedelta(days=1),
        )
        assert expired_memory.is_expired() is True
        
        # Future expiration
        future_memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
            expires_at=datetime.now() + timedelta(days=1),
        )
        assert future_memory.is_expired() is False
    
    def test_deactivate_activate(self):
        """Test deactivation and activation."""
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        assert memory.is_active is True
        
        memory.deactivate()
        assert memory.is_active is False
        
        memory.activate()
        assert memory.is_active is True
    
    def test_to_prompt_format(self):
        """Test conversion to prompt format."""
        # High confidence
        memory = MemoryItem(
            user_id="u1",
            content="User is vegetarian",
            type=MemoryType.SEMANTIC,
            confidence_score=1.0,
        )
        assert memory.to_prompt_format() == "[Semantic]: User is vegetarian"
        
        # Medium confidence
        memory.confidence_score = 0.8
        assert "(likely)" in memory.to_prompt_format()
        
        # Low confidence
        memory.confidence_score = 0.5
        assert "(uncertain)" in memory.to_prompt_format()
    
    def test_similarity_score(self):
        """Test cosine similarity calculation."""
        memory = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
            embedding=[1.0, 0.0, 0.0],
        )
        
        # Same direction = 1.0
        assert abs(memory.similarity_score([1.0, 0.0, 0.0]) - 1.0) < 0.001
        
        # Orthogonal = 0.0
        assert abs(memory.similarity_score([0.0, 1.0, 0.0]) - 0.0) < 0.001
        
        # No embedding
        memory_no_emb = MemoryItem(
            user_id="u1",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        assert memory_no_emb.similarity_score([1.0, 0.0, 0.0]) == 0.0


class TestMemoryFactory:
    """Tests for MemoryFactory."""
    
    def test_semantic_memory(self):
        """Test creating semantic memory."""
        memory = MemoryFactory.semantic_memory(
            user_id="u1",
            content="User prefers dark mode",
            source=MemorySource.USER_EXPLICIT,
            confidence=0.9,
            tags=["preference", "ui"],
        )
        
        assert memory.type == MemoryType.SEMANTIC
        assert memory.content == "User prefers dark mode"
        assert memory.source == MemorySource.USER_EXPLICIT
        assert memory.confidence_score == 0.9
        assert "preference" in memory.tags
    
    def test_episodic_memory(self):
        """Test creating episodic memory."""
        memory = MemoryFactory.episodic_memory(
            user_id="u1",
            content="User traveled to Paris in June",
            source_event_id="evt_123",
            tags=["travel"],
        )
        
        assert memory.type == MemoryType.EPISODIC
        assert memory.source_event_id == "evt_123"
        assert "travel" in memory.tags
    
    def test_procedural_memory(self):
        """Test creating procedural memory."""
        memory = MemoryFactory.procedural_memory(
            user_id="u1",
            content="To reset password: go to settings...",
            source=MemorySource.SYSTEM,
            metadata={"category": "help"},
        )
        
        assert memory.type == MemoryType.PROCEDURAL
        assert memory.source == MemorySource.SYSTEM
        assert memory.metadata["category"] == "help"


class TestMemoryQuery:
    """Tests for MemoryQuery."""
    
    def test_default_query(self):
        """Test default query values."""
        query = MemoryQuery(user_id="u1")
        
        assert query.user_id == "u1"
        assert query.query_text is None
        assert query.types is None
        assert query.limit == 10
        assert query.offset == 0
        assert query.min_confidence == 0.0
        assert query.include_inactive is False
    
    def test_query_with_filters(self):
        """Test query with filters."""
        query = MemoryQuery(
            user_id="u1",
            query_text="vegetarian food",
            types=[MemoryType.SEMANTIC],
            tags=["food", "preference"],
            min_confidence=0.5,
            limit=5,
        )
        
        assert query.query_text == "vegetarian food"
        assert MemoryType.SEMANTIC in query.types
        assert "food" in query.tags
        assert query.min_confidence == 0.5
        assert query.limit == 5


class TestRestatement:
    """Tests for lossless restatement fields on MemoryItem."""

    def test_restatement_defaults_to_none(self):
        memory = MemoryItem(
            user_id="u1",
            content="He likes coffee",
            type=MemoryType.SEMANTIC,
        )
        assert memory.restatement is None
        assert memory.extracted_entities == {}

    def test_display_content_prefers_restatement(self):
        memory = MemoryItem(
            user_id="u1",
            content="He likes coffee",
            type=MemoryType.SEMANTIC,
            restatement="Bob likes coffee",
        )
        assert memory.display_content == "Bob likes coffee"

    def test_display_content_falls_back_to_content(self):
        memory = MemoryItem(
            user_id="u1",
            content="He likes coffee",
            type=MemoryType.SEMANTIC,
        )
        assert memory.display_content == "He likes coffee"

    def test_to_prompt_format_uses_restatement(self):
        memory = MemoryItem(
            user_id="u1",
            content="He said he'd move there tomorrow",
            type=MemoryType.EPISODIC,
            restatement="Bob will move to Seattle on 2026-02-16",
        )
        prompt = memory.to_prompt_format()
        assert "Bob will move to Seattle on 2026-02-16" in prompt
        assert "He said" not in prompt

    def test_to_prompt_format_without_restatement(self):
        memory = MemoryItem(
            user_id="u1",
            content="User is vegetarian",
            type=MemoryType.SEMANTIC,
        )
        prompt = memory.to_prompt_format()
        assert "User is vegetarian" in prompt

    def test_extracted_entities_stored(self):
        memory = MemoryItem(
            user_id="u1",
            content="He met Alice in Paris",
            type=MemoryType.EPISODIC,
            extracted_entities={
                "persons": ["Alice"],
                "locations": ["Paris"],
                "timestamps": ["2025-06-15"],
            },
        )
        assert memory.extracted_entities["persons"] == ["Alice"]
        assert memory.extracted_entities["locations"] == ["Paris"]

