"""
Tests for User Preference Memory Type.

Tests the PREFERENCE memory type for storing user preferences.
"""

from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType


class TestPreferenceMemoryType:
    """Tests for the PREFERENCE memory type."""
    
    def test_preference_type_exists(self):
        """Test that PREFERENCE type exists in MemoryType enum."""
        assert hasattr(MemoryType, 'PREFERENCE')
        assert MemoryType.PREFERENCE == "preference"
    
    def test_create_preference_memory(self):
        """Test creating a preference memory item."""
        memory = MemoryItem(
            user_id="user-123",
            content="User prefers dark mode",
            type=MemoryType.PREFERENCE,
            source=MemorySource.USER_EXPLICIT,
        )
        
        assert memory.type == MemoryType.PREFERENCE
        assert memory.content == "User prefers dark mode"
        assert memory.user_id == "user-123"
    
    def test_preference_memory_with_metadata(self):
        """Test preference memory with additional metadata."""
        memory = MemoryItem(
            user_id="user-123",
            content="User prefers concise answers",
            type=MemoryType.PREFERENCE,
            source=MemorySource.USER_EXPLICIT,
            confidence_score=1.0,
            tags=["communication", "style"],
            metadata={
                "preference_category": "communication_style",
                "strength": "strong",
            },
        )
        
        assert memory.type == MemoryType.PREFERENCE
        assert "communication" in memory.tags
        assert memory.metadata["preference_category"] == "communication_style"
    
    def test_preference_memory_inferred(self):
        """Test preference memory inferred from behavior."""
        memory = MemoryItem(
            user_id="user-123",
            content="User tends to ask for examples",
            type=MemoryType.PREFERENCE,
            source=MemorySource.USER_IMPLICIT,
            confidence_score=0.7,  # Lower confidence for inferred
        )
        
        assert memory.type == MemoryType.PREFERENCE
        assert memory.source == MemorySource.USER_IMPLICIT
        assert memory.confidence_score == 0.7


class TestAllMemoryTypes:
    """Tests to verify all memory types work correctly together."""
    
    def test_all_memory_types_exist(self):
        """Test that all expected memory types exist."""
        expected_types = ["semantic", "episodic", "procedural", "preference"]
        actual_types = [t.value for t in MemoryType]
        
        for expected in expected_types:
            assert expected in actual_types, f"Missing memory type: {expected}"
    
    def test_create_memories_of_each_type(self):
        """Test creating memories of each type."""
        types_and_content = [
            (MemoryType.SEMANTIC, "User is vegetarian"),
            (MemoryType.EPISODIC, "User went to Paris in 2023"),
            (MemoryType.PROCEDURAL, "Step 1: Open settings. Step 2: Click save."),
            (MemoryType.PREFERENCE, "User prefers bullet points"),
        ]
        
        for memory_type, content in types_and_content:
            memory = MemoryItem(
                user_id="user-123",
                content=content,
                type=memory_type,
            )
            assert memory.type == memory_type
            assert memory.content == content
    
    def test_memory_type_string_values(self):
        """Test that memory types have correct string values."""
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.PREFERENCE.value == "preference"
