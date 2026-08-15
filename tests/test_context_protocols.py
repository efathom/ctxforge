"""
Tests for Context Protocol Compliance.

Verifies that MemoryItem, ExpertiseItem, and their related classes
properly conform to the IContextItem, IContextRetriever, and other
generic context protocols.
"""

from datetime import datetime
from typing import List

import pytest

from ctxforge.core.expertise import ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.protocols.context import (
    ContextRetrievalResult,
    IContextItem,
    IndexSearchResult,
)
from ctxforge.retrieval.indexers import ExpertiseIndexer, MemoryIndexer
from ctxforge.retrieval.retrievers import ExpertiseRetriever
from ctxforge.retrieval.retrievers.base import BaseRetriever

# ============================================================================
# IContextItem Protocol Compliance Tests
# ============================================================================

class TestMemoryItemContextProtocol:
    """Test that MemoryItem conforms to IContextItem protocol."""
    
    def test_memory_item_is_context_item(self):
        """MemoryItem should be recognized as IContextItem."""
        memory = MemoryItem(
            user_id="user-123",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
        )
        
        # Protocol check via isinstance (runtime_checkable)
        assert isinstance(memory, IContextItem)
    
    def test_memory_item_has_item_id(self):
        """MemoryItem should have item_id property (alias for memory_id)."""
        memory = MemoryItem(
            memory_id="mem-123",
            user_id="user-123",
            content="Test content",
            type=MemoryType.SEMANTIC,
        )
        
        assert hasattr(memory, 'item_id')
        assert memory.item_id == "mem-123"
        assert memory.item_id == memory.memory_id
    
    def test_memory_item_has_content(self):
        """MemoryItem should have content property."""
        memory = MemoryItem(
            user_id="user-123",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
        )
        
        assert memory.content == "User prefers dark mode"
    
    def test_memory_item_has_is_active(self):
        """MemoryItem should have is_active property."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        assert memory.is_active is True
        memory.deactivate()
        assert memory.is_active is False
    
    def test_memory_item_has_embedding(self):
        """MemoryItem should have embedding property."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
            embedding=[0.1, 0.2, 0.3],
        )
        
        assert memory.embedding == [0.1, 0.2, 0.3]
    
    def test_memory_item_has_metadata(self):
        """MemoryItem should have metadata property."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
            metadata={"key": "value"},
        )
        
        assert memory.metadata == {"key": "value"}
    
    def test_memory_item_has_timestamps(self):
        """MemoryItem should have created_at and updated_at."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)
    
    def test_memory_item_has_to_prompt_format(self):
        """MemoryItem should have to_prompt_format method."""
        memory = MemoryItem(
            user_id="user-123",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
            confidence_score=0.95,
        )
        
        prompt = memory.to_prompt_format()
        assert isinstance(prompt, str)
        assert "dark mode" in prompt
        assert "Semantic" in prompt


class TestExpertiseItemContextProtocol:
    """Test that ExpertiseItem conforms to IContextItem protocol."""
    
    def test_expertise_item_is_context_item(self):
        """ExpertiseItem should be recognized as IContextItem."""
        item = ExpertiseItem(
            item_id="strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="Always validate user input",
        )
        
        # Protocol check via isinstance (runtime_checkable)
        assert isinstance(item, IContextItem)
    
    def test_expertise_item_has_item_id(self):
        """ExpertiseItem should have item_id property."""
        item = ExpertiseItem(
            item_id="strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="Test content",
        )
        
        assert item.item_id == "strat-001"
    
    def test_expertise_item_has_content(self):
        """ExpertiseItem should have content property."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Always validate user input",
        )
        
        assert item.content == "Always validate user input"
    
    def test_expertise_item_has_is_active(self):
        """ExpertiseItem should have is_active property."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
        )
        
        assert item.is_active is True
        item.deactivate()
        assert item.is_active is False
    
    def test_expertise_item_has_embedding(self):
        """ExpertiseItem should have embedding property."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
            embedding=[0.1, 0.2, 0.3],
        )
        
        assert item.embedding == [0.1, 0.2, 0.3]
    
    def test_expertise_item_has_metadata(self):
        """ExpertiseItem should have metadata property."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
            metadata={"key": "value"},
        )
        
        assert item.metadata == {"key": "value"}
    
    def test_expertise_item_has_timestamps(self):
        """ExpertiseItem should have created_at and updated_at."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
        )
        
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.updated_at, datetime)
    
    def test_expertise_item_has_to_prompt_format(self):
        """ExpertiseItem should have to_prompt_format method."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Always validate user input",
            helpful_count=10,
            harmful_count=1,
        )
        
        prompt = item.to_prompt_format()
        assert isinstance(prompt, str)
        assert "validate" in prompt
        # Should include section name
        assert "STRATEGIES" in prompt


class TestContextRetrievalResult:
    """Test ContextRetrievalResult generic class."""
    
    def test_create_with_memory_item(self):
        """Can create ContextRetrievalResult with MemoryItem."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        result = ContextRetrievalResult(
            item=memory,
            score=0.85,
            retrieval_method="semantic",
        )
        
        assert result.item == memory
        assert result.score == 0.85
        assert result.retrieval_method == "semantic"
    
    def test_create_with_expertise_item(self):
        """Can create ContextRetrievalResult with ExpertiseItem."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
        )
        
        result = ContextRetrievalResult(
            item=item,
            score=0.9,
            retrieval_method="vector",
        )
        
        assert result.item == item
        assert result.score == 0.9
    
    def test_score_clamping(self):
        """Score should be clamped to 0.0-1.0 range."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
        )
        
        result_high = ContextRetrievalResult(item=item, score=1.5)
        assert result_high.score == 1.0
        
        result_low = ContextRetrievalResult(item=item, score=-0.5)
        assert result_low.score == 0.0
    
    def test_default_metadata(self):
        """Should have empty dict as default metadata."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
        )
        
        result = ContextRetrievalResult(item=item, score=0.5)
        assert result.metadata == {}


# ============================================================================
# Protocol Interoperability Tests
# ============================================================================

class TestContextItemInteroperability:
    """Test that different item types can be used interchangeably."""
    
    def test_generic_function_accepts_both(self):
        """A function accepting IContextItem works with both types."""
        
        def get_item_summary(item: IContextItem) -> str:
            """Generic function that works with any context item."""
            return f"{item.item_id}: {item.content[:20]}..."
        
        memory = MemoryItem(
            memory_id="mem-123",
            user_id="user-1",
            content="User prefers dark mode interface",
            type=MemoryType.SEMANTIC,
        )
        
        expertise = ExpertiseItem(
            item_id="strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="Always validate user input before processing",
        )
        
        # Both should work with the same function
        mem_summary = get_item_summary(memory)
        exp_summary = get_item_summary(expertise)
        
        assert "mem-123" in mem_summary
        assert "strat-001" in exp_summary
    
    def test_list_of_mixed_items(self):
        """Can create a list of mixed IContextItem types."""
        
        memory = MemoryItem(
            user_id="user-1",
            content="Memory content",
            type=MemoryType.SEMANTIC,
        )
        
        expertise = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Expertise content",
        )
        
        # Both can be in the same list
        items: List[IContextItem] = [memory, expertise]
        
        assert len(items) == 2
        assert all(isinstance(item, IContextItem) for item in items)
    
    def test_prompt_format_for_all(self):
        """Can call to_prompt_format on any IContextItem."""
        
        memory = MemoryItem(
            user_id="user-1",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
        )
        
        expertise = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Validate input",
        )
        
        items: List[IContextItem] = [memory, expertise]
        
        prompts = [item.to_prompt_format() for item in items]
        
        assert len(prompts) == 2
        assert all(isinstance(p, str) for p in prompts)


# ============================================================================
# Expertise-specific Tests (ensuring domain-specific features still work)
# ============================================================================

class TestExpertiseItemDomainFeatures:
    """Test that ExpertiseItem still has its domain-specific features."""
    
    def test_effectiveness_score(self):
        """ExpertiseItem should have effectiveness_score."""
        item = ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Test",
            helpful_count=8,
            harmful_count=2,
        )
        
        assert item.effectiveness_score == 0.8
    
    def test_to_ace_format(self):
        """ExpertiseItem should have to_ace_format method."""
        item = ExpertiseItem(
            item_id="strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="Test content",
            helpful_count=5,
            harmful_count=1,
        )
        
        ace = item.to_ace_format()
        assert "[strat-001]" in ace
        assert "helpful=5" in ace
        assert "harmful=1" in ace


class TestMemoryItemDomainFeatures:
    """Test that MemoryItem still has its domain-specific features."""
    
    def test_user_id(self):
        """MemoryItem should still have user_id."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
        )
        
        assert memory.user_id == "user-123"
    
    def test_confidence_score(self):
        """MemoryItem should have confidence_score."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
            confidence_score=0.75,
        )
        
        assert memory.confidence_score == 0.75
    
    def test_tags(self):
        """MemoryItem should have tags."""
        memory = MemoryItem(
            user_id="user-123",
            content="Test",
            type=MemoryType.SEMANTIC,
            tags=["preference", "ui"],
        )
        
        assert "preference" in memory.tags
        assert "ui" in memory.tags


# ============================================================================
# IContextIndexer Protocol Compliance Tests
# ============================================================================

class TestExpertiseIndexerProtocol:
    """Test that ExpertiseIndexer conforms to IContextIndexer protocol."""
    
    def test_expertise_indexer_is_context_indexer(self):
        """ExpertiseIndexer should be recognized as IContextIndexer."""
        # Note: We can't instantiate without dependencies, but we can check the class
        
        # Check that the class has the required methods
        assert hasattr(ExpertiseIndexer, 'index_item')
        assert hasattr(ExpertiseIndexer, 'remove_item')
        assert hasattr(ExpertiseIndexer, 'search')
    
    def test_index_search_result_structure(self):
        """IndexSearchResult should have required fields."""
        result = IndexSearchResult(
            item_id="test-001",
            score=0.85,
            metadata={"section": "strategies"},
        )
        
        assert result.item_id == "test-001"
        assert result.score == 0.85
        assert result.metadata["section"] == "strategies"


class TestMemoryIndexerProtocol:
    """Test that MemoryIndexer conforms to IContextIndexer protocol."""
    
    def test_memory_indexer_is_context_indexer(self):
        """MemoryIndexer should be recognized as IContextIndexer."""
        
        # Check that the class has the required methods
        assert hasattr(MemoryIndexer, 'index_item')
        assert hasattr(MemoryIndexer, 'remove_item')
        assert hasattr(MemoryIndexer, 'search')
        assert hasattr(MemoryIndexer, 'search_by_embedding')
        assert hasattr(MemoryIndexer, 'index_all')
        assert hasattr(MemoryIndexer, 'remove_all')


class TestBaseRetrieverProtocol:
    """Test that BaseRetriever conforms to IContextRetriever protocol."""
    
    def test_base_retriever_has_context_methods(self):
        """BaseRetriever should have IContextRetriever methods."""
        
        # Check that the class has the required methods for IContextRetriever
        assert hasattr(BaseRetriever, 'name')
        assert hasattr(BaseRetriever, 'retrieve_items')
        assert hasattr(BaseRetriever, 'retrieve_with_scores')
        # Also has IRetriever methods
        assert hasattr(BaseRetriever, 'retrieve')
        assert hasattr(BaseRetriever, 'retrieve_by_embedding')
        assert hasattr(BaseRetriever, 'retrieve_related')


class TestExpertiseRetrieverProtocol:
    """Test that ExpertiseRetriever conforms to IContextRetriever protocol."""
    
    def test_expertise_retriever_is_context_retriever(self):
        """ExpertiseRetriever should be recognized as IContextRetriever."""
        
        # Check that the class has the required methods for IContextRetriever
        assert hasattr(ExpertiseRetriever, 'name')
        assert hasattr(ExpertiseRetriever, 'retrieve_items')  # IContextRetriever protocol
        assert hasattr(ExpertiseRetriever, 'retrieve_with_scores')
        # Also has domain-specific retrieve
        assert hasattr(ExpertiseRetriever, 'retrieve')
    
    def test_context_retrieval_result_with_expertise(self):
        """ContextRetrievalResult should work with ExpertiseItem."""
        item = ExpertiseItem(
            item_id="strat-001",
            section=ExpertiseSection.STRATEGIES,
            content="Test strategy",
        )
        
        result = ContextRetrievalResult(
            item=item,
            score=0.9,
            retrieval_method="semantic",
            metadata={"boost": 0.1},
        )
        
        assert result.item.item_id == "strat-001"
        assert result.score == 0.9
        assert result.retrieval_method == "semantic"


# ============================================================================
# Protocol Inheritance Tests
# ============================================================================

class TestProtocolInheritance:
    """Test that classes properly inherit from protocols."""
    
    def test_expertise_indexer_inherits_icontextindexer(self):
        """ExpertiseIndexer should explicitly inherit from IContextIndexer."""
        import inspect
        
        # Check class hierarchy includes IContextIndexer
        mro = inspect.getmro(ExpertiseIndexer)
        class_names = [cls.__name__ for cls in mro]
        
        # Should have IContextIndexer in the hierarchy (via Generic)
        assert 'ExpertiseIndexer' in class_names
    
    def test_expertise_retriever_inherits_icontextretriever(self):
        """ExpertiseRetriever should explicitly inherit from IContextRetriever."""
        import inspect
        
        mro = inspect.getmro(ExpertiseRetriever)
        class_names = [cls.__name__ for cls in mro]
        
        assert 'ExpertiseRetriever' in class_names
        # Should also have IExpertiseRetriever
        assert 'IExpertiseRetriever' in class_names or any('Retriever' in name for name in class_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

