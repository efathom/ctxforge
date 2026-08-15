"""
Tests for Structured Knowledge Types.

Tests knowledge type classification, filtering, and priority boosting.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from ctxforge.core.knowledge_types import (
    HeuristicKnowledgeClassifier,
    KnowledgeScope,
    KnowledgeType,
    StructuredKnowledge,
    classify_knowledge,
    classify_knowledge_async,
)
from ctxforge.retrieval.filters.knowledge_type import KnowledgeTypeFilter


class TestKnowledgeType:
    """Tests for KnowledgeType enum."""
    
    def test_all_types_exist(self):
        """Test all knowledge types exist."""
        assert KnowledgeType.RULE == "rule"
        assert KnowledgeType.PATTERN == "pattern"
        assert KnowledgeType.GOTCHA == "gotcha"
        assert KnowledgeType.EXAMPLE == "example"
        assert KnowledgeType.DEFINITION == "definition"
        assert KnowledgeType.PROCEDURE == "procedure"
        assert KnowledgeType.INSIGHT == "insight"
        assert KnowledgeType.CONSTRAINT == "constraint"


class TestKnowledgeScope:
    """Tests for KnowledgeScope enum."""
    
    def test_all_scopes_exist(self):
        """Test all knowledge scopes exist."""
        assert KnowledgeScope.GLOBAL == "global"
        assert KnowledgeScope.ENTITY == "entity"
        assert KnowledgeScope.OPERATION == "operation"
        assert KnowledgeScope.USER == "user"


class TestStructuredKnowledge:
    """Tests for StructuredKnowledge model."""
    
    def test_create_rule(self):
        """Test creating a rule knowledge item."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.RULE,
            content="Always validate user input",
            priority=10,
        )
        
        assert knowledge.knowledge_type == KnowledgeType.RULE
        assert knowledge.content == "Always validate user input"
        assert knowledge.priority == 10
        assert knowledge.scope == KnowledgeScope.GLOBAL  # default
    
    def test_create_pattern(self):
        """Test creating a pattern knowledge item."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.PATTERN,
            name="User Query",
            content="SELECT * FROM users WHERE id = ?",
            scope=KnowledgeScope.ENTITY,
            applies_to=["users", "user_profiles"],
            source_question="How do I get a user by ID?",
        )
        
        assert knowledge.knowledge_type == KnowledgeType.PATTERN
        assert knowledge.name == "User Query"
        assert len(knowledge.applies_to) == 2
    
    def test_matches_entity_global_scope(self):
        """Test entity matching with global scope."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.RULE,
            content="Global rule",
            scope=KnowledgeScope.GLOBAL,
        )
        
        assert knowledge.matches_entity("any_entity") is True
        assert knowledge.matches_entity("other_entity") is True
    
    def test_matches_entity_specific_scope(self):
        """Test entity matching with entity scope."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.PATTERN,
            content="Pattern",
            scope=KnowledgeScope.ENTITY,
            applies_to=["users", "orders"],
        )
        
        assert knowledge.matches_entity("users") is True
        assert knowledge.matches_entity("orders") is True
        assert knowledge.matches_entity("products") is False
    
    def test_matches_entity_prefix(self):
        """Test entity matching with prefix."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.PATTERN,
            content="Pattern",
            scope=KnowledgeScope.ENTITY,
            applies_to=["user"],
        )
        
        assert knowledge.matches_entity("user_profiles") is True
        assert knowledge.matches_entity("users") is True
        assert knowledge.matches_entity("products") is False
    
    def test_to_prompt_format_with_name(self):
        """Test prompt format generation with name."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.RULE,
            name="Input Validation",
            content="Always validate user input",
        )
        
        prompt = knowledge.to_prompt_format()
        
        assert "📋 RULE" in prompt
        assert "Input Validation" in prompt
        assert "Always validate user input" in prompt
    
    def test_to_prompt_format_without_name(self):
        """Test prompt format generation without name."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.GOTCHA,
            content="Don't forget null checks",
        )
        
        prompt = knowledge.to_prompt_format()
        
        assert "⚠️ GOTCHA" in prompt
        assert "Don't forget null checks" in prompt
    
    def test_priority_validation(self):
        """Test priority validation (1-10)."""
        knowledge = StructuredKnowledge(
            knowledge_type=KnowledgeType.RULE,
            content="Content",
            priority=5,
        )
        assert knowledge.priority == 5
        
        # Test bounds
        with pytest.raises(ValueError):
            StructuredKnowledge(
                knowledge_type=KnowledgeType.RULE,
                content="Content",
                priority=0,  # Below minimum
            )
        
        with pytest.raises(ValueError):
            StructuredKnowledge(
                knowledge_type=KnowledgeType.RULE,
                content="Content",
                priority=11,  # Above maximum
            )


class TestClassifyKnowledge:
    """Tests for classify_knowledge function."""
    
    def test_classify_rule(self):
        """Test classifying rules."""
        assert classify_knowledge("You must always validate input") == KnowledgeType.RULE
        assert classify_knowledge("Never store passwords in plaintext") == KnowledgeType.RULE
        assert classify_knowledge("This is required for all requests") == KnowledgeType.RULE
    
    def test_classify_gotcha(self):
        """Test classifying gotchas."""
        assert classify_knowledge("Don't forget to close connections") == KnowledgeType.GOTCHA
        assert classify_knowledge("Avoid using SELECT *") == KnowledgeType.GOTCHA
        assert classify_knowledge("Common mistake: not handling null") == KnowledgeType.GOTCHA
    
    def test_classify_pattern(self):
        """Test classifying patterns."""
        assert classify_knowledge("SELECT id FROM users WHERE email = ?") == KnowledgeType.PATTERN
        assert classify_knowledge("def process_data(input):") == KnowledgeType.PATTERN
    
    def test_classify_procedure(self):
        """Test classifying procedures."""
        assert classify_knowledge("Step 1: Open the file") == KnowledgeType.PROCEDURE
        assert classify_knowledge("First, validate the input. Then, process it.") == KnowledgeType.PROCEDURE
    
    def test_classify_definition(self):
        """Test classifying definitions."""
        assert classify_knowledge("A transaction means an atomic operation") == KnowledgeType.DEFINITION
        assert classify_knowledge("ACID is defined as Atomicity, Consistency...") == KnowledgeType.DEFINITION
    
    def test_classify_constraint(self):
        """Test classifying constraints."""
        assert classify_knowledge("Limit results to 100 rows maximum") == KnowledgeType.CONSTRAINT
        assert classify_knowledge("Cannot exceed 5 retries") == KnowledgeType.CONSTRAINT
    
    def test_classify_example(self):
        """Test classifying examples."""
        assert classify_knowledge("For example, use a prepared statement") == KnowledgeType.EXAMPLE
        assert classify_knowledge("Such as: user.name, user.email") == KnowledgeType.EXAMPLE
    
    def test_classify_default_insight(self):
        """Test default classification to insight."""
        assert classify_knowledge("The system works well under load") == KnowledgeType.INSIGHT
        assert classify_knowledge("Users tend to prefer dark mode") == KnowledgeType.INSIGHT


class TestKnowledgeTypeFilter:
    """Tests for KnowledgeTypeFilter."""
    
    def test_should_include_all_by_default(self):
        """Test that all types are included by default."""
        filter = KnowledgeTypeFilter()
        
        for kt in KnowledgeType:
            assert filter.should_include(kt) is True
    
    def test_should_include_only_specified(self):
        """Test including only specified types."""
        filter = KnowledgeTypeFilter(
            include_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA]
        )
        
        assert filter.should_include(KnowledgeType.RULE) is True
        assert filter.should_include(KnowledgeType.GOTCHA) is True
        assert filter.should_include(KnowledgeType.PATTERN) is False
        assert filter.should_include(KnowledgeType.INSIGHT) is False
    
    def test_should_exclude_specified(self):
        """Test excluding specified types."""
        filter = KnowledgeTypeFilter(
            exclude_types=[KnowledgeType.INSIGHT]
        )
        
        assert filter.should_include(KnowledgeType.RULE) is True
        assert filter.should_include(KnowledgeType.INSIGHT) is False
    
    def test_exclude_takes_precedence(self):
        """Test that exclude takes precedence over include."""
        filter = KnowledgeTypeFilter(
            include_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA],
            exclude_types=[KnowledgeType.GOTCHA],
        )
        
        assert filter.should_include(KnowledgeType.RULE) is True
        assert filter.should_include(KnowledgeType.GOTCHA) is False
    
    def test_get_priority_boost(self):
        """Test priority boost calculation."""
        filter = KnowledgeTypeFilter(
            priority_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA, KnowledgeType.CONSTRAINT]
        )
        
        # First priority gets highest boost
        assert filter.get_priority_boost(KnowledgeType.RULE) == pytest.approx(0.2)
        assert filter.get_priority_boost(KnowledgeType.GOTCHA) == pytest.approx(0.15)
        assert filter.get_priority_boost(KnowledgeType.CONSTRAINT) == pytest.approx(0.1)
        
        # Non-priority types get no boost
        assert filter.get_priority_boost(KnowledgeType.INSIGHT) == 0.0
    
    def test_filter_items(self):
        """Test filtering a list of items."""
        @dataclass
        class MockItem:
            knowledge_type: KnowledgeType
            content: str
        
        items = [
            MockItem(KnowledgeType.RULE, "Rule 1"),
            MockItem(KnowledgeType.GOTCHA, "Gotcha 1"),
            MockItem(KnowledgeType.INSIGHT, "Insight 1"),
            MockItem(KnowledgeType.PATTERN, "Pattern 1"),
        ]
        
        filter = KnowledgeTypeFilter(
            include_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA]
        )
        
        filtered = filter.filter_items(items)
        
        assert len(filtered) == 2
        assert all(item.knowledge_type in [KnowledgeType.RULE, KnowledgeType.GOTCHA] for item in filtered)
    
    def test_sort_by_priority(self):
        """Test sorting items by priority."""
        @dataclass
        class MockItem:
            knowledge_type: KnowledgeType
            content: str
            score: float = 0.5
        
        items = [
            MockItem(KnowledgeType.INSIGHT, "Insight 1", 0.9),
            MockItem(KnowledgeType.RULE, "Rule 1", 0.5),
            MockItem(KnowledgeType.GOTCHA, "Gotcha 1", 0.7),
        ]
        
        filter = KnowledgeTypeFilter(
            priority_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA]
        )
        
        sorted_items = filter.sort_by_priority(items)
        
        # Insight has highest base score but no priority boost
        # Rule: 0.5 + 0.2 = 0.7
        # Gotcha: 0.7 + 0.15 = 0.85
        # Insight: 0.9 + 0 = 0.9
        
        # Order should be: Insight (0.9), Gotcha (0.85), Rule (0.7)
        assert sorted_items[0].content == "Insight 1"
        assert sorted_items[1].content == "Gotcha 1"
        assert sorted_items[2].content == "Rule 1"
    
    def test_filter_items_with_metadata(self):
        """Test filtering items with knowledge_type in metadata."""
        @dataclass
        class MockItemWithMetadata:
            content: str
            metadata: dict
            knowledge_type: Optional[KnowledgeType] = None
        
        items = [
            MockItemWithMetadata("Item 1", {"knowledge_type": "rule"}),
            MockItemWithMetadata("Item 2", {"knowledge_type": "insight"}),
            MockItemWithMetadata("Item 3", {}),  # Will default to INSIGHT
        ]
        
        filter = KnowledgeTypeFilter(
            include_types=[KnowledgeType.RULE]
        )
        
        filtered = filter.filter_items(items)
        
        assert len(filtered) == 1
        assert filtered[0].content == "Item 1"


class TestHeuristicKnowledgeClassifier:
    """Tests for the improved HeuristicKnowledgeClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        return HeuristicKnowledgeClassifier()
    
    @pytest.mark.asyncio
    async def test_classify_with_confidence(self, classifier):
        """Test classification returns confidence score."""
        kt, confidence = await classifier.classify("Always validate user input")
        
        assert kt == KnowledgeType.RULE
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.5  # Should have reasonable confidence
    
    @pytest.mark.asyncio
    async def test_classify_sql_pattern(self, classifier):
        """Test SQL pattern detection."""
        kt, confidence = await classifier.classify("SELECT * FROM users WHERE id = ?")
        
        assert kt == KnowledgeType.PATTERN
        assert confidence >= 0.9  # High confidence for SQL
    
    @pytest.mark.asyncio
    async def test_classify_code_pattern(self, classifier):
        """Test code pattern detection."""
        kt, confidence = await classifier.classify("def process_data(input):\n    return input.strip()")
        
        assert kt == KnowledgeType.PATTERN
    
    @pytest.mark.asyncio
    async def test_classify_gotcha_with_warning(self, classifier):
        """Test gotcha detection with warning phrases."""
        kt, confidence = await classifier.classify("Watch out for null values in joins")
        
        assert kt == KnowledgeType.GOTCHA
    
    @pytest.mark.asyncio
    async def test_classify_numbered_procedure(self, classifier):
        """Test procedure detection with numbered steps."""
        kt, confidence = await classifier.classify("1. Open the file\n2. Read contents\n3. Process data")
        
        assert kt == KnowledgeType.PROCEDURE
    
    @pytest.mark.asyncio
    async def test_classify_low_confidence_returns_insight(self):
        """Test that low confidence returns INSIGHT."""
        classifier = HeuristicKnowledgeClassifier(min_confidence=0.99)
        
        kt, confidence = await classifier.classify("Some general text")
        
        assert kt == KnowledgeType.INSIGHT
    
    @pytest.mark.asyncio
    async def test_multiple_matches_boost_confidence(self, classifier):
        """Test that multiple pattern matches increase confidence."""
        # Contains multiple rule indicators
        kt1, conf1 = await classifier.classify("always validate")
        kt2, conf2 = await classifier.classify("you must always validate and never skip")
        
        assert kt1 == kt2 == KnowledgeType.RULE
        # Multiple matches should give higher confidence
        assert conf2 >= conf1


class TestClassifyKnowledgeAsync:
    """Tests for async classification function."""
    
    @pytest.mark.asyncio
    async def test_classify_async_default_classifier(self):
        """Test async classification with default classifier."""
        kt, confidence = await classify_knowledge_async(
            "Don't forget to close database connections"
        )
        
        assert kt == KnowledgeType.GOTCHA
        assert confidence > 0
    
    @pytest.mark.asyncio
    async def test_classify_async_custom_classifier(self):
        """Test async classification with custom classifier."""
        classifier = HeuristicKnowledgeClassifier(min_confidence=0.1)
        
        kt, confidence = await classify_knowledge_async(
            "SELECT name FROM products",
            classifier=classifier,
        )
        
        assert kt == KnowledgeType.PATTERN
