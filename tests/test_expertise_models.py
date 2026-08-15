"""
Tests for Expertise system core models.

Tests cover:
- ExpertiseSection enum
- ExpertiseItem model
- Expertise model
- CompletedTurn model
- ExpertiseUsageLog model
- ReflectionResult model
- CurationOp and CurationPlan models
- ExpertiseStats model
- SimilarGroup model
"""


import pytest

from ctxforge.core.expertise import (
    CompletedTurn,
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseStats,
    ExpertiseUsageLog,
    ReflectionResult,
    SimilarGroup,
    TurnOutcome,
    UsageFeedback,
)


class TestExpertiseSection:
    """Tests for ExpertiseSection enum."""
    
    def test_section_values(self):
        """Test that all sections have expected values."""
        assert ExpertiseSection.STRATEGIES.value == "strategies_and_insights"
        assert ExpertiseSection.FORMULAS.value == "formulas_and_calculations"
        assert ExpertiseSection.CODE_SNIPPETS.value == "code_snippets_and_templates"
        assert ExpertiseSection.COMMON_MISTAKES.value == "common_mistakes_to_avoid"
        assert ExpertiseSection.HEURISTICS.value == "problem_solving_heuristics"
        assert ExpertiseSection.CONTEXT_CLUES.value == "context_clues_and_indicators"
        assert ExpertiseSection.CUSTOM.value == "custom"
    
    def test_from_string(self):
        """Test parsing section from string."""
        assert ExpertiseSection.from_string("strategies_and_insights") == ExpertiseSection.STRATEGIES
        assert ExpertiseSection.from_string("Strategies And Insights") == ExpertiseSection.STRATEGIES
        assert ExpertiseSection.from_string("FORMULAS & CALCULATIONS") == ExpertiseSection.FORMULAS
        assert ExpertiseSection.from_string("unknown_section") == ExpertiseSection.CUSTOM
    
    def test_to_display_name(self):
        """Test conversion to display name."""
        assert ExpertiseSection.STRATEGIES.to_display_name() == "STRATEGIES AND INSIGHTS"
        assert ExpertiseSection.COMMON_MISTAKES.to_display_name() == "COMMON MISTAKES TO AVOID"
    
    def test_to_slug(self):
        """Test conversion to slug for item IDs."""
        assert ExpertiseSection.STRATEGIES.to_slug() == "strat"
        assert ExpertiseSection.FORMULAS.to_slug() == "form"
        assert ExpertiseSection.CODE_SNIPPETS.to_slug() == "code"
        assert ExpertiseSection.COMMON_MISTAKES.to_slug() == "mist"
        assert ExpertiseSection.HEURISTICS.to_slug() == "heur"
        assert ExpertiseSection.CONTEXT_CLUES.to_slug() == "clue"
        assert ExpertiseSection.CUSTOM.to_slug() == "cust"


class TestUsageFeedback:
    """Tests for UsageFeedback enum."""
    
    def test_feedback_values(self):
        """Test that all feedback types have expected values."""
        assert UsageFeedback.HELPFUL.value == "helpful"
        assert UsageFeedback.HARMFUL.value == "harmful"
        assert UsageFeedback.NEUTRAL.value == "neutral"


class TestTurnOutcome:
    """Tests for TurnOutcome enum."""
    
    def test_outcome_values(self):
        """Test that all outcomes have expected values."""
        assert TurnOutcome.SUCCESS.value == "success"
        assert TurnOutcome.FAILURE.value == "failure"
        assert TurnOutcome.PARTIAL.value == "partial"
        assert TurnOutcome.UNKNOWN.value == "unknown"


class TestCuratorOperation:
    """Tests for CuratorOperation enum."""
    
    def test_operation_values(self):
        """Test that all operations have expected values."""
        assert CuratorOperation.ADD.value == "add"
        assert CuratorOperation.UPDATE.value == "update"
        assert CuratorOperation.MERGE.value == "merge"
        assert CuratorOperation.DELETE.value == "delete"


class TestExpertiseItem:
    """Tests for ExpertiseItem model."""
    
    def test_create_item(self):
        """Test creating an expertise item."""
        item = ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Always greet the customer by name",
        )
        
        assert item.item_id == "strat-00001"
        assert item.section == ExpertiseSection.STRATEGIES
        assert item.content == "Always greet the customer by name"
        assert item.helpful_count == 0
        assert item.harmful_count == 0
        assert item.is_active is True
        assert item.source is None
    
    def test_create_item_with_all_fields(self):
        """Test creating an item with all fields."""
        item = ExpertiseItem(
            item_id="form-00001",
            section=ExpertiseSection.FORMULAS,
            content="Revenue = Units Sold * Price",
            helpful_count=10,
            harmful_count=2,
            source="manual",
            is_active=True,
            embedding=[0.1, 0.2, 0.3],
            metadata={"category": "finance"},
        )
        
        assert item.helpful_count == 10
        assert item.harmful_count == 2
        assert item.source == "manual"
        assert item.embedding == [0.1, 0.2, 0.3]
        assert item.metadata == {"category": "finance"}
    
    def test_content_validation(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            ExpertiseItem(
                item_id="test",
                section=ExpertiseSection.CUSTOM,
                content="",
            )
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            ExpertiseItem(
                item_id="test",
                section=ExpertiseSection.CUSTOM,
                content="   ",
            )
    
    def test_content_stripped(self):
        """Test that content is stripped of whitespace."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="  Hello World  ",
        )
        assert item.content == "Hello World"
    
    def test_effectiveness_score_unused(self):
        """Test effectiveness score for unused items."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
        )
        assert item.effectiveness_score == 0.5
    
    def test_effectiveness_score_all_helpful(self):
        """Test effectiveness score when all uses are helpful."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=10,
            harmful_count=0,
        )
        assert item.effectiveness_score == 1.0
    
    def test_effectiveness_score_all_harmful(self):
        """Test effectiveness score when all uses are harmful."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=0,
            harmful_count=10,
        )
        assert item.effectiveness_score == 0.0
    
    def test_effectiveness_score_mixed(self):
        """Test effectiveness score with mixed usage."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=8,
            harmful_count=2,
        )
        assert item.effectiveness_score == 0.8
    
    def test_total_usage(self):
        """Test total usage calculation."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=5,
            harmful_count=3,
        )
        assert item.total_usage == 8
    
    def test_is_high_performing(self):
        """Test high performing detection."""
        # High performing: helpful > 5, harmful < 2
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=10,
            harmful_count=1,
        )
        assert item.is_high_performing is True
        
        # Not high performing (not enough helpful)
        item2 = ExpertiseItem(
            item_id="test2",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=3,
            harmful_count=0,
        )
        assert item2.is_high_performing is False
        
        # Not high performing (too many harmful)
        item3 = ExpertiseItem(
            item_id="test3",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=10,
            harmful_count=5,
        )
        assert item3.is_high_performing is False
    
    def test_is_problematic(self):
        """Test problematic detection."""
        # Problematic: harmful >= helpful and harmful > 0
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=2,
            harmful_count=5,
        )
        assert item.is_problematic is True
        
        # Not problematic (more helpful)
        item2 = ExpertiseItem(
            item_id="test2",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=10,
            harmful_count=2,
        )
        assert item2.is_problematic is False
        
        # Not problematic (no harmful)
        item3 = ExpertiseItem(
            item_id="test3",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
            helpful_count=0,
            harmful_count=0,
        )
        assert item3.is_problematic is False
    
    def test_is_unused(self):
        """Test unused detection."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
        )
        assert item.is_unused is True
        
        item.helpful_count = 1
        assert item.is_unused is False
    
    def test_increment_helpful(self):
        """Test incrementing helpful count."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
        )
        original_updated = item.updated_at
        
        item.increment_helpful()
        
        assert item.helpful_count == 1
        assert item.updated_at >= original_updated
    
    def test_increment_harmful(self):
        """Test incrementing harmful count."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
        )
        
        item.increment_harmful()
        
        assert item.harmful_count == 1
    
    def test_update_content(self):
        """Test updating content."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Original content",
        )
        
        item.update_content("  New content  ")
        
        assert item.content == "New content"
    
    def test_deactivate_activate(self):
        """Test deactivating and activating."""
        item = ExpertiseItem(
            item_id="test",
            section=ExpertiseSection.CUSTOM,
            content="Test content",
        )
        
        assert item.is_active is True
        
        item.deactivate()
        assert item.is_active is False
        
        item.activate()
        assert item.is_active is True
    
    def test_to_ace_format(self):
        """Test conversion to ACE format."""
        item = ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Always greet the customer",
            helpful_count=5,
            harmful_count=1,
        )
        
        expected = "[strat-00001] helpful=5 harmful=1 :: Always greet the customer"
        assert item.to_ace_format() == expected
    
    def test_from_ace_format(self):
        """Test parsing from ACE format."""
        line = "[strat-00001] helpful=5 harmful=1 :: Always greet the customer"
        
        item = ExpertiseItem.from_ace_format(line, ExpertiseSection.STRATEGIES)
        
        assert item is not None
        assert item.item_id == "strat-00001"
        assert item.content == "Always greet the customer"
        assert item.helpful_count == 5
        assert item.harmful_count == 1
        assert item.section == ExpertiseSection.STRATEGIES
    
    def test_from_ace_format_invalid(self):
        """Test parsing invalid ACE format."""
        invalid_line = "This is not valid ACE format"
        
        item = ExpertiseItem.from_ace_format(invalid_line)
        
        assert item is None


class TestExpertise:
    """Tests for Expertise model."""
    
    def test_create_expertise(self):
        """Test creating an expertise."""
        expertise = Expertise(
            expertise_id="test-001",
            name="Customer Support",
        )
        
        assert expertise.expertise_id == "test-001"
        assert expertise.name == "Customer Support"
        assert expertise.domain is None
        assert expertise.items == []
        assert expertise.version == 1
        assert expertise.token_budget == 80000
        assert expertise.next_item_id == 1
    
    def test_create_expertise_with_domain(self):
        """Test creating expertise with domain."""
        expertise = Expertise(
            expertise_id="test-001",
            name="Finance Expert",
            domain="finance",
        )
        
        assert expertise.domain == "finance"
    
    def test_name_validation(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Expertise(expertise_id="test", name="")
        
        with pytest.raises(ValueError, match="name cannot be empty"):
            Expertise(expertise_id="test", name="   ")
    
    def test_active_items(self):
        """Test getting active items."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        item1 = ExpertiseItem(
            item_id="item1",
            section=ExpertiseSection.STRATEGIES,
            content="Active item",
            is_active=True,
        )
        item2 = ExpertiseItem(
            item_id="item2",
            section=ExpertiseSection.STRATEGIES,
            content="Inactive item",
            is_active=False,
        )
        
        expertise.items = [item1, item2]
        
        assert len(expertise.active_items) == 1
        assert expertise.active_items[0].item_id == "item1"
    
    def test_item_count(self):
        """Test item count properties."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        expertise.items = [
            ExpertiseItem(item_id="1", section=ExpertiseSection.CUSTOM, content="A", is_active=True),
            ExpertiseItem(item_id="2", section=ExpertiseSection.CUSTOM, content="B", is_active=False),
            ExpertiseItem(item_id="3", section=ExpertiseSection.CUSTOM, content="C", is_active=True),
        ]
        
        assert expertise.item_count == 3
        assert expertise.active_item_count == 2
    
    def test_get_item(self):
        """Test getting item by ID."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        item = ExpertiseItem(
            item_id="target",
            section=ExpertiseSection.STRATEGIES,
            content="Target item",
        )
        expertise.items = [item]
        
        assert expertise.get_item("target") == item
        assert expertise.get_item("nonexistent") is None
    
    def test_get_items_by_section(self):
        """Test getting items by section."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        expertise.items = [
            ExpertiseItem(item_id="1", section=ExpertiseSection.STRATEGIES, content="A"),
            ExpertiseItem(item_id="2", section=ExpertiseSection.FORMULAS, content="B"),
            ExpertiseItem(item_id="3", section=ExpertiseSection.STRATEGIES, content="C"),
        ]
        
        strategies = expertise.get_items_by_section(ExpertiseSection.STRATEGIES)
        
        assert len(strategies) == 2
        assert all(item.section == ExpertiseSection.STRATEGIES for item in strategies)
    
    def test_add_item(self):
        """Test adding an item."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        item = expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="New strategy",
            source="manual",
        )
        
        assert item.item_id == "strat-00001"
        assert item.section == ExpertiseSection.STRATEGIES
        assert item.content == "New strategy"
        assert item.source == "manual"
        assert len(expertise.items) == 1
        assert expertise.next_item_id == 2
    
    def test_add_multiple_items(self):
        """Test adding multiple items with incrementing IDs."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        item1 = expertise.add_item(ExpertiseSection.STRATEGIES, "First")
        item2 = expertise.add_item(ExpertiseSection.STRATEGIES, "Second")
        item3 = expertise.add_item(ExpertiseSection.FORMULAS, "Third")
        
        assert item1.item_id == "strat-00001"
        assert item2.item_id == "strat-00002"
        assert item3.item_id == "form-00003"
        assert expertise.next_item_id == 4
    
    def test_remove_item_soft_delete(self):
        """Test removing item with soft delete."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        item = expertise.add_item(ExpertiseSection.STRATEGIES, "To remove")
        
        result = expertise.remove_item(item.item_id, soft_delete=True)
        
        assert result is True
        assert len(expertise.items) == 1  # Still there
        assert expertise.items[0].is_active is False  # But inactive
    
    def test_remove_item_hard_delete(self):
        """Test removing item with hard delete."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        item = expertise.add_item(ExpertiseSection.STRATEGIES, "To remove")
        
        result = expertise.remove_item(item.item_id, soft_delete=False)
        
        assert result is True
        assert len(expertise.items) == 0
    
    def test_remove_nonexistent_item(self):
        """Test removing nonexistent item."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        result = expertise.remove_item("nonexistent")
        
        assert result is False
    
    def test_update_item_counts(self):
        """Test updating item counts."""
        expertise = Expertise(expertise_id="test", name="Test")
        item = expertise.add_item(ExpertiseSection.STRATEGIES, "Test item")
        
        result = expertise.update_item_counts(item.item_id, helpful_delta=3, harmful_delta=1)
        
        assert result is True
        assert item.helpful_count == 3
        assert item.harmful_count == 1
    
    def test_update_item_counts_nonexistent(self):
        """Test updating counts for nonexistent item."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        result = expertise.update_item_counts("nonexistent", helpful_delta=1)
        
        assert result is False
    
    def test_increment_version(self):
        """Test incrementing version."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        assert expertise.version == 1
        
        expertise.increment_version()
        
        assert expertise.version == 2
    
    def test_to_ace_format(self):
        """Test conversion to ACE format."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        expertise.add_item(ExpertiseSection.STRATEGIES, "Strategy 1")
        expertise.add_item(ExpertiseSection.FORMULAS, "Formula 1")
        
        ace_format = expertise.to_ace_format()
        
        assert "## STRATEGIES AND INSIGHTS" in ace_format
        assert "[strat-00001]" in ace_format
        assert "Strategy 1" in ace_format
        assert "## FORMULAS AND CALCULATIONS" in ace_format
        assert "[form-00002]" in ace_format
        assert "Formula 1" in ace_format
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        expertise.add_item(ExpertiseSection.STRATEGIES, "This is a test strategy with some content")
        
        tokens = expertise.estimate_tokens()
        
        assert tokens > 0


class TestCompletedTurn:
    """Tests for CompletedTurn model."""
    
    def test_create_turn(self):
        """Test creating a completed turn."""
        turn = CompletedTurn(
            user_input="What is the weather?",
            assistant_response="The weather is sunny.",
        )
        
        assert turn.user_input == "What is the weather?"
        assert turn.assistant_response == "The weather is sunny."
        assert turn.expected_output is None
        assert turn.actual_outcome is None
        assert turn.metadata == {}
    
    def test_create_turn_with_all_fields(self):
        """Test creating turn with all fields."""
        turn = CompletedTurn(
            user_input="Calculate 2+2",
            assistant_response="The answer is 4",
            expected_output="4",
            actual_outcome="correct",
            metadata={"latency_ms": 100},
        )
        
        assert turn.expected_output == "4"
        assert turn.actual_outcome == "correct"
        assert turn.metadata == {"latency_ms": 100}


class TestExpertiseUsageLog:
    """Tests for ExpertiseUsageLog model."""
    
    def test_create_log(self):
        """Test creating a usage log."""
        log = ExpertiseUsageLog(
            session_id="session-001",
            expertise_id="expertise-001",
            items_used=["strat-00001", "strat-00002"],
            feedback={"strat-00001": UsageFeedback.HELPFUL},
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert log.session_id == "session-001"
        assert log.expertise_id == "expertise-001"
        assert len(log.items_used) == 2
        assert log.feedback["strat-00001"] == UsageFeedback.HELPFUL
        assert log.outcome == TurnOutcome.SUCCESS


class TestReflectionResult:
    """Tests for ReflectionResult model."""
    
    def test_create_result(self):
        """Test creating a reflection result."""
        result = ReflectionResult(
            item_feedback={
                "strat-00001": UsageFeedback.HELPFUL,
                "strat-00002": UsageFeedback.HARMFUL,
            },
            insights="The first strategy worked well, but the second was misleading.",
            suggested_additions=["Add a strategy for edge cases"],
            suggested_removals=["strat-00002"],
            confidence=0.85,
        )
        
        assert result.item_feedback["strat-00001"] == UsageFeedback.HELPFUL
        assert "first strategy" in result.insights
        assert len(result.suggested_additions) == 1
        assert len(result.suggested_removals) == 1
        assert result.confidence == 0.85
    
    def test_has_suggestions(self):
        """Test has_suggestions property."""
        # No suggestions
        result1 = ReflectionResult()
        assert result1.has_suggestions is False
        
        # With additions
        result2 = ReflectionResult(suggested_additions=["new item"])
        assert result2.has_suggestions is True
        
        # With removals
        result3 = ReflectionResult(suggested_removals=["old item"])
        assert result3.has_suggestions is True
    
    def test_helpful_harmful_items(self):
        """Test helpful and harmful item properties."""
        result = ReflectionResult(
            item_feedback={
                "item1": UsageFeedback.HELPFUL,
                "item2": UsageFeedback.HARMFUL,
                "item3": UsageFeedback.HELPFUL,
                "item4": UsageFeedback.NEUTRAL,
            }
        )
        
        assert set(result.helpful_items) == {"item1", "item3"}
        assert result.harmful_items == ["item2"]


class TestCurationOp:
    """Tests for CurationOp model."""
    
    def test_create_add_op(self):
        """Test creating an ADD operation."""
        op = CurationOp(
            type=CuratorOperation.ADD,
            section=ExpertiseSection.STRATEGIES,
            content="New strategy to add",
            reason="Identified from reflection",
        )
        
        assert op.type == CuratorOperation.ADD
        assert op.section == ExpertiseSection.STRATEGIES
        assert op.content == "New strategy to add"
        assert op.reason == "Identified from reflection"
    
    def test_create_merge_op(self):
        """Test creating a MERGE operation."""
        op = CurationOp(
            type=CuratorOperation.MERGE,
            item_ids=["strat-00001", "strat-00002"],
            reason="These items are duplicates",
        )
        
        assert op.type == CuratorOperation.MERGE
        assert len(op.item_ids) == 2


class TestCurationPlan:
    """Tests for CurationPlan model."""
    
    def test_create_plan(self):
        """Test creating a curation plan."""
        plan = CurationPlan(
            operations=[
                CurationOp(type=CuratorOperation.ADD, content="New item"),
                CurationOp(type=CuratorOperation.DELETE, item_ids=["old-item"]),
            ],
            reasoning="Based on recent reflection feedback",
        )
        
        assert plan.operation_count == 2
        assert plan.has_operations is True
        assert "reflection" in plan.reasoning
    
    def test_empty_plan(self):
        """Test empty curation plan."""
        plan = CurationPlan()
        
        assert plan.operation_count == 0
        assert plan.has_operations is False
    
    def test_get_operations_by_type(self):
        """Test getting operations by type."""
        plan = CurationPlan(
            operations=[
                CurationOp(type=CuratorOperation.ADD, content="Item 1"),
                CurationOp(type=CuratorOperation.ADD, content="Item 2"),
                CurationOp(type=CuratorOperation.DELETE, item_ids=["old"]),
            ]
        )
        
        add_ops = plan.get_operations_by_type(CuratorOperation.ADD)
        
        assert len(add_ops) == 2


class TestExpertiseStats:
    """Tests for ExpertiseStats model."""
    
    def test_from_expertise(self):
        """Test calculating stats from expertise."""
        expertise = Expertise(expertise_id="test", name="Test")
        
        # Add various items
        expertise.items = [
            ExpertiseItem(
                item_id="1",
                section=ExpertiseSection.STRATEGIES,
                content="High performer",
                helpful_count=10,
                harmful_count=1,
            ),
            ExpertiseItem(
                item_id="2",
                section=ExpertiseSection.STRATEGIES,
                content="Problematic",
                helpful_count=2,
                harmful_count=5,
            ),
            ExpertiseItem(
                item_id="3",
                section=ExpertiseSection.FORMULAS,
                content="Unused",
                helpful_count=0,
                harmful_count=0,
            ),
            ExpertiseItem(
                item_id="4",
                section=ExpertiseSection.FORMULAS,
                content="Inactive",
                helpful_count=5,
                harmful_count=0,
                is_active=False,
            ),
        ]
        
        stats = ExpertiseStats.from_expertise(expertise)
        
        assert stats.total_items == 4
        assert stats.active_items == 3
        assert stats.high_performing == 1
        assert stats.problematic == 1
        assert stats.unused == 1
        assert stats.total_helpful == 12
        assert stats.total_harmful == 6
        assert stats.items_by_section["strategies_and_insights"] == 2
        assert stats.items_by_section["formulas_and_calculations"] == 1
        assert stats.average_effectiveness > 0


class TestSimilarGroup:
    """Tests for SimilarGroup model."""
    
    def test_create_group(self):
        """Test creating a similar group."""
        items = [
            ExpertiseItem(
                item_id="1",
                section=ExpertiseSection.STRATEGIES,
                content="Similar item 1",
                helpful_count=5,
                harmful_count=1,
            ),
            ExpertiseItem(
                item_id="2",
                section=ExpertiseSection.STRATEGIES,
                content="Similar item 2",
                helpful_count=3,
                harmful_count=2,
            ),
        ]
        
        group = SimilarGroup(
            items=items,
            similarity_scores=[1.0, 0.95],
        )
        
        assert group.item_count == 2
        assert group.item_ids == ["1", "2"]
        assert group.primary_item == items[0]
        assert group.total_helpful == 8
        assert group.total_harmful == 3
    
    def test_empty_group(self):
        """Test empty similar group."""
        group = SimilarGroup()
        
        assert group.item_count == 0
        assert group.item_ids == []
        assert group.primary_item is None
        assert group.total_helpful == 0
        assert group.total_harmful == 0

