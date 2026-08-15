"""
Tests for the Expertise Curator.

Tests curation operations, validation, and application.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.expertise import (
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ReflectionResult,
    UsageFeedback,
)
from ctxforge.expertise.curator import (
    ExpertiseCurator,
    MockCurator,
    RuleBasedCurator,
)
from ctxforge.expertise.operations import (
    apply_add_operation,
    apply_curation_plan,
    apply_delete_operation,
    apply_merge_operation,
    apply_update_operation,
    format_item_line,
    generate_item_id,
    parse_item_line,
    validate_operation,
)


# Test fixtures
@pytest.fixture
def expertise():
    """Create sample expertise for testing."""
    exp = Expertise(
        expertise_id="test-expertise",
        name="Test Expertise",
        domain="testing",
    )
    
    # Add some items
    exp.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="Always start with a greeting",
    )
    exp.add_item(
        section=ExpertiseSection.FORMULAS,
        content="Use formula: price = cost * (1 + margin)",
    )
    exp.add_item(
        section=ExpertiseSection.COMMON_MISTAKES,
        content="Don't forget to validate input",
    )
    
    # Set some usage counts
    exp.items[0].helpful_count = 10
    exp.items[0].harmful_count = 2
    exp.items[2].helpful_count = 2
    exp.items[2].harmful_count = 8  # Problematic item
    
    return exp


@pytest.fixture
def reflection_result():
    """Create sample reflection result."""
    return ReflectionResult(
        item_feedback={
            "strat-00001": UsageFeedback.HELPFUL,
            "form-00002": UsageFeedback.NEUTRAL,
            "mist-00003": UsageFeedback.HARMFUL,
        },
        insights="The greeting helped establish rapport, but input validation was misapplied.",
        suggested_additions=["Add error handling for edge cases"],
        suggested_removals=["mist-00003 - consistently harmful"],
        confidence=0.85,
    )


@pytest.fixture
def mock_curator_response():
    """Create a mock curator LLM response."""
    return json.dumps({
        "reasoning": "Based on reflection, we should add error handling guidance.",
        "operations": [
            {
                "type": "ADD",
                "section": "strategies_and_insights",
                "content": "Handle edge cases with proper error handling",
                "reason": "Suggested by reflection"
            }
        ]
    })


class TestOperationHelpers:
    """Tests for operation helper functions."""
    
    def test_generate_item_id(self):
        """Test item ID generation."""
        assert generate_item_id(ExpertiseSection.STRATEGIES, 1) == "strat-00001"
        assert generate_item_id(ExpertiseSection.FORMULAS, 42) == "form-00042"
        assert generate_item_id(ExpertiseSection.COMMON_MISTAKES, 100) == "mist-00100"
    
    def test_format_item_line(self):
        """Test item line formatting."""
        item = ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Test content",
            helpful_count=5,
            harmful_count=2,
        )
        
        result = format_item_line(item)
        
        assert result == "[strat-00001] helpful=5 harmful=2 :: Test content"
    
    def test_parse_item_line(self):
        """Test item line parsing."""
        line = "[strat-00001] helpful=5 harmful=2 :: Test content here"
        
        item = parse_item_line(line, ExpertiseSection.STRATEGIES)
        
        assert item is not None
        assert item.item_id == "strat-00001"
        assert item.helpful_count == 5
        assert item.harmful_count == 2
        assert item.content == "Test content here"
        assert item.section == ExpertiseSection.STRATEGIES
    
    def test_parse_item_line_invalid(self):
        """Test parsing invalid line returns None."""
        assert parse_item_line("invalid line") is None
        assert parse_item_line("") is None


class TestApplyOperations:
    """Tests for operation application functions."""
    
    def test_apply_add_operation(self, expertise):
        """Test applying ADD operation."""
        initial_count = len(expertise.items)
        
        op = CurationOp(
            type=CuratorOperation.ADD,
            section=ExpertiseSection.HEURISTICS,
            content="New heuristic content",
            reason="Test add",
        )
        
        item = apply_add_operation(expertise, op)
        
        assert item is not None
        assert len(expertise.items) == initial_count + 1
        assert item.section == ExpertiseSection.HEURISTICS
        assert item.content == "New heuristic content"
        assert item.source == "curator"
    
    def test_apply_add_operation_no_content(self, expertise):
        """Test ADD operation with no content returns None."""
        op = CurationOp(
            type=CuratorOperation.ADD,
            section=ExpertiseSection.STRATEGIES,
            content=None,
        )
        
        result = apply_add_operation(expertise, op)
        
        assert result is None
    
    def test_apply_update_operation(self, expertise):
        """Test applying UPDATE operation."""
        original_content = expertise.items[0].content
        
        op = CurationOp(
            type=CuratorOperation.UPDATE,
            item_ids=["strat-00001"],
            content="Updated greeting strategy",
        )
        
        result = apply_update_operation(expertise, op)
        
        assert result is True
        assert expertise.items[0].content == "Updated greeting strategy"
        assert expertise.items[0].content != original_content
    
    def test_apply_update_operation_invalid_id(self, expertise):
        """Test UPDATE with invalid ID returns False."""
        op = CurationOp(
            type=CuratorOperation.UPDATE,
            item_ids=["invalid-99999"],
            content="Updated content",
        )
        
        result = apply_update_operation(expertise, op)
        
        assert result is False
    
    def test_apply_merge_operation(self, expertise):
        """Test applying MERGE operation."""
        # Add another formula item to merge
        expertise.add_item(
            section=ExpertiseSection.FORMULAS,
            content="Another formula item",
        )
        expertise.items[-1].helpful_count = 3
        expertise.items[-1].harmful_count = 1
        
        _initial_active = expertise.active_item_count
        item1_helpful = expertise.items[1].helpful_count
        item2_helpful = expertise.items[-1].helpful_count
        
        op = CurationOp(
            type=CuratorOperation.MERGE,
            item_ids=["form-00002", "form-00004"],
            content="Merged formula content",
        )
        
        merged = apply_merge_operation(expertise, op)
        
        assert merged is not None
        assert merged.content == "Merged formula content"
        assert merged.helpful_count == item1_helpful + item2_helpful
        # Original items should be deactivated
        assert not expertise.get_item("form-00002").is_active
        assert not expertise.get_item("form-00004").is_active
    
    def test_apply_delete_operation(self, expertise):
        """Test applying DELETE operation."""
        assert expertise.get_item("strat-00001").is_active is True
        
        op = CurationOp(
            type=CuratorOperation.DELETE,
            item_ids=["strat-00001"],
        )
        
        result = apply_delete_operation(expertise, op)
        
        assert result is True
        assert expertise.get_item("strat-00001").is_active is False


class TestCurationPlanApplication:
    """Tests for full curation plan application."""
    
    def test_apply_empty_plan(self, expertise):
        """Test applying empty plan."""
        plan = CurationPlan(operations=[], reasoning="No changes")
        original_version = expertise.version
        
        updated, adds, updates, merges, deletes = apply_curation_plan(expertise, plan)
        
        assert adds == 0
        assert updates == 0
        assert merges == 0
        assert deletes == 0
        assert expertise.version == original_version  # No version bump
    
    def test_apply_plan_with_multiple_operations(self, expertise):
        """Test applying plan with multiple operations."""
        plan = CurationPlan(
            operations=[
                CurationOp(
                    type=CuratorOperation.ADD,
                    section=ExpertiseSection.STRATEGIES,
                    content="New strategy",
                ),
                CurationOp(
                    type=CuratorOperation.DELETE,
                    item_ids=["mist-00003"],
                ),
            ],
            reasoning="Test plan",
        )
        
        original_version = expertise.version
        updated, adds, updates, merges, deletes = apply_curation_plan(expertise, plan)
        
        assert adds == 1
        assert deletes == 1
        assert expertise.version == original_version + 1


class TestValidateOperation:
    """Tests for operation validation."""
    
    def test_validate_add_operation_valid(self):
        """Test validating valid ADD operation."""
        op = CurationOp(
            type=CuratorOperation.ADD,
            section=ExpertiseSection.STRATEGIES,
            content="New content",
        )
        
        is_valid, error = validate_operation(op)
        
        assert is_valid is True
        assert error == ""
    
    def test_validate_add_operation_missing_content(self):
        """Test validating ADD without content."""
        op = CurationOp(
            type=CuratorOperation.ADD,
            section=ExpertiseSection.STRATEGIES,
            content=None,
        )
        
        is_valid, error = validate_operation(op)
        
        assert is_valid is False
        assert "content" in error.lower()
    
    def test_validate_add_operation_missing_section(self):
        """Test validating ADD without section."""
        op = CurationOp(
            type=CuratorOperation.ADD,
            section=None,
            content="Some content",
        )
        
        is_valid, error = validate_operation(op)
        
        assert is_valid is False
        assert "section" in error.lower()
    
    def test_validate_update_operation_valid(self):
        """Test validating valid UPDATE operation."""
        op = CurationOp(
            type=CuratorOperation.UPDATE,
            item_ids=["strat-00001"],
            content="Updated content",
        )
        
        is_valid, error = validate_operation(op)
        
        assert is_valid is True
    
    def test_validate_merge_operation_requires_two_items(self):
        """Test validating MERGE requires at least 2 items."""
        op = CurationOp(
            type=CuratorOperation.MERGE,
            item_ids=["strat-00001"],
            content="Merged",
        )
        
        is_valid, error = validate_operation(op)
        
        assert is_valid is False
        assert "2" in error
    
    def test_validate_delete_operation_valid(self):
        """Test validating valid DELETE operation."""
        op = CurationOp(
            type=CuratorOperation.DELETE,
            item_ids=["strat-00001"],
        )
        
        is_valid, error = validate_operation(op)
        
        assert is_valid is True


class TestExpertiseCurator:
    """Tests for ExpertiseCurator class."""
    
    @pytest.mark.asyncio
    async def test_initialization_with_llm_provider(self):
        """Test initialization with LLM provider."""
        mock_provider = MagicMock()
        mock_provider.name = "mock-llm"
        
        curator = ExpertiseCurator(llm_provider=mock_provider)
        
        assert curator.name == "expertise-curator:mock-llm"
    
    def test_initialization_requires_llm(self):
        """Test that initialization requires LLM."""
        with pytest.raises(ValueError):
            ExpertiseCurator()
    
    @pytest.mark.asyncio
    async def test_curate_no_suggestions(self, expertise):
        """Test curation with no suggestions returns unchanged."""
        empty_reflection = ReflectionResult(
            item_feedback={},
            insights="",
            suggested_additions=[],
            suggested_removals=[],
            confidence=0.5,
        )
        
        async def mock_func(prompt: str) -> str:
            return "{}"
        
        curator = ExpertiseCurator(llm_func=mock_func)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=empty_reflection,
            usage_stats={},
        )
        
        assert plan.operations == []
    
    @pytest.mark.asyncio
    async def test_curate_success(self, expertise, reflection_result, mock_curator_response):
        """Test successful curation."""
        initial_count = len(expertise.items)
        
        async def mock_func(prompt: str) -> str:
            return mock_curator_response
        
        curator = ExpertiseCurator(llm_func=mock_func)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={"total_turns": 100},
        )
        
        assert len(plan.operations) == 1
        assert plan.operations[0].type == CuratorOperation.ADD
        assert len(expertise.items) == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_curate_handles_invalid_json(self, expertise, reflection_result):
        """Test curation handles invalid JSON gracefully."""
        async def mock_func(prompt: str) -> str:
            return "This is not JSON at all"
        
        curator = ExpertiseCurator(llm_func=mock_func)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        assert plan.operations == []
        assert "Failed" in plan.reasoning or "parse" in plan.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_curate_add_only_mode(self, expertise, reflection_result):
        """Test curator in add-only mode."""
        response = json.dumps({
            "reasoning": "Adding new insight",
            "operations": [
                {
                    "type": "ADD",
                    "section": "strategies_and_insights",
                    "content": "New insight from add-only mode",
                }
            ]
        })
        
        async def mock_func(prompt: str) -> str:
            return response
        
        curator = ExpertiseCurator(llm_func=mock_func, add_only_mode=True)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        assert len(plan.operations) == 1
        assert plan.operations[0].type == CuratorOperation.ADD
    
    @pytest.mark.asyncio
    async def test_curate_validates_item_ids(self, expertise, reflection_result):
        """Test that curator validates item IDs exist."""
        response = json.dumps({
            "reasoning": "Updating item",
            "operations": [
                {
                    "type": "UPDATE",
                    "item_id": "invalid-99999",  # Does not exist
                    "content": "Updated content",
                }
            ]
        })
        
        async def mock_func(prompt: str) -> str:
            return response
        
        curator = ExpertiseCurator(llm_func=mock_func)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        # Invalid operation should be filtered out
        assert len(plan.operations) == 0


class TestMockCurator:
    """Tests for MockCurator class."""
    
    @pytest.mark.asyncio
    async def test_mock_curator_predefined_operations(self, expertise, reflection_result):
        """Test MockCurator with predefined operations."""
        operations = [
            CurationOp(
                type=CuratorOperation.ADD,
                section=ExpertiseSection.STRATEGIES,
                content="Predefined content",
            )
        ]
        
        curator = MockCurator(operations=operations, auto_add_from_reflection=False)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        assert len(plan.operations) == 1
        assert plan.operations[0].content == "Predefined content"
    
    @pytest.mark.asyncio
    async def test_mock_curator_auto_add_from_reflection(self, expertise, reflection_result):
        """Test MockCurator auto-generates ADD from reflection."""
        curator = MockCurator(auto_add_from_reflection=True)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        # Should have ADD operations from suggested_additions
        assert len(plan.operations) >= 1
        assert any(
            op.type == CuratorOperation.ADD and "error handling" in op.content.lower()
            for op in plan.operations
        )


class TestRuleBasedCurator:
    """Tests for RuleBasedCurator class."""
    
    @pytest.mark.asyncio
    async def test_rule_based_adds_suggestions(self, expertise, reflection_result):
        """Test RuleBasedCurator adds from suggestions."""
        curator = RuleBasedCurator()
        
        initial_count = len(expertise.items)
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        # Should have ADD operations from suggested_additions
        add_ops = [op for op in plan.operations if op.type == CuratorOperation.ADD]
        assert len(add_ops) >= 1
        assert len(expertise.items) > initial_count
    
    @pytest.mark.asyncio
    async def test_rule_based_auto_delete_harmful(self, expertise, reflection_result):
        """Test RuleBasedCurator auto-deletes harmful items."""
        curator = RuleBasedCurator(harmful_threshold=0.3, auto_delete_harmful=True)
        
        # mist-00003 has effectiveness 0.2 (2 helpful, 8 harmful)
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        delete_ops = [op for op in plan.operations if op.type == CuratorOperation.DELETE]
        assert len(delete_ops) >= 1
    
    @pytest.mark.asyncio
    async def test_rule_based_no_auto_delete_by_default(self, expertise, reflection_result):
        """Test RuleBasedCurator doesn't auto-delete by default."""
        curator = RuleBasedCurator(auto_delete_harmful=False)
        
        # Create reflection without suggested removals
        reflection = ReflectionResult(
            item_feedback={},
            insights="",
            suggested_additions=[],
            suggested_removals=[],
            confidence=0.5,
        )
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection,
            usage_stats={},
        )
        
        delete_ops = [op for op in plan.operations if op.type == CuratorOperation.DELETE]
        assert len(delete_ops) == 0


class TestCuratorWithLLMProvider:
    """Tests for ExpertiseCurator with ILLMProvider."""
    
    @pytest.mark.asyncio
    async def test_curate_with_llm_provider(self, expertise, reflection_result):
        """Test curation using ILLMProvider interface."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "reasoning": "Test reasoning",
            "operations": [
                {
                    "type": "ADD",
                    "section": "strategies_and_insights",
                    "content": "New strategy",
                }
            ]
        })
        
        mock_provider = MagicMock()
        mock_provider.name = "test-provider"
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        curator = ExpertiseCurator(
            llm_provider=mock_provider,
            temperature=0.2,
            max_tokens=1500,
        )
        
        updated, plan = await curator.curate(
            expertise=expertise,
            reflection=reflection_result,
            usage_stats={},
        )
        
        # Verify LLM was called
        mock_provider.chat.assert_called_once()
        
        # Verify result
        assert len(plan.operations) == 1

