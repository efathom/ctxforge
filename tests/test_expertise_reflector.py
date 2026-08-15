"""
Tests for the Expertise Reflector.

Tests reflection, parsing, and feedback extraction.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.expertise import (
    CompletedTurn,
    ExpertiseItem,
    ExpertiseSection,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.expertise.prompts.reflector import (
    REFLECTOR_SUCCESS_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
    REFLECTOR_USER_PROMPT_NO_GT,
)
from ctxforge.expertise.reflector import (
    ExpertiseReflector,
    MockReflector,
    RuleBasedReflector,
)


# Test fixtures
@pytest.fixture
def sample_items():
    """Create sample expertise items for testing."""
    return [
        ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Always start with a greeting",
            helpful_count=10,
            harmful_count=2,
        ),
        ExpertiseItem(
            item_id="form-00002",
            section=ExpertiseSection.FORMULAS,
            content="Use the formula: price = cost * (1 + margin)",
            helpful_count=5,
            harmful_count=0,
        ),
        ExpertiseItem(
            item_id="mist-00003",
            section=ExpertiseSection.COMMON_MISTAKES,
            content="Don't forget to validate input",
            helpful_count=3,
            harmful_count=5,
        ),
    ]


@pytest.fixture
def completed_turn():
    """Create a sample completed turn."""
    return CompletedTurn(
        user_input="What is the price if cost is $100 and margin is 20%?",
        assistant_response="The price would be $120.",
        expected_output="$120",
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response in JSON format."""
    return json.dumps({
        "reasoning": "The assistant correctly applied the pricing formula.",
        "error_identification": "No errors identified.",
        "root_cause_analysis": "N/A - successful turn.",
        "correct_approach": "The approach was correct.",
        "bullet_tags": [
            {"id": "strat-00001", "tag": "helpful"},
            {"id": "form-00002", "tag": "helpful"},
            {"id": "mist-00003", "tag": "neutral"},
        ],
        "insights": "The pricing formula was applied correctly. The greeting strategy helped establish rapport.",
        "suggested_additions": ["Add example for complex margin calculations"],
        "suggested_removals": [],
        "confidence": 0.92,
    })


class TestExpertiseReflector:
    """Tests for ExpertiseReflector class."""
    
    @pytest.mark.asyncio
    async def test_initialization_with_llm_provider(self):
        """Test initialization with LLM provider."""
        mock_provider = MagicMock()
        mock_provider.name = "mock-llm"
        
        reflector = ExpertiseReflector(llm_provider=mock_provider)
        
        assert reflector.name == "expertise-reflector:mock-llm"
    
    @pytest.mark.asyncio
    async def test_initialization_with_llm_func(self):
        """Test initialization with LLM function."""
        async def mock_func(prompt: str) -> str:
            return "{}"
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        assert reflector.name == "expertise-reflector:custom"
    
    def test_initialization_requires_llm(self):
        """Test that initialization requires either llm_provider or llm_func."""
        with pytest.raises(ValueError, match="Either llm_provider or llm_func must be provided"):
            ExpertiseReflector()
    
    @pytest.mark.asyncio
    async def test_reflect_with_no_items(self, completed_turn):
        """Test reflection with no items returns empty result."""
        async def mock_func(prompt: str) -> str:
            return "{}"
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=[],
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.item_feedback == {}
        assert "No expertise items" in result.insights
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_reflect_success(self, sample_items, completed_turn, mock_llm_response):
        """Test successful reflection with LLM."""
        async def mock_func(prompt: str) -> str:
            return mock_llm_response
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.item_feedback["strat-00001"] == UsageFeedback.HELPFUL
        assert result.item_feedback["form-00002"] == UsageFeedback.HELPFUL
        assert result.item_feedback["mist-00003"] == UsageFeedback.NEUTRAL
        assert "pricing formula" in result.insights.lower() or len(result.insights) > 0
        assert result.confidence == 0.92
        assert len(result.suggested_additions) == 1
    
    @pytest.mark.asyncio
    async def test_reflect_failure_with_ground_truth(self, sample_items):
        """Test reflection on failure with ground truth."""
        turn = CompletedTurn(
            user_input="Calculate 2 + 2",
            assistant_response="5",
            expected_output="4",
        )
        
        response = json.dumps({
            "reasoning": "The calculation was incorrect.",
            "error_identification": "Basic addition error.",
            "bullet_tags": [
                {"id": "strat-00001", "tag": "neutral"},
                {"id": "form-00002", "tag": "harmful"},
            ],
            "insights": "Need to double-check arithmetic.",
            "suggested_additions": [],
            "suggested_removals": ["form-00002 - led to incorrect calculation"],
            "confidence": 0.85,
        })
        
        async def mock_func(prompt: str) -> str:
            return response
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=turn,
            items_used=sample_items[:2],  # Use only first 2 items
            outcome=TurnOutcome.FAILURE,
        )
        
        assert result.item_feedback["strat-00001"] == UsageFeedback.NEUTRAL
        assert result.item_feedback["form-00002"] == UsageFeedback.HARMFUL
        assert len(result.suggested_removals) == 1
    
    @pytest.mark.asyncio
    async def test_reflect_without_ground_truth(self, sample_items):
        """Test reflection when no ground truth is available."""
        turn = CompletedTurn(
            user_input="Help me write an email",
            assistant_response="Dear Sir/Madam...",
            actual_outcome="Customer complained about formal tone",
        )
        
        response = json.dumps({
            "bullet_tags": [
                {"id": "strat-00001", "tag": "harmful"},
            ],
            "insights": "The formal greeting was inappropriate for this context.",
            "suggested_additions": ["Consider casual greetings for informal contexts"],
            "suggested_removals": [],
            "confidence": 0.7,
        })
        
        async def mock_func(prompt: str) -> str:
            return response
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=turn,
            items_used=sample_items[:1],
            outcome=TurnOutcome.FAILURE,
        )
        
        assert result.item_feedback["strat-00001"] == UsageFeedback.HARMFUL
    
    @pytest.mark.asyncio
    async def test_reflect_handles_invalid_json(self, sample_items, completed_turn):
        """Test that reflection handles invalid JSON gracefully."""
        async def mock_func(prompt: str) -> str:
            return "This is not JSON at all. strat-00001: helpful, form-00002: neutral"
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items[:2],
            outcome=TurnOutcome.SUCCESS,
        )
        
        # Should fallback to regex-based parsing
        assert result.item_feedback.get("strat-00001") == UsageFeedback.HELPFUL
        assert result.confidence == 0.3  # Low confidence for fallback
    
    @pytest.mark.asyncio
    async def test_reflect_handles_markdown_json(self, sample_items, completed_turn):
        """Test that reflection extracts JSON from markdown code blocks."""
        response = """Here's my analysis:

```json
{
    "bullet_tags": [
        {"id": "strat-00001", "tag": "helpful"}
    ],
    "insights": "Good use of greeting",
    "suggested_additions": [],
    "suggested_removals": [],
    "confidence": 0.8
}
```

That's my reflection."""
        
        async def mock_func(prompt: str) -> str:
            return response
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items[:1],
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.item_feedback["strat-00001"] == UsageFeedback.HELPFUL
        assert result.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_reflect_ignores_invalid_item_ids(self, sample_items, completed_turn):
        """Test that reflection ignores item IDs not in items_used."""
        response = json.dumps({
            "bullet_tags": [
                {"id": "strat-00001", "tag": "helpful"},
                {"id": "invalid-99999", "tag": "harmful"},  # Not in items_used
            ],
            "insights": "Test",
            "suggested_additions": [],
            "suggested_removals": [],
            "confidence": 0.8,
        })
        
        async def mock_func(prompt: str) -> str:
            return response
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items[:1],
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert "strat-00001" in result.item_feedback
        assert "invalid-99999" not in result.item_feedback
    
    @pytest.mark.asyncio
    async def test_reflect_handles_llm_error(self, sample_items, completed_turn):
        """Test that reflection handles LLM errors gracefully."""
        async def mock_func(prompt: str) -> str:
            raise RuntimeError("LLM service unavailable")
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.item_feedback == {}
        assert "failed" in result.insights.lower()
        assert result.confidence == 0.0


class TestMockReflector:
    """Tests for MockReflector class."""
    
    @pytest.mark.asyncio
    async def test_mock_reflector_success_outcome(self, sample_items, completed_turn):
        """Test MockReflector marks items as helpful on success."""
        reflector = MockReflector()
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.SUCCESS,
        )
        
        # All items should be helpful on success
        for item in sample_items:
            assert result.item_feedback[item.item_id] == UsageFeedback.HELPFUL
    
    @pytest.mark.asyncio
    async def test_mock_reflector_with_feedback_map(self, sample_items, completed_turn):
        """Test MockReflector with custom feedback map."""
        feedback_map = {
            "strat-00001": UsageFeedback.HELPFUL,
            "form-00002": UsageFeedback.HARMFUL,
        }
        
        reflector = MockReflector(feedback_map=feedback_map)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items[:2],
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.item_feedback["strat-00001"] == UsageFeedback.HELPFUL
        assert result.item_feedback["form-00002"] == UsageFeedback.HARMFUL
    
    @pytest.mark.asyncio
    async def test_mock_reflector_custom_insights(self, sample_items, completed_turn):
        """Test MockReflector with custom insights."""
        reflector = MockReflector(
            insights="Custom test insights",
            suggested_additions=["New item 1", "New item 2"],
            confidence=0.95,
        )
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.insights == "Custom test insights"
        assert len(result.suggested_additions) == 2
        assert result.confidence == 0.95


class TestRuleBasedReflector:
    """Tests for RuleBasedReflector class."""
    
    @pytest.mark.asyncio
    async def test_rule_based_success(self, sample_items, completed_turn):
        """Test RuleBasedReflector marks all items helpful on success."""
        reflector = RuleBasedReflector()
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.SUCCESS,
        )
        
        for item in sample_items:
            assert result.item_feedback[item.item_id] == UsageFeedback.HELPFUL
        
        assert "successful" in result.insights.lower()
    
    @pytest.mark.asyncio
    async def test_rule_based_failure_low_effectiveness(self, sample_items, completed_turn):
        """Test RuleBasedReflector marks low-effectiveness items harmful on failure."""
        reflector = RuleBasedReflector(effectiveness_threshold=0.5)
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.FAILURE,
        )
        
        # mist-00003 has effectiveness < 0.5 (3 helpful, 5 harmful = 0.375)
        assert result.item_feedback["mist-00003"] == UsageFeedback.HARMFUL
        
        # strat-00001 has effectiveness > 0.5 (10 helpful, 2 harmful = 0.833)
        assert result.item_feedback["strat-00001"] == UsageFeedback.NEUTRAL
        
        # form-00002 has effectiveness 1.0 (5 helpful, 0 harmful)
        assert result.item_feedback["form-00002"] == UsageFeedback.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_rule_based_partial(self, sample_items, completed_turn):
        """Test RuleBasedReflector on partial success."""
        reflector = RuleBasedReflector()
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.PARTIAL,
        )
        
        # strat-00001: effectiveness 0.833 >= 0.7 -> helpful
        assert result.item_feedback["strat-00001"] == UsageFeedback.HELPFUL
        
        # form-00002: effectiveness 1.0 >= 0.7 -> helpful
        assert result.item_feedback["form-00002"] == UsageFeedback.HELPFUL
        
        # mist-00003: effectiveness 0.375 < 0.3 -> harmful (actually it's 3/8=0.375 which is > 0.3)
        # So it should be neutral
        assert result.item_feedback["mist-00003"] == UsageFeedback.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_rule_based_unknown_outcome(self, sample_items, completed_turn):
        """Test RuleBasedReflector on unknown outcome."""
        reflector = RuleBasedReflector()
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.UNKNOWN,
        )
        
        for item in sample_items:
            assert result.item_feedback[item.item_id] == UsageFeedback.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_rule_based_moderate_confidence(self, sample_items, completed_turn):
        """Test RuleBasedReflector has moderate confidence."""
        reflector = RuleBasedReflector()
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items,
            outcome=TurnOutcome.SUCCESS,
        )
        
        assert result.confidence == 0.6


class TestPromptTemplates:
    """Tests for prompt templates."""
    
    def test_system_prompt_exists(self):
        """Test system prompt is defined."""
        assert REFLECTOR_SYSTEM_PROMPT
        assert "analyze" in REFLECTOR_SYSTEM_PROMPT.lower()
        assert "helpful" in REFLECTOR_SYSTEM_PROMPT.lower()
        assert "harmful" in REFLECTOR_SYSTEM_PROMPT.lower()
    
    def test_user_prompt_has_placeholders(self):
        """Test user prompt has all required placeholders."""
        assert "{user_input}" in REFLECTOR_USER_PROMPT
        assert "{assistant_response}" in REFLECTOR_USER_PROMPT
        assert "{expected_output}" in REFLECTOR_USER_PROMPT
        assert "{outcome}" in REFLECTOR_USER_PROMPT
        assert "{expertise_items}" in REFLECTOR_USER_PROMPT
    
    def test_user_prompt_no_gt_has_placeholders(self):
        """Test no-ground-truth prompt has correct placeholders."""
        assert "{user_input}" in REFLECTOR_USER_PROMPT_NO_GT
        assert "{assistant_response}" in REFLECTOR_USER_PROMPT_NO_GT
        assert "{actual_outcome}" in REFLECTOR_USER_PROMPT_NO_GT
        assert "{outcome}" in REFLECTOR_USER_PROMPT_NO_GT
        assert "{expertise_items}" in REFLECTOR_USER_PROMPT_NO_GT
        # Should NOT have expected_output
        assert "{expected_output}" not in REFLECTOR_USER_PROMPT_NO_GT
    
    def test_success_prompt_has_placeholders(self):
        """Test success prompt has correct placeholders."""
        assert "{user_input}" in REFLECTOR_SUCCESS_PROMPT
        assert "{assistant_response}" in REFLECTOR_SUCCESS_PROMPT
        assert "{expertise_items}" in REFLECTOR_SUCCESS_PROMPT


class TestReflectorItemFormatting:
    """Tests for expertise item formatting."""
    
    @pytest.mark.asyncio
    async def test_items_formatted_in_ace_style(self, sample_items, completed_turn):
        """Test that items are formatted in ACE style for the prompt."""
        captured_prompt = None
        
        async def mock_func(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return json.dumps({
                "bullet_tags": [],
                "insights": "",
                "suggested_additions": [],
                "suggested_removals": [],
                "confidence": 0.5,
            })
        
        reflector = ExpertiseReflector(llm_func=mock_func)
        
        await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items[:1],
            outcome=TurnOutcome.SUCCESS,
        )
        
        # Check that the prompt contains ACE-style formatting
        assert "[strat-00001]" in captured_prompt
        assert "helpful=10" in captured_prompt
        assert "harmful=2" in captured_prompt
        assert "Always start with a greeting" in captured_prompt


class TestReflectorWithLLMProvider:
    """Tests for ExpertiseReflector with ILLMProvider."""
    
    @pytest.mark.asyncio
    async def test_reflect_with_llm_provider(self, sample_items, completed_turn):
        """Test reflection using ILLMProvider interface."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "bullet_tags": [{"id": "strat-00001", "tag": "helpful"}],
            "insights": "Good",
            "suggested_additions": [],
            "suggested_removals": [],
            "confidence": 0.9,
        })
        
        mock_provider = MagicMock()
        mock_provider.name = "test-provider"
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        reflector = ExpertiseReflector(
            llm_provider=mock_provider,
            temperature=0.2,
            max_tokens=1500,
            model="test-model",
        )
        
        result = await reflector.reflect(
            turn=completed_turn,
            items_used=sample_items[:1],
            outcome=TurnOutcome.SUCCESS,
        )
        
        # Verify LLM was called correctly
        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 1500
        assert call_kwargs["model"] == "test-model"
        
        # Verify result
        assert result.item_feedback["strat-00001"] == UsageFeedback.HELPFUL
        assert result.confidence == 0.9

