"""
Expertise Reflector Implementation.

The reflector analyzes completed conversation turns and provides
feedback on which expertise items were helpful or harmful.

Inspired by ACE framework's Reflector agent.
"""

import json
import logging
import re
from typing import Awaitable, Callable, Dict, List, Optional

from ctxforge.core.expertise import (
    CompletedTurn,
    ExpertiseItem,
    ReflectionResult,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.expertise.prompts.reflector import (
    REFLECTOR_SUCCESS_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
    REFLECTOR_USER_PROMPT_NO_GT,
)
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

logger = logging.getLogger(__name__)


class ExpertiseReflector:
    """
    Analyzes conversation turns and provides feedback on expertise effectiveness.
    
    The reflector evaluates each expertise item that was used in a turn,
    tagging it as helpful, harmful, or neutral based on the turn outcome.
    It also provides insights and suggestions for improving the expertise.
    
    Implements the IReflector protocol.
    
    Example:
        >>> reflector = ExpertiseReflector(llm_provider=my_llm)
        >>> result = await reflector.reflect(
        ...     turn=CompletedTurn(user_input="...", assistant_response="..."),
        ...     items_used=[item1, item2],
        ...     outcome=TurnOutcome.FAILURE
        ... )
        >>> print(result.item_feedback)  # {"strat-00001": "helpful", "form-00002": "harmful"}
    """
    
    def __init__(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        llm_func: Optional[Callable[[str], Awaitable[str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ):
        """
        Initialize the reflector.
        
        Args:
            llm_provider: An ILLMProvider implementation
            llm_func: Alternative: a simple async function(prompt) -> response
            system_prompt: Custom system prompt (uses default if not provided)
            temperature: LLM sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in LLM response
            model: Specific model to use (defaults to provider's default)
        """
        self._llm_provider = llm_provider
        self._llm_func = llm_func
        self._system_prompt = system_prompt or REFLECTOR_SYSTEM_PROMPT
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model = model
        
        if not llm_provider and not llm_func:
            raise ValueError("Either llm_provider or llm_func must be provided")
    
    @property
    def name(self) -> str:
        """The name of this reflector."""
        if self._llm_provider:
            return f"expertise-reflector:{self._llm_provider.name}"
        return "expertise-reflector:custom"
    
    async def reflect(
        self,
        turn: CompletedTurn,
        items_used: List[ExpertiseItem],
        outcome: TurnOutcome,
    ) -> ReflectionResult:
        """
        Analyze a turn and provide feedback on expertise items.
        
        Args:
            turn: The completed conversation turn
            items_used: List of expertise items that were used
            outcome: The outcome of the turn
            
        Returns:
            ReflectionResult with item feedback and insights
        """
        if not items_used:
            # No items used, return empty result
            return ReflectionResult(
                item_feedback={},
                insights="No expertise items were used in this turn.",
                suggested_additions=[],
                suggested_removals=[],
                confidence=1.0,
            )
        
        # Format items for prompt
        items_text = self._format_items_for_prompt(items_used)
        
        # Build the appropriate prompt
        user_prompt = self._build_user_prompt(turn, items_text, outcome)
        
        try:
            # Call LLM
            response = await self._call_llm(user_prompt)
            
            # Parse response
            result = self._parse_response(response, items_used)
            
            return result
            
        except Exception as e:
            # Log error but return partial result
            logger.warning("Reflection error: %s", e)
            return ReflectionResult(
                item_feedback={},
                insights=f"Reflection failed: {str(e)}",
                suggested_additions=[],
                suggested_removals=[],
                confidence=0.0,
            )
    
    def _format_items_for_prompt(self, items: List[ExpertiseItem]) -> str:
        """
        Format expertise items for inclusion in the prompt.
        
        Uses ACE-style format: [id] helpful=X harmful=Y :: content
        """
        lines = []
        for item in items:
            line = (
                f"[{item.item_id}] "
                f"helpful={item.helpful_count} harmful={item.harmful_count} :: "
                f"{item.content}"
            )
            lines.append(line)
        return "\n".join(lines) if lines else "(no items)"
    
    def _build_user_prompt(
        self,
        turn: CompletedTurn,
        items_text: str,
        outcome: TurnOutcome,
    ) -> str:
        """Build the user prompt based on turn context."""
        
        if outcome == TurnOutcome.SUCCESS:
            # Use success prompt for successful turns
            return REFLECTOR_SUCCESS_PROMPT.format(
                user_input=turn.user_input,
                assistant_response=turn.assistant_response,
                expected_output=turn.expected_output or "(not provided)",
                expertise_items=items_text,
            )
        elif turn.expected_output:
            # Use ground truth prompt when expected output is available
            return REFLECTOR_USER_PROMPT.format(
                user_input=turn.user_input,
                assistant_response=turn.assistant_response,
                expected_output=turn.expected_output,
                outcome=outcome.value,
                expertise_items=items_text,
            )
        else:
            # Use no-ground-truth prompt
            return REFLECTOR_USER_PROMPT_NO_GT.format(
                user_input=turn.user_input,
                assistant_response=turn.assistant_response,
                actual_outcome=turn.actual_outcome or "(not provided)",
                outcome=outcome.value,
                expertise_items=items_text,
            )
    
    async def _call_llm(self, user_prompt: str) -> str:
        """
        Call the LLM with the reflection prompt.
        
        Args:
            user_prompt: The user prompt
            
        Returns:
            LLM response text
        """
        if self._llm_provider:
            messages = [
                ChatMessage(role="system", content=self._system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
            
            response = await self._llm_provider.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            
            return response.content
        else:
            # Use simple function
            full_prompt = f"{self._system_prompt}\n\n{user_prompt}"
            return await self._llm_func(full_prompt)
    
    def _parse_response(
        self,
        response: str,
        items_used: List[ExpertiseItem],
    ) -> ReflectionResult:
        """
        Parse the LLM response into a ReflectionResult.
        
        Args:
            response: The LLM response text
            items_used: The items that were used (for validation)
            
        Returns:
            ReflectionResult with parsed data
        """
        # Extract JSON from response
        json_str = self._extract_json(response)
        
        if not json_str:
            # Fallback: try to extract bullet tags manually
            return self._fallback_parse(response, items_used)
        
        try:
            data = json.loads(json_str)
            
            # Parse item tags
            item_feedback = self._parse_item_tags(
                data.get("bullet_tags", []),
                items_used,
            )
            
            # Parse other fields
            insights = data.get("insights", "")
            if not insights:
                # Try to build insights from other fields
                parts = []
                if data.get("reasoning"):
                    parts.append(data["reasoning"])
                if data.get("error_identification"):
                    parts.append(f"Error: {data['error_identification']}")
                if data.get("correct_approach"):
                    parts.append(f"Should have: {data['correct_approach']}")
                insights = " ".join(parts)
            
            suggested_additions = data.get("suggested_additions", [])
            if isinstance(suggested_additions, str):
                suggested_additions = [suggested_additions] if suggested_additions else []
            
            suggested_removals = data.get("suggested_removals", [])
            if isinstance(suggested_removals, str):
                suggested_removals = [suggested_removals] if suggested_removals else []
            
            confidence = data.get("confidence", 0.7)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.7
            confidence = max(0.0, min(1.0, confidence))
            
            return ReflectionResult(
                item_feedback=item_feedback,
                insights=insights,
                suggested_additions=suggested_additions,
                suggested_removals=suggested_removals,
                confidence=confidence,
            )
            
        except json.JSONDecodeError:
            return self._fallback_parse(response, items_used)
    
    def _parse_item_tags(
        self,
        tags_list: List[Dict[str, str]],
        items_used: List[ExpertiseItem],
    ) -> Dict[str, UsageFeedback]:
        """
        Parse item tags from LLM response.
        
        Args:
            tags_list: List of {"id": "...", "tag": "..."} dicts
            items_used: Items that were used (for validation)
            
        Returns:
            Dict mapping item_id to UsageFeedback
        """
        # Build set of valid item IDs
        valid_ids = {item.item_id for item in items_used}
        
        # Map tag strings to UsageFeedback enum
        tag_map = {
            "helpful": UsageFeedback.HELPFUL,
            "harmful": UsageFeedback.HARMFUL,
            "neutral": UsageFeedback.NEUTRAL,
        }
        
        feedback = {}
        
        for tag_entry in tags_list:
            if not isinstance(tag_entry, dict):
                continue
            
            item_id = tag_entry.get("id", "")
            tag_str = tag_entry.get("tag", "").lower()
            
            # Validate item ID exists in used items
            if item_id not in valid_ids:
                continue
            
            # Map to enum
            if tag_str in tag_map:
                feedback[item_id] = tag_map[tag_str]
        
        return feedback
    
    def _fallback_parse(
        self,
        response: str,
        items_used: List[ExpertiseItem],
    ) -> ReflectionResult:
        """
        Fallback parsing when JSON extraction fails.
        
        Tries to extract bullet tags using regex patterns.
        """
        feedback = {}
        
        # Try to find tags in the response
        # Look for patterns like: "strat-00001": "helpful" or id: strat-00001, tag: helpful
        for item in items_used:
            item_id = item.item_id
            
            # Check for helpful mentions
            helpful_pattern = rf'{re.escape(item_id)}["\s:,]+helpful'
            if re.search(helpful_pattern, response, re.IGNORECASE):
                feedback[item_id] = UsageFeedback.HELPFUL
                continue
            
            # Check for harmful mentions
            harmful_pattern = rf'{re.escape(item_id)}["\s:,]+harmful'
            if re.search(harmful_pattern, response, re.IGNORECASE):
                feedback[item_id] = UsageFeedback.HARMFUL
                continue
            
            # Check for neutral mentions
            neutral_pattern = rf'{re.escape(item_id)}["\s:,]+neutral'
            if re.search(neutral_pattern, response, re.IGNORECASE):
                feedback[item_id] = UsageFeedback.NEUTRAL
        
        return ReflectionResult(
            item_feedback=feedback,
            insights="(Fallback parsing: JSON extraction failed)",
            suggested_additions=[],
            suggested_removals=[],
            confidence=0.3,
        )
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from LLM response.
        
        Delegates to shared utility function.
        
        Args:
            text: The response text
            
        Returns:
            JSON string if found, None otherwise
        """
        return extract_json_from_text(text)


class MockReflector:
    """
    A mock reflector for testing.
    
    Returns predefined feedback based on configuration.
    Useful for testing the expertise system without LLM calls.
    """
    
    def __init__(
        self,
        default_feedback: UsageFeedback = UsageFeedback.NEUTRAL,
        feedback_map: Optional[Dict[str, UsageFeedback]] = None,
        insights: str = "Mock reflection",
        suggested_additions: Optional[List[str]] = None,
        confidence: float = 0.8,
    ):
        """
        Initialize mock reflector.
        
        Args:
            default_feedback: Default feedback for items not in feedback_map
            feedback_map: Specific feedback for item IDs
            insights: Insights string to return
            suggested_additions: Suggested additions to return
            confidence: Confidence score to return
        """
        self._default_feedback = default_feedback
        self._feedback_map = feedback_map or {}
        self._insights = insights
        self._suggested_additions = suggested_additions or []
        self._confidence = confidence
    
    @property
    def name(self) -> str:
        """The name of this reflector."""
        return "mock-reflector"
    
    async def reflect(
        self,
        turn: CompletedTurn,
        items_used: List[ExpertiseItem],
        outcome: TurnOutcome,
    ) -> ReflectionResult:
        """Return mock reflection result."""
        feedback = {}
        
        for item in items_used:
            if item.item_id in self._feedback_map:
                feedback[item.item_id] = self._feedback_map[item.item_id]
            else:
                # Assign based on outcome
                if outcome == TurnOutcome.SUCCESS:
                    feedback[item.item_id] = UsageFeedback.HELPFUL
                elif outcome == TurnOutcome.FAILURE:
                    # Alternate between harmful and neutral
                    feedback[item.item_id] = self._default_feedback
                else:
                    feedback[item.item_id] = UsageFeedback.NEUTRAL
        
        return ReflectionResult(
            item_feedback=feedback,
            insights=self._insights,
            suggested_additions=self._suggested_additions,
            suggested_removals=[],
            confidence=self._confidence,
        )


class RuleBasedReflector:
    """
    A rule-based reflector that doesn't require an LLM.
    
    Uses simple rules based on the turn outcome to assign feedback:
    - SUCCESS: All items marked helpful
    - FAILURE: Items with low effectiveness marked harmful
    - PARTIAL: Items marked based on effectiveness threshold
    
    Useful for testing and as a fallback when LLM is unavailable.
    """
    
    def __init__(
        self,
        effectiveness_threshold: float = 0.5,
    ):
        """
        Initialize rule-based reflector.
        
        Args:
            effectiveness_threshold: Below this, items are marked harmful on failure
        """
        self._threshold = effectiveness_threshold
    
    @property
    def name(self) -> str:
        """The name of this reflector."""
        return "rule-based-reflector"
    
    async def reflect(
        self,
        turn: CompletedTurn,
        items_used: List[ExpertiseItem],
        outcome: TurnOutcome,
    ) -> ReflectionResult:
        """Apply rule-based reflection."""
        feedback = {}
        
        for item in items_used:
            if outcome == TurnOutcome.SUCCESS:
                # All items get credit for success
                feedback[item.item_id] = UsageFeedback.HELPFUL
                
            elif outcome == TurnOutcome.FAILURE:
                # Check effectiveness to determine if harmful
                if item.effectiveness_score < self._threshold:
                    feedback[item.item_id] = UsageFeedback.HARMFUL
                else:
                    feedback[item.item_id] = UsageFeedback.NEUTRAL
                    
            elif outcome == TurnOutcome.PARTIAL:
                # High-effectiveness items are helpful, others neutral
                if item.effectiveness_score >= 0.7:
                    feedback[item.item_id] = UsageFeedback.HELPFUL
                elif item.effectiveness_score < 0.3:
                    feedback[item.item_id] = UsageFeedback.HARMFUL
                else:
                    feedback[item.item_id] = UsageFeedback.NEUTRAL
                    
            else:  # UNKNOWN
                feedback[item.item_id] = UsageFeedback.NEUTRAL
        
        # Generate insights based on outcome
        if outcome == TurnOutcome.SUCCESS:
            insights = "Turn was successful. All used expertise items contributed positively."
        elif outcome == TurnOutcome.FAILURE:
            low_eff = [i.item_id for i in items_used if i.effectiveness_score < self._threshold]
            if low_eff:
                insights = f"Turn failed. Low-effectiveness items used: {', '.join(low_eff)}"
            else:
                insights = "Turn failed despite using well-performing items. May need new expertise."
        else:
            insights = "Partial success. Review item relevance to the query."
        
        return ReflectionResult(
            item_feedback=feedback,
            insights=insights,
            suggested_additions=[],
            suggested_removals=[],
            confidence=0.6,  # Rule-based has moderate confidence
        )

