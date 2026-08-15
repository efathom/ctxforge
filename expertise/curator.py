"""
Expertise Curator Implementation.

The curator evolves expertise based on reflection feedback,
performing ADD, UPDATE, MERGE, and DELETE operations.

Inspired by ACE framework's Curator agent.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ctxforge.core.expertise import (
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    ExpertiseSection,
    ExpertiseStats,
    ReflectionResult,
)
from ctxforge.expertise.operations import (
    apply_curation_plan,
    format_expertise_for_prompt,
    parse_section_from_string,
    validate_operation,
)
from ctxforge.expertise.prompts.curator import (
    CURATOR_ADD_ONLY_PROMPT,
    CURATOR_SYSTEM_PROMPT,
    CURATOR_USER_PROMPT,
)
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

logger = logging.getLogger(__name__)


class ExpertiseCurator:
    """
    Curates and evolves expertise based on reflection feedback.
    
    The curator analyzes reflection results and usage statistics
    to generate operations that improve the expertise:
    - ADD: Create new expertise items from insights
    - UPDATE: Improve existing items based on feedback
    - MERGE: Combine similar or redundant items
    - DELETE: Remove harmful or outdated items
    
    Implements the ICurator protocol.
    
    Example:
        >>> curator = ExpertiseCurator(llm_provider=my_llm)
        >>> updated, plan = await curator.curate(
        ...     expertise=expertise,
        ...     reflection=reflection_result,
        ...     usage_stats={"total_turns": 100}
        ... )
        >>> print(f"Applied {plan.operation_count} operations")
    """
    
    def __init__(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        llm_func: Optional[Callable[[str], Awaitable[str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        model: Optional[str] = None,
        add_only_mode: bool = False,
    ):
        """
        Initialize the curator.
        
        Args:
            llm_provider: An ILLMProvider implementation
            llm_func: Alternative: a simple async function(prompt) -> response
            system_prompt: Custom system prompt (uses default if not provided)
            temperature: LLM sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in LLM response
            model: Specific model to use (defaults to provider's default)
            add_only_mode: If True, only generate ADD operations (simpler)
        """
        self._llm_provider = llm_provider
        self._llm_func = llm_func
        self._system_prompt = system_prompt or CURATOR_SYSTEM_PROMPT
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model = model
        self._add_only_mode = add_only_mode
        
        if not llm_provider and not llm_func:
            raise ValueError("Either llm_provider or llm_func must be provided")
    
    @property
    def name(self) -> str:
        """The name of this curator."""
        if self._llm_provider:
            return f"expertise-curator:{self._llm_provider.name}"
        return "expertise-curator:custom"
    
    async def curate(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> Tuple[Expertise, CurationPlan]:
        """
        Generate and apply curation plan to evolve expertise.
        
        Args:
            expertise: The current expertise
            reflection: Recent reflection result
            usage_stats: Usage statistics
            
        Returns:
            Tuple of (updated expertise, curation plan applied)
        """
        # Check if there's anything to curate
        if not reflection.has_suggestions and not reflection.item_feedback:
            return expertise, CurationPlan(operations=[], reasoning="No changes needed")
        
        # Build the prompt
        user_prompt = self._build_user_prompt(expertise, reflection, usage_stats)
        
        try:
            # Call LLM
            response = await self._call_llm(user_prompt)
            
            # Parse response into curation plan
            plan = self._parse_response(response, expertise)
            
            # Apply the plan
            updated_expertise, adds, updates, merges, deletes = apply_curation_plan(
                expertise, plan
            )
            
            # Update plan reasoning with summary
            if plan.operations:
                summary = f"Applied: {adds} adds, {updates} updates, {merges} merges, {deletes} deletes"
                plan.reasoning = f"{plan.reasoning} [{summary}]"
            
            return updated_expertise, plan
            
        except Exception as e:
            # Log error but return unchanged expertise
            logger.warning("Curation error: %s", e)
            return expertise, CurationPlan(
                operations=[],
                reasoning=f"Curation failed: {str(e)}",
            )
    
    def _build_user_prompt(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> str:
        """Build the user prompt for curation."""
        
        # Get expertise stats
        stats = ExpertiseStats.from_expertise(expertise)
        
        if self._add_only_mode:
            # Use simplified ADD-only prompt
            return CURATOR_ADD_ONLY_PROMPT.format(
                total_items=stats.total_items,
                token_budget=expertise.token_budget,
                current_tokens=stats.estimated_tokens,
                reflection=reflection.insights,
                expertise_content=format_expertise_for_prompt(expertise),
                insights="\n".join(reflection.suggested_additions) or "(none)",
            )
        
        # Full curation prompt
        return CURATOR_USER_PROMPT.format(
            expertise_stats=self._format_stats(stats),
            reflection=reflection.insights,
            expertise_content=format_expertise_for_prompt(expertise),
            helpful_items=", ".join(reflection.helpful_items) or "(none)",
            harmful_items=", ".join(reflection.harmful_items) or "(none)",
            suggested_additions="\n".join(f"- {s}" for s in reflection.suggested_additions) or "(none)",
            suggested_removals="\n".join(f"- {s}" for s in reflection.suggested_removals) or "(none)",
        )
    
    def _format_stats(self, stats: ExpertiseStats) -> str:
        """Format stats for prompt."""
        return f"""- Total items: {stats.total_items}
- Active items: {stats.active_items}
- High performing: {stats.high_performing}
- Problematic: {stats.problematic}
- Unused: {stats.unused}
- Average effectiveness: {stats.average_effectiveness:.2f}
- Estimated tokens: {stats.estimated_tokens}"""
    
    async def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM with the curation prompt."""
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
            full_prompt = f"{self._system_prompt}\n\n{user_prompt}"
            return await self._llm_func(full_prompt)
    
    def _parse_response(
        self,
        response: str,
        expertise: Expertise,
    ) -> CurationPlan:
        """
        Parse the LLM response into a CurationPlan.
        
        Args:
            response: The LLM response text
            expertise: Current expertise (for validation)
            
        Returns:
            CurationPlan with validated operations
        """
        # Extract JSON from response
        json_str = extract_json_from_text(response)
        
        if not json_str:
            return CurationPlan(
                operations=[],
                reasoning="Failed to parse JSON from response",
            )
        
        try:
            data = json.loads(json_str)
            
            reasoning = data.get("reasoning", "")
            raw_operations = data.get("operations", [])
            
            # Parse and validate operations
            operations = self._parse_operations(raw_operations, expertise)
            
            return CurationPlan(
                operations=operations,
                reasoning=reasoning,
            )
            
        except json.JSONDecodeError as e:
            return CurationPlan(
                operations=[],
                reasoning=f"JSON parse error: {str(e)}",
            )
    
    def _parse_operations(
        self,
        raw_operations: List[Dict[str, Any]],
        expertise: Expertise,
    ) -> List[CurationOp]:
        """
        Parse and validate raw operations from LLM response.
        
        Args:
            raw_operations: List of operation dicts from LLM
            expertise: Current expertise (for validation)
            
        Returns:
            List of validated CurationOp objects
        """
        operations = []
        
        for raw_op in raw_operations:
            if not isinstance(raw_op, dict):
                continue
            
            op = self._parse_single_operation(raw_op, expertise)
            if op:
                # Validate operation
                is_valid, error = validate_operation(op)
                if is_valid:
                    operations.append(op)
                else:
                    logger.warning("Invalid operation skipped: %s", error)
        
        return operations
    
    def _parse_single_operation(
        self,
        raw_op: Dict[str, Any],
        expertise: Expertise,
    ) -> Optional[CurationOp]:
        """Parse a single operation from raw dict."""
        op_type_str = raw_op.get("type", "").upper()
        
        # Map string to enum
        type_map = {
            "ADD": CuratorOperation.ADD,
            "UPDATE": CuratorOperation.UPDATE,
            "MERGE": CuratorOperation.MERGE,
            "DELETE": CuratorOperation.DELETE,
        }
        
        op_type = type_map.get(op_type_str)
        if not op_type:
            return None
        
        # Parse section
        section = None
        if "section" in raw_op:
            section = parse_section_from_string(raw_op["section"])
        
        # Parse item_ids
        item_ids = []
        if "item_id" in raw_op:
            item_ids = [raw_op["item_id"]]
        elif "item_ids" in raw_op:
            item_ids = raw_op["item_ids"]
            if isinstance(item_ids, str):
                item_ids = [item_ids]
        
        # Validate item_ids exist (except for ADD)
        if op_type != CuratorOperation.ADD:
            valid_ids = {item.item_id for item in expertise.items}
            item_ids = [id for id in item_ids if id in valid_ids]
        
        # Parse content
        content = raw_op.get("content", "")
        if isinstance(content, str):
            content = content.strip()
        
        # Parse reason
        reason = raw_op.get("reason", "")
        
        return CurationOp(
            type=op_type,
            section=section,
            item_ids=item_ids,
            content=content or None,
            reason=reason,
        )


class MockCurator:
    """
    A mock curator for testing.
    
    Returns predefined curation plans based on configuration.
    Useful for testing the expertise system without LLM calls.
    """
    
    def __init__(
        self,
        operations: Optional[List[CurationOp]] = None,
        reasoning: str = "Mock curation",
        auto_add_from_reflection: bool = True,
    ):
        """
        Initialize mock curator.
        
        Args:
            operations: Predefined operations to return
            reasoning: Reasoning string to return
            auto_add_from_reflection: If True, auto-generate ADD ops from reflection
        """
        self._operations = operations or []
        self._reasoning = reasoning
        self._auto_add = auto_add_from_reflection
    
    @property
    def name(self) -> str:
        """The name of this curator."""
        return "mock-curator"
    
    async def curate(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> Tuple[Expertise, CurationPlan]:
        """Return mock curation result."""
        operations = list(self._operations)
        
        # Auto-generate ADD operations from reflection suggestions
        if self._auto_add and reflection.suggested_additions:
            for suggestion in reflection.suggested_additions:
                operations.append(CurationOp(
                    type=CuratorOperation.ADD,
                    section=ExpertiseSection.STRATEGIES,
                    content=suggestion,
                    reason="From reflection suggestion",
                ))
        
        plan = CurationPlan(
            operations=operations,
            reasoning=self._reasoning,
        )
        
        # Apply the plan
        updated_expertise, _, _, _, _ = apply_curation_plan(expertise, plan)
        
        return updated_expertise, plan


class RuleBasedCurator:
    """
    A rule-based curator that doesn't require an LLM.
    
    Uses simple rules based on reflection feedback:
    - Adds items from suggested_additions
    - Removes items from suggested_removals if they're problematic
    - Marks highly harmful items for deletion
    
    Useful for testing and as a fallback when LLM is unavailable.
    """
    
    def __init__(
        self,
        harmful_threshold: float = 0.3,
        auto_delete_harmful: bool = False,
    ):
        """
        Initialize rule-based curator.
        
        Args:
            harmful_threshold: Effectiveness below this triggers deletion
            auto_delete_harmful: If True, auto-delete highly harmful items
        """
        self._threshold = harmful_threshold
        self._auto_delete = auto_delete_harmful
    
    @property
    def name(self) -> str:
        """The name of this curator."""
        return "rule-based-curator"
    
    async def curate(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> Tuple[Expertise, CurationPlan]:
        """Apply rule-based curation."""
        operations = []
        
        # ADD operations from suggested additions
        for suggestion in reflection.suggested_additions:
            if suggestion.strip():
                operations.append(CurationOp(
                    type=CuratorOperation.ADD,
                    section=ExpertiseSection.STRATEGIES,
                    content=suggestion.strip(),
                    reason="Suggested by reflection",
                ))
        
        # DELETE operations for harmful items
        if self._auto_delete:
            for item in expertise.active_items:
                if item.total_usage >= 3 and item.effectiveness_score < self._threshold:
                    operations.append(CurationOp(
                        type=CuratorOperation.DELETE,
                        item_ids=[item.item_id],
                        reason=f"Low effectiveness ({item.effectiveness_score:.2f})",
                    ))
        
        # Check suggested removals
        for removal in reflection.suggested_removals:
            # Try to extract item ID from removal suggestion
            item_id = self._extract_item_id(removal)
            if item_id and expertise.get_item(item_id):
                operations.append(CurationOp(
                    type=CuratorOperation.DELETE,
                    item_ids=[item_id],
                    reason=removal,
                ))
        
        plan = CurationPlan(
            operations=operations,
            reasoning="Rule-based curation applied",
        )
        
        # Apply the plan
        updated_expertise, adds, updates, merges, deletes = apply_curation_plan(
            expertise, plan
        )
        
        return updated_expertise, plan
    
    def _extract_item_id(self, text: str) -> Optional[str]:
        """Try to extract an item ID from text."""
        import re
        # Look for patterns like "strat-00001" or "[strat-00001]"
        match = re.search(r'\[?([a-z]{4}-\d{5})\]?', text)
        if match:
            return match.group(1)
        return None

