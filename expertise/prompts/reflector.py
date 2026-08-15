"""
Reflector prompts for the Expertise system.

These prompts are used by the ExpertiseReflector to analyze turns
and provide feedback on expertise item effectiveness.

Adapted from ACE framework's reflector prompts.
"""

# System prompt for the reflector
REFLECTOR_SYSTEM_PROMPT = """You are an expert at analyzing AI assistant responses and evaluating which knowledge items were helpful or harmful.

Your task is to:
1. Analyze the conversation turn outcome (success, failure, partial)
2. Evaluate each expertise item that was used
3. Tag each item as "helpful", "harmful", or "neutral"
4. Provide insights for improvement
5. Suggest new expertise items to add or existing ones to remove

Be specific and actionable in your analysis. Focus on the root cause of any issues, not just surface-level errors."""


# User prompt template when ground truth is available
REFLECTOR_USER_PROMPT = """Analyze this conversation turn and evaluate the effectiveness of the expertise items used.

## USER INPUT
{user_input}

## ASSISTANT RESPONSE
{assistant_response}

## EXPECTED OUTPUT (Ground Truth)
{expected_output}

## TURN OUTCOME
{outcome}

## EXPERTISE ITEMS USED
{expertise_items}

---

Provide your analysis in the following JSON format:
{{
    "reasoning": "Your detailed analysis of what happened in this turn...",
    "error_identification": "What specifically went wrong (if anything)?",
    "root_cause_analysis": "Why did this error occur? What was misunderstood?",
    "correct_approach": "What should have been done instead?",
    "bullet_tags": [
        {{"id": "strat-00001", "tag": "helpful"}},
        {{"id": "form-00002", "tag": "harmful"}}
    ],
    "insights": "Key takeaways and lessons learned from this turn.",
    "suggested_additions": [
        "New expertise item content to add..."
    ],
    "suggested_removals": [
        "ID of item to remove and reason..."
    ],
    "confidence": 0.85
}}

Note: tag must be one of: "helpful", "harmful", "neutral"
Only include bullet_tags for items that were actually used."""


# User prompt template when no ground truth is available
REFLECTOR_USER_PROMPT_NO_GT = """Analyze this conversation turn and evaluate the effectiveness of the expertise items used.

## USER INPUT
{user_input}

## ASSISTANT RESPONSE
{assistant_response}

## ACTUAL OUTCOME
{actual_outcome}

## TURN OUTCOME
{outcome}

## EXPERTISE ITEMS USED
{expertise_items}

---

Provide your analysis in the following JSON format:
{{
    "reasoning": "Your detailed analysis of what happened in this turn...",
    "error_identification": "What specifically went wrong (if anything)?",
    "root_cause_analysis": "Why did this error occur? What was misunderstood?",
    "correct_approach": "What should have been done instead?",
    "bullet_tags": [
        {{"id": "strat-00001", "tag": "helpful"}},
        {{"id": "form-00002", "tag": "harmful"}}
    ],
    "insights": "Key takeaways and lessons learned from this turn.",
    "suggested_additions": [
        "New expertise item content to add..."
    ],
    "suggested_removals": [
        "ID of item to remove and reason..."
    ],
    "confidence": 0.85
}}

Note: tag must be one of: "helpful", "harmful", "neutral"
Only include bullet_tags for items that were actually used."""


# User prompt for successful turns
REFLECTOR_SUCCESS_PROMPT = """Analyze this successful conversation turn and identify which expertise items contributed to the success.

## USER INPUT
{user_input}

## ASSISTANT RESPONSE
{assistant_response}

## EXPECTED OUTPUT (if available)
{expected_output}

## EXPERTISE ITEMS USED
{expertise_items}

---

Provide your analysis in the following JSON format:
{{
    "reasoning": "Your analysis of why this turn was successful...",
    "bullet_tags": [
        {{"id": "strat-00001", "tag": "helpful"}},
        {{"id": "form-00002", "tag": "neutral"}}
    ],
    "insights": "What made this successful? What patterns should be reinforced?",
    "suggested_additions": [
        "New expertise item to capture this success pattern..."
    ],
    "suggested_removals": [],
    "confidence": 0.90
}}

Note: tag must be one of: "helpful", "harmful", "neutral"
For successful turns, focus on identifying which items were most helpful."""

