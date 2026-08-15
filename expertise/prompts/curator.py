"""
Curator prompts for the Expertise system.

These prompts are used by the ExpertiseCurator to generate and apply
curation operations based on reflection feedback.

Adapted from ACE framework's curator prompts.
"""

# System prompt for the curator
CURATOR_SYSTEM_PROMPT = """You are a master curator of knowledge. Your job is to evolve an expertise knowledge base based on reflection feedback from conversation turns.

Your task is to:
1. Analyze the current expertise and recent reflection feedback
2. Identify operations to improve the expertise:
   - ADD: Create new expertise items from insights
   - UPDATE: Improve existing items based on feedback
   - MERGE: Combine similar or redundant items
   - DELETE: Remove harmful or outdated items
3. Focus on quality over quantity - a focused, well-organized expertise is better than an exhaustive one
4. Avoid redundancy - only add new content that complements existing items
5. Be concise and specific - each operation should be actionable

You MUST respond with valid JSON only."""


# User prompt for curation
CURATOR_USER_PROMPT = """Curate the expertise based on the reflection feedback.

## EXPERTISE STATS
{expertise_stats}

## RECENT REFLECTION
{reflection}

## CURRENT EXPERTISE
{expertise_content}

## REFLECTION INSIGHTS
Helpful items: {helpful_items}
Harmful items: {harmful_items}
Suggested additions: {suggested_additions}
Suggested removals: {suggested_removals}

---

Provide your curation plan in the following JSON format:
{{
    "reasoning": "Your analysis of what changes are needed and why...",
    "operations": [
        {{
            "type": "ADD",
            "section": "strategies_and_insights",
            "content": "New expertise content to add...",
            "reason": "Why this should be added"
        }},
        {{
            "type": "UPDATE",
            "item_id": "strat-00001",
            "content": "Updated content for existing item...",
            "reason": "Why this update is needed"
        }},
        {{
            "type": "MERGE",
            "item_ids": ["form-00001", "form-00002"],
            "content": "Merged content combining both items...",
            "reason": "Why these should be merged"
        }},
        {{
            "type": "DELETE",
            "item_id": "mist-00001",
            "reason": "Why this should be removed"
        }}
    ]
}}

Notes:
- type must be one of: ADD, UPDATE, MERGE, DELETE
- For ADD: requires section and content
- For UPDATE: requires item_id and content
- For MERGE: requires item_ids (list) and content
- For DELETE: requires item_id
- section must be one of: strategies_and_insights, formulas_and_calculations, code_snippets_and_templates, common_mistakes_to_avoid, problem_solving_heuristics, context_clues_and_indicators, custom
- Return empty operations list if no changes are needed"""


# Simplified prompt focused on ADD operations (for initial learning)
CURATOR_ADD_ONLY_PROMPT = """Review the reflection feedback and identify NEW expertise items to add.

## EXPERTISE STATS
- Total items: {total_items}
- Token budget: {token_budget} tokens
- Current token usage: ~{current_tokens} tokens

## RECENT REFLECTION
{reflection}

## CURRENT EXPERTISE
{expertise_content}

## INSIGHTS FROM REFLECTION
{insights}

---

Identify ONLY NEW insights that are MISSING from the current expertise.
Avoid redundancy - only add content that complements existing items.

Respond in JSON format:
{{
    "reasoning": "Your analysis of what new insights should be added...",
    "operations": [
        {{
            "type": "ADD",
            "section": "strategies_and_insights",
            "content": "New expertise content...",
            "reason": "Why this is valuable"
        }}
    ]
}}

Notes:
- Only include ADD operations
- section must be one of: strategies_and_insights, formulas_and_calculations, code_snippets_and_templates, common_mistakes_to_avoid, problem_solving_heuristics, context_clues_and_indicators, custom
- Return empty operations list if nothing new to add"""


# Prompt for maintenance operations (UPDATE, MERGE, DELETE)
CURATOR_MAINTENANCE_PROMPT = """Review the expertise and identify maintenance operations needed.

## EXPERTISE STATS
{expertise_stats}

## USAGE STATISTICS
{usage_stats}

## PROBLEMATIC ITEMS (high harmful count)
{problematic_items}

## UNUSED ITEMS (never used)
{unused_items}

## POTENTIAL DUPLICATES
{similar_items}

---

Identify maintenance operations to improve expertise quality:
- UPDATE: Improve items that are partially effective
- MERGE: Combine similar or redundant items
- DELETE: Remove harmful or outdated items

Respond in JSON format:
{{
    "reasoning": "Your analysis of maintenance needs...",
    "operations": [
        {{
            "type": "UPDATE",
            "item_id": "strat-00001",
            "content": "Improved content...",
            "reason": "Why this update helps"
        }},
        {{
            "type": "MERGE",
            "item_ids": ["form-00001", "form-00002"],
            "content": "Combined content...",
            "reason": "Why these are redundant"
        }},
        {{
            "type": "DELETE",
            "item_id": "mist-00001",
            "reason": "Why this should be removed"
        }}
    ]
}}

Notes:
- Focus on quality improvements, not additions
- Only delete items that are consistently harmful
- Merge items with >80% content similarity
- Return empty operations list if no maintenance needed"""

