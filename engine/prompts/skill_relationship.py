"""
Prompt templates for skill relationship analysis.

These prompts guide an LLM to infer typed relationships between skills.
"""

RELATIONSHIP_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert at analyzing relationships between agent skills. \
Given a list of skills (name + description), infer meaningful relationships.

Valid relationship types:
- similar_to: Skills that are functionally interchangeable or very close in purpose.
- belong_to: One skill is a sub-task or specialization of another (child -> parent).
- compose_with: Independent skills that are often used together in a workflow.
- depend_on: One skill requires another as a hard prerequisite (dependent -> prerequisite).

Respond ONLY with a valid JSON array (no markdown fences, no commentary). \
Each element must have this schema:

{
  "source": "<skill-name>",
  "target": "<skill-name>",
  "relation_type": "<similar_to|belong_to|compose_with|depend_on>",
  "reason": "<one-sentence justification>",
  "confidence": <0.0-1.0>
}

Only include relationships you are confident about (confidence >= 0.6). \
If no relationships exist, return an empty array: []
"""

RELATIONSHIP_ANALYSIS_USER_PROMPT_TEMPLATE = """\
Analyze the relationships between the following skills:

{skills_list}

Identify all meaningful relationships between these skills. \
Consider how they might depend on, compose with, or relate to each other.
"""


def build_relationship_prompt(skills_info: str) -> str:
    """Build the user prompt for relationship analysis.

    Args:
        skills_info: Formatted string listing skill names and descriptions.

    Returns:
        Formatted user prompt string.
    """
    return RELATIONSHIP_ANALYSIS_USER_PROMPT_TEMPLATE.format(
        skills_list=skills_info,
    )
