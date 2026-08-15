"""
Prompt templates for skill quality evaluation.

These prompts guide an LLM to evaluate a skill across five dimensions:
Safety, Completeness, Executability, Maintainability, and Cost-Awareness.
"""

SKILL_EVALUATION_SYSTEM_PROMPT = """\
You are an expert skill evaluator. Your job is to assess the quality of \
agent skills across five dimensions. Each dimension must be rated as \
"good", "average", or "poor" with a brief reason.

Respond ONLY with a valid JSON object (no markdown fences, no commentary) \
using this exact schema:

{
  "safety": "<good|average|poor>",
  "safety_reason": "<one-sentence justification>",
  "completeness": "<good|average|poor>",
  "completeness_reason": "<one-sentence justification>",
  "executability": "<good|average|poor>",
  "executability_reason": "<one-sentence justification>",
  "maintainability": "<good|average|poor>",
  "maintainability_reason": "<one-sentence justification>",
  "cost_awareness": "<good|average|poor>",
  "cost_awareness_reason": "<one-sentence justification>"
}
"""

SKILL_EVALUATION_USER_PROMPT_TEMPLATE = """\
Evaluate the following skill.

## Skill Name
{name}

## Description
{description}

## Instructions / Content
{content}

{scripts_section}

{references_section}

{tools_section}

{script_execution_section}

## Evaluation Criteria

1. **Safety**: Does the skill avoid destructive actions, protect sensitive \
data, and include appropriate guardrails?
2. **Completeness**: Does the skill cover the full workflow end-to-end, \
including edge cases and error handling?
3. **Executability**: Can an agent follow the instructions step-by-step \
without ambiguity? Are the steps concrete and actionable?
4. **Maintainability**: Is the skill well-structured, modular, and easy \
to update or extend?
5. **Cost-Awareness**: Does the skill minimize unnecessary LLM calls, \
tool invocations, and token usage?

Rate each dimension as "good", "average", or "poor" with a brief reason.
"""


def build_evaluation_prompt(
    name: str,
    description: str,
    content: str,
    scripts: str = "",
    references: str = "",
    allowed_tools: str = "",
    script_execution_results: str = "",
) -> str:
    """Build the user prompt for skill evaluation.

    Args:
        name: Skill name.
        description: Skill description.
        content: Main skill content / instructions.
        scripts: Formatted scripts section (may be empty).
        references: Formatted references section (may be empty).
        allowed_tools: Comma-separated list of allowed tools (may be empty).
        script_execution_results: Formatted script execution results (may be empty).

    Returns:
        Formatted user prompt string.
    """
    scripts_section = ""
    if scripts:
        scripts_section = f"## Scripts\n{scripts}"

    references_section = ""
    if references:
        references_section = f"## References\n{references}"

    tools_section = ""
    if allowed_tools:
        tools_section = f"## Allowed Tools\n{allowed_tools}"

    script_execution_section = ""
    if script_execution_results:
        script_execution_section = (
            "## Script Execution Results\n"
            f"{script_execution_results}\n\n"
            "Use these results to inform your Executability rating:\n"
            "- All scripts succeeded: strong positive signal\n"
            "- Compile-only (no execution): neutral (may need inputs)\n"
            "- Failures with clear errors: negative signal\n"
            "- Timeouts: potential infinite loop or heavy computation\n"
            "- Skipped (placeholders): neutral (documentation examples)"
        )

    return SKILL_EVALUATION_USER_PROMPT_TEMPLATE.format(
        name=name,
        description=description,
        content=content,
        scripts_section=scripts_section,
        references_section=references_section,
        tools_section=tools_section,
        script_execution_section=script_execution_section,
    )
