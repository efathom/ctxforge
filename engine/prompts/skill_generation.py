"""
Prompt templates for skill generation from sessions, observations, and prompts.

Two-phase approach:
  Phase 1 - Identify candidate skill metadata from session events.
  Phase 2 - Generate full skill content for each candidate.
"""

CANDIDATE_METADATA_SYSTEM_PROMPT = """\
You are an expert at identifying reusable agent skills from session transcripts.

Given a list of session events (user messages, agent actions, tool calls, \
observations), identify potential reusable skills that could be extracted.

For each candidate skill, provide:
- name: kebab-case identifier (e.g., "fix-import-errors")
- description: One-sentence description (max 256 chars)
- when_to_use: When an agent should activate this skill
- category: One of: debugging, testing, deployment, refactoring, documentation, coding, other

Respond ONLY with a valid JSON array (no markdown fences, no commentary). \
Each element must have this schema:

{
  "name": "<kebab-case-name>",
  "description": "<short description>",
  "when_to_use": "<trigger description>",
  "category": "<category>"
}

If no reusable skills can be extracted, return an empty array: []
"""

CANDIDATE_METADATA_USER_PROMPT_TEMPLATE = """\
Analyze the following session events and identify reusable skills:

{events_text}

Extract skills that represent repeatable workflows, patterns, or procedures \
that an agent could reuse in future sessions.
"""

SKILL_CONTENT_SYSTEM_PROMPT = """\
You are an expert at writing agent skill instructions.

Given a skill name, description, and context, generate clear, step-by-step \
instructions that an AI agent can follow.

Respond ONLY with a valid JSON object (no markdown fences, no commentary) \
using this exact schema:

{
  "instructions": "<markdown instructions with numbered steps>",
  "scripts": {},
  "references": {}
}

The instructions should be:
- Concrete and actionable (no vague steps)
- Include error handling guidance
- Mention relevant tools when applicable
"""

SKILL_CONTENT_USER_PROMPT_TEMPLATE = """\
Generate skill content for:

**Name:** {name}
**Description:** {description}
**When to use:** {when_to_use}
**Category:** {category}

{context_section}

Write clear, step-by-step instructions that an AI agent can follow to \
perform this skill.
"""

PROMPT_SKILL_SYSTEM_PROMPT = """\
You are an expert at creating agent skills from natural language descriptions.

Given a description of what the skill should do, generate a complete skill \
definition.

Respond ONLY with a valid JSON object (no markdown fences, no commentary) \
using this exact schema:

{
  "name": "<kebab-case-name>",
  "description": "<short description, max 256 chars>",
  "when_to_use": "<trigger description>",
  "category": "<category>",
  "instructions": "<markdown instructions with numbered steps>",
  "triggers": ["<keyword1>", "<keyword2>"]
}
"""

PROMPT_SKILL_USER_PROMPT_TEMPLATE = """\
Create a reusable agent skill based on this description:

{description}

The skill should be concrete, actionable, and reusable across sessions.
"""


def build_candidate_prompt(events_text: str) -> str:
    """Build the user prompt for candidate metadata extraction."""
    return CANDIDATE_METADATA_USER_PROMPT_TEMPLATE.format(
        events_text=events_text,
    )


def build_content_prompt(
    name: str,
    description: str,
    when_to_use: str,
    category: str,
    context: str = "",
) -> str:
    """Build the user prompt for skill content generation."""
    context_section = ""
    if context:
        context_section = f"**Additional Context:**\n{context}"
    return SKILL_CONTENT_USER_PROMPT_TEMPLATE.format(
        name=name,
        description=description,
        when_to_use=when_to_use,
        category=category,
        context_section=context_section,
    )


def build_prompt_skill_prompt(description: str) -> str:
    """Build the user prompt for generating a skill from a text description."""
    return PROMPT_SKILL_USER_PROMPT_TEMPLATE.format(description=description)
