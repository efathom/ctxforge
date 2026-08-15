"""
Prompt templates for GitHub repository skill generation.

These prompts guide an LLM to generate a comprehensive skill package
from GitHub repository data including README, file tree, and code analysis.
"""

GITHUB_SKILL_SYSTEM_PROMPT = """\
You are an expert Technical Writer specializing in creating Skills for AI agents.
Your task is to analyze a GitHub repository and generate a comprehensive skill \
package that captures the repository's functionality and usage patterns.

CRITICAL REQUIREMENTS:
1. Generate COMPLETE content - do not truncate or abbreviate sections
2. Include ALL installation steps with actual commands from README
3. Extract CONCRETE code examples from README - copy them exactly
4. List specific models, APIs, or tools mentioned in the repository
5. For scripts: Generate REAL, RUNNABLE Python code that demonstrates library usage
6. For references: Generate DETAILED API documentation with actual function signatures
7. Follow the skill structure standard exactly

SCRIPT QUALITY REQUIREMENTS:
- Scripts must be self-contained and runnable
- Scripts should demonstrate actual library API usage, not shell command wrappers
- Include proper imports, error handling, and docstrings
- If the library requires specific data, use placeholder paths with clear comments

REFERENCE QUALITY REQUIREMENTS:
- API references must include actual function signatures from code analysis
- Include parameter types, return types, and brief descriptions
- Organize by module/class hierarchy
- Reference the source file locations

Output STRICT JSON only (no prose, no markdown fences) using this schema:
{
  "name": "<kebab-case-name>",
  "description": "<when-to-use trigger statement>",
  "when_to_use": "<clear description of when this skill should be activated>",
  "triggers": ["<keyword1>", "<keyword2>", ...],
  "category": "<category>",
  "instructions": "<comprehensive markdown instructions>",
  "scripts": {"<filename.py>": "<script content>", ...},
  "references": {"<filename.md>": "<reference content>", ...}
}
"""

GITHUB_SKILL_USER_PROMPT_TEMPLATE = """\
Generate a complete skill package from the provided GitHub repository information.

## Repository Info
- **Name:** {repo_name}
- **URL:** {repo_url}
- **Description:** {repo_description}
- **Primary Language:** {language}
- **Languages Breakdown:** {languages_breakdown}
- **Stars:** {stars}
- **Topics:** {topics}

## README Content
{readme_content}

## File Structure
{file_tree}

## Code Analysis Summary
{code_summary}

## Required Skill Sections
1. YAML-like frontmatter (name in kebab-case, description)
2. When to Use (primary use cases, trigger conditions)
3. Quick Reference (official docs, resources)
4. Installation/Setup (actual commands from README)
5. Core Features
6. Usage Examples (extracted from README)
7. Key APIs/Models (from code analysis)
8. Common Patterns & Best Practices

Generate the JSON output now.\
"""


def build_github_skill_prompt(
    repo_name: str,
    repo_url: str,
    repo_description: str,
    language: str,
    languages_breakdown: str,
    stars: int,
    topics: str,
    readme_content: str,
    file_tree: str,
    code_summary: str,
) -> str:
    """Build the user prompt for GitHub skill generation.

    Args:
        repo_name: Full repository name (owner/repo).
        repo_url: Repository URL.
        repo_description: Repository description.
        language: Primary language.
        languages_breakdown: Language percentage breakdown.
        stars: Star count.
        topics: Comma-separated topics.
        readme_content: README content (may be truncated).
        file_tree: Formatted file tree.
        code_summary: Code analysis summary.

    Returns:
        Formatted user prompt string.
    """
    return GITHUB_SKILL_USER_PROMPT_TEMPLATE.format(
        repo_name=repo_name,
        repo_url=repo_url,
        repo_description=repo_description,
        language=language,
        languages_breakdown=languages_breakdown,
        stars=stars,
        topics=topics,
        readme_content=readme_content,
        file_tree=file_tree,
        code_summary=code_summary,
    )
