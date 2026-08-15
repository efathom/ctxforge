"""
Prompt templates for office document skill generation.

These prompts guide an LLM to generate a skill package from text content
extracted from office documents (PDF, DOCX, PPTX).
"""

DOCUMENT_SKILL_SYSTEM_PROMPT = """\
You are an expert Technical Writer specializing in creating Skills for AI agents.
Your task is to analyze text content extracted from an office document \
(PDF, PPT, or Word) and convert it into a structured skill package.

CRITICAL REQUIREMENTS:
1. Identify the core knowledge, procedures, or guidelines from the document
2. Structure the content as a reusable AI skill
3. Extract actionable instructions that an AI agent can follow
4. Preserve key information while organizing it into the skill format
5. Generate appropriate scripts if the document describes code procedures
6. Create reference files for supplementary information

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

DOCUMENT_SKILL_USER_PROMPT_TEMPLATE = """\
Convert the following document content into a structured skill package.

## Source File
**Filename:** {file_name}
**File Type:** {file_type}

## Extracted Text Content
{document_text}

## Content Analysis Guidelines
1. **Identify the Skill Name**: Derive from document title or main topic \
(kebab-case)
2. **Create Description**: Write a "when-to-use" trigger statement
3. **Extract Procedures**: Convert step-by-step instructions into actionable \
format
4. **Identify Code/Commands**: If the document contains code, create scripts
5. **Supplementary Info**: Move detailed references to references

## Required Sections in instructions
- **Overview**: Brief summary of what this skill covers
- **When to Use**: Clear triggers for skill activation
- **Prerequisites**: Any required knowledge, tools, or setup
- **Instructions/Procedures**: Main actionable content from document
- **Examples**: Practical examples if available in source
- **References**: Links to additional resources mentioned

Generate the JSON output now.\
"""


def build_document_skill_prompt(
    file_name: str,
    file_type: str,
    document_text: str,
) -> str:
    """Build the user prompt for document skill generation.

    Args:
        file_name: Name of the source document file.
        file_type: Human-readable file type (e.g., "PDF Document").
        document_text: Extracted text content from the document.

    Returns:
        Formatted user prompt string.
    """
    return DOCUMENT_SKILL_USER_PROMPT_TEMPLATE.format(
        file_name=file_name,
        file_type=file_type,
        document_text=document_text,
    )
