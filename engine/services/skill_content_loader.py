"""
Skill Content Loader.

Parses SKILL.md files with YAML frontmatter and loads scripts/references
from a skill directory. Supports progressive disclosure formatting.
"""
import logging
import os
import re
from typing import Dict, Tuple

from ctxforge.core.skill import SkillContent

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)


def _parse_yaml_simple(text: str) -> Dict[str, str]:
    """Minimal YAML-like key: value parser (avoids PyYAML dependency).

    Handles simple scalar values and comma-separated lists.
    Does NOT handle nested structures or multi-line values.
    """
    result: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip()
    return result


class SkillContentLoader:
    """Load and format structured skill content."""

    @staticmethod
    def parse_skill_markdown(raw_content: str) -> SkillContent:
        """Parse a SKILL.md string into a SkillContent.

        Extracts YAML frontmatter (if present) and treats the remainder
        as the instructions body.

        Args:
            raw_content: Raw markdown string, optionally with YAML frontmatter.

        Returns:
            A SkillContent with instructions populated.
        """
        instructions, _ = SkillContentLoader._split_frontmatter(raw_content)
        return SkillContent(instructions=instructions.strip())

    @staticmethod
    def parse_frontmatter(raw_content: str) -> Dict[str, str]:
        """Extract YAML frontmatter fields from a SKILL.md string.

        Args:
            raw_content: Raw markdown string.

        Returns:
            Dictionary of frontmatter key-value pairs (empty if no frontmatter).
        """
        _, frontmatter = SkillContentLoader._split_frontmatter(raw_content)
        return frontmatter

    @staticmethod
    def load_from_directory(
        skill_dir: str,
        max_script_chars: int = 4000,
        max_ref_chars: int = 8000,
    ) -> SkillContent:
        """Load a SkillContent from a skill directory.

        Expected layout:
            skill_dir/
                SKILL.md          (required)
                scripts/          (optional)
                    setup.sh
                    ...
                references/       (optional)
                    guide.md
                    ...

        Args:
            skill_dir: Path to the skill directory.
            max_script_chars: Maximum characters per script file.
            max_ref_chars: Maximum characters per reference file.

        Returns:
            A SkillContent with instructions, scripts, and references.

        Raises:
            FileNotFoundError: If SKILL.md is not found.
        """
        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md_path):
            raise FileNotFoundError(
                f"SKILL.md not found in {skill_dir}"
            )

        with open(skill_md_path, "r", encoding="utf-8") as f:
            raw = f.read()

        instructions, _ = SkillContentLoader._split_frontmatter(raw)

        scripts: Dict[str, str] = {}
        scripts_dir = os.path.join(skill_dir, "scripts")
        if os.path.isdir(scripts_dir):
            for fname in sorted(os.listdir(scripts_dir)):
                fpath = os.path.join(scripts_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    scripts[fname] = content[:max_script_chars]

        references: Dict[str, str] = {}
        refs_dir = os.path.join(skill_dir, "references")
        if os.path.isdir(refs_dir):
            for fname in sorted(os.listdir(refs_dir)):
                fpath = os.path.join(refs_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    references[fname] = content[:max_ref_chars]

        return SkillContent(
            instructions=instructions.strip(),
            scripts=scripts,
            references=references,
        )

    @staticmethod
    def format_for_prompt(
        content: SkillContent,
        include_scripts: bool = False,
        include_references: bool = False,
    ) -> str:
        """Format a SkillContent for prompt injection.

        Supports progressive disclosure levels:
        - Level 1 (default): instructions only
        - Level 2 (include_scripts=True): instructions + scripts
        - Level 3 (include_scripts=True, include_references=True): full

        Args:
            content: The SkillContent to format.
            include_scripts: Whether to include scripts.
            include_references: Whether to include references.

        Returns:
            Formatted string for prompt injection.
        """
        parts = [content.instructions]

        if include_scripts and content.scripts:
            parts.append("")
            parts.append("### Scripts")
            for name, code in content.scripts.items():
                parts.append(f"\n**{name}**:")
                parts.append(f"```\n{code}\n```")

        if include_references and content.references:
            parts.append("")
            parts.append("### References")
            for name, ref_text in content.references.items():
                parts.append(f"\n**{name}**:")
                parts.append(ref_text)

        return "\n".join(parts)

    @staticmethod
    def _split_frontmatter(
        raw_content: str,
    ) -> Tuple[str, Dict[str, str]]:
        """Split raw content into body and frontmatter dict.

        Args:
            raw_content: Raw markdown string.

        Returns:
            Tuple of (body_text, frontmatter_dict).
        """
        match = _FRONTMATTER_RE.match(raw_content)
        if match:
            frontmatter_text = match.group(1)
            body = match.group(2)
            frontmatter = _parse_yaml_simple(frontmatter_text)
            return body, frontmatter
        return raw_content, {}
