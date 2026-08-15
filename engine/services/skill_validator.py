"""
Skill Validator.

Validates skill content for structural completeness and quality before
registration/persistence. Prevents malformed skills from entering the store.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List

from ctxforge.core.skill import Skill, lint_skill_description

logger = logging.getLogger(__name__)

_KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass
class ValidationResult:
    """Result of validating a skill."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SkillValidator:
    """Validate skill content for structural completeness and quality.

    Runs after generation, before registration/persistence.
    """

    def __init__(self, min_instruction_length: int = 50):
        self._min_instruction_length = min_instruction_length

    def validate(self, skill: Skill) -> ValidationResult:
        """Validate a skill and return a result with errors and warnings.

        Checks:
        1. Name is kebab-case.
        2. Description is non-empty.
        3. Description passes CSO lint (warnings only).
        4. Content/instructions is non-empty and >= min_instruction_length.
        5. If structured_content.scripts present, each has non-empty content.
        6. If structured_content.references present, each has non-empty content.
        7. triggers list is non-empty.
        8. when_to_use is non-empty.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Name format
        if not _KEBAB_RE.match(skill.name):
            errors.append(
                f"Name '{skill.name}' must be lowercase, start with a letter, "
                "and contain only letters, numbers, and hyphens."
            )

        # 2. Description
        if not skill.description or not skill.description.strip():
            errors.append("Description must not be empty.")

        # 3. CSO lint (warnings, not errors)
        if skill.description:
            cso_warnings = lint_skill_description(skill.description)
            warnings.extend(cso_warnings)

        # 4. Content / instructions
        content_text = skill.content or ""
        if skill.structured_content and skill.structured_content.instructions:
            content_text = skill.structured_content.instructions

        if not content_text.strip():
            errors.append("Skill content/instructions must not be empty.")
        elif len(content_text.strip()) < self._min_instruction_length:
            errors.append(
                f"Skill content/instructions is too short "
                f"({len(content_text.strip())} chars, "
                f"minimum {self._min_instruction_length})."
            )

        # 5. Scripts with empty content
        if skill.structured_content and skill.structured_content.scripts:
            for name, code in skill.structured_content.scripts.items():
                if not code or not code.strip():
                    errors.append(f"Script '{name}' has empty content.")

        # 6. References with empty content
        if skill.structured_content and skill.structured_content.references:
            for name, ref in skill.structured_content.references.items():
                if not ref or not ref.strip():
                    errors.append(f"Reference '{name}' has empty content.")

        # 7. Triggers
        if not skill.triggers:
            errors.append("Skill must have at least one trigger.")

        # 8. when_to_use
        if not skill.when_to_use or not skill.when_to_use.strip():
            errors.append("Skill must have a when_to_use description.")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_and_raise(self, skill: Skill) -> None:
        """Validate a skill and raise ValueError if invalid.

        Raises:
            ValueError: With all validation errors joined by newline.
        """
        result = self.validate(skill)
        if not result.is_valid:
            msg = "Skill validation failed:\n" + "\n".join(
                f"  - {e}" for e in result.errors
            )
            raise ValueError(msg)
