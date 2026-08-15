"""
Tests for Skill Validator.
"""
import pytest

from ctxforge.core.skill import Skill, SkillContent, SkillScope
from ctxforge.engine.services.skill_validator import SkillValidator


def _make_valid_skill(**overrides) -> Skill:
    """Create a valid skill with all required fields populated."""
    defaults = dict(
        name="my-test-skill",
        description="Use when the user asks to run tests",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Test Skill\n\nThis is a valid skill with sufficient content length to pass validation checks.",
        triggers=["run tests", "test suite"],
        when_to_use="Use when the user asks to run tests",
    )
    defaults.update(overrides)
    return Skill(**defaults)


class TestSkillValidator:
    """Tests for SkillValidator."""

    def test_valid_skill_passes(self):
        """A well-formed skill passes all checks."""
        validator = SkillValidator()
        result = validator.validate(_make_valid_skill())
        assert result.is_valid is True
        assert result.errors == []

    def test_name_uppercase_fails(self):
        """Name with uppercase letters is rejected by Skill.__post_init__."""
        with pytest.raises(ValueError):
            _make_valid_skill(name="MySkill")

    def test_name_with_spaces_fails(self):
        """Name with spaces is rejected by Skill.__post_init__."""
        with pytest.raises(ValueError):
            _make_valid_skill(name="my skill")

    def test_empty_description_fails(self):
        """Empty description is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(description="")
        # Skill allows empty description; validator catches it
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("Description" in e for e in result.errors)

    def test_empty_content_fails(self):
        """Empty content is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(content="")
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("content" in e.lower() for e in result.errors)

    def test_short_content_fails(self):
        """Content below min length is rejected."""
        validator = SkillValidator(min_instruction_length=100)
        skill = _make_valid_skill(content="Too short.")
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("too short" in e.lower() for e in result.errors)

    def test_structured_content_instructions_used(self):
        """When structured_content has instructions, they are checked instead of content."""
        validator = SkillValidator(min_instruction_length=50)
        skill = _make_valid_skill(
            content="",  # empty content
            structured_content=SkillContent(
                instructions="This is a detailed instruction set that is well above the minimum length requirement for validation.",
            ),
        )
        result = validator.validate(skill)
        assert result.is_valid is True

    def test_empty_script_content_fails(self):
        """Script with empty content is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(
            structured_content=SkillContent(
                instructions="Valid instructions with enough content to pass the minimum length check.",
                scripts={"setup.py": ""},
            ),
        )
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("setup.py" in e for e in result.errors)

    def test_empty_reference_content_fails(self):
        """Reference with empty content is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(
            structured_content=SkillContent(
                instructions="Valid instructions with enough content to pass the minimum length check.",
                references={"guide.md": "  "},
            ),
        )
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("guide.md" in e for e in result.errors)

    def test_missing_triggers_fails(self):
        """Skill without triggers is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(triggers=[])
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("trigger" in e.lower() for e in result.errors)

    def test_missing_when_to_use_fails(self):
        """Skill without when_to_use is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(when_to_use="")
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("when_to_use" in e for e in result.errors)

    def test_when_to_use_none_fails(self):
        """Skill with when_to_use=None is rejected."""
        validator = SkillValidator()
        skill = _make_valid_skill(when_to_use=None)
        result = validator.validate(skill)
        assert result.is_valid is False
        assert any("when_to_use" in e for e in result.errors)

    def test_multiple_errors_collected(self):
        """Multiple errors are collected in a single result."""
        validator = SkillValidator()
        skill = _make_valid_skill(
            description="",
            content="",
            triggers=[],
            when_to_use="",
        )
        result = validator.validate(skill)
        assert result.is_valid is False
        assert len(result.errors) >= 4

    def test_cso_warnings_populated(self):
        """CSO lint warnings are included (but don't cause failure)."""
        validator = SkillValidator()
        skill = _make_valid_skill(
            description="Use when implementing features, then deploy to production",
        )
        result = validator.validate(skill)
        # CSO violations are warnings, not errors
        assert len(result.warnings) > 0
        # Skill may still be valid (CSO issues are warnings)
        # The important thing is warnings are populated

    def test_validate_and_raise_on_invalid(self):
        """validate_and_raise raises ValueError with all errors."""
        validator = SkillValidator()
        skill = _make_valid_skill(
            content="",
            triggers=[],
            when_to_use="",
        )
        with pytest.raises(ValueError, match="Skill validation failed"):
            validator.validate_and_raise(skill)

    def test_validate_and_raise_on_valid(self):
        """validate_and_raise does not raise for valid skill."""
        validator = SkillValidator()
        skill = _make_valid_skill()
        validator.validate_and_raise(skill)  # Should not raise

    def test_valid_scripts_and_references_pass(self):
        """Non-empty scripts and references pass validation."""
        validator = SkillValidator()
        skill = _make_valid_skill(
            structured_content=SkillContent(
                instructions="Valid instructions with enough content to pass the minimum length check.",
                scripts={"setup.py": "print('setup')"},
                references={"guide.md": "# Guide\nSome content here."},
            ),
        )
        result = validator.validate(skill)
        assert result.is_valid is True
        assert result.errors == []

    def test_custom_min_instruction_length(self):
        """Custom min_instruction_length is respected."""
        validator = SkillValidator(min_instruction_length=10)
        skill = _make_valid_skill(content="Short ok.")
        # 9 chars < 10
        result = validator.validate(skill)
        assert result.is_valid is False

        validator2 = SkillValidator(min_instruction_length=5)
        result2 = validator2.validate(skill)
        assert result2.is_valid is True
