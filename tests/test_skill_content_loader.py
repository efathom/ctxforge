"""
Tests for Skill Content Loader.
"""
import os
import tempfile

import pytest

from ctxforge.core.skill import Skill, SkillContent, SkillScope
from ctxforge.engine.services.skill_content_loader import SkillContentLoader
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.storage.memory.skill import InMemorySkillStore


class TestParseSkillMarkdown:
    """Tests for SkillContentLoader.parse_skill_markdown."""

    def test_extracts_instructions_no_frontmatter(self):
        raw = "# Steps\n1. Do A\n2. Do B"
        sc = SkillContentLoader.parse_skill_markdown(raw)
        assert sc.instructions == "# Steps\n1. Do A\n2. Do B"
        assert sc.scripts == {}
        assert sc.references == {}

    def test_extracts_frontmatter_fields(self):
        raw = "---\nname: my-skill\ndescription: Does things\ncategory: testing\n---\n# Instructions\nDo it."
        fm = SkillContentLoader.parse_frontmatter(raw)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Does things"
        assert fm["category"] == "testing"

    def test_extracts_instructions_with_frontmatter(self):
        raw = "---\nname: my-skill\n---\n# Instructions\nStep 1."
        sc = SkillContentLoader.parse_skill_markdown(raw)
        assert sc.instructions == "# Instructions\nStep 1."

    def test_handles_missing_frontmatter(self):
        raw = "Just plain markdown content."
        sc = SkillContentLoader.parse_skill_markdown(raw)
        assert sc.instructions == "Just plain markdown content."
        fm = SkillContentLoader.parse_frontmatter(raw)
        assert fm == {}

    def test_handles_empty_content(self):
        sc = SkillContentLoader.parse_skill_markdown("")
        assert sc.instructions == ""

    def test_handles_frontmatter_only(self):
        raw = "---\nname: test\n---\n"
        sc = SkillContentLoader.parse_skill_markdown(raw)
        assert sc.instructions == ""


class TestLoadFromDirectory:
    """Tests for SkillContentLoader.load_from_directory."""

    def test_loads_skill_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "SKILL.md"), "w") as f:
                f.write("---\nname: test\n---\n# Instructions\nDo it.")
            sc = SkillContentLoader.load_from_directory(tmpdir)
            assert "Instructions" in sc.instructions
            assert sc.scripts == {}
            assert sc.references == {}

    def test_loads_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "SKILL.md"), "w") as f:
                f.write("# Inst")
            scripts_dir = os.path.join(tmpdir, "scripts")
            os.makedirs(scripts_dir)
            with open(os.path.join(scripts_dir, "setup.sh"), "w") as f:
                f.write("#!/bin/bash\necho hello")
            sc = SkillContentLoader.load_from_directory(tmpdir)
            assert "setup.sh" in sc.scripts
            assert "echo hello" in sc.scripts["setup.sh"]

    def test_loads_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "SKILL.md"), "w") as f:
                f.write("# Inst")
            refs_dir = os.path.join(tmpdir, "references")
            os.makedirs(refs_dir)
            with open(os.path.join(refs_dir, "guide.md"), "w") as f:
                f.write("# Guide\nDetailed info.")
            sc = SkillContentLoader.load_from_directory(tmpdir)
            assert "guide.md" in sc.references
            assert "Detailed info" in sc.references["guide.md"]

    def test_respects_max_script_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "SKILL.md"), "w") as f:
                f.write("# Inst")
            scripts_dir = os.path.join(tmpdir, "scripts")
            os.makedirs(scripts_dir)
            with open(os.path.join(scripts_dir, "big.sh"), "w") as f:
                f.write("x" * 10000)
            sc = SkillContentLoader.load_from_directory(
                tmpdir, max_script_chars=100
            )
            assert len(sc.scripts["big.sh"]) == 100

    def test_respects_max_ref_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "SKILL.md"), "w") as f:
                f.write("# Inst")
            refs_dir = os.path.join(tmpdir, "references")
            os.makedirs(refs_dir)
            with open(os.path.join(refs_dir, "big.md"), "w") as f:
                f.write("y" * 20000)
            sc = SkillContentLoader.load_from_directory(
                tmpdir, max_ref_chars=500
            )
            assert len(sc.references["big.md"]) == 500

    def test_raises_if_no_skill_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="SKILL.md"):
                SkillContentLoader.load_from_directory(tmpdir)


class TestFormatForPrompt:
    """Tests for SkillContentLoader.format_for_prompt."""

    def test_level1_instructions_only(self):
        sc = SkillContent(
            instructions="Do the thing.",
            scripts={"a.sh": "echo a"},
            references={"b.md": "# B"},
        )
        result = SkillContentLoader.format_for_prompt(sc)
        assert "Do the thing." in result
        assert "Scripts" not in result
        assert "References" not in result

    def test_level2_includes_scripts(self):
        sc = SkillContent(
            instructions="Do the thing.",
            scripts={"a.sh": "echo a"},
            references={"b.md": "# B"},
        )
        result = SkillContentLoader.format_for_prompt(
            sc, include_scripts=True
        )
        assert "Do the thing." in result
        assert "Scripts" in result
        assert "echo a" in result
        assert "References" not in result

    def test_level3_includes_scripts_and_references(self):
        sc = SkillContent(
            instructions="Do the thing.",
            scripts={"a.sh": "echo a"},
            references={"b.md": "# B"},
        )
        result = SkillContentLoader.format_for_prompt(
            sc, include_scripts=True, include_references=True
        )
        assert "Do the thing." in result
        assert "Scripts" in result
        assert "References" in result
        assert "# B" in result

    def test_empty_scripts_and_refs(self):
        sc = SkillContent(instructions="Just instructions.")
        result = SkillContentLoader.format_for_prompt(
            sc, include_scripts=True, include_references=True
        )
        assert result == "Just instructions."


class TestFormatSkillWorkflowBackwardCompat:
    """Tests for SkillService.format_skill_workflow backward compatibility."""

    def test_falls_back_to_content_when_no_structured_content(self):
        store = InMemorySkillStore()
        svc = SkillService(store)
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="# Old Content\nStep 1",
        )
        result = svc.format_skill_workflow(skill)
        assert "# Old Content" in result
        assert "Step 1" in result

    def test_uses_structured_content_when_available(self):
        store = InMemorySkillStore()
        svc = SkillService(store)
        sc = SkillContent(
            instructions="Structured instructions.",
            scripts={"run.sh": "echo run"},
            references={"doc.md": "# Doc"},
        )
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="fallback content",
            structured_content=sc,
        )
        result = svc.format_skill_workflow(skill, detail_level=1)
        assert "Structured instructions." in result
        assert "Scripts" not in result

    def test_detail_level_2_includes_scripts(self):
        store = InMemorySkillStore()
        svc = SkillService(store)
        sc = SkillContent(
            instructions="Instructions.",
            scripts={"run.sh": "echo run"},
        )
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="fallback",
            structured_content=sc,
        )
        result = svc.format_skill_workflow(skill, detail_level=2)
        assert "Instructions." in result
        assert "Scripts" in result
        assert "echo run" in result

    def test_detail_level_3_includes_references(self):
        store = InMemorySkillStore()
        svc = SkillService(store)
        sc = SkillContent(
            instructions="Instructions.",
            scripts={"run.sh": "echo run"},
            references={"doc.md": "# Documentation"},
        )
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="fallback",
            structured_content=sc,
        )
        result = svc.format_skill_workflow(skill, detail_level=3)
        assert "Instructions." in result
        assert "Scripts" in result
        assert "References" in result
        assert "# Documentation" in result
