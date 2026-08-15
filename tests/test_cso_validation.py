"""
Tests for CSO (Claude Search Optimization) Validation.

Tests that skill descriptions are validated for CSO compliance:
descriptions should contain only triggering conditions, not workflow
summaries that LLMs might follow as shortcuts.
"""

import logging

from ctxforge.core.skill import (
    Skill,
    SkillMetadata,
    SkillScope,
    lint_skill_description,
)


class TestLintSkillDescription:
    """Tests for the lint_skill_description function."""

    def test_good_description_no_warnings(self):
        """A trigger-only description should produce no warnings."""
        desc = "Use when encountering any bug, test failure, or unexpected behavior"
        warnings = lint_skill_description(desc)
        assert warnings == []

    def test_good_description_with_use_when(self):
        """Standard 'Use when ...' pattern should pass."""
        desc = "Use when tests have race conditions or pass/fail inconsistently"
        assert lint_skill_description(desc) == []

    def test_bad_description_with_process_verbs(self):
        """Descriptions with process verbs should produce warnings."""
        desc = "Dispatches subagent per task with code review between tasks"
        warnings = lint_skill_description(desc)
        assert len(warnings) > 0
        assert "process verbs" in warnings[0].lower() or "process verb" in warnings[0].lower()

    def test_bad_description_with_execute(self):
        """'execute' is a process verb that should trigger a warning."""
        desc = "Executes plan by running tests and deploying code"
        warnings = lint_skill_description(desc)
        assert len(warnings) > 0

    def test_bad_description_with_sequence_markers(self):
        """Descriptions with 'then', 'followed by' etc. suggest workflow."""
        desc = "Use when building features, then review code, followed by deployment"
        warnings = lint_skill_description(desc)
        # Should flag both process verbs and sequence markers
        assert len(warnings) >= 1

    def test_bad_description_with_then(self):
        """'then' as a sequence marker should be flagged."""
        desc = "First analyzes the code then refactors it"
        warnings = lint_skill_description(desc)
        assert len(warnings) > 0

    def test_bad_description_with_step_numbers(self):
        """Step numbers indicate a workflow summary."""
        desc = "Step 1 gather data step 2 analyze results"
        warnings = lint_skill_description(desc)
        assert len(warnings) > 0

    def test_empty_description_no_warnings(self):
        """Empty description should produce no warnings."""
        assert lint_skill_description("") == []

    def test_short_trigger_description(self):
        """Short, focused trigger descriptions should pass."""
        good_descriptions = [
            "Use when implementing any feature or bugfix",
            "Use when tests are flaky or timing-dependent",
            "Use when starting a new development branch",
            "Use when encountering merge conflicts",
        ]
        for desc in good_descriptions:
            assert lint_skill_description(desc) == [], f"False positive: {desc}"

    def test_workflow_summary_descriptions(self):
        """Descriptions that summarize workflows should all be flagged."""
        bad_descriptions = [
            "Dispatches fresh subagent per task with two-stage review",
            "Creates a plan then executes each step sequentially",
            "Generates test cases and reviews code quality",
            "Analyzes codebase, then refactors and deploys",
        ]
        for desc in bad_descriptions:
            warnings = lint_skill_description(desc)
            assert len(warnings) > 0, f"False negative: {desc}"


class TestCSOIntegrationWithSkillMetadata:
    """Tests that CSO validation is integrated into SkillMetadata creation."""

    def test_good_description_logs_no_warnings(self, caplog):
        """Creating metadata with a good description should not log warnings."""
        with caplog.at_level(logging.WARNING, logger="ctxforge.core.skill"):
            SkillMetadata(
                name="good-skill",
                description="Use when tests fail unexpectedly",
                scope=SkillScope.BASE,
                scope_id="system",
            )
        cso_messages = [r for r in caplog.records if "CSO" in r.message]
        assert len(cso_messages) == 0

    def test_bad_description_logs_warning(self, caplog):
        """Creating metadata with a bad description should log CSO warnings."""
        with caplog.at_level(logging.WARNING, logger="ctxforge.core.skill"):
            SkillMetadata(
                name="bad-skill",
                description="Dispatches agents and reviews code then deploys",
                scope=SkillScope.BASE,
                scope_id="system",
            )
        cso_messages = [r for r in caplog.records if "CSO" in r.message]
        assert len(cso_messages) > 0

    def test_cso_warnings_property(self):
        """The cso_warnings property should return current warnings."""
        meta = SkillMetadata(
            name="test-skill",
            description="Executes tests and generates reports",
            scope=SkillScope.BASE,
            scope_id="system",
        )
        assert len(meta.cso_warnings) > 0


class TestCSOIntegrationWithSkill:
    """Tests that CSO validation is integrated into Skill creation."""

    def test_skill_cso_warnings_property(self):
        """Full Skill objects should also expose cso_warnings."""
        skill = Skill(
            name="workflow-skill",
            description="Creates plans then dispatches subagents",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# Workflow\nDo things.",
        )
        assert len(skill.cso_warnings) > 0

    def test_skill_good_description_no_warnings(self):
        """A Skill with a trigger-only description should have no CSO warnings."""
        skill = Skill(
            name="trigger-skill",
            description="Use when debugging race conditions in async code",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# Debugging\nInvestigate root cause.",
        )
        assert skill.cso_warnings == []
