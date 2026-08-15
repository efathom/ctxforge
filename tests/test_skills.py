"""
Tests for Skill Models and Protocols.

This module tests the core skill data structures
and storage protocol implementations.
"""
from datetime import datetime

import pytest

from ctxforge.config.base import (
    SkillEffectivenessConfig,
    SkillEvaluationConfig,
    SkillGenerationConfig,
    SkillRelationshipConfig,
    SkillsConfig,
)
from ctxforge.core.skill import (
    EVALUATION_LEVEL_SCORES,
    EvaluationLevel,
    Skill,
    SkillContent,
    SkillEvaluation,
    SkillMatch,
    SkillMetadata,
    SkillRelationship,
    SkillRelationType,
    SkillScope,
    SkillsIndex,
)
from ctxforge.engine.services.skill_matcher import RegexSkillMatcher, SkillMatcher
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skills import SkillRequestMiddleware, SkillsMiddleware
from ctxforge.storage.memory.skill import InMemorySkillStore


class TestSkillScope:
    """Tests for SkillScope enum."""

    def test_scope_values(self):
        """Test that all scope values are defined."""
        assert SkillScope.BASE.value == "base"
        assert SkillScope.USER.value == "user"
        assert SkillScope.PROJECT.value == "project"

    def test_scope_priority(self):
        """Test scope priority ordering."""
        assert SkillScope.priority(SkillScope.BASE) == 0
        assert SkillScope.priority(SkillScope.USER) == 1
        assert SkillScope.priority(SkillScope.PROJECT) == 2

    def test_project_overrides_user(self):
        """Test that project has higher priority than user."""
        assert SkillScope.priority(SkillScope.PROJECT) > \
               SkillScope.priority(SkillScope.USER)

    def test_user_overrides_base(self):
        """Test that user has higher priority than base."""
        assert SkillScope.priority(SkillScope.USER) > \
               SkillScope.priority(SkillScope.BASE)


class TestSkillMetadata:
    """Tests for SkillMetadata dataclass."""

    def test_create_metadata(self):
        """Test basic metadata creation."""
        meta = SkillMetadata(
            name="sql-optimize",
            description="Optimize SQL queries",
            scope=SkillScope.BASE,
            scope_id="system",
        )

        assert meta.name == "sql-optimize"
        assert meta.description == "Optimize SQL queries"
        assert meta.scope == SkillScope.BASE
        assert meta.triggers == []
        assert meta.version == "1.0"

    def test_metadata_with_triggers(self):
        """Test metadata with triggers."""
        meta = SkillMetadata(
            name="code-review",
            description="Review code for best practices",
            scope=SkillScope.USER,
            scope_id="user-123",
            triggers=["review", "check code", "code quality"],
        )

        assert len(meta.triggers) == 3
        assert "review" in meta.triggers

    def test_invalid_name_uppercase(self):
        """Test that uppercase names are rejected."""
        with pytest.raises(ValueError, match="must be lowercase"):
            SkillMetadata(
                name="SQLOptimize",
                description="Test",
                scope=SkillScope.BASE,
                scope_id="system",
            )

    def test_invalid_name_underscore(self):
        """Test that underscores in names are rejected."""
        with pytest.raises(ValueError, match="must be lowercase"):
            SkillMetadata(
                name="sql_optimize",
                description="Test",
                scope=SkillScope.BASE,
                scope_id="system",
            )

    def test_invalid_name_starts_with_number(self):
        """Test that names starting with numbers are rejected."""
        with pytest.raises(ValueError, match="must be lowercase"):
            SkillMetadata(
                name="1st-skill",
                description="Test",
                scope=SkillScope.BASE,
                scope_id="system",
            )

    def test_valid_name_with_numbers(self):
        """Test that names with numbers (not at start) are valid."""
        meta = SkillMetadata(
            name="python3-tips",
            description="Tips for Python 3",
            scope=SkillScope.BASE,
            scope_id="system",
        )
        assert meta.name == "python3-tips"

    def test_description_truncation(self):
        """Test that long descriptions are truncated."""
        long_desc = "x" * 300
        meta = SkillMetadata(
            name="test-skill",
            description=long_desc,
            scope=SkillScope.BASE,
            scope_id="system",
        )

        assert len(meta.description) == 256
        assert meta.description.endswith("...")

    def test_to_dict(self):
        """Test serialization to dictionary."""
        meta = SkillMetadata(
            name="api-design",
            description="Design REST APIs",
            scope=SkillScope.PROJECT,
            scope_id="proj-1",
            triggers=["api", "rest"],
            version="2.0",
        )

        data = meta.to_dict()

        assert data["name"] == "api-design"
        assert data["scope"] == "project"
        assert data["version"] == "2.0"
        assert data["triggers"] == ["api", "rest"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "test-skill",
            "description": "A test skill",
            "scope": "user",
            "scope_id": "user-1",
            "triggers": ["test"],
            "version": "1.5",
        }

        meta = SkillMetadata.from_dict(data)

        assert meta.name == "test-skill"
        assert meta.scope == SkillScope.USER
        assert meta.version == "1.5"

    def test_matches_trigger(self):
        """Test trigger matching."""
        meta = SkillMetadata(
            name="sql-skill",
            description="SQL operations",
            scope=SkillScope.BASE,
            scope_id="system",
            triggers=["optimize query", "slow sql", "database"],
        )

        # Exact match
        assert meta.matches_trigger("optimize query") == "optimize query"

        # Case insensitive
        assert meta.matches_trigger("OPTIMIZE QUERY") == "optimize query"

        # Partial match (trigger in query)
        assert meta.matches_trigger("Help me optimize query performance") == "optimize query"

        # No match
        assert meta.matches_trigger("write python code") is None


class TestSkill:
    """Tests for Skill dataclass."""

    def test_create_skill(self):
        """Test basic skill creation."""
        skill = Skill(
            name="sql-optimize",
            description="Optimize SQL queries for performance",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# SQL Optimization\n\n1. Check indexes...",
        )

        assert skill.name == "sql-optimize"
        assert skill.content.startswith("# SQL Optimization")
        assert skill.prerequisites == []
        assert skill.allowed_tools == []

    def test_skill_with_all_fields(self):
        """Test skill with all optional fields."""
        now = datetime.now()
        skill = Skill(
            name="code-review",
            description="Review code for best practices",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="# Code Review Workflow\n\n...",
            triggers=["review", "check code"],
            prerequisites=["code-style"],
            allowed_tools=["read_file", "grep"],
            metadata={"author": "team-lead"},
            version="2.0",
            created_at=now,
            updated_at=now,
        )

        assert len(skill.triggers) == 2
        assert skill.prerequisites == ["code-style"]
        assert skill.allowed_tools == ["read_file", "grep"]
        assert skill.metadata["author"] == "team-lead"
        assert skill.version == "2.0"

    def test_skill_metadata_property(self):
        """Test extracting metadata from skill."""
        skill = Skill(
            name="test-skill",
            description="A test skill",
            scope=SkillScope.PROJECT,
            scope_id="proj-1",
            content="Content...",
            triggers=["test"],
            version="1.5",
        )

        meta = skill.skill_metadata

        assert isinstance(meta, SkillMetadata)
        assert meta.name == "test-skill"
        assert meta.scope == SkillScope.PROJECT
        assert meta.triggers == ["test"]
        assert meta.version == "1.5"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        skill = Skill(
            name="api-skill",
            description="API operations",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="# API Guide\n\n...",
            triggers=["api"],
            prerequisites=["auth-skill"],
            allowed_tools=["curl"],
        )

        data = skill.to_dict()

        assert data["name"] == "api-skill"
        assert data["scope"] == "user"
        assert data["content"] == "# API Guide\n\n..."
        assert data["prerequisites"] == ["auth-skill"]
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "test-skill",
            "description": "Test",
            "scope": "project",
            "scope_id": "proj-1",
            "content": "Test content",
            "triggers": ["test"],
            "prerequisites": [],
            "allowed_tools": [],
            "metadata": {},
            "version": "1.0",
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T10:00:00",
        }

        skill = Skill.from_dict(data)

        assert skill.name == "test-skill"
        assert skill.scope == SkillScope.PROJECT
        assert skill.content == "Test content"

    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {
            "name": "minimal-skill",
            "description": "Minimal",
            "scope": "base",
            "scope_id": "system",
            "content": "Minimal content",
        }

        skill = Skill.from_dict(data)

        assert skill.name == "minimal-skill"
        assert skill.triggers == []
        assert skill.prerequisites == []
        assert skill.version == "1.0"


class TestSkillMatch:
    """Tests for SkillMatch dataclass."""

    def test_create_match(self):
        """Test basic match creation."""
        meta = SkillMetadata(
            name="test-skill",
            description="Test",
            scope=SkillScope.BASE,
            scope_id="system",
        )

        match = SkillMatch(
            skill=meta,
            confidence=0.85,
            matched_trigger="test",
            match_reason="Trigger 'test' found in query",
        )

        assert match.confidence == 0.85
        assert match.matched_trigger == "test"
        assert match.match_reason == "Trigger 'test' found in query"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        meta = SkillMetadata(
            name="test-skill",
            description="Test",
            scope=SkillScope.BASE,
            scope_id="system",
        )

        match = SkillMatch(skill=meta, confidence=0.9)
        data = match.to_dict()

        assert data["confidence"] == 0.9
        assert data["skill"]["name"] == "test-skill"


class TestSkillsIndex:
    """Tests for SkillsIndex dataclass."""

    def test_empty_index(self):
        """Test empty skills index."""
        index = SkillsIndex(skills=[])

        assert index.total_count == 0
        assert index.format_for_prompt() == ""
        assert index.format_compact() == "No skills available."

    def test_index_with_skills(self):
        """Test index with multiple skills."""
        skills = [
            SkillMetadata(
                name="sql-optimize",
                description="Optimize SQL queries",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=["slow query", "optimize"],
            ),
            SkillMetadata(
                name="code-review",
                description="Review code for best practices",
                scope=SkillScope.USER,
                scope_id="user-1",
                triggers=["review"],
            ),
        ]

        index = SkillsIndex(skills=skills)

        assert index.total_count == 2

    def test_format_for_prompt(self):
        """Test prompt formatting."""
        skills = [
            SkillMetadata(
                name="sql-skill",
                description="Work with databases",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=["sql", "database", "query"],
            ),
        ]

        index = SkillsIndex(skills=skills)
        formatted = index.format_for_prompt()

        assert "## Available Skills" in formatted
        assert "**sql-skill**" in formatted
        assert "Work with databases" in formatted
        assert "sql, database, query" in formatted

    def test_format_compact(self):
        """Test compact formatting."""
        skills = [
            SkillMetadata(
                name="skill-a",
                description="A",
                scope=SkillScope.BASE,
                scope_id="system",
            ),
            SkillMetadata(
                name="skill-b",
                description="B",
                scope=SkillScope.USER,
                scope_id="user-1",
            ),
        ]

        index = SkillsIndex(skills=skills)
        compact = index.format_compact()

        assert "`skill-a`" in compact
        assert "`skill-b`" in compact
        assert ", " in compact


# =============================================================================
# New Dataclass Tests (Feature 1)
# =============================================================================


class TestSkillRelationType:
    """Tests for SkillRelationType enum."""

    def test_all_values(self):
        assert SkillRelationType.SIMILAR_TO.value == "similar_to"
        assert SkillRelationType.BELONG_TO.value == "belong_to"
        assert SkillRelationType.COMPOSE_WITH.value == "compose_with"
        assert SkillRelationType.DEPEND_ON.value == "depend_on"

    def test_from_value(self):
        assert SkillRelationType("similar_to") == SkillRelationType.SIMILAR_TO


class TestSkillRelationship:
    """Tests for SkillRelationship dataclass."""

    def test_create(self):
        rel = SkillRelationship(
            source="skill-a",
            target="skill-b",
            relation_type=SkillRelationType.DEPEND_ON,
            reason="A needs B",
            confidence=0.9,
        )
        assert rel.source == "skill-a"
        assert rel.target == "skill-b"
        assert rel.relation_type == SkillRelationType.DEPEND_ON
        assert rel.confidence == 0.9

    def test_defaults(self):
        rel = SkillRelationship(
            source="a", target="b", relation_type=SkillRelationType.SIMILAR_TO
        )
        assert rel.reason == ""
        assert rel.confidence == 0.8

    def test_to_dict(self):
        rel = SkillRelationship(
            source="a", target="b",
            relation_type=SkillRelationType.COMPOSE_WITH,
            reason="often used together",
            confidence=0.75,
        )
        d = rel.to_dict()
        assert d["source"] == "a"
        assert d["relation_type"] == "compose_with"
        assert d["confidence"] == 0.75

    def test_from_dict(self):
        d = {
            "source": "x",
            "target": "y",
            "relation_type": "belong_to",
            "reason": "child of",
            "confidence": 0.95,
        }
        rel = SkillRelationship.from_dict(d)
        assert rel.relation_type == SkillRelationType.BELONG_TO
        assert rel.confidence == 0.95

    def test_round_trip(self):
        original = SkillRelationship(
            source="a", target="b",
            relation_type=SkillRelationType.DEPEND_ON,
            reason="prereq", confidence=0.88,
        )
        restored = SkillRelationship.from_dict(original.to_dict())
        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.relation_type == original.relation_type
        assert restored.reason == original.reason
        assert restored.confidence == original.confidence


class TestEvaluationLevel:
    """Tests for EvaluationLevel enum."""

    def test_all_values(self):
        assert EvaluationLevel.GOOD.value == "good"
        assert EvaluationLevel.AVERAGE.value == "average"
        assert EvaluationLevel.POOR.value == "poor"

    def test_scores(self):
        assert EVALUATION_LEVEL_SCORES[EvaluationLevel.GOOD] == 1.0
        assert EVALUATION_LEVEL_SCORES[EvaluationLevel.AVERAGE] == 0.5
        assert EVALUATION_LEVEL_SCORES[EvaluationLevel.POOR] == 0.0


class TestSkillEvaluation:
    """Tests for SkillEvaluation dataclass."""

    def _make_evaluation(self, **overrides):
        defaults = dict(
            safety=EvaluationLevel.GOOD,
            safety_reason="safe",
            completeness=EvaluationLevel.GOOD,
            completeness_reason="complete",
            executability=EvaluationLevel.GOOD,
            executability_reason="executable",
            maintainability=EvaluationLevel.GOOD,
            maintainability_reason="maintainable",
            cost_awareness=EvaluationLevel.GOOD,
            cost_awareness_reason="cost-aware",
            evaluated_at=datetime(2025, 1, 1),
            overall_score=1.0,
        )
        defaults.update(overrides)
        return SkillEvaluation(**defaults)

    def test_create(self):
        ev = self._make_evaluation()
        assert ev.safety == EvaluationLevel.GOOD
        assert ev.overall_score == 1.0

    def test_compute_overall_score_all_good(self):
        score = SkillEvaluation.compute_overall_score(
            EvaluationLevel.GOOD, EvaluationLevel.GOOD,
            EvaluationLevel.GOOD, EvaluationLevel.GOOD,
            EvaluationLevel.GOOD,
        )
        assert score == 1.0

    def test_compute_overall_score_all_poor(self):
        score = SkillEvaluation.compute_overall_score(
            EvaluationLevel.POOR, EvaluationLevel.POOR,
            EvaluationLevel.POOR, EvaluationLevel.POOR,
            EvaluationLevel.POOR,
        )
        assert score == 0.0

    def test_compute_overall_score_mixed(self):
        score = SkillEvaluation.compute_overall_score(
            EvaluationLevel.GOOD, EvaluationLevel.AVERAGE,
            EvaluationLevel.POOR, EvaluationLevel.GOOD,
            EvaluationLevel.AVERAGE,
        )
        # 1.0*0.25 + 0.5*0.25 + 0.0*0.20 + 1.0*0.15 + 0.5*0.15
        expected = 0.25 + 0.125 + 0.0 + 0.15 + 0.075
        assert score == round(expected, 4)

    def test_to_dict(self):
        ev = self._make_evaluation()
        d = ev.to_dict()
        assert d["safety"] == "good"
        assert d["overall_score"] == 1.0
        assert "evaluated_at" in d

    def test_from_dict(self):
        d = {
            "safety": "average", "safety_reason": "ok",
            "completeness": "good", "completeness_reason": "ok",
            "executability": "poor", "executability_reason": "ok",
            "maintainability": "good", "maintainability_reason": "ok",
            "cost_awareness": "average", "cost_awareness_reason": "ok",
            "evaluated_at": "2025-06-01T00:00:00",
            "overall_score": 0.6,
        }
        ev = SkillEvaluation.from_dict(d)
        assert ev.safety == EvaluationLevel.AVERAGE
        assert ev.executability == EvaluationLevel.POOR
        assert ev.overall_score == 0.6

    def test_round_trip(self):
        original = self._make_evaluation(
            safety=EvaluationLevel.AVERAGE,
            overall_score=0.75,
        )
        restored = SkillEvaluation.from_dict(original.to_dict())
        assert restored.safety == original.safety
        assert restored.overall_score == original.overall_score
        assert restored.evaluated_at == original.evaluated_at


class TestSkillContent:
    """Tests for SkillContent dataclass."""

    def test_create_minimal(self):
        sc = SkillContent(instructions="Do the thing.")
        assert sc.instructions == "Do the thing."
        assert sc.scripts == {}
        assert sc.references == {}

    def test_create_full(self):
        sc = SkillContent(
            instructions="Step 1",
            scripts={"setup.sh": "#!/bin/bash\necho hi"},
            references={"guide.md": "# Guide\n..."},
        )
        assert "setup.sh" in sc.scripts
        assert "guide.md" in sc.references

    def test_to_dict(self):
        sc = SkillContent(
            instructions="inst",
            scripts={"a.py": "print(1)"},
            references={"b.md": "ref"},
        )
        d = sc.to_dict()
        assert d["instructions"] == "inst"
        assert d["scripts"]["a.py"] == "print(1)"

    def test_from_dict(self):
        d = {
            "instructions": "hello",
            "scripts": {"x.sh": "echo x"},
            "references": {},
        }
        sc = SkillContent.from_dict(d)
        assert sc.instructions == "hello"
        assert sc.scripts == {"x.sh": "echo x"}

    def test_round_trip(self):
        original = SkillContent(
            instructions="workflow",
            scripts={"run.py": "import os"},
            references={"doc.md": "# Doc"},
        )
        restored = SkillContent.from_dict(original.to_dict())
        assert restored.instructions == original.instructions
        assert restored.scripts == original.scripts
        assert restored.references == original.references


class TestSkillMetadataNewFields:
    """Tests for new fields on SkillMetadata."""

    def test_new_fields_defaults(self):
        meta = SkillMetadata(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
        )
        assert meta.category is None
        assert meta.tags == []
        assert meta.when_to_use is None

    def test_new_fields_set(self):
        meta = SkillMetadata(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            category="debugging",
            tags=["python", "testing"],
            when_to_use="When debugging Python tests",
        )
        assert meta.category == "debugging"
        assert meta.tags == ["python", "testing"]
        assert meta.when_to_use == "When debugging Python tests"

    def test_to_dict_includes_new_fields(self):
        meta = SkillMetadata(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            category="deployment", tags=["ci"],
            when_to_use="During CI/CD",
        )
        d = meta.to_dict()
        assert d["category"] == "deployment"
        assert d["tags"] == ["ci"]
        assert d["when_to_use"] == "During CI/CD"

    def test_from_dict_with_new_fields(self):
        d = {
            "name": "test-skill", "description": "Test",
            "scope": "base", "scope_id": "system",
            "category": "testing", "tags": ["unit"],
            "when_to_use": "When writing tests",
        }
        meta = SkillMetadata.from_dict(d)
        assert meta.category == "testing"
        assert meta.tags == ["unit"]

    def test_from_dict_without_new_fields(self):
        d = {
            "name": "test-skill", "description": "Test",
            "scope": "base", "scope_id": "system",
        }
        meta = SkillMetadata.from_dict(d)
        assert meta.category is None
        assert meta.tags == []
        assert meta.when_to_use is None


class TestSkillNewFields:
    """Tests for new fields on Skill."""

    def test_new_fields_defaults(self):
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="content",
        )
        assert skill.category is None
        assert skill.tags == []
        assert skill.when_to_use is None
        assert skill.composed_of == []
        assert skill.evaluation is None
        assert skill.effectiveness is None
        assert skill.structured_content is None

    def test_new_fields_set(self):
        ev = SkillEvaluation(
            safety=EvaluationLevel.GOOD, safety_reason="safe",
            completeness=EvaluationLevel.GOOD, completeness_reason="ok",
            executability=EvaluationLevel.GOOD, executability_reason="ok",
            maintainability=EvaluationLevel.GOOD, maintainability_reason="ok",
            cost_awareness=EvaluationLevel.GOOD, cost_awareness_reason="ok",
            evaluated_at=datetime(2025, 1, 1), overall_score=1.0,
        )
        sc = SkillContent(instructions="Do it")
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="content",
            category="debugging",
            tags=["python"],
            when_to_use="When debugging",
            composed_of=["sub-skill-a", "sub-skill-b"],
            evaluation=ev,
            effectiveness={"usage_count": 5, "success_rate": 0.8},
            structured_content=sc,
        )
        assert skill.category == "debugging"
        assert skill.composed_of == ["sub-skill-a", "sub-skill-b"]
        assert skill.evaluation.overall_score == 1.0
        assert skill.effectiveness["usage_count"] == 5
        assert skill.structured_content.instructions == "Do it"

    def test_skill_metadata_property_includes_new_fields(self):
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="content",
            category="testing",
            tags=["unit", "python"],
            when_to_use="When testing",
        )
        meta = skill.skill_metadata
        assert meta.category == "testing"
        assert meta.tags == ["unit", "python"]
        assert meta.when_to_use == "When testing"

    def test_to_dict_includes_new_fields(self):
        ev = SkillEvaluation(
            safety=EvaluationLevel.GOOD, safety_reason="safe",
            completeness=EvaluationLevel.AVERAGE, completeness_reason="ok",
            executability=EvaluationLevel.GOOD, executability_reason="ok",
            maintainability=EvaluationLevel.POOR, maintainability_reason="ok",
            cost_awareness=EvaluationLevel.GOOD, cost_awareness_reason="ok",
            evaluated_at=datetime(2025, 6, 1), overall_score=0.7,
        )
        sc = SkillContent(instructions="inst", scripts={"a.py": "code"})
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="content",
            category="deployment",
            tags=["ci"],
            when_to_use="During deploy",
            composed_of=["sub-a"],
            evaluation=ev,
            effectiveness={"usage_count": 3},
            structured_content=sc,
        )
        d = skill.to_dict()
        assert d["category"] == "deployment"
        assert d["tags"] == ["ci"]
        assert d["composed_of"] == ["sub-a"]
        assert d["evaluation"]["safety"] == "good"
        assert d["effectiveness"]["usage_count"] == 3
        assert d["structured_content"]["instructions"] == "inst"

    def test_to_dict_none_evaluation(self):
        skill = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.BASE, scope_id="system",
            content="content",
        )
        d = skill.to_dict()
        assert d["evaluation"] is None
        assert d["structured_content"] is None
        assert d["effectiveness"] is None

    def test_from_dict_with_new_fields(self):
        d = {
            "name": "test-skill", "description": "Test",
            "scope": "base", "scope_id": "system",
            "content": "content",
            "category": "testing",
            "tags": ["unit"],
            "when_to_use": "When testing",
            "composed_of": ["sub-a"],
            "evaluation": {
                "safety": "good", "safety_reason": "safe",
                "completeness": "good", "completeness_reason": "ok",
                "executability": "good", "executability_reason": "ok",
                "maintainability": "good", "maintainability_reason": "ok",
                "cost_awareness": "good", "cost_awareness_reason": "ok",
                "evaluated_at": "2025-01-01T00:00:00",
                "overall_score": 1.0,
            },
            "effectiveness": {"usage_count": 10},
            "structured_content": {
                "instructions": "Do it",
                "scripts": {"run.py": "print(1)"},
                "references": {},
            },
        }
        skill = Skill.from_dict(d)
        assert skill.category == "testing"
        assert skill.composed_of == ["sub-a"]
        assert skill.evaluation is not None
        assert skill.evaluation.safety == EvaluationLevel.GOOD
        assert skill.effectiveness["usage_count"] == 10
        assert skill.structured_content.instructions == "Do it"

    def test_from_dict_without_new_fields(self):
        d = {
            "name": "test-skill", "description": "Test",
            "scope": "base", "scope_id": "system",
            "content": "content",
        }
        skill = Skill.from_dict(d)
        assert skill.category is None
        assert skill.tags == []
        assert skill.evaluation is None
        assert skill.effectiveness is None
        assert skill.structured_content is None

    def test_round_trip_with_all_new_fields(self):
        ev = SkillEvaluation(
            safety=EvaluationLevel.AVERAGE, safety_reason="ok",
            completeness=EvaluationLevel.GOOD, completeness_reason="ok",
            executability=EvaluationLevel.GOOD, executability_reason="ok",
            maintainability=EvaluationLevel.POOR, maintainability_reason="ok",
            cost_awareness=EvaluationLevel.AVERAGE, cost_awareness_reason="ok",
            evaluated_at=datetime(2025, 3, 15, 12, 0, 0),
            overall_score=0.6,
        )
        sc = SkillContent(
            instructions="workflow",
            scripts={"setup.sh": "echo setup"},
            references={"guide.md": "# Guide"},
        )
        original = Skill(
            name="test-skill", description="Test",
            scope=SkillScope.USER, scope_id="user-1",
            content="content",
            category="debugging",
            tags=["python", "testing"],
            when_to_use="When debugging",
            composed_of=["sub-a", "sub-b"],
            evaluation=ev,
            effectiveness={"usage_count": 7, "success_rate": 0.85},
            structured_content=sc,
        )
        restored = Skill.from_dict(original.to_dict())
        assert restored.category == original.category
        assert restored.tags == original.tags
        assert restored.when_to_use == original.when_to_use
        assert restored.composed_of == original.composed_of
        assert restored.evaluation.safety == original.evaluation.safety
        assert restored.evaluation.overall_score == original.evaluation.overall_score
        assert restored.effectiveness == original.effectiveness
        assert restored.structured_content.instructions == original.structured_content.instructions
        assert restored.structured_content.scripts == original.structured_content.scripts


# =============================================================================
# Config Tests (Feature 1)
# =============================================================================


class TestSkillConfigExtensions:
    """Tests for new config classes."""

    def test_evaluation_config_defaults(self):
        cfg = SkillEvaluationConfig()
        assert cfg.enabled is False
        assert cfg.auto_evaluate_on_register is False
        assert cfg.model is None

    def test_relationship_config_defaults(self):
        cfg = SkillRelationshipConfig()
        assert cfg.enabled is True
        assert cfg.auto_analyze is False
        assert cfg.auto_resolve_dependencies is True

    def test_generation_config_defaults(self):
        cfg = SkillGenerationConfig()
        assert cfg.enabled is True
        assert cfg.auto_generate_from_sessions is False
        assert cfg.min_session_events == 5
        assert cfg.model is None

    def test_effectiveness_config_defaults(self):
        cfg = SkillEffectivenessConfig()
        assert cfg.enabled is True
        assert cfg.track_usage is True
        assert cfg.weight_in_ranking == 0.3

    def test_skills_config_includes_sub_configs(self):
        cfg = SkillsConfig()
        assert isinstance(cfg.evaluation, SkillEvaluationConfig)
        assert isinstance(cfg.relationships, SkillRelationshipConfig)
        assert isinstance(cfg.generation, SkillGenerationConfig)
        assert isinstance(cfg.effectiveness, SkillEffectivenessConfig)

    def test_skills_config_from_dict(self):
        d = {
            "enabled": True,
            "evaluation": {"enabled": True, "auto_evaluate_on_register": True},
            "relationships": {"enabled": True, "auto_analyze": True},
            "generation": {"enabled": True, "min_session_events": 10},
            "effectiveness": {"enabled": True, "weight_in_ranking": 0.5},
        }
        cfg = SkillsConfig.model_validate(d)
        assert cfg.evaluation.enabled is True
        assert cfg.evaluation.auto_evaluate_on_register is True
        assert cfg.relationships.auto_analyze is True
        assert cfg.generation.min_session_events == 10
        assert cfg.effectiveness.weight_in_ranking == 0.5


# =============================================================================
# In-Memory Store Tests
# =============================================================================


class TestInMemorySkillStore:
    """Tests for InMemorySkillStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemorySkillStore()

    @pytest.fixture
    def sample_skill(self):
        """Create a sample skill for testing."""
        return Skill(
            name="sql-optimize",
            description="Optimize SQL queries for performance",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# SQL Optimization\n\n1. Check indexes\n2. Analyze queries",
            triggers=["slow query", "optimize sql", "database performance"],
        )

    async def test_initialize(self, store):
        """Test store initialization."""
        await store.initialize()
        # Should not raise

    async def test_save_and_get(self, store, sample_skill):
        """Test saving and retrieving a skill."""
        await store.save(sample_skill)

        retrieved = await store.get("sql-optimize", SkillScope.BASE, "system")

        assert retrieved is not None
        assert retrieved.name == "sql-optimize"
        assert retrieved.content.startswith("# SQL Optimization")

    async def test_get_nonexistent(self, store):
        """Test retrieving a non-existent skill."""
        result = await store.get("nonexistent", SkillScope.BASE, "system")
        assert result is None

    async def test_get_metadata(self, store, sample_skill):
        """Test retrieving only metadata."""
        await store.save(sample_skill)

        meta = await store.get_metadata("sql-optimize", SkillScope.BASE, "system")

        assert meta is not None
        assert meta.name == "sql-optimize"
        assert meta.description == "Optimize SQL queries for performance"

    async def test_save_updates_existing(self, store, sample_skill):
        """Test that save updates if name exists."""
        await store.save(sample_skill)

        # Update the same skill
        updated = Skill(
            name="sql-optimize",
            description="New description",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Updated content",
        )
        await store.save(updated)

        # Should have only one skill
        count = await store.count()
        assert count == 1

        # Should have the updated content
        retrieved = await store.get("sql-optimize", SkillScope.BASE, "system")
        assert retrieved.description == "New description"

    async def test_list_metadata(self, store):
        """Test listing skill metadata for a scope."""
        await store.save(Skill(
            name="skill-a",
            description="A",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content A",
        ))
        await store.save(Skill(
            name="skill-b",
            description="B",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content B",
        ))
        await store.save(Skill(
            name="skill-c",
            description="C",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="Content C",
        ))

        # List base skills
        base_skills = await store.list_metadata(SkillScope.BASE, "system")
        assert len(base_skills) == 2
        assert all(s.scope == SkillScope.BASE for s in base_skills)

        # List user skills
        user_skills = await store.list_metadata(SkillScope.USER, "user-1")
        assert len(user_skills) == 1

    async def test_list_all_metadata_with_layering(self, store):
        """Test listing all skills with scope layering."""
        # Base skill
        await store.save(Skill(
            name="shared-skill",
            description="Base version",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Base content",
        ))
        # User override
        await store.save(Skill(
            name="shared-skill",
            description="User version",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="User content",
        ))
        # User-only skill
        await store.save(Skill(
            name="user-only",
            description="User only",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="User only content",
        ))

        # List with just base
        skills = await store.list_all_metadata()
        assert len(skills) == 1
        assert skills[0].description == "Base version"

        # List with user (should override base)
        skills = await store.list_all_metadata(user_id="user-1")
        assert len(skills) == 2
        # shared-skill should have user version
        shared = next(s for s in skills if s.name == "shared-skill")
        assert shared.description == "User version"

    async def test_list_all_metadata_project_overrides(self, store):
        """Test that project skills override user and base."""
        await store.save(Skill(
            name="shared-skill",
            description="Base version",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
        ))
        await store.save(Skill(
            name="shared-skill",
            description="User version",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="Content",
        ))
        await store.save(Skill(
            name="shared-skill",
            description="Project version",
            scope=SkillScope.PROJECT,
            scope_id="proj-1",
            content="Content",
        ))

        skills = await store.list_all_metadata(
            user_id="user-1",
            project_id="proj-1"
        )
        assert len(skills) == 1
        assert skills[0].description == "Project version"

    async def test_delete(self, store, sample_skill):
        """Test deleting a skill."""
        await store.save(sample_skill)

        deleted = await store.delete("sql-optimize", SkillScope.BASE, "system")
        assert deleted is True

        # Should not exist
        retrieved = await store.get("sql-optimize", SkillScope.BASE, "system")
        assert retrieved is None

    async def test_delete_nonexistent(self, store):
        """Test deleting a non-existent skill."""
        deleted = await store.delete("nonexistent", SkillScope.BASE, "system")
        assert deleted is False

    async def test_search_by_trigger(self, store):
        """Test searching skills by trigger."""
        await store.save(Skill(
            name="sql-optimize",
            description="Optimize SQL",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
            triggers=["slow query", "optimize sql"],
        ))
        await store.save(Skill(
            name="code-review",
            description="Review code",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
            triggers=["review", "check code"],
        ))

        # Search with matching trigger (trigger must be substring of query)
        matches = await store.search_by_trigger("I have a slow query issue")
        assert len(matches) == 1
        assert matches[0].skill.name == "sql-optimize"
        assert matches[0].matched_trigger == "slow query"

        # Search with no match
        matches = await store.search_by_trigger("What time is it?")
        assert len(matches) == 0

    async def test_search_by_trigger_with_layering(self, store):
        """Test that trigger search respects scope layering."""
        # Base skill
        await store.save(Skill(
            name="shared-skill",
            description="Base",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
            triggers=["trigger"],
        ))
        # User override (different triggers)
        await store.save(Skill(
            name="shared-skill",
            description="User",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="Content",
            triggers=["custom trigger"],
        ))

        # Without user_id, should use base
        matches = await store.search_by_trigger("I need trigger help")
        assert len(matches) == 1
        assert matches[0].skill.description == "Base"

        # With user_id, should use user version
        matches = await store.search_by_trigger(
            "I need custom trigger help",
            user_id="user-1"
        )
        assert len(matches) == 1
        assert matches[0].skill.description == "User"

    async def test_count(self, store):
        """Test counting skills."""
        await store.save(Skill(
            name="skill-a",
            description="A",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
        ))
        await store.save(Skill(
            name="skill-b",
            description="B",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="Content",
        ))

        # Total count
        assert await store.count() == 2

        # By scope
        assert await store.count(scope=SkillScope.BASE) == 1
        assert await store.count(scope=SkillScope.USER) == 1
        assert await store.count(scope=SkillScope.PROJECT) == 0

    async def test_clear_all(self, store):
        """Test clearing all skills."""
        await store.save(Skill(
            name="skill-a",
            description="A",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
        ))
        await store.save(Skill(
            name="skill-b",
            description="B",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="Content",
        ))

        deleted_count = await store.clear()
        assert deleted_count == 2
        assert await store.count() == 0

    async def test_clear_by_scope(self, store):
        """Test clearing skills by scope."""
        await store.save(Skill(
            name="skill-a",
            description="A",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Content",
        ))
        await store.save(Skill(
            name="skill-b",
            description="B",
            scope=SkillScope.USER,
            scope_id="user-1",
            content="Content",
        ))

        deleted_count = await store.clear(scope=SkillScope.USER)
        assert deleted_count == 1
        assert await store.count() == 1
        assert await store.count(scope=SkillScope.BASE) == 1

    # --- New protocol methods (Feature 1) ---

    async def test_save_relationships(self, store):
        """Test saving relationships."""
        rels = [
            SkillRelationship(
                source="skill-a", target="skill-b",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
            SkillRelationship(
                source="skill-a", target="skill-c",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ]
        count = await store.save_relationships(rels)
        assert count == 2

    async def test_save_relationships_deduplication(self, store):
        """Test that duplicate relationships are not saved twice."""
        rel = SkillRelationship(
            source="a", target="b",
            relation_type=SkillRelationType.SIMILAR_TO,
        )
        await store.save_relationships([rel])
        count = await store.save_relationships([rel])
        assert count == 0
        all_rels = await store.get_all_relationships()
        assert len(all_rels) == 1

    async def test_get_relationships(self, store):
        """Test getting relationships for a skill."""
        rels = [
            SkillRelationship(
                source="skill-a", target="skill-b",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
            SkillRelationship(
                source="skill-c", target="skill-a",
                relation_type=SkillRelationType.SIMILAR_TO,
            ),
            SkillRelationship(
                source="skill-d", target="skill-e",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ]
        await store.save_relationships(rels)
        result = await store.get_relationships("skill-a")
        assert len(result) == 2

    async def test_get_all_relationships(self, store):
        """Test getting all relationships."""
        rels = [
            SkillRelationship(
                source="a", target="b",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
            SkillRelationship(
                source="c", target="d",
                relation_type=SkillRelationType.SIMILAR_TO,
            ),
        ]
        await store.save_relationships(rels)
        all_rels = await store.get_all_relationships()
        assert len(all_rels) == 2

    async def test_search_by_category(self, store):
        """Test searching skills by category."""
        await store.save(Skill(
            name="skill-a", description="A",
            scope=SkillScope.BASE, scope_id="system",
            content="Content", category="debugging",
        ))
        await store.save(Skill(
            name="skill-b", description="B",
            scope=SkillScope.BASE, scope_id="system",
            content="Content", category="testing",
        ))
        await store.save(Skill(
            name="skill-c", description="C",
            scope=SkillScope.BASE, scope_id="system",
            content="Content", category="debugging",
        ))
        result = await store.search_by_category("debugging")
        assert len(result) == 2
        names = {m.name for m in result}
        assert names == {"skill-a", "skill-c"}

    async def test_search_by_category_no_match(self, store):
        """Test searching by category with no matches."""
        await store.save(Skill(
            name="skill-a", description="A",
            scope=SkillScope.BASE, scope_id="system",
            content="Content", category="debugging",
        ))
        result = await store.search_by_category("deployment")
        assert len(result) == 0

    async def test_update_effectiveness(self, store):
        """Test updating effectiveness metrics."""
        await store.save(Skill(
            name="skill-a", description="A",
            scope=SkillScope.BASE, scope_id="system",
            content="Content",
        ))
        updated = await store.update_effectiveness(
            "skill-a", SkillScope.BASE, "system",
            {"usage_count": 5, "success_rate": 0.8},
        )
        assert updated is True
        skill = await store.get("skill-a", SkillScope.BASE, "system")
        assert skill.effectiveness["usage_count"] == 5
        assert skill.effectiveness["success_rate"] == 0.8

    async def test_update_effectiveness_incremental(self, store):
        """Test that update_effectiveness merges metrics."""
        await store.save(Skill(
            name="skill-a", description="A",
            scope=SkillScope.BASE, scope_id="system",
            content="Content",
        ))
        await store.update_effectiveness(
            "skill-a", SkillScope.BASE, "system",
            {"usage_count": 5},
        )
        await store.update_effectiveness(
            "skill-a", SkillScope.BASE, "system",
            {"success_rate": 0.9},
        )
        skill = await store.get("skill-a", SkillScope.BASE, "system")
        assert skill.effectiveness["usage_count"] == 5
        assert skill.effectiveness["success_rate"] == 0.9

    async def test_update_effectiveness_nonexistent(self, store):
        """Test updating effectiveness for a non-existent skill."""
        updated = await store.update_effectiveness(
            "nonexistent", SkillScope.BASE, "system",
            {"usage_count": 1},
        )
        assert updated is False


# =============================================================================
# Skill Service Tests
# =============================================================================


class TestSkillService:
    """Tests for SkillService."""

    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        store = InMemorySkillStore()
        return SkillService(store)

    async def test_initialize(self, service):
        """Test service initialization."""
        await service.initialize()
        # Should not raise

    async def test_register_base_skill(self, service):
        """Test registering a base skill."""
        skill = await service.register_base_skill(
            name="sql-optimize",
            description="Optimize SQL queries",
            content="# SQL Optimization\n\n...",
            triggers=["slow query", "optimize sql"],
        )

        assert skill.name == "sql-optimize"
        assert skill.scope == SkillScope.BASE
        assert skill.scope_id == "system"

    async def test_register_user_skill(self, service):
        """Test registering a user skill."""
        skill = await service.register_user_skill(
            user_id="user-1",
            name="my-workflow",
            description="My custom workflow",
            content="# Custom Workflow\n\n...",
        )

        assert skill.scope == SkillScope.USER
        assert skill.scope_id == "user-1"

    async def test_register_project_skill(self, service):
        """Test registering a project skill."""
        skill = await service.register_project_skill(
            project_id="proj-1",
            name="deploy-script",
            description="Deployment workflow",
            content="# Deploy\n\n...",
            triggers=["deploy", "release"],
        )

        assert skill.scope == SkillScope.PROJECT
        assert skill.scope_id == "proj-1"

    async def test_get_available_skills_base_only(self, service):
        """Test getting skills with only base scope."""
        await service.register_base_skill(
            name="skill-a", description="A", content="A content"
        )
        await service.register_base_skill(
            name="skill-b", description="B", content="B content"
        )

        skills = await service.get_available_skills()

        assert len(skills) == 2
        assert all(s.scope == SkillScope.BASE for s in skills)

    async def test_get_available_skills_with_layering(self, service):
        """Test that skill layering works correctly."""
        # Base skill
        await service.register_base_skill(
            name="shared-skill",
            description="Base version",
            content="Base content",
        )
        # User override
        await service.register_user_skill(
            user_id="user-1",
            name="shared-skill",
            description="User version",
            content="User content",
        )

        # Without user_id
        skills = await service.get_available_skills()
        assert len(skills) == 1
        assert skills[0].description == "Base version"

        # With user_id
        skills = await service.get_available_skills(user_id="user-1")
        assert len(skills) == 1
        assert skills[0].description == "User version"

    async def test_get_skills_index(self, service):
        """Test getting skills index."""
        await service.register_base_skill(
            name="skill-a", description="A", content="A"
        )
        await service.register_user_skill(
            user_id="user-1", name="skill-b", description="B", content="B"
        )

        index = await service.get_skills_index(user_id="user-1")

        assert index.total_count == 2
        assert SkillScope.BASE in index.scope_counts
        assert SkillScope.USER in index.scope_counts

    async def test_load_skill_content(self, service):
        """Test loading full skill content."""
        await service.register_base_skill(
            name="test-skill",
            description="Test",
            content="# Full Content\n\nStep 1...",
        )

        skill = await service.load_skill_content("test-skill")

        assert skill is not None
        assert skill.content == "# Full Content\n\nStep 1..."

    async def test_load_skill_content_with_layering(self, service):
        """Test that load_skill_content respects layering."""
        await service.register_base_skill(
            name="shared-skill",
            description="Base",
            content="Base content",
        )
        await service.register_project_skill(
            project_id="proj-1",
            name="shared-skill",
            description="Project",
            content="Project content",
        )

        # Without project
        skill = await service.load_skill_content("shared-skill")
        assert skill.content == "Base content"

        # With project
        skill = await service.load_skill_content(
            "shared-skill", project_id="proj-1"
        )
        assert skill.content == "Project content"

    async def test_match_skills(self, service):
        """Test matching skills to queries."""
        await service.register_base_skill(
            name="sql-optimize",
            description="Optimize SQL",
            content="Content",
            triggers=["slow query", "optimize sql"],
        )

        matches = await service.match_skills(
            "I have a slow query problem"
        )

        assert len(matches) == 1
        assert matches[0].skill.name == "sql-optimize"

    async def test_match_skills_threshold(self, service):
        """Test that threshold filters low-confidence matches."""
        await service.register_base_skill(
            name="test-skill",
            description="Test",
            content="Content",
            triggers=["exact trigger"],
        )

        # High threshold - should not match partial
        matches = await service.match_skills(
            "something unrelated",
            threshold=0.9
        )
        assert len(matches) == 0

    async def test_format_skills_index(self, service):
        """Test formatting skills for prompt."""
        skills = [
            SkillMetadata(
                name="skill-a",
                description="Do A",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=["trigger-a"],
            )
        ]

        formatted = service.format_skills_index(skills)

        assert "## Available Skills" in formatted
        assert "**skill-a**" in formatted
        assert "Do A" in formatted

    async def test_format_skill_workflow(self, service):
        """Test formatting a skill workflow."""
        skill = Skill(
            name="test-skill",
            description="A test skill",
            scope=SkillScope.BASE,
            scope_id="system",
            content="1. Step one\n2. Step two",
            prerequisites=["other-skill"],
            allowed_tools=["read_file", "write"],
        )

        formatted = service.format_skill_workflow(skill)

        assert "## Skill: test-skill" in formatted
        assert "**Description:** A test skill" in formatted
        assert "**Prerequisites:** other-skill" in formatted
        assert "**Allowed Tools:** read_file, write" in formatted
        assert "### Workflow" in formatted
        assert "1. Step one" in formatted

    async def test_cache_invalidation(self, service):
        """Test that cache is invalidated on registration."""
        service.enable_cache(True)

        await service.register_base_skill(
            name="skill-a", description="A", content="A"
        )

        # Get skills (populates cache)
        skills = await service.get_available_skills()
        assert len(skills) == 1

        # Register another skill (should invalidate cache)
        await service.register_base_skill(
            name="skill-b", description="B", content="B"
        )

        # Should now have 2 skills
        skills = await service.get_available_skills()
        assert len(skills) == 2

    async def test_delete_skill(self, service):
        """Test deleting a skill."""
        await service.register_base_skill(
            name="to-delete", description="Delete me", content="Content"
        )

        deleted = await service.delete_skill(
            "to-delete", SkillScope.BASE, "system"
        )

        assert deleted is True
        skill = await service.load_skill_content("to-delete")
        assert skill is None


# =============================================================================
# Skill Matcher Tests
# =============================================================================


class TestSkillMatcher:
    """Tests for SkillMatcher."""

    @pytest.fixture
    def matcher(self):
        """Create a matcher without embedding provider."""
        return SkillMatcher()

    @pytest.fixture
    def sample_skills(self):
        """Create sample skills for testing."""
        return [
            SkillMetadata(
                name="sql-optimize",
                description="Optimize SQL queries",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=["slow query", "optimize sql", "database performance"],
            ),
            SkillMetadata(
                name="code-review",
                description="Review code for best practices",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=["review code", "check code", "code quality"],
            ),
        ]

    async def test_match_exact_trigger(self, matcher, sample_skills):
        """Test matching with exact trigger substring."""
        matches = await matcher.match(
            "I have a slow query issue",
            sample_skills,
            threshold=0.5
        )

        assert len(matches) == 1
        assert matches[0].skill.name == "sql-optimize"
        assert matches[0].matched_trigger == "slow query"
        assert matches[0].confidence >= 0.5

    async def test_match_multiple_skills(self, matcher, sample_skills):
        """Test matching multiple skills."""
        matches = await matcher.match(
            "review my slow query code",
            sample_skills,
            threshold=0.5
        )

        # Both should match
        assert len(matches) == 2

    async def test_match_word_overlap(self, matcher, sample_skills):
        """Test matching via word overlap."""
        matches = await matcher.match(
            "please review the code",
            sample_skills,
            threshold=0.5
        )

        # Should match code-review via word overlap
        matched_names = [m.skill.name for m in matches]
        assert "code-review" in matched_names

    async def test_no_match(self, matcher, sample_skills):
        """Test when nothing matches."""
        matches = await matcher.match(
            "what is the weather today",
            sample_skills,
            threshold=0.5
        )

        assert len(matches) == 0

    async def test_threshold_filtering(self, matcher, sample_skills):
        """Test that threshold filters low-confidence matches."""
        # With low threshold
        matches_low = await matcher.match(
            "database",  # Partial word match
            sample_skills,
            threshold=0.3
        )

        # With high threshold
        matches_high = await matcher.match(
            "database",
            sample_skills,
            threshold=0.9
        )

        # Low threshold might have matches, high won't
        assert len(matches_high) <= len(matches_low)

    async def test_empty_skills(self, matcher):
        """Test matching against empty skills list."""
        matches = await matcher.match("any query", [], threshold=0.5)
        assert matches == []


class TestRegexSkillMatcher:
    """Tests for RegexSkillMatcher."""

    @pytest.fixture
    def matcher(self):
        """Create a regex matcher."""
        return RegexSkillMatcher()

    async def test_regex_trigger(self, matcher):
        """Test regex pattern matching."""
        skills = [
            SkillMetadata(
                name="error-handler",
                description="Handle errors",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=[r"error:\s*\w+", r"exception\s+in"],
            ),
        ]

        matches = await matcher.match(
            "I got error: ConnectionTimeout",
            skills,
            threshold=0.5
        )

        assert len(matches) == 1
        assert matches[0].skill.name == "error-handler"

    async def test_non_regex_trigger(self, matcher):
        """Test that non-regex triggers still work."""
        skills = [
            SkillMetadata(
                name="simple-skill",
                description="Simple",
                scope=SkillScope.BASE,
                scope_id="system",
                triggers=["help me"],
            ),
        ]

        matches = await matcher.match(
            "please help me with this",
            skills,
            threshold=0.5
        )

        assert len(matches) == 1


# =============================================================================
# Skills Middleware Tests
# =============================================================================


class TestSkillsMiddleware:
    """Tests for SkillsMiddleware."""

    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        store = InMemorySkillStore()
        return SkillService(store)

    @pytest.fixture
    def middleware(self, service):
        """Create middleware with the service."""
        return SkillsMiddleware(
            skill_service=service,
            user_id="user-1",
            project_id="proj-1",
        )

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        """Simple next function for testing."""
        return context

    async def test_middleware_name(self, middleware):
        """Test middleware has correct name."""
        assert middleware.name == "skills"

    async def test_middleware_enabled(self, middleware):
        """Test middleware enabled property."""
        assert middleware.enabled is True
        middleware.enabled = False
        assert middleware.enabled is False

    async def test_inject_skills_index(self, service, middleware):
        """Test that skills index is injected as a context section."""
        # Register some skills
        await service.register_base_skill(
            name="sql-optimize",
            description="Optimize SQL queries",
            content="# SQL Optimization\n\n...",
            triggers=["slow query"],
        )
        await service.register_base_skill(
            name="code-review",
            description="Review code",
            content="# Code Review\n\n...",
            triggers=["review"],
        )

        context = MiddlewareContext(user_input="Hello")

        result = await middleware._do_process(context, self._next_fn)

        # processed_input should remain clean
        assert result.processed_input == "Hello"
        # Skills should be in context_sections
        section_names = [s.name for s in result.context_sections]
        assert "skills_index" in section_names
        skills_section = next(s for s in result.context_sections if s.name == "skills_index")
        assert "sql-optimize" in skills_section.content
        assert "code-review" in skills_section.content
        assert result.has_flag("skills_index_injected")

    async def test_auto_activate_matching_skill(self, service, middleware):
        """Test that matching skills are auto-activated as context sections."""
        await service.register_base_skill(
            name="sql-optimize",
            description="Optimize SQL queries",
            content="# Step 1: Check indexes\n# Step 2: Analyze query",
            triggers=["slow query", "optimize sql"],
        )

        context = MiddlewareContext(
            user_input="I have a slow query problem"
        )

        result = await middleware._do_process(context, self._next_fn)

        # processed_input should remain clean
        assert result.processed_input == "I have a slow query problem"
        # Skill should be auto-activated as a section
        assert result.has_flag("skills_auto_activated")
        section_names = [s.name for s in result.context_sections]
        assert "activated_skills" in section_names
        activated_section = next(s for s in result.context_sections if s.name == "activated_skills")
        assert "Step 1: Check indexes" in activated_section.content

    async def test_no_skills(self, service, middleware):
        """Test when no skills exist."""
        context = MiddlewareContext(user_input="Hello")

        result = await middleware._do_process(context, self._next_fn)

        # Should still have original input
        assert "Hello" in result.processed_input
        assert not result.has_flag("skills_index_injected")

    async def test_disable_auto_activate(self, service):
        """Test disabling auto-activation."""
        middleware = SkillsMiddleware(
            skill_service=service,
            auto_activate=False,
        )

        await service.register_base_skill(
            name="sql-optimize",
            description="Optimize SQL",
            content="Content",
            triggers=["slow query"],
        )

        context = MiddlewareContext(
            user_input="I have a slow query"
        )

        result = await middleware._do_process(context, self._next_fn)

        # Should have index but NOT activated skills
        assert result.has_flag("skills_index_injected")
        assert not result.has_flag("skills_auto_activated")
        section_names = [s.name for s in result.context_sections]
        assert "activated_skills" not in section_names

    async def test_max_auto_skills_limit(self, service):
        """Test that max_auto_skills limits activated skills."""
        middleware = SkillsMiddleware(
            skill_service=service,
            max_auto_skills=1,
            confidence_threshold=0.5,
        )

        await service.register_base_skill(
            name="skill-a",
            description="A",
            content="Content A",
            triggers=["test trigger"],
        )
        await service.register_base_skill(
            name="skill-b",
            description="B",
            content="Content B",
            triggers=["test trigger"],
        )

        context = MiddlewareContext(
            user_input="Run test trigger now"
        )

        result = await middleware._do_process(context, self._next_fn)

        # Should only have one skill activated
        # modifications is a dict where key is middleware name, value is list of modifications
        modifications_list = result.modifications.get("skills", [])
        if modifications_list:
            # Each modification is a dict with 'activated_skills' key
            for mod in modifications_list:
                if isinstance(mod, dict) and "activated_skills" in mod:
                    activated = mod.get("activated_skills", [])
                    assert len(activated) <= 1

    async def test_disabled_middleware(self, service, middleware):
        """Test that disabled middleware is skipped."""
        await service.register_base_skill(
            name="test-skill",
            description="Test",
            content="Content",
        )

        middleware.enabled = False
        context = MiddlewareContext(user_input="Hello")

        result = await middleware.process(context, self._next_fn)

        # Skills should NOT be injected when disabled
        assert len(result.context_sections) == 0


class TestSkillRequestMiddleware:
    """Tests for SkillRequestMiddleware."""

    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        store = InMemorySkillStore()
        return SkillService(store)

    @pytest.fixture
    def middleware(self, service):
        """Create middleware with the service."""
        return SkillRequestMiddleware(
            skill_service=service,
        )

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        """Simple next function for testing."""
        return context

    async def test_middleware_name(self, middleware):
        """Test middleware has correct name."""
        assert middleware.name == "skill_request"

    async def test_explicit_skill_request(self, service, middleware):
        """Test handling explicit skill request."""
        await service.register_base_skill(
            name="deploy-script",
            description="Deployment workflow",
            content="# Deploy Steps\n\n1. Build\n2. Test\n3. Deploy",
        )

        context = MiddlewareContext(
            user_input="use skill deploy-script for my project"
        )

        result = await middleware._do_process(context, self._next_fn)

        assert result.has_flag("skill_requested")
        assert result.get_metadata("requested_skill") == "deploy-script"
        # processed_input should remain clean
        assert result.processed_input == "use skill deploy-script for my project"
        # Skill should be in context_sections
        section_names = [s.name for s in result.context_sections]
        assert "requested_skill" in section_names
        skill_section = next(s for s in result.context_sections if s.name == "requested_skill")
        assert "## Requested Skill: deploy-script" in skill_section.content
        assert "Deploy Steps" in skill_section.content

    async def test_skill_not_found(self, service, middleware):
        """Test when requested skill doesn't exist."""
        context = MiddlewareContext(
            user_input="use skill nonexistent-skill"
        )

        result = await middleware._do_process(context, self._next_fn)

        assert not result.has_flag("skill_requested")
        assert result.get_metadata("skill_not_found") == "nonexistent-skill"

    async def test_no_skill_request(self, service, middleware):
        """Test when input doesn't request a skill."""
        context = MiddlewareContext(
            user_input="Just a normal request"
        )

        result = await middleware._do_process(context, self._next_fn)

        assert not result.has_flag("skill_requested")
        assert result.processed_input == "Just a normal request"

    async def test_various_request_patterns(self, service, middleware):
        """Test different skill request patterns."""
        await service.register_base_skill(
            name="test-skill",
            description="Test",
            content="Test content",
        )

        patterns = [
            "activate skill test-skill",
            "run skill test-skill",
            "execute skill test-skill",
        ]

        for pattern in patterns:
            context = MiddlewareContext(user_input=pattern)
            result = await middleware._do_process(context, self._next_fn)
            assert result.has_flag("skill_requested"), f"Failed for: {pattern}"


# =========================================================================
# Phase 4: Enhanced Discovery and Ranking Tests
# =========================================================================


class TestSkillMatcherCompositeScoring:
    """Tests for SkillMatcher composite scoring."""

    def _make_metadata(
        self,
        name,
        triggers=None,
        category=None,
        tags=None,
        evaluation=None,
        effectiveness=None,
    ):
        """Helper to create SkillMetadata with optional enrichment data."""
        meta = {}
        if evaluation is not None:
            meta["evaluation"] = evaluation
        if effectiveness is not None:
            meta["effectiveness"] = effectiveness
        return SkillMetadata(
            name=name,
            description=f"Skill {name}",
            scope=SkillScope.BASE,
            scope_id="system",
            triggers=triggers or [],
            category=category,
            tags=tags or [],
        ), meta

    def _make_skill_meta(self, name, triggers=None, metadata=None,
                         category=None, tags=None):
        """Create a SkillMetadata with metadata dict attached."""
        sm = SkillMetadata(
            name=name,
            description=f"Skill {name}",
            scope=SkillScope.BASE,
            scope_id="system",
            triggers=triggers or [],
            category=category,
            tags=tags or [],
        )
        if metadata:
            # Attach metadata dict as an attribute for the matcher to read
            object.__setattr__(sm, "metadata", metadata)
        return sm

    async def test_composite_scoring_with_all_factors(self):
        """Test that composite scoring combines all 5 factors."""
        matcher = SkillMatcher()

        skill = self._make_skill_meta(
            "deploy-app",
            triggers=["deploy"],
            metadata={
                "evaluation": {"overall_score": 0.9},
                "effectiveness": {"success_rate": 0.8},
            },
        )

        rel = SkillRelationship(
            source="deploy-app",
            target="active-skill",
            relation_type=SkillRelationType.COMPOSE_WITH,
        )

        matches = await matcher.match(
            query="deploy",
            available_skills=[skill],
            threshold=0.0,
            active_skill_names=["active-skill"],
            relationships=[rel],
        )

        assert len(matches) == 1
        m = matches[0]
        assert m.confidence > 0
        assert "trigger=" in m.match_reason
        assert "eval=" in m.match_reason
        assert "eff=" in m.match_reason
        assert "rel_boost=" in m.match_reason

    async def test_higher_evaluation_ranks_higher(self):
        """Skills with higher evaluation scores rank above those with lower."""
        matcher = SkillMatcher()

        high_eval = self._make_skill_meta(
            "skill-high",
            triggers=["test"],
            metadata={"evaluation": {"overall_score": 0.95}},
        )
        low_eval = self._make_skill_meta(
            "skill-low",
            triggers=["test"],
            metadata={"evaluation": {"overall_score": 0.2}},
        )

        matches = await matcher.match(
            query="test",
            available_skills=[low_eval, high_eval],
            threshold=0.0,
        )

        assert len(matches) == 2
        assert matches[0].skill.name == "skill-high"
        assert matches[1].skill.name == "skill-low"

    async def test_higher_effectiveness_ranks_higher(self):
        """Skills with higher effectiveness rank above those with lower."""
        matcher = SkillMatcher()

        high_eff = self._make_skill_meta(
            "skill-high-eff",
            triggers=["build"],
            metadata={"effectiveness": {"success_rate": 0.9}},
        )
        low_eff = self._make_skill_meta(
            "skill-low-eff",
            triggers=["build"],
            metadata={"effectiveness": {"success_rate": 0.1}},
        )

        matches = await matcher.match(
            query="build",
            available_skills=[low_eff, high_eff],
            threshold=0.0,
        )

        assert len(matches) == 2
        assert matches[0].skill.name == "skill-high-eff"
        assert matches[1].skill.name == "skill-low-eff"

    async def test_relationship_boost_with_active_companion(self):
        """Skills with compose_with relationships to active skills get a boost."""
        matcher = SkillMatcher()

        boosted = self._make_skill_meta(
            "skill-boosted",
            triggers=["deploy"],
        )
        unboosted = self._make_skill_meta(
            "skill-unboosted",
            triggers=["deploy"],
        )

        rel = SkillRelationship(
            source="skill-boosted",
            target="active-one",
            relation_type=SkillRelationType.COMPOSE_WITH,
        )

        matches = await matcher.match(
            query="deploy",
            available_skills=[unboosted, boosted],
            threshold=0.0,
            active_skill_names=["active-one"],
            relationships=[rel],
        )

        assert len(matches) == 2
        assert matches[0].skill.name == "skill-boosted"

    async def test_no_relationship_boost_without_active_skills(self):
        """No relationship boost when no active skills are provided."""
        matcher = SkillMatcher()

        skill = self._make_skill_meta("skill-a", triggers=["test"])

        rel = SkillRelationship(
            source="skill-a",
            target="skill-b",
            relation_type=SkillRelationType.COMPOSE_WITH,
        )

        matches = await matcher.match(
            query="test",
            available_skills=[skill],
            threshold=0.0,
            active_skill_names=[],
            relationships=[rel],
        )

        assert len(matches) == 1
        assert "rel_boost" not in matches[0].match_reason

    async def test_empty_skills_returns_empty(self):
        """Empty available skills returns empty matches."""
        matcher = SkillMatcher()
        matches = await matcher.match("test", [], threshold=0.0)
        assert matches == []

    async def test_threshold_filters_low_scores(self):
        """Skills below threshold are filtered out."""
        matcher = SkillMatcher()

        skill = self._make_skill_meta("skill-weak", triggers=["unrelated"])

        matches = await matcher.match(
            query="deploy application",
            available_skills=[skill],
            threshold=0.9,
        )

        assert len(matches) == 0


class TestSkillServiceSearchByCategory:
    """Tests for SkillService.search_by_category."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def service(self, store):
        return SkillService(store)

    async def test_returns_only_matching_category(self, service):
        """search_by_category returns only skills matching the category."""
        await service.register_base_skill(
            name="skill-a",
            description="Skill A",
            content="content",
        )
        # Manually set category on the stored skill
        store = service._store
        key = ("skill-a", SkillScope.BASE, "system")
        store._store[key].category = "testing"

        await service.register_base_skill(
            name="skill-b",
            description="Skill B",
            content="content",
        )
        key_b = ("skill-b", SkillScope.BASE, "system")
        store._store[key_b].category = "deployment"

        results = await service.search_by_category("testing")
        assert len(results) == 1
        assert results[0].name == "skill-a"

    async def test_returns_empty_for_no_match(self, service):
        """search_by_category returns empty when no skills match."""
        await service.register_base_skill(
            name="skill-x",
            description="Skill X",
            content="content",
        )
        results = await service.search_by_category("nonexistent")
        assert results == []


class TestSkillServiceSearchByTags:
    """Tests for SkillService.search_by_tags."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def service(self, store):
        return SkillService(store)

    async def test_returns_skills_matching_any_tag(self, service):
        """search_by_tags returns skills matching any of the provided tags."""
        store = service._store

        await service.register_base_skill(
            name="skill-a",
            description="Skill A",
            content="content",
        )
        store._store[("skill-a", SkillScope.BASE, "system")].tags = [
            "python", "testing"
        ]

        await service.register_base_skill(
            name="skill-b",
            description="Skill B",
            content="content",
        )
        store._store[("skill-b", SkillScope.BASE, "system")].tags = [
            "java", "deployment"
        ]

        await service.register_base_skill(
            name="skill-c",
            description="Skill C",
            content="content",
        )
        store._store[("skill-c", SkillScope.BASE, "system")].tags = [
            "python", "deployment"
        ]

        # Invalidate cache so new tags are picked up
        service.invalidate_cache()

        results = await service.search_by_tags(["python"])
        names = [r.name for r in results]
        assert "skill-a" in names
        assert "skill-c" in names
        assert "skill-b" not in names

    async def test_case_insensitive_tag_matching(self, service):
        """Tag matching is case-insensitive."""
        store = service._store

        await service.register_base_skill(
            name="skill-ci",
            description="CI skill",
            content="content",
        )
        store._store[("skill-ci", SkillScope.BASE, "system")].tags = ["Python"]

        service.invalidate_cache()

        results = await service.search_by_tags(["python"])
        assert len(results) == 1
        assert results[0].name == "skill-ci"

    async def test_returns_empty_for_no_tag_match(self, service):
        """search_by_tags returns empty when no skills match."""
        await service.register_base_skill(
            name="skill-z",
            description="Skill Z",
            content="content",
        )
        results = await service.search_by_tags(["nonexistent"])
        assert results == []


class TestSkillServiceGetRecommendedSkills:
    """Tests for SkillService.get_recommended_skills."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def service(self, store):
        return SkillService(store)

    async def test_returns_compose_with_neighbors(self, service, store):
        """get_recommended_skills returns compose_with neighbors."""
        await service.register_base_skill(
            name="skill-a", description="A", content="c"
        )
        await service.register_base_skill(
            name="skill-b", description="B", content="c"
        )
        await service.register_base_skill(
            name="skill-c", description="C", content="c"
        )

        await store.save_relationships([
            SkillRelationship(
                source="skill-a",
                target="skill-b",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ])

        results = await service.get_recommended_skills(["skill-a"])
        names = [r.name for r in results]
        assert "skill-b" in names
        assert "skill-c" not in names

    async def test_returns_depend_on_neighbors(self, service, store):
        """get_recommended_skills returns depend_on neighbors."""
        await service.register_base_skill(
            name="skill-x", description="X", content="c"
        )
        await service.register_base_skill(
            name="skill-y", description="Y", content="c"
        )

        await store.save_relationships([
            SkillRelationship(
                source="skill-x",
                target="skill-y",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
        ])

        results = await service.get_recommended_skills(["skill-x"])
        names = [r.name for r in results]
        assert "skill-y" in names

    async def test_excludes_already_active_skills(self, service, store):
        """get_recommended_skills excludes already-active skills."""
        await service.register_base_skill(
            name="skill-p", description="P", content="c"
        )
        await service.register_base_skill(
            name="skill-q", description="Q", content="c"
        )

        await store.save_relationships([
            SkillRelationship(
                source="skill-p",
                target="skill-q",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ])

        results = await service.get_recommended_skills(["skill-p", "skill-q"])
        assert len(results) == 0

    async def test_ignores_similar_to_relationships(self, service, store):
        """get_recommended_skills ignores SIMILAR_TO relationships."""
        await service.register_base_skill(
            name="skill-m", description="M", content="c"
        )
        await service.register_base_skill(
            name="skill-n", description="N", content="c"
        )

        await store.save_relationships([
            SkillRelationship(
                source="skill-m",
                target="skill-n",
                relation_type=SkillRelationType.SIMILAR_TO,
            ),
        ])

        results = await service.get_recommended_skills(["skill-m"])
        assert len(results) == 0

    async def test_empty_active_skills_returns_empty(self, service):
        """get_recommended_skills returns empty for empty active list."""
        results = await service.get_recommended_skills([])
        assert results == []


class TestSkillsMiddlewareCompanionLoading:
    """Tests for SkillsMiddleware companion skill loading."""

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        return context

    async def test_loads_companion_skills_via_relationships(self):
        """SkillsMiddleware loads companion skills via COMPOSE_WITH."""
        store = InMemorySkillStore()
        matcher = SkillMatcher()
        service = SkillService(store, matcher=matcher)

        await service.register_base_skill(
            name="skill-main",
            description="Main skill",
            content="Main content",
            triggers=["main"],
        )
        await service.register_base_skill(
            name="skill-companion",
            description="Companion skill",
            content="Companion content",
        )

        await store.save_relationships([
            SkillRelationship(
                source="skill-main",
                target="skill-companion",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ])

        middleware = SkillsMiddleware(
            skill_service=service,
            auto_activate=True,
            max_auto_skills=5,
            confidence_threshold=0.3,
            skill_store=store,
            load_companions=True,
        )

        context = MiddlewareContext(user_input="main")
        result = await middleware._do_process(context, self._next_fn)

        assert result.has_flag("skills_auto_activated")
        skills_mods = result.modifications.get("skills", [])
        skill_mod = next(
            (m for m in skills_mods if isinstance(m, dict) and m.get("action") == "injected_skills"),
            None,
        )
        assert skill_mod is not None
        activated = skill_mod["activated_skills"]
        assert "skill-main" in activated
        assert "skill-companion" in activated

    async def test_resolves_composed_of_sub_skills(self):
        """SkillsMiddleware resolves composed_of sub-skill chains."""
        store = InMemorySkillStore()
        matcher = SkillMatcher()
        service = SkillService(store, matcher=matcher)

        # Register parent skill with composed_of
        parent = Skill(
            name="skill-parent",
            description="Parent skill",
            scope=SkillScope.BASE,
            scope_id="system",
            content="Parent content",
            triggers=["parent"],
            composed_of=["skill-child"],
        )
        await store.save(parent)

        await service.register_base_skill(
            name="skill-child",
            description="Child skill",
            content="Child content",
        )

        middleware = SkillsMiddleware(
            skill_service=service,
            auto_activate=True,
            max_auto_skills=5,
            confidence_threshold=0.3,
            skill_store=store,
            resolve_composed=True,
        )

        context = MiddlewareContext(user_input="parent")
        result = await middleware._do_process(context, self._next_fn)

        assert result.has_flag("skills_auto_activated")
        skills_mods = result.modifications.get("skills", [])
        skill_mod = next(
            (m for m in skills_mods if isinstance(m, dict) and m.get("action") == "injected_skills"),
            None,
        )
        assert skill_mod is not None
        activated = skill_mod["activated_skills"]
        assert "skill-parent" in activated
        assert "skill-child" in activated

    async def test_respects_max_auto_skills_limit(self):
        """Companion loading respects max_auto_skills limit."""
        store = InMemorySkillStore()
        matcher = SkillMatcher()
        service = SkillService(store, matcher=matcher)

        await service.register_base_skill(
            name="skill-main",
            description="Main",
            content="content",
            triggers=["main"],
        )
        await service.register_base_skill(
            name="skill-comp-a",
            description="Companion A",
            content="content",
        )
        await service.register_base_skill(
            name="skill-comp-b",
            description="Companion B",
            content="content",
        )

        await store.save_relationships([
            SkillRelationship(
                source="skill-main",
                target="skill-comp-a",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
            SkillRelationship(
                source="skill-main",
                target="skill-comp-b",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ])

        middleware = SkillsMiddleware(
            skill_service=service,
            auto_activate=True,
            max_auto_skills=2,
            confidence_threshold=0.3,
            skill_store=store,
            load_companions=True,
        )

        context = MiddlewareContext(user_input="main")
        result = await middleware._do_process(context, self._next_fn)

        skills_mods = result.modifications.get("skills", [])
        skill_mod = next(
            (m for m in skills_mods if isinstance(m, dict) and m.get("action") == "injected_skills"),
            None,
        )
        assert skill_mod is not None
        activated = skill_mod["activated_skills"]
        # max_auto_skills=2, so at most 2 skills total
        assert len(activated) <= 2


class TestSkillsIndexCategoryBadges:
    """Tests for SkillsIndex category badges in format_for_prompt."""

    def test_format_includes_category_badge(self):
        """format_for_prompt includes category badge when present."""
        skills = [
            SkillMetadata(
                name="skill-a",
                description="Skill A",
                scope=SkillScope.BASE,
                scope_id="system",
                category="testing",
            ),
        ]
        index = SkillsIndex(skills=skills)
        output = index.format_for_prompt()
        assert "[testing]" in output
        assert "skill-a" in output

    def test_format_no_badge_when_no_category(self):
        """format_for_prompt omits badge when category is None."""
        skills = [
            SkillMetadata(
                name="skill-b",
                description="Skill B",
                scope=SkillScope.BASE,
                scope_id="system",
            ),
        ]
        index = SkillsIndex(skills=skills)
        output = index.format_for_prompt()
        assert "[" not in output or "skill-b" in output
        assert "**skill-b**" in output


class TestInMemorySkillStoreSearchByTags:
    """Tests for InMemorySkillStore.search_by_tags."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    async def test_search_by_tags_returns_matching(self, store):
        """search_by_tags returns skills matching any tag."""
        skill = Skill(
            name="skill-tagged",
            description="Tagged skill",
            scope=SkillScope.BASE,
            scope_id="system",
            content="content",
            tags=["python", "testing"],
        )
        await store.save(skill)

        results = await store.search_by_tags(["python"])
        assert len(results) == 1
        assert results[0].name == "skill-tagged"

    async def test_search_by_tags_case_insensitive(self, store):
        """search_by_tags is case-insensitive."""
        skill = Skill(
            name="skill-ci",
            description="CI skill",
            scope=SkillScope.BASE,
            scope_id="system",
            content="content",
            tags=["Python"],
        )
        await store.save(skill)

        results = await store.search_by_tags(["python"])
        assert len(results) == 1

    async def test_search_by_tags_no_match(self, store):
        """search_by_tags returns empty for no match."""
        skill = Skill(
            name="skill-nm",
            description="No match",
            scope=SkillScope.BASE,
            scope_id="system",
            content="content",
            tags=["java"],
        )
        await store.save(skill)

        results = await store.search_by_tags(["python"])
        assert results == []
