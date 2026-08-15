"""
Tests for Skill Lifecycle Service.
"""
import json
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from ctxforge.config.base import SkillGenerationConfig
from ctxforge.core.events import Event, EventType
from ctxforge.core.skill import (
    EvaluationLevel,
    Skill,
    SkillContent,
    SkillScope,
)
from ctxforge.engine.services.skill_effectiveness_service import (
    SkillEffectivenessService,
)
from ctxforge.engine.services.skill_evaluation_service import (
    SkillEvaluationService,
)
from ctxforge.engine.services.skill_generator_service import (
    SkillGeneratorService,
)
from ctxforge.engine.services.skill_lifecycle_service import (
    SkillLifecycleService,
)
from ctxforge.engine.services.skill_relationship_service import (
    SkillRelationshipService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.engine.services.skill_validator import SkillValidator
from ctxforge.protocols.llm import ChatMessage, LLMResponse
from ctxforge.storage.memory.skill import InMemorySkillStore


class FakeLLMProvider:
    """Fake LLM that returns configurable responses."""

    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = list(responses or [])
        self._call_index = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-model"

    async def generate(
        self, prompt: str, model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None, **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._next(), model="fake-model")

    async def chat(
        self, messages: List[ChatMessage], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        functions: Optional[List[Dict[str, Any]]] = None, **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._next(), model="fake-model")

    async def stream(
        self, messages: List[ChatMessage], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield self._next()

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text.split())

    def count_message_tokens(
        self, messages: List[ChatMessage], model: Optional[str] = None,
    ) -> int:
        return sum(len(m.content.split()) for m in messages)

    def _next(self) -> str:
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return "[]"


def _make_events(n: int = 5) -> List[Event]:
    events = []
    for i in range(n):
        et = EventType.USER if i % 2 == 0 else EventType.AGENT
        events.append(Event(type=et, content=f"Event {i} content"))
    return events


def _candidate_response() -> str:
    return json.dumps([
        {
            "name": "fix-import-errors",
            "description": "Fix Python import errors",
            "when_to_use": "When import errors are detected",
            "category": "debugging",
            "triggers": ["import error", "fix imports"],
        },
    ])


def _content_response() -> str:
    return json.dumps({
        "instructions": (
            "# Fix Import Errors\n\n"
            "1. Check the import statement at the top of the file\n"
            "2. Verify the module exists in the project dependencies\n"
            "3. Fix any typos in the module name or path\n"
            "4. Run the linter to confirm the fix"
        ),
        "scripts": {},
        "references": {},
        "triggers": ["import error", "fix imports"],
    })


def _good_evaluation_response() -> str:
    return json.dumps({
        "safety": "good",
        "safety_reason": "No dangerous operations",
        "completeness": "good",
        "completeness_reason": "All steps covered",
        "executability": "good",
        "executability_reason": "Clear and actionable",
        "maintainability": "good",
        "maintainability_reason": "Well-structured",
        "cost_awareness": "good",
        "cost_awareness_reason": "Minimal cost",
    })


def _poor_evaluation_response() -> str:
    return json.dumps({
        "safety": "poor",
        "safety_reason": "Unsafe",
        "completeness": "poor",
        "completeness_reason": "Incomplete",
        "executability": "poor",
        "executability_reason": "Not actionable",
        "maintainability": "poor",
        "maintainability_reason": "Hard to maintain",
        "cost_awareness": "poor",
        "cost_awareness_reason": "Expensive",
    })


def _relationship_response() -> str:
    return json.dumps([
        {
            "source": "fix-import-errors",
            "target": "lint-code",
            "relation_type": "compose_with",
            "reason": "Complementary",
            "confidence": 0.9,
        },
    ])


def _make_skill(
    name: str = "fix-import-errors",
    project_id: str = "proj-1",
    effectiveness: Optional[Dict[str, Any]] = None,
) -> Skill:
    return Skill(
        name=name,
        description="Fix Python import errors",
        scope=SkillScope.PROJECT,
        scope_id=project_id,
        content=(
            "# Fix Import Errors\n\n"
            "1. Check the import statement at the top of the file\n"
            "2. Verify the module exists in the project dependencies\n"
            "3. Fix any typos in the module name or path\n"
            "4. Run the linter to confirm the fix"
        ),
        triggers=["import error", "fix imports"],
        when_to_use="When import errors are detected",
        category="debugging",
        structured_content=SkillContent(
            instructions=(
                "# Fix Import Errors\n\n"
                "1. Check the import statement at the top of the file\n"
                "2. Verify the module exists in the project dependencies\n"
                "3. Fix any typos in the module name or path\n"
                "4. Run the linter to confirm the fix"
            ),
        ),
        effectiveness=effectiveness,
    )


class TestSkillLifecycleService:
    """Tests for SkillLifecycleService."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def skill_service(self, store):
        return SkillService(store)

    def _make_lifecycle(
        self,
        store: InMemorySkillStore,
        skill_service: SkillService,
        llm_responses: Optional[List[str]] = None,
        rel_responses: Optional[List[str]] = None,
        min_evaluation_score: float = 0.4,
        with_relationships: bool = False,
        with_effectiveness: bool = False,
    ) -> SkillLifecycleService:
        llm = FakeLLMProvider(llm_responses or [])
        generator = SkillGeneratorService(
            llm, skill_service,
            config=SkillGenerationConfig(min_session_events=1),
        )
        validator = SkillValidator()
        evaluator = SkillEvaluationService(llm)

        rel_svc = None
        if with_relationships:
            rel_llm = FakeLLMProvider(rel_responses or ["[]"])
            rel_svc = SkillRelationshipService(rel_llm, store)

        eff_svc = None
        if with_effectiveness:
            eff_svc = SkillEffectivenessService(store)

        return SkillLifecycleService(
            generator=generator,
            validator=validator,
            evaluator=evaluator,
            skill_service=skill_service,
            relationship_service=rel_svc,
            effectiveness_service=eff_svc,
            min_evaluation_score=min_evaluation_score,
        )

    async def test_create_from_session_full_pipeline(
        self, store, skill_service,
    ):
        """Events -> skills generated, validated, evaluated, persisted."""
        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[
                _candidate_response(),
                _content_response(),
                _good_evaluation_response(),
            ],
        )

        skills = await lifecycle.create_from_session(
            _make_events(5), "proj-1",
        )
        assert len(skills) == 1
        assert skills[0].name == "fix-import-errors"
        assert skills[0].evaluation is not None
        assert skills[0].evaluation.overall_score >= 0.4

        # Verify persisted to store
        saved = await store.get(
            "fix-import-errors", SkillScope.PROJECT, "proj-1",
        )
        assert saved is not None
        assert saved.evaluation is not None

    async def test_create_from_session_rejected_by_evaluation(
        self, store, skill_service,
    ):
        """Skill scoring below min_evaluation_score is not persisted."""
        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[
                _candidate_response(),
                _content_response(),
                _poor_evaluation_response(),
            ],
            min_evaluation_score=0.5,
        )

        skills = await lifecycle.create_from_session(
            _make_events(5), "proj-1",
        )
        assert len(skills) == 0

    async def test_create_from_session_with_relationships(
        self, store, skill_service,
    ):
        """Relationship analysis runs after skill creation."""
        # Pre-register a second skill so relationships can be analyzed
        await skill_service.register_project_skill(
            project_id="proj-1",
            name="lint-code",
            description="Run code linter",
            content=(
                "# Lint Code\n\n"
                "1. Run flake8 on the entire project\n"
                "2. Fix all reported issues\n"
                "3. Verify no remaining warnings"
            ),
            triggers=["lint", "flake8"],
        )

        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[
                _candidate_response(),
                _content_response(),
                _good_evaluation_response(),
            ],
            rel_responses=[_relationship_response()],
            with_relationships=True,
        )

        skills = await lifecycle.create_from_session(
            _make_events(5), "proj-1",
        )
        assert len(skills) == 1

        # Verify relationships were persisted
        rels = await store.get_all_relationships()
        assert len(rels) >= 1

    async def test_create_from_session_empty_events(
        self, store, skill_service,
    ):
        """No events -> no skills."""
        lifecycle = self._make_lifecycle(store, skill_service)
        skills = await lifecycle.create_from_session([], "proj-1")
        assert skills == []

    async def test_create_from_github(self, store, skill_service):
        """GitHub skill goes through validate -> evaluate -> persist."""
        skill = _make_skill()

        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[_good_evaluation_response()],
        )

        # Mock generator.generate_from_github to return a skill
        lifecycle._generator.generate_from_github = AsyncMock(
            return_value=skill,
        )

        result = await lifecycle.create_from_github(
            "https://github.com/example/repo", "proj-1",
        )
        assert result is not None
        assert result.name == "fix-import-errors"
        assert result.evaluation is not None

        saved = await store.get(
            "fix-import-errors", SkillScope.PROJECT, "proj-1",
        )
        assert saved is not None

    async def test_create_from_github_returns_none(
        self, store, skill_service,
    ):
        """Generator returns None -> lifecycle returns None."""
        lifecycle = self._make_lifecycle(store, skill_service)
        lifecycle._generator.generate_from_github = AsyncMock(
            return_value=None,
        )

        result = await lifecycle.create_from_github(
            "https://github.com/example/repo", "proj-1",
        )
        assert result is None

    async def test_create_from_document(self, store, skill_service):
        """Document skill goes through validate -> evaluate -> persist."""
        skill = _make_skill()

        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[_good_evaluation_response()],
        )

        lifecycle._generator.generate_from_document = AsyncMock(
            return_value=skill,
        )

        result = await lifecycle.create_from_document(
            "/tmp/doc.pdf", "proj-1",
        )
        assert result is not None
        assert result.evaluation is not None

        saved = await store.get(
            "fix-import-errors", SkillScope.PROJECT, "proj-1",
        )
        assert saved is not None

    async def test_create_from_document_returns_none(
        self, store, skill_service,
    ):
        """Generator returns None -> lifecycle returns None."""
        lifecycle = self._make_lifecycle(store, skill_service)
        lifecycle._generator.generate_from_document = AsyncMock(
            return_value=None,
        )

        result = await lifecycle.create_from_document(
            "/tmp/doc.pdf", "proj-1",
        )
        assert result is None

    async def test_retire_underperforming(self, store, skill_service):
        """Skills below threshold are deleted from store."""
        bad_skill = _make_skill(
            name="bad-skill",
            effectiveness={
                "usage_count": 10,
                "success_count": 1,
                "failure_count": 9,
                "success_rate": 0.1,
            },
        )
        good_skill = _make_skill(
            name="good-skill",
            effectiveness={
                "usage_count": 10,
                "success_count": 9,
                "failure_count": 1,
                "success_rate": 0.9,
            },
        )
        await store.save(bad_skill)
        await store.save(good_skill)

        lifecycle = self._make_lifecycle(store, skill_service)
        retired = await lifecycle.retire_underperforming(
            SkillScope.PROJECT, "proj-1",
            min_success_rate=0.3, min_usage_count=5,
        )

        assert "bad-skill" in retired
        assert "good-skill" not in retired

        # Verify bad-skill is deleted
        assert await store.get(
            "bad-skill", SkillScope.PROJECT, "proj-1",
        ) is None
        # Verify good-skill still exists
        assert await store.get(
            "good-skill", SkillScope.PROJECT, "proj-1",
        ) is not None

    async def test_retire_insufficient_usage_not_deleted(
        self, store, skill_service,
    ):
        """Skills with usage below min_usage_count are not retired."""
        low_usage_skill = _make_skill(
            name="low-usage-skill",
            effectiveness={
                "usage_count": 2,
                "success_count": 0,
                "failure_count": 2,
                "success_rate": 0.0,
            },
        )
        await store.save(low_usage_skill)

        lifecycle = self._make_lifecycle(store, skill_service)
        retired = await lifecycle.retire_underperforming(
            SkillScope.PROJECT, "proj-1",
            min_success_rate=0.3, min_usage_count=5,
        )

        assert retired == []
        assert await store.get(
            "low-usage-skill", SkillScope.PROJECT, "proj-1",
        ) is not None

    async def test_refresh_evaluation(self, store, skill_service):
        """All skills in scope are re-evaluated and persisted."""
        skill1 = _make_skill(name="skill-one")
        skill2 = _make_skill(name="skill-two")
        await store.save(skill1)
        await store.save(skill2)

        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[
                _good_evaluation_response(),
                _good_evaluation_response(),
            ],
        )

        results = await lifecycle.refresh_evaluation(
            SkillScope.PROJECT, "proj-1",
        )

        assert "skill-one" in results
        assert "skill-two" in results

        # Verify evaluations were persisted
        saved1 = await store.get(
            "skill-one", SkillScope.PROJECT, "proj-1",
        )
        assert saved1 is not None
        assert saved1.evaluation is not None

        saved2 = await store.get(
            "skill-two", SkillScope.PROJECT, "proj-1",
        )
        assert saved2 is not None
        assert saved2.evaluation is not None

    async def test_validation_failure_rejects_skill(
        self, store, skill_service,
    ):
        """Skill that fails validation is not evaluated or persisted."""
        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[_good_evaluation_response()],
        )

        # Create a skill with empty content (fails validation)
        bad_skill = Skill(
            name="bad-content",
            description="Has no real content",
            scope=SkillScope.PROJECT,
            scope_id="proj-1",
            content="",
            triggers=["test"],
            when_to_use="Testing",
        )

        result = await lifecycle._validate_evaluate_persist(bad_skill)
        assert result is None

    async def test_evaluation_error_handled_gracefully(
        self, store, skill_service,
    ):
        """Evaluation error results in None, no crash."""
        # LLM returns invalid JSON -> evaluator raises ValueError
        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=["not valid json"],
        )

        skill = _make_skill()
        result = await lifecycle._validate_evaluate_persist(skill)
        assert result is None

    async def test_persistence_verification_session(
        self, store, skill_service,
    ):
        """Verify store.save() is called with evaluation attached."""
        lifecycle = self._make_lifecycle(
            store, skill_service,
            llm_responses=[
                _candidate_response(),
                _content_response(),
                _good_evaluation_response(),
            ],
        )

        skills = await lifecycle.create_from_session(
            _make_events(5), "proj-1",
        )
        assert len(skills) == 1

        # Verify the stored skill has evaluation
        saved = await store.get(
            "fix-import-errors", SkillScope.PROJECT, "proj-1",
        )
        assert saved is not None
        assert saved.evaluation is not None
        assert saved.evaluation.safety == EvaluationLevel.GOOD
