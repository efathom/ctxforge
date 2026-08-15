"""
Tests for Skill Generator Service and Middleware.
"""
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from ctxforge.config.base import SkillGenerationConfig
from ctxforge.core.events import Event, EventType
from ctxforge.core.observation import Observation, ObservationType
from ctxforge.core.scoped_memory import MemoryCategory, MemoryScope, ScopedMemory
from ctxforge.core.skill import SkillScope
from ctxforge.engine.services.skill_generator_service import SkillGeneratorService
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skill_generation import SkillGenerationMiddleware
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


def _make_observations() -> List[Observation]:
    return [
        Observation(
            type=ObservationType.BUGFIX,
            summary="Fixed import error",
            detail="Moved import to top of file",
        ),
        Observation(
            type=ObservationType.DISCOVERY,
            summary="Found caching pattern",
        ),
    ]


def _make_memories() -> List[ScopedMemory]:
    return [
        ScopedMemory(
            id="mem-1",
            scope=MemoryScope.PROJECT,
            scope_id="proj-1",
            category=MemoryCategory.CONVENTION,
            key="type-hints",
            content="Always use type hints in function signatures",
        ),
    ]


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


def _prompt_skill_response() -> str:
    return json.dumps({
        "name": "deploy-to-staging",
        "description": "Deploy application to staging environment",
        "when_to_use": "When deploying to staging",
        "category": "deployment",
        "instructions": "1. Build\n2. Test\n3. Deploy",
        "triggers": ["deploy staging", "staging deploy"],
    })


class TestSkillGeneratorService:
    """Tests for SkillGeneratorService."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def skill_service(self, store):
        return SkillService(store)

    async def test_generate_from_session_produces_skills(self, store, skill_service):
        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(events, "proj-1")
        assert len(skills) == 1
        assert skills[0].name == "fix-import-errors"
        assert skills[0].scope == SkillScope.PROJECT
        assert skills[0].structured_content is not None

    async def test_generate_from_session_too_few_events(self, skill_service):
        llm = FakeLLMProvider()
        config = SkillGenerationConfig(min_session_events=10)
        svc = SkillGeneratorService(llm, skill_service, config=config)
        events = _make_events(3)
        skills = await svc.generate_from_session(events, "proj-1")
        assert skills == []

    async def test_generate_from_observations(self, store, skill_service):
        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        skills = await svc.generate_from_observations(_make_observations(), "proj-1")
        assert len(skills) == 1

    async def test_generate_from_observations_empty(self, skill_service):
        llm = FakeLLMProvider()
        svc = SkillGeneratorService(llm, skill_service)
        skills = await svc.generate_from_observations([], "proj-1")
        assert skills == []

    async def test_generate_from_memories(self, store, skill_service):
        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        skills = await svc.generate_from_memories(_make_memories(), "proj-1")
        assert len(skills) == 1

    async def test_generate_from_prompt(self, store, skill_service):
        llm = FakeLLMProvider([_prompt_skill_response()])
        svc = SkillGeneratorService(llm, skill_service)
        skill = await svc.generate_from_prompt(
            "Create a skill for deploying to staging", "proj-1"
        )
        assert skill is not None
        assert skill.name == "deploy-to-staging"
        assert skill.triggers == ["deploy staging", "staging deploy"]

    async def test_generate_from_prompt_invalid_name(self, skill_service):
        bad_response = json.dumps({
            "name": "INVALID_NAME",
            "description": "Bad",
            "instructions": "Bad",
        })
        llm = FakeLLMProvider([bad_response])
        svc = SkillGeneratorService(llm, skill_service)
        skill = await svc.generate_from_prompt("test", "proj-1")
        assert skill is None

    async def test_candidate_extraction_validates_names(self, skill_service):
        response = json.dumps([
            {"name": "valid-name", "description": "Good", "when_to_use": "x", "category": "other"},
            {"name": "INVALID", "description": "Bad", "when_to_use": "x", "category": "other"},
        ])
        llm = FakeLLMProvider([response, _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(events, "proj-1")
        assert len(skills) == 1
        assert skills[0].name == "valid-name"

    async def test_handles_malformed_json_gracefully(self, skill_service):
        llm = FakeLLMProvider(["not json at all"])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(events, "proj-1")
        assert skills == []


class TestDeduplicationAndRetry:
    """Tests for deduplication and retry in generate_from_session."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def skill_service(self, store):
        return SkillService(store)

    async def test_dedup_skips_existing_skills(self, store, skill_service):
        """Existing skill name is skipped during deduplication."""
        # Pre-register a skill with the same name as the candidate
        await skill_service.register_base_skill(
            name="fix-import-errors",
            description="Already exists",
            content="# Existing\n\nThis skill already exists and has enough content.",
        )

        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(events, "proj-1")
        # Candidate "fix-import-errors" should be skipped (dedup)
        assert len(skills) == 0

    async def test_dedup_false_allows_duplicates(self, store, skill_service):
        """deduplicate=False skips the dedup step."""
        await skill_service.register_base_skill(
            name="fix-import-errors",
            description="Already exists",
            content="# Existing\n\nThis skill already exists with sufficient content.",
        )

        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(
            events, "proj-1", deduplicate=False,
        )
        # Without dedup, the skill should still be generated
        assert len(skills) == 1

    async def test_retry_on_validation_failure(self, store, skill_service):
        """First generation fails validation, second succeeds."""
        # First content response has empty instructions (fails validation)
        bad_content = json.dumps({
            "instructions": "",
            "scripts": {},
            "references": {},
        })
        good_content = _content_response()
        llm = FakeLLMProvider([
            _candidate_response(),
            bad_content,
            good_content,
        ])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(
            events, "proj-1", max_retries=2,
        )
        assert len(skills) == 1
        assert skills[0].name == "fix-import-errors"

    async def test_all_retries_fail_no_error(self, store, skill_service):
        """All retries fail validation -> no skill persisted, no crash."""
        bad_content = json.dumps({
            "instructions": "",
            "scripts": {},
            "references": {},
        })
        llm = FakeLLMProvider([
            _candidate_response(),
            bad_content,
            bad_content,
            bad_content,
        ])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(
            events, "proj-1", max_retries=2,
        )
        assert len(skills) == 0

    async def test_max_retries_zero_means_no_retries(self, store, skill_service):
        """max_retries=0 means one attempt only."""
        bad_content = json.dumps({
            "instructions": "",
            "scripts": {},
            "references": {},
        })
        good_content = _content_response()
        llm = FakeLLMProvider([
            _candidate_response(),
            bad_content,
            good_content,  # This should NOT be reached
        ])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(
            events, "proj-1", max_retries=0,
        )
        assert len(skills) == 0

    async def test_skills_persisted_to_store(self, store, skill_service):
        """Verify generated skills are persisted via store.save()."""
        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        svc = SkillGeneratorService(llm, skill_service)
        events = _make_events(5)
        skills = await svc.generate_from_session(events, "proj-1")
        assert len(skills) == 1

        # Verify it's actually in the store
        saved = await store.get(
            "fix-import-errors", SkillScope.PROJECT, "proj-1",
        )
        assert saved is not None
        assert saved.name == "fix-import-errors"


class TestSkillGenerationMiddleware:
    """Tests for SkillGenerationMiddleware."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def skill_service(self, store):
        return SkillService(store)

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        return context

    async def test_triggers_on_session_complete(self, skill_service):
        llm = FakeLLMProvider([_candidate_response(), _content_response()])
        generator = SkillGeneratorService(llm, skill_service)
        config = SkillGenerationConfig(
            auto_generate_from_sessions=True,
            min_session_events=1,
        )
        mw = SkillGenerationMiddleware(
            generator, config=config, project_id="proj-1"
        )

        context = MiddlewareContext(user_input="test")
        context.add_flag("session_complete")

        # Simulate session with events
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        mock_session.events = _make_events(5)
        context.session = mock_session

        result = await mw._do_process(context, self._next_fn)
        assert result.has_flag("skills_generated")

    async def test_does_nothing_when_disabled(self, skill_service):
        llm = FakeLLMProvider()
        generator = SkillGeneratorService(llm, skill_service)
        config = SkillGenerationConfig(auto_generate_from_sessions=False)
        mw = SkillGenerationMiddleware(generator, config=config)

        context = MiddlewareContext(user_input="test")
        context.add_flag("session_complete")

        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("skills_generated")

    async def test_does_nothing_when_not_session_complete(self, skill_service):
        llm = FakeLLMProvider()
        generator = SkillGeneratorService(llm, skill_service)
        config = SkillGenerationConfig(auto_generate_from_sessions=True)
        mw = SkillGenerationMiddleware(generator, config=config)

        context = MiddlewareContext(user_input="test")
        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("skills_generated")

    async def test_does_nothing_when_too_few_events(self, skill_service):
        llm = FakeLLMProvider()
        generator = SkillGeneratorService(llm, skill_service)
        config = SkillGenerationConfig(
            auto_generate_from_sessions=True,
            min_session_events=100,
        )
        mw = SkillGenerationMiddleware(generator, config=config)

        context = MiddlewareContext(user_input="test")
        context.add_flag("session_complete")

        from unittest.mock import MagicMock
        mock_session = MagicMock()
        mock_session.events = _make_events(3)
        context.session = mock_session

        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("skills_generated")
