"""
Tests for Skill Relationship Middleware.
"""
import json
from typing import AsyncIterator

import pytest

from ctxforge.core.skill import SkillScope
from ctxforge.engine.services.skill_relationship_service import (
    SkillRelationshipService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skill_relationship import (
    SkillRelationshipMiddleware,
)
from ctxforge.protocols.llm import LLMResponse
from ctxforge.storage.memory.skill import InMemorySkillStore


class FakeLLMProvider:
    """Fake LLM that returns relationship JSON."""

    def __init__(self, response: str = "[]"):
        self._response = response

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-model"

    async def generate(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(content=self._response, model="fake-model")

    async def chat(self, messages, **kwargs) -> LLMResponse:
        return LLMResponse(content=self._response, model="fake-model")

    async def stream(self, messages, **kwargs) -> AsyncIterator[str]:
        yield self._response

    def count_tokens(self, text, model=None) -> int:
        return len(text.split())

    def count_message_tokens(self, messages, model=None) -> int:
        return sum(len(m.content.split()) for m in messages)


async def _make_service_with_skills(
    store: InMemorySkillStore, n: int = 3,
) -> SkillService:
    """Create a SkillService with n skills pre-registered."""
    from ctxforge.core.skill import Skill
    svc = SkillService(store)
    for i in range(n):
        skill = Skill(
            name=f"skill-{i}",
            description=f"Skill number {i}",
            scope=SkillScope.BASE,
            scope_id="system",
            content=f"# Skill {i}\n\nContent for skill {i}.",
        )
        await svc.register_skill(skill)
    return svc


class TestSkillRelationshipMiddleware:
    """Tests for SkillRelationshipMiddleware."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    async def _next_fn(
        self, context: MiddlewareContext,
    ) -> MiddlewareContext:
        return context

    async def test_fires_when_skills_generated(self, store):
        """Middleware triggers analysis when skills_generated flag is set."""
        rel_json = json.dumps([
            {
                "source": "skill-0",
                "target": "skill-1",
                "relation_type": "compose_with",
                "reason": "Work together",
                "confidence": 0.9,
            },
        ])
        llm = FakeLLMProvider(response=rel_json)
        skill_svc = await _make_service_with_skills(store, n=3)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(skill_svc, rel_svc)

        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_generated")
        context.set_metadata("generated_skill_names", ["skill-0"])

        result = await mw._do_process(context, self._next_fn)
        assert result.has_flag("relationships_analyzed")
        assert result.get_metadata("relationships_count") >= 1

    async def test_no_flag_does_nothing(self, store):
        """Middleware does nothing when skills_generated flag is not set."""
        llm = FakeLLMProvider()
        skill_svc = await _make_service_with_skills(store, n=3)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(skill_svc, rel_svc)

        context = MiddlewareContext(user_input="test")
        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("relationships_analyzed")

    async def test_no_generated_names_does_nothing(self, store):
        """Middleware does nothing when generated_skill_names is empty."""
        llm = FakeLLMProvider()
        skill_svc = await _make_service_with_skills(store, n=3)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(skill_svc, rel_svc)

        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_generated")
        context.set_metadata("generated_skill_names", [])

        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("relationships_analyzed")

    async def test_fewer_than_two_skills_skips(self, store):
        """Middleware skips analysis when fewer than 2 skills exist."""
        llm = FakeLLMProvider()
        skill_svc = await _make_service_with_skills(store, n=1)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(skill_svc, rel_svc)

        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_generated")
        context.set_metadata("generated_skill_names", ["skill-0"])

        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("relationships_analyzed")

    async def test_relationships_persisted(self, store):
        """Verify relationships are persisted to the store."""
        rel_json = json.dumps([
            {
                "source": "skill-0",
                "target": "skill-1",
                "relation_type": "similar_to",
                "reason": "They are similar",
                "confidence": 0.85,
            },
        ])
        llm = FakeLLMProvider(response=rel_json)
        skill_svc = await _make_service_with_skills(store, n=3)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(skill_svc, rel_svc)

        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_generated")
        context.set_metadata("generated_skill_names", ["skill-0"])

        await mw._do_process(context, self._next_fn)

        # Verify relationships were persisted
        rels = await store.get_all_relationships()
        assert len(rels) >= 1
        assert rels[0].source == "skill-0"
        assert rels[0].target == "skill-1"

    async def test_disabled_middleware_skips(self, store):
        """Disabled middleware passes through without analysis."""
        llm = FakeLLMProvider()
        skill_svc = await _make_service_with_skills(store, n=3)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(
            skill_svc, rel_svc, enabled=False,
        )

        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_generated")
        context.set_metadata("generated_skill_names", ["skill-0"])

        result = await mw.process(context, self._next_fn)
        assert not result.has_flag("relationships_analyzed")

    async def test_analysis_error_handled_gracefully(self, store):
        """Errors during analysis don't crash the middleware."""
        llm = FakeLLMProvider(response="invalid json!!!")
        skill_svc = await _make_service_with_skills(store, n=3)
        rel_svc = SkillRelationshipService(llm, store)
        mw = SkillRelationshipMiddleware(skill_svc, rel_svc)

        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_generated")
        context.set_metadata("generated_skill_names", ["skill-0"])

        result = await mw._do_process(context, self._next_fn)
        assert not result.has_flag("relationships_analyzed")
        assert result.get_metadata("relationship_analysis_error") is not None
