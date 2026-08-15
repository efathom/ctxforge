"""
Tests for Skill Effectiveness Service and Middleware.
"""
import pytest

from ctxforge.config.base import SkillEffectivenessConfig
from ctxforge.core.skill import Skill, SkillScope
from ctxforge.engine.services.skill_effectiveness_service import (
    DEFAULT_EFFECTIVENESS,
    SkillEffectivenessService,
)
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skill_effectiveness import SkillEffectivenessMiddleware
from ctxforge.storage.memory.skill import InMemorySkillStore


def _make_skill(name: str = "test-skill") -> Skill:
    return Skill(
        name=name, description="Test",
        scope=SkillScope.BASE, scope_id="system",
        content="content",
    )


class TestSkillEffectivenessService:
    """Tests for SkillEffectivenessService."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def service(self, store):
        return SkillEffectivenessService(store)

    async def test_record_usage_increments_count(self, store, service):
        await store.save(_make_skill())
        await service.record_usage("test-skill", SkillScope.BASE, "system", 0.9)
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff["usage_count"] == 1

    async def test_record_usage_updates_last_used_at(self, store, service):
        await store.save(_make_skill())
        await service.record_usage("test-skill", SkillScope.BASE, "system", 0.9)
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff["last_used_at"] is not None

    async def test_record_usage_running_average_confidence(self, store, service):
        await store.save(_make_skill())
        await service.record_usage("test-skill", SkillScope.BASE, "system", 0.8)
        await service.record_usage("test-skill", SkillScope.BASE, "system", 1.0)
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff["avg_confidence_at_match"] == 0.9

    async def test_record_outcome_success(self, store, service):
        await store.save(_make_skill())
        await service.record_outcome("test-skill", SkillScope.BASE, "system", True)
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff["success_count"] == 1
        assert eff["failure_count"] == 0
        assert eff["success_rate"] == 1.0

    async def test_record_outcome_failure(self, store, service):
        await store.save(_make_skill())
        await service.record_outcome("test-skill", SkillScope.BASE, "system", False)
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff["success_count"] == 0
        assert eff["failure_count"] == 1
        assert eff["success_rate"] == 0.0

    async def test_record_outcome_mixed(self, store, service):
        await store.save(_make_skill())
        await service.record_outcome("test-skill", SkillScope.BASE, "system", True)
        await service.record_outcome("test-skill", SkillScope.BASE, "system", True)
        await service.record_outcome("test-skill", SkillScope.BASE, "system", False)
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff["success_count"] == 2
        assert eff["failure_count"] == 1
        assert abs(eff["success_rate"] - 2 / 3) < 0.01

    async def test_get_effectiveness_defaults(self, store, service):
        await store.save(_make_skill())
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert eff == DEFAULT_EFFECTIVENESS

    async def test_get_effectiveness_nonexistent(self, service):
        eff = await service.get_effectiveness("nope", SkillScope.BASE, "system")
        assert eff == DEFAULT_EFFECTIVENESS

    async def test_get_ranking_boost_no_usage(self, store, service):
        await store.save(_make_skill())
        boost = await service.get_ranking_boost("test-skill", SkillScope.BASE, "system")
        assert boost == 0.0

    async def test_get_ranking_boost_with_usage(self, store, service):
        await store.save(_make_skill())
        await service.record_usage("test-skill", SkillScope.BASE, "system", 0.9)
        await service.record_outcome("test-skill", SkillScope.BASE, "system", True)
        boost = await service.get_ranking_boost("test-skill", SkillScope.BASE, "system")
        # success_rate=1.0, weight=0.3 (default) => 0.3
        assert boost == 0.3

    async def test_get_ranking_boost_custom_weight(self, store):
        config = SkillEffectivenessConfig(weight_in_ranking=0.5)
        service = SkillEffectivenessService(store, config=config)
        await store.save(_make_skill())
        await service.record_usage("test-skill", SkillScope.BASE, "system", 0.9)
        await service.record_outcome("test-skill", SkillScope.BASE, "system", True)
        boost = await service.get_ranking_boost("test-skill", SkillScope.BASE, "system")
        assert boost == 0.5

    async def test_record_usage_with_session_id(self, store, service):
        await store.save(_make_skill())
        await service.record_usage(
            "test-skill", SkillScope.BASE, "system", 0.9,
            session_id="sess-1",
        )
        eff = await service.get_effectiveness("test-skill", SkillScope.BASE, "system")
        assert "sess-1" in eff["sessions_used_in"]


class TestSkillEffectivenessMiddleware:
    """Tests for SkillEffectivenessMiddleware."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    @pytest.fixture
    def service(self, store):
        return SkillEffectivenessService(store)

    @pytest.fixture
    def middleware(self, service):
        return SkillEffectivenessMiddleware(service)

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        return context

    async def test_records_usage_when_skills_activated(self, store, service, middleware):
        await store.save(_make_skill(name="skill-a"))
        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_auto_activated")
        context.record_modification("skills", {
            "action": "injected_skills",
            "activated_skills": ["skill-a"],
        })
        await middleware._do_process(context, self._next_fn)
        eff = await service.get_effectiveness("skill-a", SkillScope.BASE, "system")
        assert eff["usage_count"] == 1

    async def test_records_outcome_on_session_complete(self, store, service, middleware):
        await store.save(_make_skill(name="skill-a"))
        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_auto_activated")
        context.add_flag("session_complete")
        context.record_modification("skills", {
            "action": "injected_skills",
            "activated_skills": ["skill-a"],
        })
        await middleware._do_process(context, self._next_fn)
        eff = await service.get_effectiveness("skill-a", SkillScope.BASE, "system")
        assert eff["success_count"] == 1

    async def test_records_failure_on_session_failed(self, store, service, middleware):
        await store.save(_make_skill(name="skill-a"))
        context = MiddlewareContext(user_input="test")
        context.add_flag("skills_auto_activated")
        context.add_flag("session_complete")
        context.add_flag("session_failed")
        context.record_modification("skills", {
            "action": "injected_skills",
            "activated_skills": ["skill-a"],
        })
        await middleware._do_process(context, self._next_fn)
        eff = await service.get_effectiveness("skill-a", SkillScope.BASE, "system")
        assert eff["failure_count"] == 1
        assert eff["success_count"] == 0

    async def test_does_nothing_when_no_skills_activated(self, store, service, middleware):
        await store.save(_make_skill(name="skill-a"))
        context = MiddlewareContext(user_input="test")
        await middleware._do_process(context, self._next_fn)
        eff = await service.get_effectiveness("skill-a", SkillScope.BASE, "system")
        assert eff["usage_count"] == 0
