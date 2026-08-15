"""
Tests for Skill Prerequisites Enforcement in SkillsMiddleware.

Tests that the middleware correctly blocks skills whose prerequisites
have not been completed, and allows them once prerequisites are marked done.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.session import Session, SessionState
from ctxforge.core.skill import Skill, SkillMatch, SkillScope
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skills import (
    COMPLETED_SKILLS_KEY,
    SkillsMiddleware,
)


def _make_skill(name, prerequisites=None, triggers=None):
    """Helper to create a Skill with minimal boilerplate."""
    return Skill(
        name=name,
        description=f"Use when {name} is needed",
        scope=SkillScope.BASE,
        scope_id="system",
        content=f"# {name}\nWorkflow content here.",
        prerequisites=prerequisites or [],
        triggers=triggers or [name],
    )


def _make_context(user_input="do something", session=None):
    """Helper to create a MiddlewareContext."""
    return MiddlewareContext(
        user_input=user_input,
        session=session,
    )


def _make_session(completed_skills=None):
    """Helper to create a Session with completed skills in state."""
    session = MagicMock(spec=Session)
    state = SessionState()
    if completed_skills:
        state.set(COMPLETED_SKILLS_KEY, list(completed_skills))
    session.state = state
    session.session_id = "test-session"
    return session


def _make_service(skills=None, matches=None):
    """Helper to create a mock SkillService."""
    service = AsyncMock()
    service.get_available_skills = AsyncMock(return_value=[
        s.skill_metadata for s in (skills or [])
    ])
    service.format_skills_index = MagicMock(return_value="## Skills Index")
    service.match_skills = AsyncMock(return_value=[
        SkillMatch(skill=s.skill_metadata, confidence=0.9)
        for s in (matches or skills or [])
    ])
    service.load_skill_content = AsyncMock(
        side_effect=lambda name, **kw: next(
            (s for s in (skills or []) if s.name == name), None
        )
    )
    service.format_skill_workflow = MagicMock(
        side_effect=lambda s, **kw: f"Workflow: {s.name}"
    )
    return service


class TestPrerequisiteEnforcement:
    """Tests for prerequisite checking in SkillsMiddleware."""

    @pytest.mark.asyncio
    async def test_skill_without_prerequisites_is_allowed(self):
        """A skill with no prerequisites should always be activated."""
        skill = _make_skill("basic-skill")
        service = _make_service(skills=[skill])

        middleware = SkillsMiddleware(
            skill_service=service,
            enforce_prerequisites=True,
        )

        ctx = _make_context("basic-skill")
        result = await middleware.process(ctx, AsyncMock(side_effect=lambda c: c))

        assert result.has_flag("skills_auto_activated")
        assert not result.has_flag("skills_blocked_by_prerequisites")

    @pytest.mark.asyncio
    async def test_skill_blocked_when_prerequisite_not_completed(self):
        """A skill should be blocked when its prerequisite is not done."""
        prereq = _make_skill("prereq-skill")
        dependent = _make_skill("dependent-skill", prerequisites=["prereq-skill"])
        service = _make_service(skills=[prereq, dependent], matches=[dependent])

        middleware = SkillsMiddleware(
            skill_service=service,
            enforce_prerequisites=True,
        )

        session = _make_session(completed_skills=[])
        ctx = _make_context("dependent-skill", session=session)
        result = await middleware.process(ctx, AsyncMock(side_effect=lambda c: c))

        assert result.has_flag("skills_blocked_by_prerequisites")
        assert not result.has_flag("skills_auto_activated")
        # blocked_skills section should list the blocked skill
        blocked_section = next(
            (s for s in result.context_sections if s.name == "blocked_skills"),
            None,
        )
        assert blocked_section is not None
        assert "dependent-skill" in blocked_section.content
        assert "prereq-skill" in blocked_section.content

    @pytest.mark.asyncio
    async def test_skill_allowed_when_prerequisite_completed(self):
        """A skill should activate when its prerequisites are done."""
        prereq = _make_skill("prereq-skill")
        dependent = _make_skill("dependent-skill", prerequisites=["prereq-skill"])
        service = _make_service(skills=[prereq, dependent], matches=[dependent])

        middleware = SkillsMiddleware(
            skill_service=service,
            enforce_prerequisites=True,
        )

        session = _make_session(completed_skills=["prereq-skill"])
        ctx = _make_context("dependent-skill", session=session)
        result = await middleware.process(ctx, AsyncMock(side_effect=lambda c: c))

        assert result.has_flag("skills_auto_activated")
        assert not result.has_flag("skills_blocked_by_prerequisites")

    @pytest.mark.asyncio
    async def test_multiple_prerequisites_all_must_be_met(self):
        """All prerequisites must be completed, not just some."""
        skill = _make_skill(
            "multi-prereq-skill",
            prerequisites=["step-one", "step-two"],
        )
        service = _make_service(skills=[skill])

        middleware = SkillsMiddleware(
            skill_service=service,
            enforce_prerequisites=True,
        )

        # Only one of two prerequisites completed
        session = _make_session(completed_skills=["step-one"])
        ctx = _make_context("multi-prereq-skill", session=session)
        result = await middleware.process(ctx, AsyncMock(side_effect=lambda c: c))

        assert result.has_flag("skills_blocked_by_prerequisites")

        # Both completed
        session2 = _make_session(completed_skills=["step-one", "step-two"])
        ctx2 = _make_context("multi-prereq-skill", session=session2)
        result2 = await middleware.process(ctx2, AsyncMock(side_effect=lambda c: c))

        assert result2.has_flag("skills_auto_activated")

    @pytest.mark.asyncio
    async def test_enforcement_disabled(self):
        """When enforce_prerequisites=False, skills activate regardless."""
        skill = _make_skill("gated-skill", prerequisites=["missing-prereq"])
        service = _make_service(skills=[skill])

        middleware = SkillsMiddleware(
            skill_service=service,
            enforce_prerequisites=False,
        )

        ctx = _make_context("gated-skill")
        result = await middleware.process(ctx, AsyncMock(side_effect=lambda c: c))

        assert result.has_flag("skills_auto_activated")
        assert not result.has_flag("skills_blocked_by_prerequisites")

    @pytest.mark.asyncio
    async def test_mixed_skills_partial_blocking(self):
        """When multiple skills match, only those with unmet prereqs are blocked."""
        allowed_skill = _make_skill("free-skill")
        blocked_skill = _make_skill("gated-skill", prerequisites=["missing"])
        service = _make_service(
            skills=[allowed_skill, blocked_skill],
            matches=[allowed_skill, blocked_skill],
        )

        middleware = SkillsMiddleware(
            skill_service=service,
            enforce_prerequisites=True,
            max_auto_skills=5,
        )

        session = _make_session(completed_skills=[])
        ctx = _make_context("do both", session=session)
        result = await middleware.process(ctx, AsyncMock(side_effect=lambda c: c))

        # One activated, one blocked
        assert result.has_flag("skills_auto_activated")
        assert result.has_flag("skills_blocked_by_prerequisites")


class TestMarkSkillCompleted:
    """Tests for the mark_skill_completed static method."""

    def test_mark_completed_in_session_state(self):
        """Marking a skill completed stores it in session state."""
        session = _make_session()
        ctx = _make_context(session=session)

        SkillsMiddleware.mark_skill_completed(ctx, "brainstorming")

        completed = session.state.get(COMPLETED_SKILLS_KEY)
        assert "brainstorming" in completed

    def test_mark_completed_idempotent(self):
        """Marking the same skill twice should not create duplicates."""
        session = _make_session()
        ctx = _make_context(session=session)

        SkillsMiddleware.mark_skill_completed(ctx, "brainstorming")
        SkillsMiddleware.mark_skill_completed(ctx, "brainstorming")

        completed = session.state.get(COMPLETED_SKILLS_KEY)
        assert completed.count("brainstorming") == 1

    def test_mark_completed_falls_back_to_metadata(self):
        """Without a session, completed skills go into context metadata."""
        ctx = _make_context()  # no session

        SkillsMiddleware.mark_skill_completed(ctx, "debugging")

        completed = ctx.get_metadata(COMPLETED_SKILLS_KEY)
        assert "debugging" in completed

    def test_multiple_skills_accumulated(self):
        """Multiple different skills can be marked completed."""
        session = _make_session()
        ctx = _make_context(session=session)

        SkillsMiddleware.mark_skill_completed(ctx, "brainstorming")
        SkillsMiddleware.mark_skill_completed(ctx, "writing-plans")

        completed = session.state.get(COMPLETED_SKILLS_KEY)
        assert "brainstorming" in completed
        assert "writing-plans" in completed
