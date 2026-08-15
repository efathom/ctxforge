"""Tests for SkillInheritanceService."""

import pytest

from ctxforge.config.base import SkillInheritanceConfig
from ctxforge.core.skill import Skill, SkillScope
from ctxforge.engine.services.skill_inheritance_service import (
    SkillInheritanceService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.storage.memory.skill import InMemorySkillStore

# ---- helpers ---------------------------------------------------------------


def _make_skill(
    name: str,
    scope: SkillScope,
    scope_id: str,
    effectiveness: dict | None = None,
) -> Skill:
    return Skill(
        name=name,
        description=f"Use when needing {name}",
        scope=scope,
        scope_id=scope_id,
        content=f"# {name}\nDo the thing.",
        effectiveness=effectiveness,
    )


async def _seed_store(store: InMemorySkillStore) -> None:
    """Populate a store with a known set of skills across scopes."""
    await store.save(_make_skill("base-one", SkillScope.BASE, "system"))
    await store.save(_make_skill("base-two", SkillScope.BASE, "system"))
    await store.save(_make_skill("user-one", SkillScope.USER, "alice"))
    await store.save(_make_skill("user-two", SkillScope.USER, "alice"))
    await store.save(_make_skill("proj-one", SkillScope.PROJECT, "proj-x"))


def _make_service(
    store: InMemorySkillStore,
    config: SkillInheritanceConfig | None = None,
) -> SkillInheritanceService:
    return SkillInheritanceService(
        skill_store=store,
        config=config or SkillInheritanceConfig(enabled=True),
    )


# ---- inheritance visibility tests ------------------------------------------


class TestInheritedSkillVisibility:

    @pytest.mark.asyncio
    async def test_base_scope_sees_own_skills_only(self):
        store = InMemorySkillStore()
        await _seed_store(store)
        svc = _make_service(store)

        skills = await svc.get_inherited_skills(
            scope=SkillScope.BASE, scope_id="system",
        )
        names = {s.name for s in skills}
        assert names == {"base-one", "base-two"}

    @pytest.mark.asyncio
    async def test_user_scope_inherits_base(self):
        store = InMemorySkillStore()
        await _seed_store(store)
        svc = _make_service(store)

        skills = await svc.get_inherited_skills(
            scope=SkillScope.USER, scope_id="alice",
        )
        names = {s.name for s in skills}
        assert "base-one" in names
        assert "base-two" in names
        assert "user-one" in names
        assert "user-two" in names

    @pytest.mark.asyncio
    async def test_project_scope_inherits_user_and_base(self):
        store = InMemorySkillStore()
        await _seed_store(store)
        svc = _make_service(store)

        skills = await svc.get_inherited_skills(
            scope=SkillScope.PROJECT,
            scope_id="proj-x",
            user_id="alice",
        )
        names = {s.name for s in skills}
        assert names == {
            "base-one", "base-two",
            "user-one", "user-two",
            "proj-one",
        }

    @pytest.mark.asyncio
    async def test_name_collision_higher_scope_wins(self):
        store = InMemorySkillStore()
        # base skill and project skill with the same name
        await store.save(_make_skill("shared", SkillScope.BASE, "system"))
        await store.save(_make_skill("shared", SkillScope.PROJECT, "proj-x"))

        svc = _make_service(store)
        skills = await svc.get_inherited_skills(
            scope=SkillScope.PROJECT,
            scope_id="proj-x",
        )
        shared = [s for s in skills if s.name == "shared"]
        assert len(shared) == 1
        assert shared[0].scope == SkillScope.PROJECT

    @pytest.mark.asyncio
    async def test_no_inheritance_when_disabled(self):
        """When inheritance is disabled, get_inherited_skills still works
        but the service config flag should be checked by callers."""
        store = InMemorySkillStore()
        await _seed_store(store)
        config = SkillInheritanceConfig(enabled=False)
        svc = _make_service(store, config=config)
        assert svc.config.enabled is False


# ---- graduation tests ------------------------------------------------------


class TestGraduation:

    @pytest.mark.asyncio
    async def test_graduate_skill_copies_to_target_scope(self):
        store = InMemorySkillStore()
        await store.save(_make_skill("my-skill", SkillScope.USER, "alice"))

        svc = _make_service(store)
        result = await svc.graduate_skill(
            name="my-skill",
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            to_scope=SkillScope.BASE,
            to_scope_id="system",
        )
        assert result is not None
        assert result.scope == SkillScope.BASE
        assert result.scope_id == "system"

        # Verify it exists in the store at the new scope
        stored = await store.get("my-skill", SkillScope.BASE, "system")
        assert stored is not None

    @pytest.mark.asyncio
    async def test_graduate_skill_preserves_provenance(self):
        store = InMemorySkillStore()
        await store.save(_make_skill("my-skill", SkillScope.USER, "alice"))

        svc = _make_service(store)
        result = await svc.graduate_skill(
            name="my-skill",
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            to_scope=SkillScope.BASE,
            to_scope_id="system",
        )
        assert result.source_scope == SkillScope.USER
        assert result.source_scope_id == "alice"

    @pytest.mark.asyncio
    async def test_graduate_skill_sets_promoted_fields(self):
        store = InMemorySkillStore()
        await store.save(_make_skill("my-skill", SkillScope.USER, "alice"))

        svc = _make_service(store)
        result = await svc.graduate_skill(
            name="my-skill",
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            to_scope=SkillScope.PROJECT,
            to_scope_id="proj-x",
        )
        assert result.promoted_from == "user"
        assert result.promoted_at is not None

    @pytest.mark.asyncio
    async def test_graduate_skill_original_remains(self):
        store = InMemorySkillStore()
        await store.save(_make_skill("my-skill", SkillScope.USER, "alice"))

        svc = _make_service(store)
        await svc.graduate_skill(
            name="my-skill",
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            to_scope=SkillScope.BASE,
            to_scope_id="system",
        )
        # Original still exists
        original = await store.get("my-skill", SkillScope.USER, "alice")
        assert original is not None

    @pytest.mark.asyncio
    async def test_graduate_skill_not_found(self):
        store = InMemorySkillStore()
        svc = _make_service(store)

        result = await svc.graduate_skill(
            name="nonexistent",
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            to_scope=SkillScope.BASE,
            to_scope_id="system",
        )
        assert result is None


# ---- graduation candidates tests ------------------------------------------


class TestGraduationCandidates:

    @pytest.mark.asyncio
    async def test_graduation_candidates_filters_by_usage(self):
        store = InMemorySkillStore()
        await store.save(_make_skill(
            "high-usage", SkillScope.USER, "alice",
            effectiveness={"usage_count": 10, "success_rate": 0.9},
        ))
        await store.save(_make_skill(
            "low-usage", SkillScope.USER, "alice",
            effectiveness={"usage_count": 2, "success_rate": 0.9},
        ))

        svc = _make_service(store)
        candidates = await svc.get_graduation_candidates(
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            min_usage_count=5,
            min_success_rate=0.8,
        )
        names = [c.name for c in candidates]
        assert "high-usage" in names
        assert "low-usage" not in names

    @pytest.mark.asyncio
    async def test_graduation_candidates_filters_by_success_rate(self):
        store = InMemorySkillStore()
        await store.save(_make_skill(
            "high-rate", SkillScope.USER, "alice",
            effectiveness={"usage_count": 10, "success_rate": 0.95},
        ))
        await store.save(_make_skill(
            "low-rate", SkillScope.USER, "alice",
            effectiveness={"usage_count": 10, "success_rate": 0.3},
        ))

        svc = _make_service(store)
        candidates = await svc.get_graduation_candidates(
            from_scope=SkillScope.USER,
            from_scope_id="alice",
            min_usage_count=5,
            min_success_rate=0.8,
        )
        names = [c.name for c in candidates]
        assert "high-rate" in names
        assert "low-rate" not in names

    @pytest.mark.asyncio
    async def test_graduation_candidates_empty_when_none_qualify(self):
        store = InMemorySkillStore()
        await store.save(_make_skill(
            "mediocre", SkillScope.USER, "alice",
            effectiveness={"usage_count": 1, "success_rate": 0.5},
        ))

        svc = _make_service(store)
        candidates = await svc.get_graduation_candidates(
            from_scope=SkillScope.USER,
            from_scope_id="alice",
        )
        assert candidates == []


# ---- provenance round-trip tests -------------------------------------------


class TestProvenanceRoundTrip:

    def test_provenance_round_trip_to_dict(self):
        from datetime import datetime
        skill = _make_skill("test-prov", SkillScope.PROJECT, "proj-x")
        skill.source_scope = SkillScope.USER
        skill.source_scope_id = "alice"
        skill.source_context = "project:acme/api-v2"
        skill.promoted_from = "user"
        skill.promoted_at = datetime(2025, 1, 15, 12, 0, 0)

        d = skill.to_dict()
        assert d["source_scope"] == "user"
        assert d["source_scope_id"] == "alice"
        assert d["source_context"] == "project:acme/api-v2"
        assert d["promoted_from"] == "user"
        assert "2025-01-15" in d["promoted_at"]

    def test_provenance_round_trip_from_dict(self):
        from datetime import datetime
        skill = _make_skill("test-prov", SkillScope.PROJECT, "proj-x")
        skill.source_scope = SkillScope.USER
        skill.source_scope_id = "alice"
        skill.promoted_from = "user"
        skill.promoted_at = datetime(2025, 1, 15, 12, 0, 0)

        d = skill.to_dict()
        restored = Skill.from_dict(d)
        assert restored.source_scope == SkillScope.USER
        assert restored.source_scope_id == "alice"
        assert restored.promoted_from == "user"
        assert restored.promoted_at == datetime(2025, 1, 15, 12, 0, 0)

    def test_provenance_none_round_trip(self):
        skill = _make_skill("no-prov", SkillScope.BASE, "system")
        d = skill.to_dict()
        assert d["source_scope"] is None
        assert d["promoted_at"] is None

        restored = Skill.from_dict(d)
        assert restored.source_scope is None
        assert restored.promoted_at is None


# ---- integration with SkillService -----------------------------------------


class TestSkillServiceInheritance:

    @pytest.mark.asyncio
    async def test_inherited_skills_in_prompt_index(self):
        store = InMemorySkillStore()
        await store.save(_make_skill("base-skill", SkillScope.BASE, "system"))
        await store.save(
            _make_skill("user-skill", SkillScope.USER, "alice"),
        )
        await store.save(
            _make_skill("proj-skill", SkillScope.PROJECT, "proj-x"),
        )

        inheritance_svc = _make_service(store)
        skill_svc = SkillService(
            store=store,
            inheritance_service=inheritance_svc,
        )

        # Without inheritance: uses default store layering
        default_skills = await skill_svc.get_available_skills(
            user_id="alice", project_id="proj-x",
            include_inherited=False,
        )

        # With inheritance: uses SkillInheritanceService
        inherited_skills = await skill_svc.get_available_skills(
            user_id="alice", project_id="proj-x",
            include_inherited=True,
        )

        default_names = {s.name for s in default_skills}
        inherited_names = {s.name for s in inherited_skills}

        # Both should contain all three skills (no shadowing in this case)
        assert "base-skill" in default_names
        assert "base-skill" in inherited_names
        assert "user-skill" in inherited_names
        assert "proj-skill" in inherited_names
