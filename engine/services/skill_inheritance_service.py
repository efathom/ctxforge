"""
Skill Inheritance Service.

Manages cross-scope skill visibility and graduation.
Skills flow downward through the scope hierarchy (BASE -> USER -> PROJECT)
and can be promoted upward when they demonstrate sufficient effectiveness.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

from ctxforge.config.base import SkillInheritanceConfig
from ctxforge.core.skill import Skill, SkillMetadata, SkillScope
from ctxforge.protocols.skill import ISkillStore

logger = logging.getLogger(__name__)


class SkillInheritanceService:
    """Manages cross-scope skill inheritance and graduation.

    Inheritance chain:
      PROJECT sees: own skills + USER skills (if user_id) + BASE skills
      USER sees: own skills + BASE skills
      BASE sees: own skills only

    Name collisions are resolved by scope priority (PROJECT > USER > BASE).
    """

    def __init__(
        self,
        *,
        skill_store: ISkillStore,
        config: Optional[SkillInheritanceConfig] = None,
    ):
        self._store = skill_store
        self._config = config or SkillInheritanceConfig()

    @property
    def config(self) -> SkillInheritanceConfig:
        return self._config

    async def get_inherited_skills(
        self,
        scope: SkillScope,
        scope_id: str,
        user_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Return skills visible to this scope through inheritance.

        Inherited skills are additive: a project sees ALL user skills,
        not just those with non-colliding names. Name collisions are
        resolved by scope priority (PROJECT > USER > BASE).

        Args:
            scope: The requesting scope.
            scope_id: The requesting scope's identifier.
            user_id: User ID (needed when scope is PROJECT to find
                     user-level skills).

        Returns:
            De-duplicated list of SkillMetadata with higher scopes
            winning name collisions.
        """
        skills_by_name: Dict[str, SkillMetadata] = {}

        # BASE skills are always visible
        base_skills = await self._store.list_metadata(
            SkillScope.BASE, "system",
        )
        for s in base_skills:
            skills_by_name[s.name] = s

        # USER skills visible to USER and PROJECT scopes
        if scope in (SkillScope.USER, SkillScope.PROJECT):
            uid = scope_id if scope == SkillScope.USER else user_id
            if uid:
                user_skills = await self._store.list_metadata(
                    SkillScope.USER, uid,
                )
                for s in user_skills:
                    skills_by_name[s.name] = s

        # PROJECT skills visible only to PROJECT scope
        if scope == SkillScope.PROJECT:
            project_skills = await self._store.list_metadata(
                SkillScope.PROJECT, scope_id,
            )
            for s in project_skills:
                skills_by_name[s.name] = s

        return list(skills_by_name.values())

    async def graduate_skill(
        self,
        name: str,
        from_scope: SkillScope,
        from_scope_id: str,
        to_scope: SkillScope,
        to_scope_id: str,
    ) -> Optional[Skill]:
        """Promote a skill from a lower scope to a higher scope.

        Copies the skill to the target scope, preserving provenance.
        The original skill remains in the source scope.

        Args:
            name: Skill name.
            from_scope: Source scope.
            from_scope_id: Source scope identifier.
            to_scope: Target scope.
            to_scope_id: Target scope identifier.

        Returns:
            The graduated skill in the new scope, or None if the
            source skill was not found.
        """
        source_skill = await self._store.get(name, from_scope, from_scope_id)
        if source_skill is None:
            logger.warning(
                "Cannot graduate skill '%s': not found in %s/%s",
                name, from_scope.value, from_scope_id,
            )
            return None

        graduated = deepcopy(source_skill)
        graduated.scope = to_scope
        graduated.scope_id = to_scope_id
        graduated.updated_at = datetime.now()

        # Preserve provenance: track where the skill originated
        if graduated.source_scope is None:
            graduated.source_scope = from_scope
            graduated.source_scope_id = from_scope_id
        graduated.promoted_from = from_scope.value
        graduated.promoted_at = datetime.now()

        await self._store.save(graduated)

        logger.info(
            "Graduated skill '%s' from %s/%s to %s/%s",
            name,
            from_scope.value, from_scope_id,
            to_scope.value, to_scope_id,
        )
        return graduated

    async def get_graduation_candidates(
        self,
        from_scope: SkillScope,
        from_scope_id: str,
        min_usage_count: Optional[int] = None,
        min_success_rate: Optional[float] = None,
    ) -> List[Skill]:
        """Find skills that meet graduation criteria.

        Returns skills in the source scope whose effectiveness metrics
        exceed the configured thresholds.

        Args:
            from_scope: Scope to search for candidates.
            from_scope_id: Scope identifier.
            min_usage_count: Override config min_usage_count.
            min_success_rate: Override config min_success_rate.

        Returns:
            List of skills that qualify for graduation.
        """
        usage_threshold = (
            min_usage_count
            if min_usage_count is not None
            else self._config.graduation.min_usage_count
        )
        rate_threshold = (
            min_success_rate
            if min_success_rate is not None
            else self._config.graduation.min_success_rate
        )

        metadata_list = await self._store.list_metadata(
            from_scope, from_scope_id,
        )

        candidates: List[Skill] = []
        for meta in metadata_list:
            skill = await self._store.get(
                meta.name, from_scope, from_scope_id,
            )
            if skill is None:
                continue

            eff = skill.effectiveness or {}
            usage_count = eff.get("usage_count", 0)
            success_rate = eff.get("success_rate", 0.0)

            if (
                usage_count >= usage_threshold
                and success_rate >= rate_threshold
            ):
                candidates.append(skill)

        return candidates
