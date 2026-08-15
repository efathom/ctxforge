"""
Skill Effectiveness Service.

Tracks skill usage and outcomes to build effectiveness metrics
that feed into ranking.
"""
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ctxforge.config.base import SkillEffectivenessConfig
from ctxforge.core.skill import SkillScope
from ctxforge.protocols.skill import ISkillStore

logger = logging.getLogger(__name__)

DEFAULT_EFFECTIVENESS: Dict[str, Any] = {
    "usage_count": 0,
    "success_count": 0,
    "failure_count": 0,
    "success_rate": 0.0,
    "avg_confidence_at_match": 0.0,
    "last_used_at": None,
    "sessions_used_in": [],
}


class SkillEffectivenessService:
    """Track and query skill effectiveness metrics."""

    def __init__(
        self,
        skill_store: ISkillStore,
        config: Optional[SkillEffectivenessConfig] = None,
    ):
        self._store = skill_store
        self._config = config or SkillEffectivenessConfig()

    async def record_usage(
        self,
        skill_name: str,
        scope: SkillScope,
        scope_id: str,
        confidence: float,
        session_id: Optional[str] = None,
    ) -> None:
        """Record that a skill was used.

        Increments usage_count, updates avg_confidence_at_match as a
        running average, and sets last_used_at.

        Args:
            skill_name: Name of the skill.
            scope: Skill scope.
            scope_id: Scope identifier.
            confidence: Confidence score at match time.
            session_id: Optional session identifier.
        """
        skill = await self._store.get(skill_name, scope, scope_id)
        if skill is None:
            logger.debug("Skill '%s' not found for usage recording", skill_name)
            return

        eff = dict(skill.effectiveness or DEFAULT_EFFECTIVENESS)
        old_count = eff.get("usage_count", 0)
        new_count = old_count + 1
        old_avg = eff.get("avg_confidence_at_match", 0.0)
        new_avg = (old_avg * old_count + confidence) / new_count

        eff["usage_count"] = new_count
        eff["avg_confidence_at_match"] = round(new_avg, 4)
        eff["last_used_at"] = datetime.now().isoformat()

        if session_id:
            sessions = list(eff.get("sessions_used_in", []))
            if session_id not in sessions:
                sessions.append(session_id)
            eff["sessions_used_in"] = sessions

        await self._store.update_effectiveness(
            skill_name, scope, scope_id, eff
        )

    async def record_outcome(
        self,
        skill_name: str,
        scope: SkillScope,
        scope_id: str,
        success: bool,
    ) -> None:
        """Record the outcome of a skill usage.

        Increments success_count or failure_count and recomputes
        success_rate.

        Args:
            skill_name: Name of the skill.
            scope: Skill scope.
            scope_id: Scope identifier.
            success: Whether the skill usage was successful.
        """
        skill = await self._store.get(skill_name, scope, scope_id)
        if skill is None:
            logger.debug("Skill '%s' not found for outcome recording", skill_name)
            return

        eff = dict(skill.effectiveness or DEFAULT_EFFECTIVENESS)

        if success:
            eff["success_count"] = eff.get("success_count", 0) + 1
        else:
            eff["failure_count"] = eff.get("failure_count", 0) + 1

        total = eff.get("success_count", 0) + eff.get("failure_count", 0)
        if total > 0:
            eff["success_rate"] = round(eff.get("success_count", 0) / total, 4)
        else:
            eff["success_rate"] = 0.0

        await self._store.update_effectiveness(
            skill_name, scope, scope_id, eff
        )

    async def get_effectiveness(
        self,
        skill_name: str,
        scope: SkillScope,
        scope_id: str,
    ) -> Dict[str, Any]:
        """Get effectiveness metrics for a skill.

        Args:
            skill_name: Name of the skill.
            scope: Skill scope.
            scope_id: Scope identifier.

        Returns:
            Effectiveness metrics dict (or defaults if not tracked).
        """
        skill = await self._store.get(skill_name, scope, scope_id)
        if skill is None or skill.effectiveness is None:
            return dict(DEFAULT_EFFECTIVENESS)
        return dict(skill.effectiveness)

    async def get_ranking_boost(
        self,
        skill_name: str,
        scope: SkillScope,
        scope_id: str,
    ) -> float:
        """Get a ranking boost factor based on effectiveness.

        Returns 0.0 for skills with no usage data, and a value
        proportional to success_rate for skills with usage.

        Args:
            skill_name: Name of the skill.
            scope: Skill scope.
            scope_id: Scope identifier.

        Returns:
            A float between 0.0 and 1.0.
        """
        eff = await self.get_effectiveness(skill_name, scope, scope_id)
        usage_count = eff.get("usage_count", 0)
        if usage_count == 0:
            return 0.0
        success_rate = eff.get("success_rate", 0.0)
        return round(success_rate * self._config.weight_in_ranking, 4)
