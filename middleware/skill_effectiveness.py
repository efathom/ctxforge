"""
Skill Effectiveness Middleware.

Runs on the 'record' phase. Checks context flags for activated skills
and records usage. On session_complete, infers success/failure from
session observations.
"""
import logging
from typing import List

from ctxforge.core.skill import SkillScope
from ctxforge.engine.services.skill_effectiveness_service import (
    SkillEffectivenessService,
)
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction

logger = logging.getLogger(__name__)


class SkillEffectivenessMiddleware(BaseMiddleware):
    """Record skill usage and outcomes during the record phase."""

    def __init__(
        self,
        effectiveness_service: SkillEffectivenessService,
        default_scope: SkillScope = SkillScope.BASE,
        default_scope_id: str = "system",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._eff_service = effectiveness_service
        self._default_scope = default_scope
        self._default_scope_id = default_scope_id

    @property
    def name(self) -> str:
        return "skill_effectiveness"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """Record usage for activated skills and outcomes on session complete."""

        if context.has_flag("skills_auto_activated"):
            activated_names = self._extract_activated_names(context)
            for skill_name in activated_names:
                await self._eff_service.record_usage(
                    skill_name=skill_name,
                    scope=self._default_scope,
                    scope_id=self._default_scope_id,
                    confidence=0.8,
                    session_id=context.session_id,
                )
            logger.debug(
                "Recorded usage for %d activated skills", len(activated_names)
            )

        if context.has_flag("session_complete"):
            activated_names = self._extract_activated_names(context)
            success = not context.has_flag("session_failed")
            for skill_name in activated_names:
                await self._eff_service.record_outcome(
                    skill_name=skill_name,
                    scope=self._default_scope,
                    scope_id=self._default_scope_id,
                    success=success,
                )
            logger.debug(
                "Recorded outcome (success=%s) for %d skills",
                success, len(activated_names),
            )

        return await next(context)

    def _extract_activated_names(
        self, context: MiddlewareContext
    ) -> List[str]:
        """Extract activated skill names from context modifications."""
        mods = context.modifications.get("skills", [])
        names: List[str] = []
        for mod in mods:
            if isinstance(mod, dict):
                activated = mod.get("activated_skills", [])
                names.extend(activated)
        return names
