"""
Skill Relationship Middleware.

Triggers automatic relationship analysis when new skills are generated.
Runs after SkillGenerationMiddleware in the record chain.
Discovered relationships are persisted via ISkillStore.save_relationships().
"""
import logging

from ctxforge.engine.services.skill_relationship_service import (
    SkillRelationshipService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction

logger = logging.getLogger(__name__)


class SkillRelationshipMiddleware(BaseMiddleware):
    """Trigger relationship analysis when skills_generated flag is set.

    Should be placed after SkillGenerationMiddleware in the chain.
    When the ``skills_generated`` flag is present and there are at least
    two skills in the store, triggers relationship analysis and persists
    results via ``ISkillStore.save_relationships()``.
    """

    def __init__(
        self,
        skill_service: SkillService,
        relationship_service: SkillRelationshipService,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._skill_service = skill_service
        self._relationship_service = relationship_service

    @property
    def name(self) -> str:
        return "skill_relationship"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """Analyze relationships if skills were generated."""
        result = await next(context)

        if not result.has_flag("skills_generated"):
            return result

        generated_names = result.get_metadata("generated_skill_names", [])
        if not generated_names:
            return result

        try:
            all_skills = await self._skill_service.get_available_skills()
            if len(all_skills) >= 2:
                relationships = (
                    await self._relationship_service.analyze_relationships(
                        all_skills,
                    )
                )
                result.add_flag("relationships_analyzed")
                result.set_metadata(
                    "relationships_count", len(relationships),
                )
                logger.info(
                    "Discovered %d relationships after skill generation",
                    len(relationships),
                )
        except Exception as exc:
            logger.warning("Relationship analysis failed: %s", exc)
            result.set_metadata("relationship_analysis_error", str(exc))

        return result
