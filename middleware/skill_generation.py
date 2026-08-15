"""
Skill Generation Middleware.

Runs on the 'record' phase after SessionObservationMiddleware.
When a session completes with enough events and observations,
attempts to generate reusable skills.
"""
import logging
from typing import List, Optional

from ctxforge.config.base import SkillGenerationConfig
from ctxforge.core.events import Event
from ctxforge.engine.services.skill_generator_service import (
    SkillGeneratorService,
)
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction

logger = logging.getLogger(__name__)


class SkillGenerationMiddleware(BaseMiddleware):
    """Generate skills from completed sessions during the record phase."""

    def __init__(
        self,
        generator_service: SkillGeneratorService,
        config: Optional[SkillGenerationConfig] = None,
        project_id: str = "default",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._generator = generator_service
        self._config = config or SkillGenerationConfig()
        self._project_id = project_id

    @property
    def name(self) -> str:
        return "skill_generation"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """Attempt skill generation on session_complete."""

        if not context.has_flag("session_complete"):
            return await next(context)

        if not self._config.auto_generate_from_sessions:
            return await next(context)

        events = self._extract_events(context)
        if len(events) < self._config.min_session_events:
            logger.debug(
                "Session has %d events, need %d for skill generation",
                len(events), self._config.min_session_events,
            )
            return await next(context)

        try:
            skills = await self._generator.generate_from_session(
                events, self._project_id
            )
            if skills:
                context.add_flag("skills_generated")
                context.set_metadata(
                    "generated_skill_names",
                    [s.name for s in skills],
                )
                logger.info(
                    "Generated %d skills from session", len(skills)
                )
        except Exception as exc:
            logger.warning("Skill generation failed: %s", exc)
            context.set_metadata("skill_generation_error", str(exc))

        return await next(context)

    def _extract_events(self, context: MiddlewareContext) -> List[Event]:
        """Extract events from the context session."""
        if context.session and hasattr(context.session, "events"):
            return list(context.session.events)
        return []
