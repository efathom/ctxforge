"""
Session observation middleware.

Runs on the ``record`` phase.  When it detects session completion
(via ``session_complete=True`` in metadata), it extracts structured
observations from the session events and saves them as scoped memories
at the PROJECT scope.
"""

import uuid
from typing import Optional

from ctxforge.core.observation import ObservationType
from ctxforge.core.scoped_memory import MemoryCategory, MemoryScope, ScopedMemory
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.extraction.observation_extractor import ObservationExtractor
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction

_OBS_TYPE_TO_CATEGORY = {
    ObservationType.DECISION: MemoryCategory.DECISION,
    ObservationType.BUGFIX: MemoryCategory.BUGFIX,
    ObservationType.DISCOVERY: MemoryCategory.DISCOVERY,
    ObservationType.FEATURE: MemoryCategory.FEATURE,
    ObservationType.REFACTOR: MemoryCategory.REFACTOR,
    ObservationType.CHANGE: MemoryCategory.CONTEXT,
}


class SessionObservationMiddleware(BaseMiddleware):
    """Extract and persist structured observations at session end."""

    def __init__(
        self,
        observation_extractor: ObservationExtractor,
        scoped_memory_service: ScopedMemoryService,
        project_id: Optional[str] = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._extractor = observation_extractor
        self._scoped = scoped_memory_service
        self._project_id = project_id or "default"

    @property
    def name(self) -> str:
        return "session_observation"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next_fn: NextFunction,
    ) -> MiddlewareContext:
        result = await next_fn(context)

        # Only run on record phase when session is marked complete.
        if context.phase != "record":
            return result
        if not context.get_metadata("session_complete", False):
            return result

        session = context.session
        if session is None:
            return result

        events = list(session.events)
        if not events:
            return result

        observations = await self._extractor.extract(events)
        if not observations:
            return result

        scoped_memories = []
        for obs in observations:
            category = _OBS_TYPE_TO_CATEGORY.get(obs.type, MemoryCategory.CONTEXT)
            obs_id = str(uuid.uuid4())
            scoped_memories.append(ScopedMemory(
                id=obs_id,
                scope=MemoryScope.PROJECT,
                scope_id=self._project_id,
                category=category,
                key=f"obs_{obs.type.value}_{obs_id[:8]}",
                content=obs.summary,
                metadata={
                    "detail": obs.detail or "",
                    "observation_type": obs.type.value,
                    "confidence": obs.confidence,
                    "source_session_id": context.session_id or "",
                },
            ))

        if scoped_memories:
            saved = await self._scoped.save_observations(scoped_memories)
            context.set_metadata("observations_saved", saved)

        return result
