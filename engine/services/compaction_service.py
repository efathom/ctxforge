"""
Compaction service.

This service owns compaction logic and the related persistence side-effects.
It exists to keep `ctxforge` thin and to consolidate compaction behaviors
behind a stable API.
"""

from __future__ import annotations

import logging
from typing import Union

from ctxforge.compaction.view import (
    CompactionView,
    CondensationResult,
    ICondenser,
)
from ctxforge.config.base import EngineConfig
from ctxforge.core.session import Session
from ctxforge.engine.services.session_service import SessionService
from ctxforge.protocols.compactor import CompactionConfig

logger = logging.getLogger(__name__)


class CompactionService:
    """
    Owns condenser dependency and compaction policy wiring.

    Handles the conversion between Session and CompactionView,
    allowing condensers to work with immutable views while
    the service applies changes back to sessions.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        condenser: ICondenser,
        session_service: SessionService,
    ):
        self._cfg = config
        self._condenser = condenser
        self._sessions = session_service

    async def maybe_compact(self, *, session: Session) -> None:
        """Run compaction if needed (best-effort)."""
        try:
            config = CompactionConfig(
                event_threshold=self._cfg.compaction.event_threshold,
                token_threshold=self._cfg.compaction.token_threshold,
                keep_recent=self._cfg.compaction.keep_recent,
            )

            # Convert session to view
            view = CompactionView.from_session(session)

            if self._condenser.should_condense(view, config):
                result = await self._condenser.condense(view, config)
                self._apply_result_to_session(session, result)

                if isinstance(result, CondensationResult):
                    logger.debug(
                        f"Condensed {result.events_forgotten_count} events, "
                        f"saved ~{result.tokens_saved} tokens"
                    )
        except Exception as e:
            logger.warning(f"Compaction failed: {e}")

    async def compact_session(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> CondensationResult:
        """Manually trigger session compaction and persist on success."""
        session = await self._sessions.fetch(session_id=session_id, user_id=user_id)

        config = CompactionConfig(
            event_threshold=self._cfg.compaction.event_threshold,
            token_threshold=self._cfg.compaction.token_threshold,
            keep_recent=self._cfg.compaction.keep_recent,
        )

        # Convert session to view
        view = CompactionView.from_session(session)

        result = await self._condenser.condense(view, config)
        self._apply_result_to_session(session, result)

        # Persist changes
        await self._sessions.save(session)

        # Ensure we return a CondensationResult
        if isinstance(result, CompactionView):
            return CondensationResult(view=result)
        return result

    async def run_async_compaction(self, *, session: Session) -> None:
        """Run compaction in the background and persist on success (best-effort)."""
        try:
            config = CompactionConfig(
                event_threshold=self._cfg.compaction.event_threshold,
                token_threshold=self._cfg.compaction.token_threshold,
                keep_recent=self._cfg.compaction.keep_recent,
            )

            # Convert session to view
            view = CompactionView.from_session(session)

            if self._condenser.should_condense(view, config):
                result = await self._condenser.condense(view, config)
                self._apply_result_to_session(session, result)
                await self._sessions.save(session)
                logger.debug("Background compaction completed")
        except Exception as e:
            logger.warning(f"Background compaction failed: {e}")

    def _apply_result_to_session(
        self,
        session: Session,
        result: Union[CompactionView, CondensationResult],
    ) -> None:
        """
        Apply condensation result back to the session.

        Updates the session's events and summary based on the result.
        """
        if isinstance(result, CondensationResult):
            view = result.view
        else:
            view = result

        # Update session events
        session.events.clear()
        session.events.extend(view.to_context_events())

        # Update summary if changed
        if view.summary is not None:
            session.summary = view.summary