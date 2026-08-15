"""
Intent Notes Middleware.

Generates structured intent notes for newly recorded events and attaches them to
`Event.metadata.custom["intent_note"]` so they can be persisted with the session.

Designed to run in the record pipeline, ideally at the `record_pre_persist` phase
after new events are appended to the session but before the session is saved.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ctxforge.core.events import Event
from ctxforge.core.scoped_memory import MemoryCategory, MemoryScope
from ctxforge.engine.services.intent_note_service import IntentNoteService
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction

logger = logging.getLogger(__name__)


DEFAULT_SEEDS_KEY = "intent_note.functional_type_seeds"


class IntentNotesMiddleware(BaseMiddleware):
    """
    Record-pipeline middleware that attaches intent notes to recorded events.

    Expected flow (record_turn):
    - record_input_output: preprocess input/response (PII, etc.)
    - events appended to session
    - record_pre_persist: attach intent notes (this middleware)
    - session saved once
    - record_persisted: post-persist hooks
    """

    def __init__(
        self,
        *,
        intent_note_service: IntentNoteService,
        scoped_memory_service: Optional[ScopedMemoryService] = None,
        project_id: Optional[str] = None,
        enabled: bool = True,
        allow_overwrite: bool = False,
        generate_for_event_types: Optional[List[str]] = None,
        functional_seed_scoped_memory_key: str = DEFAULT_SEEDS_KEY,
        functional_seed_category: MemoryCategory = MemoryCategory.CONVENTION,
    ) -> None:
        super().__init__(enabled=enabled)
        self._svc = intent_note_service
        self._scoped_mem = scoped_memory_service
        self._project_id = project_id
        self._allow_overwrite = allow_overwrite
        self._event_types = set(generate_for_event_types or ["user", "agent"])
        self._seeds_key = functional_seed_scoped_memory_key
        self._seed_category = functional_seed_category

    @property
    def name(self) -> str:
        return "intent_notes"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next_fn: NextFunction,
    ) -> MiddlewareContext:
        phase = context.get_metadata("phase") or context.phase
        if phase != "record_pre_persist":
            return await next_fn(context)

        session = context.session
        if session is None:
            return await next_fn(context)

        if not self._svc.enabled:
            return await next_fn(context)

        recorded_ids = context.get_metadata("recorded_event_ids")
        target_events = self._select_target_events(session.events, recorded_ids)
        if not target_events:
            return await next_fn(context)

        functional_seeds = await self._load_functional_seeds()

        updated = False
        new_events: List[Event] = []
        target_ids = {e.event_id for e in target_events}
        for ev in session.events:
            replacement = ev
            if ev.event_id in target_ids:
                replacement = await self._maybe_attach_note(
                    session_events=session.events,
                    event=ev,
                    functional_seeds=functional_seeds,
                )
                if replacement is not ev:
                    updated = True
            new_events.append(replacement)

        if updated:
            session.events = new_events
            context.add_flag("intent_notes_attached")
            context.record_modification(self.name, {"attached": True, "count": len(target_events)})

        return await next_fn(context)

    def _select_target_events(
        self,
        events: List[Event],
        recorded_event_ids: Optional[object],
    ) -> List[Event]:
        # Prefer explicit recorded ids (set by TurnRecordingService)
        ids: List[str] = []
        if isinstance(recorded_event_ids, list):
            ids = [str(x) for x in recorded_event_ids if x]

        if ids:
            by_id = {e.event_id: e for e in events}
            return [by_id[i] for i in ids if i in by_id]

        # Fallback: last user/agent pair
        tail = events[-4:] if len(events) >= 4 else list(events)
        return [e for e in tail if e.type.value in self._event_types]

    async def _maybe_attach_note(
        self,
        *,
        session_events: List[Event],
        event: Event,
        functional_seeds: Optional[List[str]],
    ) -> Event:
        if event.type.value not in self._event_types:
            return event

        if not self._allow_overwrite and event.get_intent_note() is not None:
            return event

        # Provide recent context up to (but not including) this event
        idx = next((i for i, e in enumerate(session_events) if e.event_id == event.event_id), None)
        if idx is None:
            keep = self._svc.config.max_history_events_for_prompt
            recent = session_events[-keep:] if keep > 0 else []
        else:
            keep = self._svc.config.max_history_events_for_prompt
            recent = session_events[max(0, idx - keep):idx] if keep > 0 else []

        note = await self._svc.generate_for_event(
            event=event,
            recent_events=recent,
            functional_type_seeds=functional_seeds,
        )
        if note is None:
            return event
        return event.with_intent_note(note)

    async def _load_functional_seeds(self) -> Optional[List[str]]:
        if self._scoped_mem is None or not self._project_id:
            return None
        try:
            mem = await self._scoped_mem.get(MemoryScope.PROJECT, self._project_id, self._seeds_key)
        except Exception as e:
            logger.debug("Failed to load functional seeds from scoped memory: %s", e)
            return None
        if mem is None or not mem.content:
            return None
        seeds = [line.strip() for line in str(mem.content).splitlines() if line.strip()]
        return seeds or None

