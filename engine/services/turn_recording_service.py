from __future__ import annotations

"""
Turn recording service.

This service owns:
- record middleware pipeline execution
- event creation and session persistence
- scheduling background tasks (extraction, async compaction, graph ingestion)

It exists to keep `ctxforge` thin and to consolidate turn-recording behavior
behind a stable API.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Set, Tuple

from ctxforge.config.base import EngineConfig
from ctxforge.core.events import Event, EventFactory
from ctxforge.core.session import Session
from ctxforge.engine.services.compaction_service import CompactionService
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.engine.services.session_service import SessionService
from ctxforge.middleware import MiddlewareChain, MiddlewareContext

logger = logging.getLogger(__name__)


class TurnRecordingService:
    """Owns recording and background scheduling dependencies."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        session_service: SessionService,
        record_chain_provider: Callable[[], Optional[MiddlewareChain]],
        run_chain: Callable[[Optional[MiddlewareChain], MiddlewareContext], Awaitable[MiddlewareContext]],
        background_tasks: Set[asyncio.Task],
        extraction_enabled_provider: Callable[[], bool],
        run_extraction: Callable[[Session, str, str], Awaitable[None]],
        compaction_service_provider: Callable[[], Optional[CompactionService]],
        graph_service_provider: Callable[[], Optional[GraphService]],
    ):
        self._cfg = config
        self._sessions = session_service
        self._get_record_chain = record_chain_provider
        self._run_chain = run_chain
        self._background_tasks = background_tasks
        self._extraction_enabled = extraction_enabled_provider
        self._run_extraction = run_extraction
        self._get_compaction = compaction_service_provider
        self._get_graph = graph_service_provider

    async def record_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        pipeline_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        session = await self._sessions.fetch(session_id=session_id, user_id=user_id)

        mw_ctx = MiddlewareContext(
            user_input=user_input,
            agent_response=assistant_response,
            user_id=user_id,
            session_id=session_id,
            session=session,
        )
        if pipeline_metadata:
            for k, v in pipeline_metadata.items():
                mw_ctx.set_metadata(k, v)
        mw_ctx.phase = "record_input_output"
        mw_ctx = await self._run_chain(self._get_record_chain(), mw_ctx)

        processed_input = mw_ctx.processed_input or user_input
        processed_response = mw_ctx.processed_response or assistant_response

        user_event = EventFactory.user_message(processed_input)
        session.add_event(user_event)

        assistant_event = EventFactory.agent_message(
            content=processed_response,
            **(metadata or {}),
        )
        session.add_event(assistant_event)

        # Allow record middleware to annotate/transform the appended events before persistence.
        mw_ctx.phase = "record_pre_persist"
        mw_ctx.session = session
        mw_ctx.set_metadata("recorded_event_ids", [user_event.event_id, assistant_event.event_id])
        await self._run_chain(self._get_record_chain(), mw_ctx)

        await self._sessions.save(session)

        mw_ctx.phase = "record_persisted"
        mw_ctx.session = session
        await self._run_chain(self._get_record_chain(), mw_ctx)

        self._schedule_background_tasks(
            session=session,
            user_input=processed_input,
            assistant_response=processed_response,
        )

        logger.debug(f"Turn recorded for session {session_id}")

    async def record_user_message(
        self,
        *,
        session_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        session = await self._sessions.fetch(session_id=session_id, user_id=user_id)
        event = EventFactory.user_message(content)
        if metadata:
            event = event.with_metadata(custom=metadata)
        session.add_event(event)
        await self._sessions.save(session)
        return event

    async def record_assistant_message(
        self,
        *,
        session_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        session = await self._sessions.fetch(session_id=session_id, user_id=user_id)
        event = EventFactory.agent_message(content, **(metadata or {}))
        session.add_event(event)
        await self._sessions.save(session)
        return event

    async def record_tool_use(
        self,
        *,
        session_id: str,
        user_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: str,
        tool_call_id: Optional[str] = None,
    ) -> Tuple[Event, Event]:
        session = await self._sessions.fetch(session_id=session_id, user_id=user_id)

        call_event = EventFactory.tool_call(
            tool_name=tool_name,
            tool_args=tool_input,
        )
        session.add_event(call_event)

        output_event = EventFactory.tool_output(
            content=tool_output,
            tool_name=tool_name,
            result_type="success",
            parent_id=call_event.event_id,
        )
        if tool_call_id:
            output_event = output_event.with_metadata(custom={"tool_call_id": tool_call_id})
        session.add_event(output_event)

        await self._sessions.save(session)
        return call_event, output_event

    def _schedule_background_tasks(self, *, session: Session, user_input: str, assistant_response: str) -> None:
        """
        Schedule best-effort background processing tasks.

        Note: we intentionally preserve the engine's behavior:
        - extraction tasks are tracked only when async_processing=True
        - compaction and graph ingestion tasks are tracked when scheduled
        """
        # Memory extraction
        if self._cfg.extraction.enabled and self._extraction_enabled():
            if self._cfg.extraction.async_processing:
                task = asyncio.create_task(self._run_extraction(session, user_input, assistant_response))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                # Sync extraction (for testing)
                asyncio.create_task(self._run_extraction(session, user_input, assistant_response))

        # Async compaction
        compaction = self._get_compaction()
        if self._cfg.compaction.async_compaction and compaction is not None:
            task = asyncio.create_task(compaction.run_async_compaction(session=session))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Graph ingestion
        graph = self._get_graph()
        if getattr(self._cfg, "graph", None) is not None and getattr(self._cfg.graph, "enabled", False):
            if graph is not None:
                task = asyncio.create_task(
                    graph.ingest_turn(
                        session=session,
                        user_input=user_input,
                        assistant_response=assistant_response,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)


