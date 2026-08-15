from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.events import Event, EventMetadata, EventType
from ctxforge.core.session import Session
from ctxforge.engine.services.intent_note_service import IntentNoteService, IntentNoteServiceConfig
from ctxforge.engine.services.session_service import SessionService
from ctxforge.engine.services.turn_recording_service import TurnRecordingService
from ctxforge.middleware import MiddlewareChain, MiddlewareContext
from ctxforge.middleware.intent_notes import IntentNotesMiddleware
from ctxforge.protocols.llm import ILLMProvider, LLMResponse
from ctxforge.storage.memory.session import InMemorySessionStore


@dataclass
class MockLLM(ILLMProvider):
    responses: List[str]
    calls: int = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        idx = self.calls
        self.calls += 1
        content = self.responses[idx] if idx < len(self.responses) else self.responses[-1]
        model = kwargs.get("model") or self.default_model
        return LLMResponse(content=content, model=model)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text.split())

    def count_message_tokens(self, messages, model: Optional[str] = None) -> int:
        return 0


async def _run_chain(chain: Optional[MiddlewareChain], ctx: MiddlewareContext) -> MiddlewareContext:
    if chain is None:
        return ctx
    res = await chain.execute(ctx)
    return res.context


@pytest.mark.asyncio
async def test_intent_notes_middleware_attaches_notes_on_record_pre_persist() -> None:
    llm = MockLLM(
        responses=[
            json.dumps(
                {
                    "act": "ask",
                    "target": "postgres config",
                    "note_text": "User asks about POSTGRES_DB.",
                    "context_scope": None,
                    "event_types": [],
                    "functional_types": [],
                    "confidence": 0.9,
                    "source": "llm",
                }
            ),
            json.dumps(
                {
                    "act": "explain",
                    "target": "postgres config",
                    "note_text": "Assistant explains POSTGRES_DB usage.",
                    "context_scope": None,
                    "event_types": [],
                    "functional_types": [],
                    "confidence": 0.9,
                    "source": "llm",
                }
            ),
        ]
    )
    svc = IntentNoteService(llm_provider=llm, config=IntentNoteServiceConfig(min_content_length=1))
    mw = IntentNotesMiddleware(intent_note_service=svc)

    s = Session(user_id="u1")
    user_ev = Event(type=EventType.USER, content="What should POSTGRES_DB be?")
    agent_ev = Event(type=EventType.AGENT, content="Set it to your database name.")
    s.add_event(user_ev)
    s.add_event(agent_ev)

    ctx = MiddlewareContext(user_input="x", agent_response="y", session=s, user_id="u1", session_id="s1")
    ctx.phase = "record_pre_persist"
    ctx.set_metadata("recorded_event_ids", [user_ev.event_id, agent_ev.event_id])

    async def _next(c: MiddlewareContext) -> MiddlewareContext:
        return c

    out = await mw.process(ctx, _next)
    assert out.session is not None
    assert out.session.events[-2].get_intent_note() is not None
    assert out.session.events[-1].get_intent_note() is not None
    assert out.has_flag("intent_notes_attached")
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_intent_notes_middleware_does_not_overwrite_existing_note() -> None:
    llm = MockLLM(
        responses=[
            json.dumps(
                {
                    "act": "ask",
                    "target": None,
                    "note_text": "This should not be used.",
                    "context_scope": None,
                    "event_types": [],
                    "functional_types": [],
                    "confidence": 0.9,
                    "source": "llm",
                }
            )
        ]
    )
    svc = IntentNoteService(llm_provider=llm, config=IntentNoteServiceConfig(min_content_length=1))
    mw = IntentNotesMiddleware(intent_note_service=svc, allow_overwrite=False)

    existing_note = {"act": "ask", "target": None, "note_text": "Existing note."}
    ev = Event(
        type=EventType.USER,
        content="Hello there",
        metadata=EventMetadata(custom={"intent_note": existing_note}),
    )
    s = Session(user_id="u1", events=[ev])
    ctx = MiddlewareContext(user_input="x", agent_response="y", session=s, user_id="u1", session_id="s1")
    ctx.phase = "record_pre_persist"
    ctx.set_metadata("recorded_event_ids", [ev.event_id])

    async def _next(c: MiddlewareContext) -> MiddlewareContext:
        return c

    out = await mw.process(ctx, _next)
    assert out.session is not None
    assert out.session.events[0].get_intent_note() is not None
    assert out.session.events[0].get_intent_note().note_text == "Existing note."
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_turn_recording_service_persists_intent_notes() -> None:
    llm = MockLLM(
        responses=[
            json.dumps({"act": "ask", "target": None, "note_text": "User asks.", "confidence": 0.9}),
            json.dumps({"act": "explain", "target": None, "note_text": "Assistant explains.", "confidence": 0.9}),
        ]
    )
    svc = IntentNoteService(llm_provider=llm, config=IntentNoteServiceConfig(min_content_length=1))
    mw = IntentNotesMiddleware(intent_note_service=svc)
    chain = MiddlewareChain().add(mw)

    store = InMemorySessionStore()
    sessions = SessionService(session_store=store)

    tr = TurnRecordingService(
        config=DEFAULT_CONFIG,
        session_service=sessions,
        record_chain_provider=lambda: chain,
        run_chain=_run_chain,
        background_tasks=set(),  # type: ignore[arg-type]
        extraction_enabled_provider=lambda: False,
        run_extraction=lambda *_: None,  # type: ignore[return-value]
        compaction_service_provider=lambda: None,
        graph_service_provider=lambda: None,
    )

    await tr.record_turn(
        session_id="s1",
        user_id="u1",
        user_input="What is POSTGRES_DB?",
        assistant_response="It's the database name.",
    )

    persisted = await sessions.fetch(session_id="s1", user_id="u1")
    assert persisted.events[-2].get_intent_note() is not None
    assert persisted.events[-1].get_intent_note() is not None

