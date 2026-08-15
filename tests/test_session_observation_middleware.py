"""Tests for SessionObservationMiddleware."""

import json
from unittest.mock import AsyncMock

import pytest

from ctxforge.core.events import Event, EventType
from ctxforge.core.scoped_memory import MemoryCategory
from ctxforge.core.session import Session
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.extraction.observation_extractor import ObservationExtractor
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.session_observation import SessionObservationMiddleware
from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore


def _mock_llm(response_text):
    llm = AsyncMock()
    resp = AsyncMock()
    resp.content = response_text
    llm.chat = AsyncMock(return_value=resp)
    return llm


async def _passthrough(ctx):
    return ctx


@pytest.mark.asyncio
async def test_middleware_saves_observations_on_session_complete():
    llm = _mock_llm(json.dumps([
        {"type": "decision", "summary": "Use Redis for caching", "confidence": 0.9},
        {"type": "discovery", "summary": "Found race condition in auth", "confidence": 0.8},
    ]))

    store = InMemoryScopedMemoryStore()
    scoped_svc = ScopedMemoryService(store)
    extractor = ObservationExtractor(llm)

    mw = SessionObservationMiddleware(
        observation_extractor=extractor,
        scoped_memory_service=scoped_svc,
        project_id="proj1",
    )

    session = Session(session_id="s1", user_id="u1")
    session.add_event(Event(type=EventType.USER, content="Let's use Redis"))
    session.add_event(Event(type=EventType.AGENT, content="Good idea"))

    ctx = MiddlewareContext(
        user_input="done",
        agent_response="Session complete",
        session=session,
        user_id="u1",
        session_id="s1",
        phase="record",
        metadata={"session_complete": True},
    )

    result = await mw.process(ctx, _passthrough)

    assert result.get_metadata("observations_saved") == 2

    # Verify scoped memories were saved
    from ctxforge.core.scoped_memory import ScopedMemoryQuery
    memories = await store.query(ScopedMemoryQuery(
        project_id="proj1",
        include_global=False,
        include_session=False,
    ))
    assert len(memories) == 2
    categories = {m.category for m in memories}
    assert MemoryCategory.DECISION in categories
    assert MemoryCategory.DISCOVERY in categories


@pytest.mark.asyncio
async def test_middleware_skips_non_record_phase():
    llm = _mock_llm("[]")
    store = InMemoryScopedMemoryStore()
    scoped_svc = ScopedMemoryService(store)
    extractor = ObservationExtractor(llm)

    mw = SessionObservationMiddleware(
        observation_extractor=extractor,
        scoped_memory_service=scoped_svc,
    )

    ctx = MiddlewareContext(
        user_input="test",
        phase="prepare",
        metadata={"session_complete": True},
    )

    result = await mw.process(ctx, _passthrough)
    assert result.get_metadata("observations_saved") is None


@pytest.mark.asyncio
async def test_middleware_skips_when_session_not_complete():
    llm = _mock_llm("[]")
    store = InMemoryScopedMemoryStore()
    scoped_svc = ScopedMemoryService(store)
    extractor = ObservationExtractor(llm)

    mw = SessionObservationMiddleware(
        observation_extractor=extractor,
        scoped_memory_service=scoped_svc,
    )

    session = Session(session_id="s1", user_id="u1")
    ctx = MiddlewareContext(
        user_input="test",
        session=session,
        phase="record",
        metadata={},  # no session_complete flag
    )

    result = await mw.process(ctx, _passthrough)
    assert result.get_metadata("observations_saved") is None
