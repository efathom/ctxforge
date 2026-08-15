"""Tests for ObservationExtractor."""

import json
from unittest.mock import AsyncMock

import pytest

from ctxforge.core.events import Event, EventType
from ctxforge.core.observation import ObservationType
from ctxforge.extraction.observation_extractor import ObservationExtractor


def _make_event(content, event_type=EventType.USER):
    return Event(type=event_type, content=content)


def _mock_llm(response_text):
    llm = AsyncMock()
    resp = AsyncMock()
    resp.content = response_text
    llm.chat = AsyncMock(return_value=resp)
    return llm


@pytest.mark.asyncio
async def test_extract_observations():
    llm = _mock_llm(json.dumps([
        {"type": "decision", "summary": "Use PostgreSQL", "confidence": 0.9},
        {"type": "bugfix", "summary": "Fixed null pointer", "detail": "In auth module"},
    ]))

    extractor = ObservationExtractor(llm)
    events = [_make_event("Let's use PostgreSQL"), _make_event("Fixed the null pointer bug")]

    observations = await extractor.extract(events)

    assert len(observations) == 2
    assert observations[0].type == ObservationType.DECISION
    assert observations[0].summary == "Use PostgreSQL"
    assert observations[1].type == ObservationType.BUGFIX
    assert observations[1].detail == "In auth module"


@pytest.mark.asyncio
async def test_extract_empty_events():
    llm = _mock_llm("[]")
    extractor = ObservationExtractor(llm)
    assert await extractor.extract([]) == []


@pytest.mark.asyncio
async def test_extract_handles_llm_error():
    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
    extractor = ObservationExtractor(llm)
    assert await extractor.extract([_make_event("test")]) == []


@pytest.mark.asyncio
async def test_summarize_session():
    llm = _mock_llm(json.dumps({
        "request": "Add auth",
        "investigated": "OAuth patterns",
        "learned": "JWT is simpler",
        "completed": "Added JWT auth",
        "next_steps": "Add refresh tokens",
    }))

    extractor = ObservationExtractor(llm)
    events = [_make_event("Add authentication")]

    report = await extractor.summarize_session(events)

    assert report.request == "Add auth"
    assert report.completed == "Added JWT auth"
    assert report.next_steps == "Add refresh tokens"


@pytest.mark.asyncio
async def test_summarize_empty_events():
    llm = _mock_llm("{}")
    extractor = ObservationExtractor(llm)
    report = await extractor.summarize_session([])
    assert report.request == ""
