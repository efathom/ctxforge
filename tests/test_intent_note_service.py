from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from ctxforge.core.events import Event, EventType
from ctxforge.engine.services.intent_note_service import IntentNoteService, IntentNoteServiceConfig
from ctxforge.protocols.llm import ILLMProvider, LLMResponse


@dataclass
class MockLLM(ILLMProvider):
    responses: List[str]
    _i: int = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        idx = self._i
        self._i += 1
        content = self.responses[idx] if idx < len(self.responses) else self.responses[-1]
        return LLMResponse(content=content, model=model or self.default_model)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text.split())

    def count_message_tokens(self, messages, model: Optional[str] = None) -> int:
        return 0


@pytest.mark.asyncio
async def test_intent_note_service_parses_json_object() -> None:
    llm = MockLLM(
        responses=[
            json.dumps(
                {
                    "act": "ask",
                    "target": "postgres config",
                    "note_text": "User asks which POSTGRES_DB value to use.",
                    "context_scope": "db setup",
                    "event_types": ["configuration"],
                    "functional_types": ["requirement"],
                    "confidence": 0.9,
                    "source": "llm",
                }
            )
        ]
    )
    svc = IntentNoteService(llm, config=IntentNoteServiceConfig())

    ev = Event(type=EventType.USER, content="What should POSTGRES_DB be?")
    note = await svc.generate_for_event(
        event=ev,
        recent_events=[],
        functional_type_seeds=["requirement"],
    )
    assert note is not None
    assert note.act == "ask"
    assert note.target == "postgres config"
    assert note.context_scope == "db setup"
    assert note.functional_types == ["requirement"]


@pytest.mark.asyncio
async def test_intent_note_service_extracts_json_from_markdown() -> None:
    llm = MockLLM(
        responses=[
            "```json\n"
            + json.dumps(
                {
                    "act": "decide",
                    "target": None,
                    "note_text": "Assistant decides to proceed with the refactor.",
                    "context_scope": None,
                    "event_types": [],
                    "functional_types": [],
                    "confidence": 0.8,
                    "source": "llm",
                }
            )
            + "\n```"
        ]
    )
    svc = IntentNoteService(llm, config=IntentNoteServiceConfig())
    ev = Event(type=EventType.AGENT, content="Let's proceed with the refactor.")
    note = await svc.generate_for_event(event=ev, recent_events=[])
    assert note is not None
    assert note.act == "decide"


@pytest.mark.asyncio
async def test_intent_note_service_enforces_functional_type_seeds() -> None:
    llm = MockLLM(
        responses=[
            json.dumps(
                {
                    "act": "explain",
                    "target": "retrieval",
                    "note_text": "Assistant explains retrieval behavior.",
                    "context_scope": None,
                    "event_types": [],
                    "functional_types": ["requirement", "made_up_label"],
                    "confidence": 1.0,
                    "source": "llm",
                }
            )
        ]
    )
    svc = IntentNoteService(llm, config=IntentNoteServiceConfig())
    ev = Event(type=EventType.AGENT, content="Here is how retrieval works...")
    note = await svc.generate_for_event(event=ev, functional_type_seeds=["requirement", "decision"])
    assert note is not None
    assert note.functional_types == ["requirement"]


@pytest.mark.asyncio
async def test_intent_note_service_skips_short_content() -> None:
    llm = MockLLM(responses=[json.dumps({"act": "ask", "target": None, "note_text": "x"})])
    cfg = IntentNoteServiceConfig(min_content_length=10)
    svc = IntentNoteService(llm, config=cfg)
    ev = Event(type=EventType.USER, content="ok")
    note = await svc.generate_for_event(event=ev)
    assert note is None


@pytest.mark.asyncio
async def test_intent_note_service_returns_none_on_invalid_json() -> None:
    llm = MockLLM(responses=["not json"])
    svc = IntentNoteService(llm, config=IntentNoteServiceConfig())
    ev = Event(type=EventType.USER, content="Explain this please.")
    note = await svc.generate_for_event(event=ev)
    assert note is None
