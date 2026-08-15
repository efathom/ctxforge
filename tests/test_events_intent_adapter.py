from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from ctxforge.core.events import Event, EventMetadata, EventType
from ctxforge.core.session import Session
from ctxforge.protocols.storage import ISessionStore
from ctxforge.retrieval.events_intent_adapter import EventsIntentAdapter
from ctxforge.retrieval.unified_retriever import ResultSource, UnifiedRetriever


@dataclass
class InMemoryStore(ISessionStore):
    session: Session

    async def load(self, session_id: str, user_id: str) -> Session:
        return self.session.model_copy(deep=True)

    async def save(self, session: Session) -> None:
        self.session = session.model_copy(deep=True)

    async def delete(self, session_id: str) -> bool:
        return False

    async def exists(self, session_id: str) -> bool:
        return True

    async def list_sessions(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Session]:
        return [self.session.model_copy(deep=True)]


@pytest.mark.asyncio
async def test_events_intent_adapter_search_returns_ranked_results() -> None:
    s = Session(user_id="u1", session_id="s1")
    s.events.append(
        Event(
            type=EventType.USER,
            content="What should POSTGRES_DB be?",
            metadata=EventMetadata(
                custom={
                    "intent_note": {
                        "act": "ask",
                        "target": "postgres config",
                        "note_text": "Ask what POSTGRES_DB should be set to.",
                        "event_types": ["configuration"],
                        "functional_types": ["requirement"],
                        "confidence": 0.9,
                        "source": "llm",
                    }
                }
            ),
        )
    )
    s.events.append(
        Event(
            type=EventType.USER,
            content="Unrelated",
            metadata=EventMetadata(
                custom={
                    "intent_note": {
                        "act": "ask",
                        "target": "python",
                        "note_text": "Ask about decorators.",
                        "event_types": [],
                        "functional_types": [],
                        "confidence": 0.9,
                        "source": "llm",
                    }
                }
            ),
        )
    )

    store = InMemoryStore(session=s)
    adapter = EventsIntentAdapter(session_store=store)
    results = await adapter.search("POSTGRES_DB value", limit=5, session_id="s1", user_id="u1")
    assert results
    assert results[0].source == ResultSource.EVENTS_INTENT
    assert results[0].knowledge_type == "event_intent"
    assert "POSTGRES_DB" in results[0].content


@pytest.mark.asyncio
async def test_unified_retriever_integration_events_intent_source() -> None:
    s = Session(user_id="u1", session_id="s1")
    s.events.append(
        Event(
            type=EventType.USER,
            content="What should POSTGRES_DB be?",
            metadata=EventMetadata(
                custom={
                    "intent_note": {
                        "act": "ask",
                        "target": "postgres config",
                        "note_text": "Ask what POSTGRES_DB should be set to.",
                        "confidence": 0.9,
                        "source": "llm",
                    }
                }
            ),
        )
    )
    store = InMemoryStore(session=s)

    retriever = UnifiedRetriever()
    retriever.register_store(
        name="events_intent",
        source=ResultSource.EVENTS_INTENT,
        adapter=EventsIntentAdapter(session_store=store),
    )

    results = await retriever.search(
        "POSTGRES_DB",
        sources=[ResultSource.EVENTS_INTENT],
        session_id="s1",
        user_id="u1",
    )
    assert results
    assert all(r.source == ResultSource.EVENTS_INTENT for r in results)
