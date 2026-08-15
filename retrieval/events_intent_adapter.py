"""
Events Intent Adapter for UnifiedRetriever.

Searches over a session's persisted structured intent notes attached to events:
`Event.metadata.custom["intent_note"]`.

This adapter is intentionally lightweight and deterministic (no embeddings/LLM).
It is meant to provide a recall-friendly way to resurface relevant event notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List

from ctxforge.core.events import Event
from ctxforge.core.intent_note import IntentNote
from ctxforge.protocols.storage import ISessionStore
from ctxforge.retrieval.unified_retriever import ResultSource, RetrievalResult

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _overlap_score(query: str, doc: str) -> float:
    q_terms = set(_tokenize(query))
    if not q_terms:
        return 0.0
    d_terms = set(_tokenize(doc))
    if not d_terms:
        return 0.0
    overlap = len(q_terms & d_terms)
    return overlap / max(1, len(q_terms))


@dataclass
class EventsIntentAdapter:
    """
    Adapter that loads a session and searches its events' intent notes.

    Required kwargs for `search()`:
    - session_id: str
    - user_id: str
    """

    session_store: ISessionStore

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> List[RetrievalResult]:
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        if not session_id or not user_id:
            return []

        session = await self.session_store.load(str(session_id), str(user_id))
        events = session.events or []

        scored: List[tuple[float, Event, IntentNote]] = []
        for ev in events:
            note = ev.get_intent_note()
            if note is None:
                continue

            doc = _note_to_search_text(note)
            if not doc:
                continue

            score = _overlap_score(query, doc)
            if score <= 0.0:
                continue

            # Small recency boost: newer events get slightly higher score.
            # Keep the score bounded in [0, 1].
            score = min(1.0, score + 0.05)
            scored.append((score, ev, note))

        scored.sort(key=lambda t: -t[0])
        results: List[RetrievalResult] = []
        for score, ev, note in scored[:limit]:
            results.append(
                RetrievalResult(
                    content=_note_to_display_text(note),
                    score=float(score),
                    source=ResultSource.EVENTS_INTENT,
                    source_id=ev.event_id,
                    knowledge_type="event_intent",
                    tags=[],
                    metadata={
                        "session_id": session_id,
                        "event_id": ev.event_id,
                        "event_type": ev.type.value,
                    },
                )
            )
        return results


def _note_to_search_text(note: IntentNote) -> str:
    parts = [note.act or "", note.target or "", note.note_text or ""]
    parts.extend(note.event_types or [])
    parts.extend(note.functional_types or [])
    return " ".join(p for p in parts if p).strip()


def _note_to_display_text(note: IntentNote) -> str:
    act = (note.act or "").strip()
    target = (note.target or "").strip()
    note_text = (note.note_text or "").strip()
    if target:
        return f"act={act} target={target} note={note_text}"
    return f"act={act} note={note_text}"
