from __future__ import annotations

import json
from datetime import datetime, timezone

from ctxforge.core.events import Event, EventMetadata, EventType
from ctxforge.core.intent_note import IntentNote


def test_eventmetadata_intent_note_roundtrip_dict() -> None:
    note = IntentNote(
        act="ask",
        target="postgres config",
        note_text="User asks which POSTGRES_DB value to use.",
        context_scope="db setup",
        event_types=["configuration"],
        functional_types=["requirement"],
        confidence=0.9,
        model="test-model",
        generated_at=datetime.now(timezone.utc),
        source="llm",
    )

    meta = EventMetadata(custom={"intent_note": note.model_dump()})
    parsed = meta.get_intent_note()
    assert parsed is not None
    assert parsed.act == "ask"
    assert parsed.target == "postgres config"
    assert parsed.note_text.startswith("User asks")
    assert parsed.context_scope == "db setup"
    assert parsed.event_types == ["configuration"]
    assert parsed.functional_types == ["requirement"]


def test_event_with_intent_note_preserves_immutability() -> None:
    ev = Event(type=EventType.USER, content="What should POSTGRES_DB be?")
    note = IntentNote(act="ask", target="postgres config", note_text="Ask about POSTGRES_DB value.")

    ev2 = ev.with_intent_note(note)
    assert ev2 is not ev
    assert ev2.event_id == ev.event_id
    assert ev2.get_intent_note() is not None
    assert ev.get_intent_note() is None


def test_event_json_roundtrip_preserves_intent_note() -> None:
    ev = Event(type=EventType.USER, content="What should POSTGRES_DB be?")
    note = IntentNote(
        act="ask",
        target="postgres config",
        note_text="Ask about POSTGRES_DB value.",
        functional_types=["requirement"],
    )
    ev = ev.with_intent_note(note)

    payload = ev.model_dump_json()
    restored = Event.model_validate(json.loads(payload))

    restored_note = restored.get_intent_note()
    assert restored_note is not None
    assert restored_note.act == "ask"
    assert restored_note.target == "postgres config"
    assert restored_note.functional_types == ["requirement"]
