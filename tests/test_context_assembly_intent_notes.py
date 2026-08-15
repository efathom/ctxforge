from __future__ import annotations

from ctxforge.compaction.assembler import DefaultContextAssembler
from ctxforge.core.events import Event, EventMetadata, EventType
from ctxforge.core.session import Session


def test_default_context_assembler_prefers_intent_note_in_history() -> None:
    s = Session(user_id="u1")
    s.events.append(
        Event(
            type=EventType.USER,
            content="What should POSTGRES_DB be?",
            metadata=EventMetadata(
                custom={
                    "intent_note": {
                        "act": "ask",
                        "target": "postgres config",
                        "note_text": "User asks which POSTGRES_DB value to use.",
                        "event_types": [],
                        "functional_types": ["requirement"],
                        "confidence": 0.9,
                        "source": "llm",
                    }
                }
            ),
        )
    )
    s.events.append(Event(type=EventType.AGENT, content="Set it to your database name."))

    assembler = DefaultContextAssembler()
    history = assembler._format_history(s.events)  # noqa: SLF001 (unit test)

    assert "[act=ask]" in history
    assert "[target=postgres config]" in history
    assert "User asks which POSTGRES_DB value to use." in history
    assert "What should POSTGRES_DB be?" not in history
