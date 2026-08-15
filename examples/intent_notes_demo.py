from __future__ import annotations

"""
Demo: Structured Turn Notes (Intent Notes)

Shows:
- Recording a turn with `IntentNotesMiddleware` attached to the record pipeline
- Persisting intent notes in session events (`metadata.custom["intent_note"]`)
- Context assembly preferring intent notes when formatting conversation history
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ctxforge.compaction.assembler import DefaultContextAssembler
from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.intent_note_service import IntentNoteService, IntentNoteServiceConfig
from ctxforge.middleware import MiddlewareChain
from ctxforge.middleware.intent_notes import IntentNotesMiddleware
from ctxforge.protocols.llm import ILLMProvider, LLMResponse
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


@dataclass
class DemoLLMProvider(ILLMProvider):
    responses: List[str]
    _i: int = 0

    @property
    def name(self) -> str:
        return "demo"

    @property
    def default_model(self) -> str:
        return "demo-model"

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        idx = self._i
        self._i += 1
        content = self.responses[idx] if idx < len(self.responses) else self.responses[-1]
        return LLMResponse(content=content, model=kwargs.get("model") or self.default_model)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text.split())

    def count_message_tokens(self, messages, model: Optional[str] = None) -> int:
        return 0


async def main() -> None:
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()

    llm = DemoLLMProvider(
        responses=[
            json.dumps(
                {
                    "act": "ask",
                    "target": "postgres config",
                    "note_text": "User asks which POSTGRES_DB value to use.",
                    "confidence": 0.9,
                    "source": "llm",
                }
            ),
            json.dumps(
                {
                    "act": "explain",
                    "target": "postgres config",
                    "note_text": "Assistant explains how POSTGRES_DB is used.",
                    "confidence": 0.9,
                    "source": "llm",
                }
            ),
        ]
    )
    svc = IntentNoteService(llm_provider=llm, config=IntentNoteServiceConfig(min_content_length=1))
    record_chain = MiddlewareChain().add(IntentNotesMiddleware(intent_note_service=svc))

    engine = CtxForge(
        config=DEFAULT_CONFIG,
        session_store=session_store,
        memory_store=memory_store,
        assembler=DefaultContextAssembler(),
        record_chain=record_chain,
    )

    session_id = "intent-notes-demo"
    user_id = "demo-user"

    await engine.record_turn(
        session_id=session_id,
        user_id=user_id,
        user_input="What should POSTGRES_DB be?",
        assistant_response="Set it to the name of your database.",
    )

    session = await engine.get_session(session_id=session_id, user_id=user_id)
    print("\nPersisted events (showing intent notes):")
    for e in session.events[-2:]:
        note = e.get_intent_note()
        print(f"- {e.type.value}: {note.model_dump() if note else None}")

    # Show history formatting prefers intent notes
    ctx = await engine.prepare_context(
        session_id=session_id,
        user_id=user_id,
        user_input="Remind me what we decided about Postgres config.",
        system_instructions="You are a helpful assistant.",
    )
    history_section = next((s for s in ctx.sections if s.name == "conversation_history"), None)
    print("\nAssembled conversation history section:")
    print(history_section.content if history_section else "(missing)")


if __name__ == "__main__":
    asyncio.run(main())
