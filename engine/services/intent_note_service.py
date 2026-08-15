"""
Intent Note Service.

Generates and validates structured turn notes ("intent notes") for events.

Notes are designed to be persisted in `Event.metadata.custom["intent_note"]`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from ctxforge.core.events import Event, EventType
from ctxforge.core.intent_note import IntentNote
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.prompts.intent_note import INTENT_NOTE_PROMPT
from ctxforge.protocols.llm import ILLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentNoteServiceConfig:
    enabled: bool = True
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 250
    max_history_events_for_prompt: int = 10
    min_content_length: int = 20
    max_note_chars: int = 240
    allow_overwrite: bool = False
    include_tool_events: bool = False


class IntentNoteService:
    """
    LLM-first structured intent note generator.

    This service returns an `IntentNote` object (or None if skipped/failed).
    Persistence is handled by callers (middleware/service) by attaching the note
    to `Event.metadata.custom["intent_note"]`.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        config: Optional[IntentNoteServiceConfig] = None,
    ) -> None:
        self._llm = llm_provider
        self._cfg = config or IntentNoteServiceConfig()

    @property
    def config(self) -> IntentNoteServiceConfig:
        return self._cfg

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    async def generate_for_event(
        self,
        *,
        event: Event,
        recent_events: Optional[List[Event]] = None,
        functional_type_seeds: Optional[List[str]] = None,
    ) -> Optional[IntentNote]:
        """
        Generate a structured intent note for a single event.

        Args:
            event: The target event
            recent_events: Optional recent events for reference resolution
            functional_type_seeds: Optional constrained vocabulary for functional_types
        """
        if not self._cfg.enabled:
            return None

        content = (event.content or "").strip()
        if len(content) < self._cfg.min_content_length:
            return None

        if not self._cfg.allow_overwrite and event.get_intent_note() is not None:
            return None

        if (
            event.type in (EventType.TOOL_CALL, EventType.TOOL_OUTPUT)
            and not self._cfg.include_tool_events
        ):
            return None

        role = self._role_for_event(event)
        recent_context = self._format_recent_context(recent_events or [])
        seeds_text = self._format_seeds(functional_type_seeds)

        prompt = INTENT_NOTE_PROMPT.format(
            recent_context=recent_context,
            functional_type_seeds=seeds_text,
            role=role,
            content=content,
        )

        try:
            resp = await self._llm.generate(
                prompt=prompt,
                model=self._cfg.model,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
            )
        except Exception as e:
            logger.warning("Intent note generation failed: %s", e)
            return None

        note = self._parse_intent_note(
            raw_text=resp.content,
            model=resp.model,
            functional_type_seeds=functional_type_seeds,
        )
        if note is None:
            return None

        # Clamp note_text length for predictable prompt usage
        if len(note.note_text) > self._cfg.max_note_chars:
            note.note_text = note.note_text[: self._cfg.max_note_chars - 3].rstrip() + "..."

        # Prefer the response model if caller didn't provide one
        if note.model is None:
            note.model = model_or_none(resp.model)

        # Stamp generation time if not set by model (should be rare; default exists)
        if note.generated_at is None:
            note.generated_at = datetime.now()

        return note

    def _parse_intent_note(
        self,
        *,
        raw_text: str,
        model: str,
        functional_type_seeds: Optional[List[str]],
    ) -> Optional[IntentNote]:
        json_text = extract_json_from_text(raw_text) or raw_text.strip()
        try:
            data = json.loads(json_text)
        except Exception as e:
            logger.warning("Failed to parse intent note JSON: %s", e)
            return None

        if not isinstance(data, dict):
            return None

        # Normalize list fields defensively
        data["event_types"] = _normalize_str_list(data.get("event_types"))
        data["functional_types"] = _normalize_str_list(data.get("functional_types"))

        # Enforce constrained vocabulary if provided
        if functional_type_seeds:
            allowed = {s.strip() for s in functional_type_seeds if s and s.strip()}
            data["functional_types"] = [t for t in data["functional_types"] if t in allowed]

        # Ensure source is set
        data.setdefault("source", "llm")
        data.setdefault("model", model_or_none(model))

        try:
            return IntentNote.model_validate(data)
        except Exception as e:
            logger.warning("IntentNote validation failed: %s", e)
            return None

    def _role_for_event(self, event: Event) -> str:
        if event.type == EventType.USER:
            return "user"
        if event.type == EventType.AGENT:
            return "assistant"
        if event.type == EventType.SYSTEM:
            return "system"
        if event.type == EventType.ERROR:
            return "error"
        if event.type == EventType.SUMMARY:
            return "summary"
        if event.type == EventType.TOOL_CALL:
            return "tool_call"
        if event.type == EventType.TOOL_OUTPUT:
            return "tool_output"
        return event.type.value

    def _format_recent_context(self, events: List[Event]) -> str:
        # Use only last N events, and keep it compact.
        keep = self._cfg.max_history_events_for_prompt
        events = events[-keep:] if keep > 0 else []

        lines: List[str] = []
        for e in events:
            if (
                e.type in (EventType.TOOL_CALL, EventType.TOOL_OUTPUT)
                and not self._cfg.include_tool_events
            ):
                continue
            role = self._role_for_event(e)
            content = (e.content or "").strip().replace("\n", " ")
            if not content:
                continue
            if len(content) > 400:
                content = content[:397] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines) if lines else "No recent context."

    def _format_seeds(self, seeds: Optional[List[str]]) -> str:
        if not seeds:
            return "None"
        cleaned = [s.strip() for s in seeds if s and s.strip()]
        return "\n".join(f"- {s}" for s in cleaned) if cleaned else "None"


def _normalize_str_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def model_or_none(model: object) -> Optional[str]:
    if model is None:
        return None
    s = str(model).strip()
    return s or None
