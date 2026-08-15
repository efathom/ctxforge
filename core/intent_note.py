"""
Intent Note - Structured turn note for events.

Intent notes are derived artifacts attached to events (via `EventMetadata.custom`)
to provide a compact, machine-usable representation of a turn's intent:
- act: pragmatic action
- target: entity/topic/object of the act
- note_text: concise resolved summary
Optionally:
- context_scope, event_types, functional_types
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class IntentNote(BaseModel):
    """Structured representation of a turn's pragmatic intent."""

    act: str
    target: Optional[str] = None
    note_text: str

    context_scope: Optional[str] = None
    event_types: List[str] = Field(default_factory=list)
    functional_types: List[str] = Field(default_factory=list)

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)
    source: str = "llm"  # "llm" | "heuristic"
