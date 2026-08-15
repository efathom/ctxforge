"""
Immutable Event Log - Represents 'what happened' in a session.

Events are the atomic units of conversation history, strictly typed and immutable
for auditing purposes. Each event captures a moment in time with full metadata.
"""

import datetime
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ctxforge.core.intent_note import IntentNote


class EventType(str, Enum):
    """Types of events that can occur in a session."""

    USER = "user"           # User input/message
    AGENT = "agent"         # Agent response
    TOOL_CALL = "tool_call"     # Tool invocation request
    TOOL_OUTPUT = "tool_output"  # Tool execution result
    SYSTEM = "system"       # System messages
    SUMMARY = "summary"     # Summarization events
    ERROR = "error"         # Error events
    METADATA = "metadata"   # Metadata updates


class EventMetadata(BaseModel):
    """Structured metadata for events."""

    # Token counts
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Tool-related metadata
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result_type: Optional[str] = None  # success, error, timeout

    # Model metadata
    model: Optional[str] = None
    temperature: Optional[float] = None

    # Performance metadata
    latency_ms: Optional[float] = None

    # Compression metadata
    compressed: bool = False
    compression_strategy: Optional[str] = None
    original_length: Optional[int] = None
    tokens_saved: Optional[int] = None

    # Custom extensions
    custom: Dict[str, Any] = Field(default_factory=dict)

    def get_intent_note(self) -> Optional[IntentNote]:
        """
        Read the structured intent note from `custom["intent_note"]` if present.

        Stored format is expected to be a JSON-serializable dict.
        """
        raw = self.custom.get("intent_note")
        if raw is None:
            return None
        if isinstance(raw, IntentNote):
            return raw
        if not isinstance(raw, dict):
            return None
        try:
            return IntentNote.model_validate(raw)
        except Exception:
            return None

    def with_intent_note(self, note: IntentNote) -> "EventMetadata":
        """Return a copy of this metadata with an updated intent note."""
        custom = dict(self.custom or {})
        custom["intent_note"] = note.model_dump()
        return self.model_copy(update={"custom": custom})


class Event(BaseModel):
    """
    Immutable Event Log entry.

    Represents a single event in the conversation history.
    Events are immutable (frozen) to ensure audit trail integrity.

    Attributes:
        event_id: Unique identifier for the event
        timestamp: When the event occurred
        type: The type of event (user, agent, tool_call, etc.)
        content: The actual content/message of the event
        metadata: Additional structured metadata
        parent_id: Optional reference to parent event (for threading)
        tags: Optional tags for filtering/categorization
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    type: EventType
    content: str
    metadata: EventMetadata = Field(default_factory=EventMetadata)
    parent_id: Optional[str] = None  # For threading/grouping events
    tags: List[str] = Field(default_factory=list)

    model_config = {"frozen": True}  # Enforces immutability for auditing

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty for most event types."""
        # Allow empty content for metadata events
        return v

    def to_prompt_format(self) -> str:
        """Convert event to a format suitable for prompt inclusion."""
        type_label = self.type.value.upper()
        return f"{type_label}: {self.content}"

    def with_metadata(self, **kwargs) -> "Event":
        """Create a new event with updated metadata (maintains immutability)."""
        current_meta = self.metadata.model_dump()
        current_meta.update(kwargs)
        return Event(
            event_id=self.event_id,
            timestamp=self.timestamp,
            type=self.type,
            content=self.content,
            metadata=EventMetadata(**current_meta),
            parent_id=self.parent_id,
            tags=self.tags,
        )

    def get_intent_note(self) -> Optional[IntentNote]:
        """Convenience wrapper for `self.metadata.get_intent_note()`."""
        return self.metadata.get_intent_note()

    def with_intent_note(self, note: IntentNote) -> "Event":
        """
        Return a new Event with `metadata.custom["intent_note"]` set.

        This preserves event immutability by creating a new Event instance.
        """
        meta = self.metadata.with_intent_note(note)
        return Event(
            event_id=self.event_id,
            timestamp=self.timestamp,
            type=self.type,
            content=self.content,
            metadata=meta,
            parent_id=self.parent_id,
            tags=self.tags,
        )


class EventFactory:
    """Factory for creating common event types."""

    @staticmethod
    def user_message(content: str, **metadata_kwargs) -> Event:
        """Create a user message event."""
        return Event(
            type=EventType.USER,
            content=content,
            metadata=EventMetadata(**metadata_kwargs) if metadata_kwargs else EventMetadata(),
        )

    @staticmethod
    def agent_message(
        content: str,
        model: Optional[str] = None,
        latency_ms: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Event:
        """Create an agent response event."""
        return Event(
            type=EventType.AGENT,
            content=content,
            metadata=EventMetadata(
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    @staticmethod
    def tool_call(
        tool_name: str,
        tool_args: Dict[str, Any],
        parent_id: Optional[str] = None,
    ) -> Event:
        """Create a tool call event."""
        return Event(
            type=EventType.TOOL_CALL,
            content=f"Calling {tool_name}",
            metadata=EventMetadata(tool_name=tool_name, tool_args=tool_args),
            parent_id=parent_id,
        )

    @staticmethod
    def tool_output(
        content: str,
        tool_name: str,
        result_type: str = "success",
        parent_id: Optional[str] = None,
    ) -> Event:
        """Create a tool output event."""
        return Event(
            type=EventType.TOOL_OUTPUT,
            content=content,
            metadata=EventMetadata(tool_name=tool_name, tool_result_type=result_type),
            parent_id=parent_id,
        )

    @staticmethod
    def system_message(content: str, **metadata_kwargs) -> Event:
        """Create a system message event."""
        return Event(
            type=EventType.SYSTEM,
            content=content,
            metadata=EventMetadata(**metadata_kwargs) if metadata_kwargs else EventMetadata(),
        )

    @staticmethod
    def error_event(content: str, **metadata_kwargs) -> Event:
        """Create an error event."""
        return Event(
            type=EventType.ERROR,
            content=content,
            metadata=EventMetadata(**metadata_kwargs) if metadata_kwargs else EventMetadata(),
        )

    @staticmethod
    def summary_event(content: str, events_summarized: int = 0) -> Event:
        """Create a summary event."""
        return Event(
            type=EventType.SUMMARY,
            content=content,
            metadata=EventMetadata(custom={"events_summarized": events_summarized}),
        )
