"""
Tests for core event data structures.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from ctxforge.core.events import (
    Event,
    EventFactory,
    EventMetadata,
    EventType,
)


class TestEventType:
    """Tests for EventType enum."""
    
    def test_event_types_exist(self):
        """Verify all expected event types exist."""
        assert EventType.USER == "user"
        assert EventType.AGENT == "agent"
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.TOOL_OUTPUT == "tool_output"
        assert EventType.SYSTEM == "system"
        assert EventType.SUMMARY == "summary"
        assert EventType.ERROR == "error"
        assert EventType.METADATA == "metadata"


class TestEventMetadata:
    """Tests for EventMetadata."""
    
    def test_default_values(self):
        """Test metadata default values."""
        meta = EventMetadata()
        
        assert meta.input_tokens is None
        assert meta.output_tokens is None
        assert meta.tool_name is None
        assert meta.model is None
        assert meta.custom == {}
    
    def test_with_values(self):
        """Test metadata with values."""
        meta = EventMetadata(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            latency_ms=250.5,
        )
        
        assert meta.input_tokens == 100
        assert meta.output_tokens == 50
        assert meta.model == "gpt-4"
        assert meta.latency_ms == 250.5


class TestEvent:
    """Tests for Event."""
    
    def test_create_basic_event(self):
        """Test creating a basic event."""
        event = Event(
            type=EventType.USER,
            content="Hello, world!",
        )
        
        assert event.type == EventType.USER
        assert event.content == "Hello, world!"
        assert event.event_id is not None
        assert isinstance(event.timestamp, datetime)
        assert event.parent_id is None
        assert event.tags == []
    
    def test_event_is_immutable(self):
        """Test that events are immutable."""
        event = Event(
            type=EventType.USER,
            content="Test",
        )
        
        with pytest.raises(ValidationError):
            event.content = "Modified"
    
    def test_event_id_is_unique(self):
        """Test that event IDs are unique."""
        events = [
            Event(type=EventType.USER, content=f"Message {i}")
            for i in range(100)
        ]
        
        event_ids = [e.event_id for e in events]
        assert len(event_ids) == len(set(event_ids))
    
    def test_to_prompt_format(self):
        """Test conversion to prompt format."""
        event = Event(
            type=EventType.USER,
            content="Hello!",
        )
        
        assert event.to_prompt_format() == "USER: Hello!"
        
        agent_event = Event(
            type=EventType.AGENT,
            content="Hi there!",
        )
        
        assert agent_event.to_prompt_format() == "AGENT: Hi there!"
    
    def test_with_metadata(self):
        """Test creating event with updated metadata."""
        event = Event(
            type=EventType.AGENT,
            content="Response",
        )
        
        updated = event.with_metadata(model="gpt-4", latency_ms=100.0)
        
        # Original is unchanged
        assert event.metadata.model is None
        
        # New event has updated metadata
        assert updated.metadata.model == "gpt-4"
        assert updated.metadata.latency_ms == 100.0
        
        # Other fields are preserved
        assert updated.event_id == event.event_id
        assert updated.content == event.content
    
    def test_event_with_tags(self):
        """Test event with tags."""
        event = Event(
            type=EventType.USER,
            content="Important message",
            tags=["important", "priority"],
        )
        
        assert "important" in event.tags
        assert "priority" in event.tags


class TestEventFactory:
    """Tests for EventFactory."""
    
    def test_user_message(self):
        """Test creating user message event."""
        event = EventFactory.user_message("Hello!")
        
        assert event.type == EventType.USER
        assert event.content == "Hello!"
    
    def test_agent_message(self):
        """Test creating agent message event."""
        event = EventFactory.agent_message(
            content="Hi there!",
            model="gpt-4",
            latency_ms=150.0,
            input_tokens=10,
            output_tokens=5,
        )
        
        assert event.type == EventType.AGENT
        assert event.content == "Hi there!"
        assert event.metadata.model == "gpt-4"
        assert event.metadata.latency_ms == 150.0
        assert event.metadata.input_tokens == 10
        assert event.metadata.output_tokens == 5
    
    def test_tool_call(self):
        """Test creating tool call event."""
        event = EventFactory.tool_call(
            tool_name="calculator",
            tool_args={"expression": "2+2"},
        )
        
        assert event.type == EventType.TOOL_CALL
        assert event.metadata.tool_name == "calculator"
        assert event.metadata.tool_args == {"expression": "2+2"}
    
    def test_tool_output(self):
        """Test creating tool output event."""
        event = EventFactory.tool_output(
            content="4",
            tool_name="calculator",
            result_type="success",
        )
        
        assert event.type == EventType.TOOL_OUTPUT
        assert event.content == "4"
        assert event.metadata.tool_name == "calculator"
        assert event.metadata.tool_result_type == "success"
    
    def test_system_message(self):
        """Test creating system message event."""
        event = EventFactory.system_message("System initialized")
        
        assert event.type == EventType.SYSTEM
        assert event.content == "System initialized"
    
    def test_error_event(self):
        """Test creating error event."""
        event = EventFactory.error_event("Something went wrong")
        
        assert event.type == EventType.ERROR
        assert event.content == "Something went wrong"
    
    def test_summary_event(self):
        """Test creating summary event."""
        event = EventFactory.summary_event(
            content="User discussed travel plans",
            events_summarized=10,
        )
        
        assert event.type == EventType.SUMMARY
        assert event.content == "User discussed travel plans"
        assert event.metadata.custom["events_summarized"] == 10

