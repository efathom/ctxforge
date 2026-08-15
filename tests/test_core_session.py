"""
Tests for core session data structures.
"""

from datetime import datetime

from ctxforge.core.events import Event, EventFactory, EventType
from ctxforge.core.session import Session, SessionBuilder, SessionState


class TestSessionState:
    """Tests for SessionState."""
    
    def test_default_state(self):
        """Test default session state."""
        state = SessionState()
        assert state.data == {}
    
    def test_get_set(self):
        """Test get and set operations."""
        state = SessionState()
        
        state.set("key", "value")
        assert state.get("key") == "value"
        assert state.get("missing") is None
        assert state.get("missing", "default") == "default"
    
    def test_delete(self):
        """Test delete operation."""
        state = SessionState()
        state.set("key", "value")
        
        assert state.delete("key") is True
        assert state.get("key") is None
        assert state.delete("missing") is False
    
    def test_has(self):
        """Test has operation."""
        state = SessionState()
        state.set("key", "value")
        
        assert state.has("key") is True
        assert state.has("missing") is False
    
    def test_clear(self):
        """Test clear operation."""
        state = SessionState()
        state.set("key1", "value1")
        state.set("key2", "value2")
        
        state.clear()
        assert state.data == {}
    
    def test_update(self):
        """Test update operation."""
        state = SessionState()
        state.set("existing", "value")
        
        state.update({"new1": "v1", "new2": "v2"})
        
        assert state.get("existing") == "value"
        assert state.get("new1") == "v1"
        assert state.get("new2") == "v2"
    
    def test_keys(self):
        """Test keys operation."""
        state = SessionState()
        state.set("a", 1)
        state.set("b", 2)
        
        keys = state.keys()
        assert "a" in keys
        assert "b" in keys


class TestSession:
    """Tests for Session."""
    
    def test_create_session(self):
        """Test creating a session."""
        session = Session(
            session_id="sess_123",
            user_id="user_456",
        )
        
        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
        assert session.version == 0
        assert session.events == []
        assert session.summary is None
        assert isinstance(session.created_at, datetime)
    
    def test_add_event(self):
        """Test adding events to session."""
        session = Session(session_id="s1", user_id="u1")
        event = Event(type=EventType.USER, content="Hello")
        
        session.add_event(event)
        
        assert len(session.events) == 1
        assert session.events[0] == event
    
    def test_add_user_message(self):
        """Test convenience method for adding user message."""
        session = Session(session_id="s1", user_id="u1")
        event = session.add_user_message("Hello!")
        
        assert len(session.events) == 1
        assert event.type == EventType.USER
        assert event.content == "Hello!"
    
    def test_add_agent_message(self):
        """Test convenience method for adding agent message."""
        session = Session(session_id="s1", user_id="u1")
        event = session.add_agent_message(
            content="Hi there!",
            model="gpt-4",
            latency_ms=100.0,
        )
        
        assert len(session.events) == 1
        assert event.type == EventType.AGENT
        assert event.content == "Hi there!"
        assert event.metadata.model == "gpt-4"
    
    def test_get_events_by_type(self):
        """Test filtering events by type."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("Hello")
        session.add_agent_message("Hi")
        session.add_user_message("How are you?")
        
        user_events = session.get_events_by_type(EventType.USER)
        assert len(user_events) == 2
        
        agent_events = session.get_events_by_type(EventType.AGENT)
        assert len(agent_events) == 1
    
    def test_get_recent_events(self):
        """Test getting recent events."""
        session = Session(session_id="s1", user_id="u1")
        for i in range(10):
            session.add_user_message(f"Message {i}")
        
        recent = session.get_recent_events(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 7"
        assert recent[2].content == "Message 9"
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("Hello")
        session.add_agent_message("Hi")
        session.add_event(Event(type=EventType.SYSTEM, content="System"))
        session.add_user_message("Bye")
        
        history = session.get_conversation_history()
        assert len(history) == 3  # Excludes system event
    
    def test_get_last_user_message(self):
        """Test getting last user message."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("First")
        session.add_agent_message("Response")
        session.add_user_message("Second")
        
        last = session.get_last_user_message()
        assert last.content == "Second"
    
    def test_get_last_agent_message(self):
        """Test getting last agent message."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("Question")
        session.add_agent_message("Answer 1")
        session.add_agent_message("Answer 2")
        
        last = session.get_last_agent_message()
        assert last.content == "Answer 2"
    
    def test_event_count(self):
        """Test event count."""
        session = Session(session_id="s1", user_id="u1")
        assert session.event_count() == 0
        
        session.add_user_message("Hello")
        assert session.event_count() == 1
    
    def test_turn_count(self):
        """Test turn count."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("Hello")
        session.add_agent_message("Hi")
        session.add_user_message("Bye")
        session.add_agent_message("Goodbye")
        
        assert session.turn_count() == 2
    
    def test_prune_events(self):
        """Test pruning events."""
        session = Session(session_id="s1", user_id="u1")
        for i in range(10):
            session.add_user_message(f"Message {i}")
        
        pruned = session.prune_events(keep_last=3)
        
        assert len(pruned) == 7
        assert len(session.events) == 3
        assert session.events[0].content == "Message 7"
    
    def test_set_summary(self):
        """Test setting summary."""
        session = Session(session_id="s1", user_id="u1")
        session.set_summary("This is a summary")
        
        assert session.summary == "This is a summary"
    
    def test_append_summary(self):
        """Test appending to summary."""
        session = Session(session_id="s1", user_id="u1")
        session.set_summary("Part 1")
        session.append_summary("Part 2")
        
        assert session.summary == "Part 1 -> Part 2"
    
    def test_increment_version(self):
        """Test version increment."""
        session = Session(session_id="s1", user_id="u1")
        assert session.version == 0
        
        new_version = session.increment_version()
        assert new_version == 1
        assert session.version == 1
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("Hello")
        session.state.set("key", "value")
        
        data = session.to_dict()
        restored = Session.from_dict(data)
        
        assert restored.session_id == session.session_id
        assert restored.user_id == session.user_id
        assert len(restored.events) == 1
        assert restored.state.get("key") == "value"
    
    def test_copy_deep(self):
        """Test deep copy."""
        session = Session(session_id="s1", user_id="u1")
        session.add_user_message("Hello")
        
        copy = session.copy_deep()
        
        # Modify original
        session.add_user_message("World")
        
        # Copy should be unchanged
        assert len(copy.events) == 1
        assert len(session.events) == 2


class TestSessionBuilder:
    """Tests for SessionBuilder."""
    
    def test_basic_build(self):
        """Test basic session building."""
        session = SessionBuilder("user_123").build()
        
        assert session.user_id == "user_123"
        assert session.session_id is not None
    
    def test_with_session_id(self):
        """Test setting session ID."""
        session = (
            SessionBuilder("user_123")
            .with_session_id("custom_id")
            .build()
        )
        
        assert session.session_id == "custom_id"
    
    def test_with_state(self):
        """Test adding state."""
        session = (
            SessionBuilder("user_123")
            .with_state("key1", "value1")
            .with_state("key2", "value2")
            .build()
        )
        
        assert session.state.get("key1") == "value1"
        assert session.state.get("key2") == "value2"
    
    def test_with_metadata(self):
        """Test adding metadata."""
        session = (
            SessionBuilder("user_123")
            .with_metadata("source", "web")
            .build()
        )
        
        assert session.metadata["source"] == "web"
    
    def test_with_event(self):
        """Test adding events."""
        event = EventFactory.user_message("Hello")
        session = (
            SessionBuilder("user_123")
            .with_event(event)
            .build()
        )
        
        assert len(session.events) == 1
        assert session.events[0].content == "Hello"
    
    def test_with_summary(self):
        """Test adding summary."""
        session = (
            SessionBuilder("user_123")
            .with_summary("Previous conversation summary")
            .build()
        )
        
        assert session.summary == "Previous conversation summary"
    
    def test_fluent_chain(self):
        """Test fluent API chain."""
        session = (
            SessionBuilder("user_123")
            .with_session_id("sess_1")
            .with_state("cart", [])
            .with_metadata("channel", "api")
            .with_summary("User is vegetarian")
            .build()
        )
        
        assert session.session_id == "sess_1"
        assert session.user_id == "user_123"
        assert session.state.get("cart") == []
        assert session.metadata["channel"] == "api"
        assert session.summary == "User is vegetarian"

