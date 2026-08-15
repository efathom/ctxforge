"""
Tests for core context data structures.
"""

import pytest

from ctxforge.core.context import Context, ContextBuilder, ContextSection
from ctxforge.core.events import EventFactory
from ctxforge.core.memory import MemoryFactory


class TestContextSection:
    """Tests for ContextSection."""
    
    def test_create_section(self):
        """Test creating a context section."""
        section = ContextSection(
            name="Test Section",
            content="This is the content",
            priority=10,
        )
        
        assert section.name == "Test Section"
        assert section.content == "This is the content"
        assert section.priority == 10
        assert section.is_required is True
    
    def test_section_with_token_estimate(self):
        """Test section with token estimate."""
        section = ContextSection(
            name="Large Section",
            content="A" * 1000,
            token_estimate=250,
        )
        
        assert section.token_estimate == 250


class TestContext:
    """Tests for Context."""
    
    def test_create_context(self):
        """Test creating a context."""
        context = Context(
            session_id="sess_123",
            user_id="user_456",
        )
        
        assert context.session_id == "sess_123"
        assert context.user_id == "user_456"
        assert context.sections == []
        assert context.memories == []
        assert context.events == []
    
    def test_add_section(self):
        """Test adding sections."""
        context = Context(session_id="s1", user_id="u1")
        
        context.add_section(
            name="Section A",
            content="Content A",
            priority=10,
        )
        context.add_section(
            name="Section B",
            content="Content B",
            priority=20,
        )
        
        # Should be sorted by priority (descending)
        assert len(context.sections) == 2
        assert context.sections[0].name == "Section B"  # Higher priority first
        assert context.sections[1].name == "Section A"
    
    def test_get_section(self):
        """Test getting section by name."""
        context = Context(session_id="s1", user_id="u1")
        context.add_section(name="Test", content="Content")
        
        section = context.get_section("Test")
        assert section is not None
        assert section.content == "Content"
        
        assert context.get_section("NonExistent") is None
    
    def test_remove_section(self):
        """Test removing section."""
        context = Context(session_id="s1", user_id="u1")
        context.add_section(name="Test", content="Content")
        
        assert context.remove_section("Test") is True
        assert context.get_section("Test") is None
        assert context.remove_section("Test") is False
    
    def test_update_section(self):
        """Test updating section content."""
        context = Context(session_id="s1", user_id="u1")
        context.add_section(name="Test", content="Original")
        
        assert context.update_section("Test", "Updated") is True
        assert context.get_section("Test").content == "Updated"
        assert context.update_section("NonExistent", "Value") is False
    
    def test_available_tokens(self):
        """Test available token calculation."""
        context = Context(
            session_id="s1",
            user_id="u1",
            total_token_budget=8000,
            reserved_output_tokens=1000,
        )
        
        assert context.get_available_tokens() == 7000
    
    def test_is_within_budget(self):
        """Test budget checking."""
        context = Context(
            session_id="s1",
            user_id="u1",
            total_token_budget=100,
            reserved_output_tokens=50,
        )
        
        # Small content should fit
        context.current_query = "Hello"
        assert context.is_within_budget() is True
        
        # Large content should not fit
        context.current_query = "Hello " * 100
        assert context.is_within_budget() is False
    
    def test_to_prompt(self):
        """Test prompt generation."""
        context = Context(
            session_id="s1",
            user_id="u1",
            system_instructions="You are helpful.",
            current_query="Hello!",
        )
        
        # Add a memory
        memory = MemoryFactory.semantic_memory(
            user_id="u1",
            content="User is vegetarian",
        )
        context.memories.append(memory)
        
        # Add an event
        event = EventFactory.user_message("Previous message")
        context.events.append(event)
        
        prompt = context.to_prompt()
        
        assert "System: You are helpful." in prompt
        assert "User is vegetarian" in prompt
        assert "USER: Previous message" in prompt
        assert "USER: Hello!" in prompt
        assert "AGENT:" in prompt
    
    def test_to_messages(self):
        """Test message list generation."""
        context = Context(
            session_id="s1",
            user_id="u1",
            system_instructions="You are helpful.",
            current_query="Hello!",
        )
        
        # Add conversation
        context.events.append(EventFactory.user_message("Hi"))
        context.events.append(EventFactory.agent_message("Hello!"))
        
        messages = context.to_messages()
        
        assert len(messages) == 4  # system, user, assistant, current user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Hello!"


class TestContextFormatConverters:
    """Tests for Context format converters."""
    
    @pytest.fixture
    def context_with_data(self):
        """Create a context with various data for testing."""
        from ctxforge.core.events import EventFactory
        
        memories = [
            MemoryFactory.semantic_memory("u1", "User is vegetarian"),
            MemoryFactory.semantic_memory("u1", "User likes hiking"),
        ]
        events = [
            EventFactory.user_message("Hello"),
            EventFactory.agent_message("Hi there!"),
        ]
        
        return (
            ContextBuilder("sess_1", "user_1")
            .with_system_instructions("You are a helpful assistant.")
            .with_section("Rules", "Be concise.", priority=10)
            .with_memories(memories)
            .with_events(events)
            .with_current_query("How are you?")
            .build()
        )
    
    def test_to_messages(self, context_with_data):
        """Test generic message format."""
        messages = context_with_data.to_messages()
        
        assert len(messages) >= 4  # system, user, assistant, current user
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert "vegetarian" in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "How are you?"
    
    def test_to_openai_messages(self, context_with_data):
        """Test OpenAI format."""
        messages = context_with_data.to_openai_messages()
        
        # Same as to_messages
        assert isinstance(messages, list)
        assert all("role" in m and "content" in m for m in messages)
    
    def test_to_anthropic_messages(self, context_with_data):
        """Test Anthropic format with separate system."""
        system, messages = context_with_data.to_anthropic_messages()
        
        # System is separate
        assert isinstance(system, str)
        assert "helpful assistant" in system
        assert "vegetarian" in system
        
        # Messages don't include system
        assert all(m["role"] != "system" for m in messages)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "How are you?"
    
    def test_to_dict(self, context_with_data):
        """Test dictionary format."""
        d = context_with_data.to_dict()
        
        assert d["session_id"] == "sess_1"
        assert d["user_id"] == "user_1"
        assert d["system_instructions"] == "You are a helpful assistant."
        assert d["current_query"] == "How are you?"
        assert len(d["memories"]) == 2
        assert len(d["events"]) == 2
        assert len(d["sections"]) == 1
    
    def test_to_prompt(self, context_with_data):
        """Test prompt string format."""
        prompt = context_with_data.to_prompt()
        
        assert "System: You are a helpful assistant." in prompt
        assert "vegetarian" in prompt
        assert "USER: How are you?" in prompt
        assert "AGENT:" in prompt
    
    def test_build_system_content(self, context_with_data):
        """Test building system content."""
        system = context_with_data._build_system_content()
        
        assert "helpful assistant" in system
        assert "[Rules]" in system
        assert "Be concise" in system
        assert "vegetarian" in system


class TestContextBuilder:
    """Tests for ContextBuilder."""
    
    def test_basic_build(self):
        """Test basic context building."""
        context = ContextBuilder("sess_1", "user_1").build()
        
        assert context.session_id == "sess_1"
        assert context.user_id == "user_1"
    
    def test_with_system_instructions(self):
        """Test setting system instructions."""
        context = (
            ContextBuilder("s1", "u1")
            .with_system_instructions("Be helpful")
            .build()
        )
        
        assert context.system_instructions == "Be helpful"
    
    def test_with_section(self):
        """Test adding sections."""
        context = (
            ContextBuilder("s1", "u1")
            .with_section("Rules", "Follow these rules...", priority=10)
            .with_section("Context", "Here is context...", priority=5)
            .build()
        )
        
        assert len(context.sections) == 2
        assert context.sections[0].name == "Rules"  # Higher priority first
    
    def test_with_memories(self):
        """Test adding memories."""
        memories = [
            MemoryFactory.semantic_memory("u1", "Fact 1"),
            MemoryFactory.semantic_memory("u1", "Fact 2"),
        ]
        
        context = (
            ContextBuilder("s1", "u1")
            .with_memories(memories)
            .build()
        )
        
        assert len(context.memories) == 2
    
    def test_with_events(self):
        """Test adding events."""
        events = [
            EventFactory.user_message("Hello"),
            EventFactory.agent_message("Hi"),
        ]
        
        context = (
            ContextBuilder("s1", "u1")
            .with_events(events)
            .build()
        )
        
        assert len(context.events) == 2
    
    def test_with_current_query(self):
        """Test setting current query."""
        context = (
            ContextBuilder("s1", "u1")
            .with_current_query("What's the weather?")
            .build()
        )
        
        assert context.current_query == "What's the weather?"
    
    def test_with_token_budget(self):
        """Test setting token budget."""
        context = (
            ContextBuilder("s1", "u1")
            .with_token_budget(16000, reserved_output=2000)
            .build()
        )
        
        assert context.total_token_budget == 16000
        assert context.reserved_output_tokens == 2000
    
    def test_with_metadata(self):
        """Test adding metadata."""
        context = (
            ContextBuilder("s1", "u1")
            .with_metadata("source", "api")
            .with_metadata("version", "1.0")
            .build()
        )
        
        assert context.metadata["source"] == "api"
        assert context.metadata["version"] == "1.0"
    
    def test_fluent_chain(self):
        """Test full fluent API chain."""
        memories = [MemoryFactory.semantic_memory("u1", "User likes pizza")]
        events = [EventFactory.user_message("Order pizza")]
        
        context = (
            ContextBuilder("sess_1", "user_1")
            .with_system_instructions("You are a food ordering assistant.")
            .with_section("Menu", "Available items: pizza, pasta", priority=10)
            .with_memories(memories)
            .with_events(events)
            .with_current_query("What pizza do you recommend?")
            .with_token_budget(8000, 1000)
            .with_metadata("restaurant", "Pizza Palace")
            .build()
        )
        
        assert context.session_id == "sess_1"
        assert context.system_instructions == "You are a food ordering assistant."
        assert len(context.sections) == 1
        assert len(context.memories) == 1
        assert len(context.events) == 1
        assert context.current_query == "What pizza do you recommend?"
        assert context.metadata["restaurant"] == "Pizza Palace"

