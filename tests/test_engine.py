"""
Tests for the ctxforge (decoupled from LLM).
"""

import pytest
import pytest_asyncio

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.core.events import EventType
from ctxforge.core.expertise import Expertise, ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import ComponentRegistry
from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class TestContextEngine:
    """Tests for ctxforge."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TESTING_CONFIG
    
    @pytest.fixture
    def session_store(self):
        """Create session store."""
        return InMemorySessionStore()
    
    @pytest.fixture
    def memory_store(self):
        """Create memory store."""
        return InMemoryMemoryStore()
    
    @pytest_asyncio.fixture
    async def engine(self, config, session_store, memory_store):
        """Create engine instance via EngineFactory (includes pipelines + assembler)."""
        factory = EngineFactory()
        return await factory.create(
            config=config,
            session_store=session_store,
            memory_store=memory_store,
        )
    
    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initializes correctly without LLM."""
        assert engine.config is not None
        assert engine.session_store is not None
        assert engine.memory_store is not None
    
    # =========================================================================
    # PREPARE_CONTEXT TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_prepare_context_basic(self, engine):
        """Test basic context preparation."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
        )
        
        assert context is not None
        assert context.session_id == "sess_1"
        assert context.user_id == "user_1"
        assert context.current_query == "Hello!"
    
    @pytest.mark.asyncio
    async def test_prepare_context_with_system_instructions(self, engine):
        """Test context with custom system instructions."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
            system_instructions="You are a helpful assistant.",
        )
        
        assert context.system_instructions == "You are a helpful assistant."
    
    @pytest.mark.asyncio
    async def test_prepare_context_includes_memories(self, engine, memory_store):
        """Test that context includes relevant memories."""
        # Add a memory
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="User is vegetarian",
        )
        await memory_store.add(memory)
        
        # Prepare context with matching query
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="vegetarian food",
        )
        
        assert len(context.memories) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_expertise_items_falls_back_on_retriever_error(self, engine):
        class FailingExpertiseRetriever:
            async def retrieve_items(self, query: str, scope_id: str, limit: int = 10, **kwargs):
                raise RuntimeError("boom")

        # Ensure expertise store is available
        engine._expertise_store = InMemoryExpertiseStore()  # type: ignore[attr-defined]

        # Seed expertise in store
        exp = Expertise(expertise_id="exp_1", name="Test")
        exp.items.append(
            ExpertiseItem(section=ExpertiseSection.STRATEGIES, content="Do the thing")
        )
        await engine.save_expertise(exp)

        # Inject failing retriever
        engine._expertise_retriever = FailingExpertiseRetriever()  # type: ignore[attr-defined]

        items = await engine.retrieve_expertise_items("exp_1", "q", limit=5)
        assert len(items) == 1
        assert items[0].content == "Do the thing"
    
    @pytest.mark.asyncio
    async def test_prepare_context_without_memories(self, engine, memory_store):
        """Test context without memory retrieval."""
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="User is vegetarian",
        )
        await memory_store.add(memory)
        
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello",
            include_memories=False,
        )
        
        assert len(context.memories) == 0
    
    @pytest.mark.asyncio
    async def test_prepare_context_has_metadata(self, engine):
        """Test that context includes preparation metadata."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
        )
        
        assert "preparation_time_ms" in context.metadata
        assert "memory_count" in context.metadata
        assert "history_event_count" in context.metadata
    
    @pytest.mark.asyncio
    async def test_prepare_context_with_session(self, engine):
        """Test prepare_context_with_session returns both context and session."""
        context, session = await engine.prepare_context_with_session(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
        )
        
        assert context is not None
        assert session is not None
        assert session.session_id == "sess_1"
    
    # =========================================================================
    # CONTEXT FORMAT CONVERTER TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_context_to_messages(self, engine):
        """Test converting context to generic messages."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
            system_instructions="You are helpful.",
        )
        
        messages = context.to_messages()
        
        assert len(messages) >= 2  # System + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello!"
    
    @pytest.mark.asyncio
    async def test_context_to_openai_messages(self, engine):
        """Test converting context to OpenAI format."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
        )
        
        messages = context.to_openai_messages()
        
        assert isinstance(messages, list)
        assert all("role" in m and "content" in m for m in messages)
    
    @pytest.mark.asyncio
    async def test_context_to_anthropic_messages(self, engine):
        """Test converting context to Anthropic format."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
            system_instructions="Be helpful.",
        )
        
        system, messages = context.to_anthropic_messages()
        
        assert isinstance(system, str)
        assert "Be helpful" in system
        assert isinstance(messages, list)
        assert messages[-1]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_context_to_dict(self, engine):
        """Test converting context to dictionary."""
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
        )
        
        d = context.to_dict()
        
        assert d["session_id"] == "sess_1"
        assert d["user_id"] == "user_1"
        assert d["current_query"] == "Hello!"
    
    # =========================================================================
    # RECORD_TURN TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_record_turn(self, engine, session_store):
        """Test recording a conversation turn."""
        await engine.record_turn(
            session_id="sess_1",
            user_id="user_1",
            user_input="Hello!",
            assistant_response="Hi there!",
        )
        
        session = await session_store.load("sess_1", "user_1")
        assert len(session.events) == 2
        
        user_events = session.get_events_by_type(EventType.USER)
        agent_events = session.get_events_by_type(EventType.AGENT)
        
        assert len(user_events) == 1
        assert len(agent_events) == 1
        assert user_events[0].content == "Hello!"
        assert agent_events[0].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_record_turn_applies_pii_redaction_pipeline(self, engine, session_store):
        """Ensure record pipeline can redact PII before persisting."""
        await engine.record_turn(
            session_id="sess_1",
            user_id="user_1",
            user_input="Email me at john@example.com",
            assistant_response="Sure, I'll email john@example.com",
        )

        session = await session_store.load("sess_1", "user_1")
        user_events = session.get_events_by_type(EventType.USER)
        agent_events = session.get_events_by_type(EventType.AGENT)

        assert "[EMAIL]" in user_events[0].content
        assert "john@example.com" not in user_events[0].content

        assert "[EMAIL]" in agent_events[0].content
        assert "john@example.com" not in agent_events[0].content
    
    @pytest.mark.asyncio
    async def test_record_turn_multiple_times(self, engine, session_store):
        """Test recording multiple turns."""
        await engine.record_turn("sess_1", "user_1", "Hello!", "Hi!")
        await engine.record_turn("sess_1", "user_1", "How are you?", "I'm good!")
        await engine.record_turn("sess_1", "user_1", "Bye!", "Goodbye!")
        
        session = await session_store.load("sess_1", "user_1")
        assert session.turn_count() == 3
        assert len(session.events) == 6  # 3 user + 3 agent
    
    @pytest.mark.asyncio
    async def test_record_user_message(self, engine, session_store):
        """Test recording only a user message."""
        event = await engine.record_user_message(
            session_id="sess_1",
            user_id="user_1",
            content="Hello!",
        )
        
        assert event.type == EventType.USER
        assert event.content == "Hello!"
        
        session = await session_store.load("sess_1", "user_1")
        assert len(session.events) == 1
    
    @pytest.mark.asyncio
    async def test_record_assistant_message(self, engine, session_store):
        """Test recording only an assistant message."""
        event = await engine.record_assistant_message(
            session_id="sess_1",
            user_id="user_1",
            content="Hi there!",
        )
        
        assert event.type == EventType.AGENT
        assert event.content == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_record_tool_use(self, engine, session_store):
        """Test recording tool usage."""
        call_event, output_event = await engine.record_tool_use(
            session_id="sess_1",
            user_id="user_1",
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            tool_output="4",
        )
        
        assert call_event.type == EventType.TOOL_CALL
        assert output_event.type == EventType.TOOL_OUTPUT
        assert call_event.metadata.tool_name == "calculator"
        assert output_event.content == "4"
    
    # =========================================================================
    # FULL WORKFLOW TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_full_workflow_without_llm(self, engine, session_store):
        """Test the full workflow: prepare -> (external LLM) -> record."""
        # 1. Prepare context
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="What's 2+2?",
            system_instructions="You are a math tutor.",
        )
        
        # 2. Simulate external LLM call
        messages = context.to_openai_messages()
        assert len(messages) >= 2
        
        # Simulate response
        assistant_response = "2+2 equals 4."
        
        # 3. Record the turn
        await engine.record_turn(
            session_id="sess_1",
            user_id="user_1",
            user_input="What's 2+2?",
            assistant_response=assistant_response,
        )
        
        # 4. Verify session state
        session = await session_store.load("sess_1", "user_1")
        assert session.turn_count() == 1
        assert session.get_last_user_message().content == "What's 2+2?"
        assert session.get_last_agent_message().content == "2+2 equals 4."
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, engine, session_store):
        """Test multi-turn conversation with history."""
        # Turn 1
        await engine.record_turn("sess_1", "user_1", "Hello!", "Hi there!")
        
        # Turn 2 - context should include history
        context = await engine.prepare_context(
            session_id="sess_1",
            user_id="user_1",
            user_input="How are you?",
        )
        
        # History should be in context
        assert len(context.events) >= 2  # Previous turn
        
        messages = context.to_messages()
        # Should have: system, user (Hello), assistant (Hi there), user (How are you?)
        assert any("Hello" in str(m) for m in messages)
        assert any("Hi there" in str(m) for m in messages)
    
    # =========================================================================
    # SESSION MANAGEMENT TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_get_session(self, engine, session_store):
        """Test getting session."""
        await engine.record_turn("sess_1", "user_1", "Hello", "Hi")
        
        session = await engine.get_session("sess_1", "user_1")
        assert session.session_id == "sess_1"
        assert len(session.events) > 0
    
    @pytest.mark.asyncio
    async def test_update_session_state(self, engine, session_store):
        """Test updating session state."""
        await engine.record_turn("sess_1", "user_1", "Hello", "Hi")
        
        await engine.update_session_state(
            session_id="sess_1",
            user_id="user_1",
            cart_items=["item1", "item2"],
            user_verified=True,
        )
        
        session = await engine.get_session("sess_1", "user_1")
        assert session.state.get("cart_items") == ["item1", "item2"]
        assert session.state.get("user_verified") is True
    
    @pytest.mark.asyncio
    async def test_delete_session(self, engine, session_store):
        """Test deleting session."""
        await engine.record_turn("sess_1", "user_1", "Hello", "Hi")
        
        assert await session_store.exists("sess_1") is True
        result = await engine.delete_session("sess_1")
        assert result is True
        assert await session_store.exists("sess_1") is False
    
    # =========================================================================
    # MEMORY MANAGEMENT TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_add_memory(self, engine, memory_store):
        """Test adding memory through engine."""
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="User likes pizza",
        )
        
        memory_id = await engine.add_memory(memory)
        assert memory_id is not None
        
        retrieved = await memory_store.get(memory_id)
        assert retrieved.content == "User likes pizza"
    
    @pytest.mark.asyncio
    async def test_search_memories(self, engine, memory_store):
        """Test searching memories through engine."""
        await engine.add_memory(MemoryFactory.semantic_memory(
            user_id="user_1",
            content="User is vegetarian",
        ))
        await engine.add_memory(MemoryFactory.semantic_memory(
            user_id="user_1",
            content="User likes hiking",
        ))
        
        results = await engine.search_memories(
            user_id="user_1",
            query="food vegetarian",
        )
        
        assert len(results) >= 1
    
    @pytest.mark.asyncio
    async def test_get_user_memories(self, engine, memory_store):
        """Test getting all user memories."""
        for i in range(5):
            await engine.add_memory(MemoryFactory.semantic_memory(
                user_id="user_1",
                content=f"Memory {i}",
            ))
        
        memories = await engine.get_user_memories("user_1")
        assert len(memories) == 5
    
    @pytest.mark.asyncio
    async def test_get_memory(self, engine, memory_store):
        """Test getting a specific memory."""
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="Test memory",
        )
        memory_id = await engine.add_memory(memory)
        
        retrieved = await engine.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Test memory"
        
        # Non-existent memory
        missing = await engine.get_memory("non_existent_id")
        assert missing is None
    
    @pytest.mark.asyncio
    async def test_update_memory(self, engine, memory_store):
        """Test updating a memory."""
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="Original content",
            confidence=0.5,
        )
        memory_id = await engine.add_memory(memory)
        
        # Get and update
        retrieved = await engine.get_memory(memory_id)
        retrieved.update_content("Updated content")
        retrieved.update_confidence(0.9)
        retrieved.add_tag("verified")
        
        result = await engine.update_memory(retrieved)
        assert result is True
        
        # Verify update
        updated = await engine.get_memory(memory_id)
        assert updated.content == "Updated content"
        assert updated.confidence_score == 0.9
        assert "verified" in updated.tags
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, engine, memory_store):
        """Test deleting memory."""
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="To delete",
        )
        memory_id = await engine.add_memory(memory)
        
        result = await engine.delete_memory(memory_id)
        assert result is True
        
        retrieved = await memory_store.get(memory_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_deactivate_memory(self, engine, memory_store):
        """Test deactivating (soft deleting) a memory."""
        memory = MemoryFactory.semantic_memory(
            user_id="user_1",
            content="To deactivate",
        )
        memory_id = await engine.add_memory(memory)
        
        result = await engine.deactivate_memory(memory_id)
        assert result is True
        
        # Memory still exists but is inactive
        retrieved = await engine.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved.is_active is False
        
        # Should not appear in searches
        results = await engine.search_memories("user_1", "deactivate")
        assert all(m.memory_id != memory_id for m in results)
    
    # =========================================================================
    # LIFECYCLE TESTS
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_close(self, engine):
        """Test engine close."""
        await engine.record_turn("sess_1", "user_1", "Hello", "Hi")
        await engine.close()
        # Should not raise


class TestContextEngineFactory:
    """Tests for ctxforge factory methods."""
    
    @pytest.mark.asyncio
    async def test_create_default(self):
        """Test creating default engine."""
        from ctxforge.engine.factory import EngineFactory
        
        engine = await EngineFactory.create_default()
        assert engine is not None
        assert engine.session_store is not None
        assert engine.memory_store is not None
    
    @pytest.mark.asyncio
    async def test_create_minimal(self):
        """Test creating minimal engine."""
        from ctxforge.engine.factory import EngineFactory
        
        engine = await EngineFactory.create_minimal()
        assert engine is not None
        
        # Should work for basic operations
        context = await engine.prepare_context(
            session_id="test",
            user_id="user",
            user_input="Hello",
        )
        assert context is not None


class TestComponentRegistry:
    """Tests for ComponentRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        reg = ComponentRegistry()
        yield reg
        reg.clear()
    
    def test_register_session_store(self, registry):
        """Test registering session store."""
        @registry.register_session_store("custom")
        class CustomStore:
            pass
        
        assert registry.get_session_store("custom") == CustomStore
        assert "custom" in registry.list_session_stores()
    
    def test_register_memory_store(self, registry):
        """Test registering memory store."""
        @registry.register_memory_store("custom")
        class CustomStore:
            pass
        
        assert registry.get_memory_store("custom") == CustomStore
        assert "custom" in registry.list_memory_stores()
    
    def test_register_retriever(self, registry):
        """Test registering retriever."""
        @registry.register_retriever("custom")
        class CustomRetriever:
            pass
        
        assert registry.get_retriever("custom") == CustomRetriever
    
    def test_register_condenser(self, registry):
        """Test registering condenser."""
        @registry.register_condenser("custom")
        class CustomCondenser:
            pass
        
        assert registry.get_condenser("custom") == CustomCondenser
        # Deprecated aliases still resolve the same registration.
        assert registry.get_compactor("custom") == CustomCondenser
    
    def test_register_extractor(self, registry):
        """Test registering extractor."""
        @registry.register_extractor("custom")
        class CustomExtractor:
            pass
        
        assert registry.get_extractor("custom") == CustomExtractor
    
    def test_register_middleware(self, registry):
        """Test registering middleware."""
        @registry.register_middleware("custom")
        class CustomMiddleware:
            pass
        
        assert registry.get_middleware("custom") == CustomMiddleware
    
    def test_case_insensitive(self, registry):
        """Test that lookups are case insensitive."""
        @registry.register_session_store("MyStore")
        class MyStore:
            pass
        
        assert registry.get_session_store("mystore") == MyStore
        assert registry.get_session_store("MYSTORE") == MyStore
    
    def test_register_component_programmatic(self, registry):
        """Test programmatic registration."""
        class MyStore:
            pass
        
        registry.register_component("session_store", "programmatic", MyStore)
        assert registry.get_session_store("programmatic") == MyStore
    
    def test_clear(self, registry):
        """Test clearing registry."""
        @registry.register_session_store("test")
        class TestStore:
            pass
        
        registry.clear()
        assert registry.get_session_store("test") is None
