"""
Tests for Expertise Integration with ctxforge.

Tests the integration of expertise with Context, ctxforge,
and ContextAssembler.
"""


import pytest

from ctxforge.compaction.assembler import DefaultContextAssembler
from ctxforge.core.context import Context, ContextBuilder
from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session


# Test fixtures
@pytest.fixture
def expertise_items():
    """Create sample expertise items."""
    return [
        ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Always start with a friendly greeting",
            helpful_count=10,
            harmful_count=1,
        ),
        ExpertiseItem(
            item_id="form-00001",
            section=ExpertiseSection.FORMULAS,
            content="Price = Cost * (1 + Margin)",
            helpful_count=5,
            harmful_count=0,
        ),
        ExpertiseItem(
            item_id="mist-00001",
            section=ExpertiseSection.COMMON_MISTAKES,
            content="Don't forget to validate input",
            helpful_count=8,
            harmful_count=2,
        ),
    ]


@pytest.fixture
def expertise():
    """Create sample expertise."""
    exp = Expertise(
        expertise_id="test-exp",
        name="Test Expertise",
        domain="testing",
    )
    exp.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="Strategy content",
    )
    exp.add_item(
        section=ExpertiseSection.FORMULAS,
        content="Formula content",
    )
    return exp


@pytest.fixture
def session():
    """Create a sample session."""
    return Session(
        session_id="test-session",
        user_id="test-user",
    )


class TestContextWithExpertise:
    """Tests for Context class with expertise fields."""
    
    def test_context_has_expertise_fields(self):
        """Test Context has expertise fields."""
        context = Context(
            session_id="test",
            user_id="user",
        )
        
        assert hasattr(context, 'expertise_items')
        assert hasattr(context, 'expertise_items_used')
        assert hasattr(context, 'expertise_id')
        assert context.expertise_items == []
        assert context.expertise_items_used == []
        assert context.expertise_id is None
    
    def test_context_with_expertise_items(self, expertise_items):
        """Test Context with expertise items."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
            expertise_id="my-expertise",
        )
        
        assert len(context.expertise_items) == 3
        assert context.expertise_id == "my-expertise"
    
    def test_format_expertise(self, expertise_items):
        """Test expertise formatting."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
        )
        
        formatted = context._format_expertise()
        
        assert "Expertise Knowledge" in formatted
        assert "STRATEGIES" in formatted.upper()
        assert "FORMULAS" in formatted.upper()
        assert "COMMON MISTAKES" in formatted.upper()
        assert "friendly greeting" in formatted
    
    def test_mark_expertise_used(self, expertise_items):
        """Test marking expertise items as used."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
        )
        
        context.mark_expertise_used(["strat-00001", "form-00001"])
        
        assert "strat-00001" in context.expertise_items_used
        assert "form-00001" in context.expertise_items_used
    
    def test_to_prompt_includes_expertise(self, expertise_items):
        """Test to_prompt includes expertise."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
            current_query="Hello",
        )
        
        prompt = context.to_prompt()
        
        assert "Expertise Knowledge" in prompt
        assert "friendly greeting" in prompt
    
    def test_build_system_content_includes_expertise(self, expertise_items):
        """Test system content includes expertise."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
            system_instructions="You are a helpful assistant.",
        )
        
        content = context._build_system_content()
        
        assert "Expertise Knowledge" in content
    
    def test_estimate_tokens_includes_expertise(self, expertise_items):
        """Test token estimation includes expertise."""
        context_without = Context(
            session_id="test",
            user_id="user",
        )
        
        context_with = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
        )
        
        tokens_without = context_without.estimate_total_tokens()
        tokens_with = context_with.estimate_total_tokens()
        
        assert tokens_with > tokens_without
    
    def test_to_dict_includes_expertise(self, expertise_items):
        """Test to_dict includes expertise fields."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=expertise_items,
            expertise_id="my-exp",
            expertise_items_used=["strat-00001"],
        )
        
        data = context.to_dict()
        
        assert "expertise_id" in data
        assert "expertise_items" in data
        assert "expertise_items_used" in data
        assert data["expertise_id"] == "my-exp"


class TestContextBuilderWithExpertise:
    """Tests for ContextBuilder with expertise support."""
    
    def test_with_expertise_items(self, expertise_items):
        """Test adding expertise items."""
        builder = ContextBuilder("session", "user")
        builder.with_expertise_items(expertise_items)
        
        context = builder.build()
        
        assert len(context.expertise_items) == 3
    
    def test_with_expertise_id(self):
        """Test setting expertise ID."""
        builder = ContextBuilder("session", "user")
        builder.with_expertise_id("my-expertise")
        
        context = builder.build()
        
        assert context.expertise_id == "my-expertise"
    
    def test_with_expertise_items_used(self):
        """Test marking items as used."""
        builder = ContextBuilder("session", "user")
        builder.with_expertise_items_used(["strat-00001"])
        
        context = builder.build()
        
        assert "strat-00001" in context.expertise_items_used
    
    def test_full_builder_with_expertise(self, expertise_items):
        """Test full builder with all expertise fields."""
        context = (
            ContextBuilder("session", "user")
            .with_system_instructions("Be helpful")
            .with_current_query("Hello")
            .with_expertise_items(expertise_items)
            .with_expertise_id("my-exp")
            .with_expertise_items_used(["strat-00001"])
            .build()
        )
        
        assert context.expertise_id == "my-exp"
        assert len(context.expertise_items) == 3
        assert "strat-00001" in context.expertise_items_used


class TestDefaultContextAssemblerWithExpertise:
    """Tests for DefaultContextAssembler with expertise."""
    
    @pytest.mark.asyncio
    async def test_assemble_with_expertise(self, session, expertise_items):
        """Test assembling context with expertise items."""
        assembler = DefaultContextAssembler()
        
        context = await assembler.assemble(
            session=session,
            current_query="Hello",
            memories=[],
            system_instructions="Be helpful",
            expertise_items=expertise_items,
            expertise_id="test-exp",
        )
        
        assert len(context.expertise_items) == 3
        assert context.expertise_id == "test-exp"
        assert "expertise_item_count" in context.metadata
        assert context.metadata["expertise_item_count"] == 3
    
    @pytest.mark.asyncio
    async def test_assemble_without_expertise(self, session):
        """Test assembling context without expertise."""
        assembler = DefaultContextAssembler()
        
        context = await assembler.assemble(
            session=session,
            current_query="Hello",
            memories=[],
            system_instructions="Be helpful",
        )
        
        assert context.expertise_items == []
        assert context.expertise_id is None
    
    @pytest.mark.asyncio
    async def test_expertise_section_created(self, session, expertise_items):
        """Test that expertise section is created."""
        assembler = DefaultContextAssembler()
        
        context = await assembler.assemble(
            session=session,
            current_query="Hello",
            memories=[],
            expertise_items=expertise_items,
        )
        
        section = context.get_section("expertise")
        assert section is not None
        assert "Expertise Knowledge" in section.content
    
    @pytest.mark.asyncio
    async def test_expertise_section_priority(self, session, expertise_items):
        """Test expertise section has correct priority."""
        assembler = DefaultContextAssembler()
        
        context = await assembler.assemble(
            session=session,
            current_query="Hello",
            memories=[MemoryItem(
                memory_id="m1",
                user_id="user",
                content="Memory content",
                type=MemoryType.SEMANTIC,
            )],
            expertise_items=expertise_items,
        )
        
        expertise_section = context.get_section("expertise")
        memory_section = context.get_section("memories")
        
        # Expertise should have higher priority than memories
        assert expertise_section.priority > memory_section.priority
    
    def test_format_expertise(self, expertise_items):
        """Test _format_expertise method."""
        assembler = DefaultContextAssembler()
        
        formatted = assembler._format_expertise(expertise_items)
        
        assert "Expertise Knowledge" in formatted
        assert "STRATEGIES" in formatted.upper()
        assert "FORMULAS" in formatted.upper()


class TestContextEngineExpertiseMethods:
    """Tests for ctxforge expertise methods."""
    
    @pytest.mark.asyncio
    async def test_create_expertise(self):
        """Test creating expertise via ctxforge."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        expertise_store = InMemoryExpertiseStore()
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=expertise_store,
        )
        
        expertise = await engine.create_expertise(
            expertise_id="test-exp",
            name="Test Expertise",
            domain="testing",
        )
        
        assert expertise.expertise_id == "test-exp"
        assert expertise.name == "Test Expertise"
        
        # Verify it was saved
        loaded = await engine.load_expertise("test-exp")
        assert loaded is not None
        assert loaded.expertise_id == "test-exp"
    
    @pytest.mark.asyncio
    async def test_add_expertise_item(self):
        """Test adding items via ctxforge."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        expertise_store = InMemoryExpertiseStore()
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=expertise_store,
        )
        
        # Create expertise
        await engine.create_expertise(
            expertise_id="test-exp",
            name="Test",
        )
        
        # Add item
        item = await engine.add_expertise_item(
            expertise_id="test-exp",
            section=ExpertiseSection.STRATEGIES,
            content="New strategy",
        )
        
        assert item is not None
        assert item.content == "New strategy"
        
        # Verify it was saved
        loaded = await engine.load_expertise("test-exp")
        assert len(loaded.items) == 1
    
    @pytest.mark.asyncio
    async def test_retrieve_expertise_items_fallback(self):
        """Test retrieving items without retriever (fallback)."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        expertise_store = InMemoryExpertiseStore()
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=expertise_store,
        )
        
        # Create expertise with items
        await engine.create_expertise("test-exp", "Test")
        await engine.add_expertise_item(
            expertise_id="test-exp",
            section=ExpertiseSection.STRATEGIES,
            content="Strategy 1",
        )
        await engine.add_expertise_item(
            expertise_id="test-exp",
            section=ExpertiseSection.FORMULAS,
            content="Formula 1",
        )
        
        # Retrieve items
        items = await engine.retrieve_expertise_items(
            expertise_id="test-exp",
            query="test",
            limit=5,
        )
        
        assert len(items) == 2
    
    @pytest.mark.asyncio
    async def test_prepare_context_with_expertise(self):
        """Test prepare_context_with_expertise method."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        expertise_store = InMemoryExpertiseStore()
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=expertise_store,
        )
        
        # Create expertise with items
        await engine.create_expertise("test-exp", "Test")
        await engine.add_expertise_item(
            expertise_id="test-exp",
            section=ExpertiseSection.STRATEGIES,
            content="Be helpful",
        )
        
        # Prepare context with expertise
        context = await engine.prepare_context_with_expertise(
            session_id="test-session",
            user_id="test-user",
            user_input="Hello",
            expertise_id="test-exp",
        )
        
        assert context.expertise_id == "test-exp"
        assert len(context.expertise_items) == 1
        assert "expertise_id" in context.metadata


class TestExpertiseFeedbackFlow:
    """Tests for expertise feedback flow."""
    
    @pytest.mark.asyncio
    async def test_record_turn_with_feedback(self):
        """Test recording turn with feedback updates items."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.expertise.reflector import MockReflector
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        expertise_store = InMemoryExpertiseStore()
        
        # Create a mock reflector that returns helpful feedback
        mock_reflector = MockReflector(
            feedback_map={"strat-00001": UsageFeedback.HELPFUL},
        )
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=expertise_store,
            reflector=mock_reflector,
        )
        
        # Create expertise with item
        _expertise = await engine.create_expertise("test-exp", "Test")
        item = await engine.add_expertise_item(
            expertise_id="test-exp",
            section=ExpertiseSection.STRATEGIES,
            content="Strategy",
        )
        
        initial_helpful = item.helpful_count
        
        # Record turn with feedback
        _updated = await engine.record_turn_with_feedback(
            session_id="test-session",
            user_id="test-user",
            user_input="Hello",
            assistant_response="Hi there!",
            expertise_items_used=[item.item_id],
            outcome=TurnOutcome.SUCCESS,
            expertise_id="test-exp",
        )
        
        # Check item was updated
        loaded = await engine.load_expertise("test-exp")
        updated_item = loaded.get_item(item.item_id)
        assert updated_item.helpful_count > initial_helpful


class TestExpertiseEdgeCases:
    """Edge case tests for expertise integration."""
    
    def test_empty_expertise_format(self):
        """Test formatting empty expertise."""
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=[],
        )
        
        formatted = context._format_expertise()
        
        assert formatted == ""
    
    def test_expertise_with_standard_item(self):
        """Test expertise items use to_prompt_format method."""
        # Create a standard ExpertiseItem
        item = ExpertiseItem(
            item_id="test-001",
            section=ExpertiseSection.STRATEGIES,
            content="Standard content for testing",
        )
        
        context = Context(
            session_id="test",
            user_id="user",
            expertise_items=[item],
        )
        
        formatted = context._format_expertise()
        
        # to_prompt_format includes the section display name
        assert "Standard content for testing" in formatted
        assert "STRATEGIES AND INSIGHTS" in formatted
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_expertise(self):
        """Test loading non-existent expertise returns None."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=InMemoryExpertiseStore(),
        )
        
        result = await engine.load_expertise("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_add_item_to_nonexistent_expertise(self):
        """Test adding item to non-existent expertise returns None."""
        from ctxforge.config.defaults import DEFAULT_CONFIG
        from ctxforge.engine.context_engine import CtxForge
        from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        engine = CtxForge(
            config=DEFAULT_CONFIG,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            expertise_store=InMemoryExpertiseStore(),
        )
        
        result = await engine.add_expertise_item(
            expertise_id="nonexistent",
            section=ExpertiseSection.STRATEGIES,
            content="Test",
        )
        
        assert result is None

