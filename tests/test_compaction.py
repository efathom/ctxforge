"""
Tests for compaction implementations.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from ctxforge.compaction.assembler import DefaultContextAssembler, MinimalContextAssembler
from ctxforge.compaction.importance import ImportanceCondenser, default_importance_scorer
from ctxforge.compaction.sliding_window import SlidingWindowCondenser
from ctxforge.compaction.summarizing import SummarizingCondenser
from ctxforge.compaction.utils import estimate_event_tokens, estimate_tokens_simple
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.core.expertise import ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session
from ctxforge.protocols.compactor import CompactionConfig

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_session():
    """Create a basic session for testing."""
    return Session(
        session_id="test_session",
        user_id="test_user",
    )


@pytest.fixture
def session_with_events():
    """Create a session with multiple events."""
    session = Session(
        session_id="test_session",
        user_id="test_user",
    )
    
    # Add events with timestamps spread over time
    now = datetime.now(timezone.utc)
    events = [
        Event(
            event_id="evt_1",
            type=EventType.USER,
            content="Hello, I need help with my order",
            timestamp=now - timedelta(minutes=10),
        ),
        Event(
            event_id="evt_2",
            type=EventType.AGENT,
            content="Of course! I'd be happy to help with your order. Could you provide your order number?",
            timestamp=now - timedelta(minutes=9),
        ),
        Event(
            event_id="evt_3",
            type=EventType.USER,
            content="My order number is 12345",
            timestamp=now - timedelta(minutes=8),
        ),
        Event(
            event_id="evt_4",
            type=EventType.TOOL_CALL,
            content="lookup_order(order_id='12345')",
            timestamp=now - timedelta(minutes=7),
        ),
        Event(
            event_id="evt_5",
            type=EventType.TOOL_OUTPUT,
            content="Order 12345: Status=Shipped, ETA=Tomorrow",
            timestamp=now - timedelta(minutes=6),
        ),
        Event(
            event_id="evt_6",
            type=EventType.AGENT,
            content="I found your order! Order #12345 has been shipped and should arrive tomorrow.",
            timestamp=now - timedelta(minutes=5),
        ),
        Event(
            event_id="evt_7",
            type=EventType.USER,
            content="Great! Can I change the delivery address?",
            timestamp=now - timedelta(minutes=4),
        ),
        Event(
            event_id="evt_8",
            type=EventType.AGENT,
            content="Let me check if that's possible. Once an order is shipped, address changes may be limited.",
            timestamp=now - timedelta(minutes=3),
        ),
        Event(
            event_id="evt_9",
            type=EventType.USER,
            content="The new address is 123 Main Street",
            timestamp=now - timedelta(minutes=2),
        ),
        Event(
            event_id="evt_10",
            type=EventType.AGENT,
            content="I've noted the address change request. I'll need to verify if it can be processed.",
            timestamp=now - timedelta(minutes=1),
        ),
        Event(
            event_id="evt_11",
            type=EventType.USER,
            content="Thanks for your help!",
            timestamp=now,
        ),
    ]
    
    session.events.extend(events)
    return session


@pytest.fixture
def session_with_system_event():
    """Session that includes a system event."""
    session = Session(
        session_id="test_session",
        user_id="test_user",
    )
    
    now = datetime.now(timezone.utc)
    events = [
        Event(
            event_id="sys_1",
            type=EventType.SYSTEM,
            content="Session initialized with admin privileges",
            timestamp=now - timedelta(minutes=20),
        ),
        Event(
            event_id="evt_1",
            type=EventType.USER,
            content="Message 1",
            timestamp=now - timedelta(minutes=10),
        ),
        Event(
            event_id="evt_2",
            type=EventType.AGENT,
            content="Response 1",
            timestamp=now - timedelta(minutes=9),
        ),
        Event(
            event_id="evt_3",
            type=EventType.USER,
            content="Message 2",
            timestamp=now - timedelta(minutes=8),
        ),
        Event(
            event_id="evt_4",
            type=EventType.AGENT,
            content="Response 2",
            timestamp=now - timedelta(minutes=7),
        ),
        Event(
            event_id="evt_5",
            type=EventType.USER,
            content="Message 3",
            timestamp=now - timedelta(minutes=6),
        ),
        Event(
            event_id="evt_6",
            type=EventType.AGENT,
            content="Response 3",
            timestamp=now - timedelta(minutes=5),
        ),
    ]
    
    session.events.extend(events)
    return session


@pytest.fixture
def sample_memories():
    """Sample memories for testing assembler."""
    return [
        MemoryItem(
            memory_id="mem_1",
            user_id="test_user",
            content="User prefers quick responses",
            type=MemoryType.SEMANTIC,
            confidence_score=0.9,
        ),
        MemoryItem(
            memory_id="mem_2",
            user_id="test_user",
            content="User is a returning customer since 2020",
            type=MemoryType.SEMANTIC,
            confidence_score=0.95,
        ),
    ]


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestEstimateTokensSimple:
    """Tests for estimate_tokens_simple function."""
    
    def test_empty_string(self):
        """Empty string returns 1 (minimum)."""
        assert estimate_tokens_simple("") == 1
    
    def test_short_string(self):
        """Short string token estimation."""
        # 4 chars -> 4 // 4 + 1 = 2
        assert estimate_tokens_simple("test") == 2
    
    def test_longer_string(self):
        """Longer string token estimation."""
        # 16 chars -> 16 // 4 + 1 = 5
        text = "1234567890123456"
        assert estimate_tokens_simple(text) == 5
    
    def test_realistic_sentence(self):
        """Realistic sentence estimation."""
        sentence = "Hello, how can I help you today?"  # 32 chars
        tokens = estimate_tokens_simple(sentence)
        assert tokens == 9  # 32 // 4 + 1


class TestEstimateEventTokens:
    """Tests for estimate_event_tokens function."""
    
    def test_empty_list(self):
        """Empty event list returns 0."""
        assert estimate_event_tokens([]) == 0
    
    def test_single_event_has_positive_tokens(self):
        """Single event returns positive token count."""
        event = Event(
            event_id="1",
            type=EventType.USER,
            content="Hello",
        )
        tokens = estimate_event_tokens([event])
        # Should include content tokens at minimum
        assert tokens > 0
        assert tokens >= estimate_tokens_simple("Hello")
    
    def test_multiple_events_additive(self):
        """Multiple events token count is additive."""
        event1 = Event(event_id="1", type=EventType.USER, content="Hello")
        event2 = Event(event_id="2", type=EventType.AGENT, content="Hi there!")
        
        single_tokens = estimate_event_tokens([event1])
        both_tokens = estimate_event_tokens([event1, event2])
        
        # Combined should be greater than single
        assert both_tokens > single_tokens
    
    def test_content_contributes_to_tokens(self):
        """Longer content contributes more tokens."""
        short_event = Event(
            event_id="1",
            type=EventType.USER,
            content="Hi",  # 2 chars
        )
        long_event = Event(
            event_id="2",
            type=EventType.USER,
            content="Hello, this is a much longer message with many words!",
        )
        
        tokens_short = estimate_event_tokens([short_event])
        tokens_long = estimate_event_tokens([long_event])
        
        # Longer content should have more tokens
        assert tokens_long > tokens_short


# =============================================================================
# Test BaseCondenser
# =============================================================================


class TestBaseCondenser:
    """Tests for BaseCondenser base class functionality."""

    @pytest.fixture
    def concrete_condenser(self):
        """Create a concrete implementation of BaseCondenser for testing."""
        # Use SlidingWindowCondenser as a concrete implementation
        return SlidingWindowCondenser()

    @pytest.fixture
    def view_with_events(self, session_with_events):
        """Create a CompactionView from a session with events."""
        return CompactionView.from_session(session_with_events)

    @pytest.fixture
    def base_view(self, base_session):
        """Create a CompactionView from a base session."""
        return CompactionView.from_session(base_session)

    def test_should_condense_by_event_threshold(
        self, concrete_condenser, view_with_events
    ):
        """should_condense returns True when event count exceeds threshold."""
        config = CompactionConfig(event_threshold=5, token_threshold=999999)

        assert len(view_with_events) > 5
        assert concrete_condenser.should_condense(view_with_events, config)

    def test_should_condense_by_token_threshold(
        self, concrete_condenser, view_with_events
    ):
        """should_condense returns True when token count exceeds threshold."""
        config = CompactionConfig(event_threshold=999999, token_threshold=10)

        # View with events should have more than 10 tokens
        assert concrete_condenser.should_condense(view_with_events, config)

    def test_should_not_condense_under_thresholds(
        self, concrete_condenser, base_session
    ):
        """should_condense returns False when under all thresholds."""
        base_session.add_event(
            Event(
                event_id="1",
                type=EventType.USER,
                content="Short",
            )
        )
        view = CompactionView.from_session(base_session)

        config = CompactionConfig(event_threshold=100, token_threshold=10000)
        assert not concrete_condenser.should_condense(view, config)

    def test_should_condense_uses_default_config(
        self, concrete_condenser, view_with_events
    ):
        """should_condense uses default config when none provided."""
        # Default config has event_threshold=10, token_threshold=4000
        # View with 11 events should trigger condensation
        result = concrete_condenser.should_condense(view_with_events)
        assert result  # Should trigger on event count

    def test_estimate_tokens_delegates_to_util(self, concrete_condenser):
        """estimate_tokens uses estimate_event_tokens utility."""
        events = [
            Event(event_id="1", type=EventType.USER, content="Hello"),
            Event(event_id="2", type=EventType.AGENT, content="Hi there!"),
        ]

        expected = estimate_event_tokens(events)
        actual = concrete_condenser.estimate_tokens(events)
        assert actual == expected

    def test_separate_events_by_type_with_preservation(self, concrete_condenser):
        """_separate_events_by_type correctly separates when preservation enabled."""
        events = [
            Event(event_id="1", type=EventType.SYSTEM, content="System init"),
            Event(event_id="2", type=EventType.USER, content="Hello"),
            Event(event_id="3", type=EventType.AGENT, content="Hi"),
            Event(event_id="4", type=EventType.SYSTEM, content="System update"),
        ]

        config = CompactionConfig(preserve_system_events=True)
        system_events, other_events = concrete_condenser._separate_events_by_type(
            events, config
        )

        assert len(system_events) == 2
        assert len(other_events) == 2
        assert all(e.type == EventType.SYSTEM for e in system_events)
        assert all(e.type != EventType.SYSTEM for e in other_events)

    def test_separate_events_by_type_without_preservation(self, concrete_condenser):
        """_separate_events_by_type treats system events as regular when not preserved."""
        events = [
            Event(event_id="1", type=EventType.SYSTEM, content="System init"),
            Event(event_id="2", type=EventType.USER, content="Hello"),
        ]

        config = CompactionConfig(preserve_system_events=False)
        system_events, other_events = concrete_condenser._separate_events_by_type(
            events, config
        )

        assert len(system_events) == 0
        assert len(other_events) == 2

    def test_create_no_op_result(self, concrete_condenser, view_with_events):
        """_create_no_op_result creates correct result."""
        result = concrete_condenser._create_no_op_result(view_with_events)

        assert isinstance(result, CondensationResult)
        assert result.view == view_with_events
        assert result.summary_generated is False
        assert result.tokens_saved == 0
        assert result.metadata.get("strategy") == concrete_condenser.name

    def test_sort_by_timestamp(self, concrete_condenser):
        """_sort_by_timestamp sorts events chronologically."""
        now = datetime.now(timezone.utc)
        events = [
            Event(event_id="3", type=EventType.USER, content="C", timestamp=now),
            Event(
                event_id="1",
                type=EventType.USER,
                content="A",
                timestamp=now - timedelta(hours=2),
            ),
            Event(
                event_id="2",
                type=EventType.USER,
                content="B",
                timestamp=now - timedelta(hours=1),
            ),
        ]

        sorted_events = concrete_condenser._sort_by_timestamp(events)

        assert sorted_events[0].event_id == "1"  # Oldest
        assert sorted_events[1].event_id == "2"
        assert sorted_events[2].event_id == "3"  # Newest


# =============================================================================
# Test SlidingWindowCondenser
# =============================================================================


class TestSlidingWindowCondenser:
    """Tests for SlidingWindowCondenser."""

    def test_should_condense_by_event_count(self, session_with_events):
        """Detects when condensation is needed by event count."""
        condenser = SlidingWindowCondenser()
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(event_threshold=5)

        assert len(view) > 5
        assert condenser.should_condense(view, config)

    def test_should_not_condense_under_threshold(self, base_session):
        """Doesn't condense when under threshold."""
        condenser = SlidingWindowCondenser()
        base_session.add_event(
            Event(
                event_id="1",
                type=EventType.USER,
                content="Short message",
            )
        )
        view = CompactionView.from_session(base_session)

        config = CompactionConfig(event_threshold=10)
        assert not condenser.should_condense(view, config)

    @pytest.mark.asyncio
    async def test_condense_keeps_recent(self, session_with_events):
        """Keeps most recent events after condensation."""
        condenser = SlidingWindowCondenser()
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=3)

        original_count = len(view)
        result = await condenser.condense(view, config)

        assert isinstance(result, CondensationResult)
        kept_count = len(result.view.to_context_events())
        assert original_count - kept_count > 0  # Some events were removed

        # Check that kept events are the most recent (evt_11 is most recent)
        kept_ids = [e.event_id for e in result.view.to_context_events()]
        assert "evt_11" in kept_ids  # Most recent

    @pytest.mark.asyncio
    async def test_condense_preserves_system_events(self, session_with_system_event):
        """Preserves system events during condensation."""
        condenser = SlidingWindowCondenser()
        view = CompactionView.from_session(session_with_system_event)
        config = CompactionConfig(
            keep_recent=2,
            preserve_system_events=True,
        )

        result = await condenser.condense(view, config)

        # System event should still be present
        system_events = [
            e for e in result.view.to_context_events() if e.type == EventType.SYSTEM
        ]
        assert len(system_events) == 1

    @pytest.mark.asyncio
    async def test_condense_no_summarization(self, session_with_events):
        """Sliding window doesn't create summaries."""
        condenser = SlidingWindowCondenser()
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=3)

        result = await condenser.condense(view, config)

        assert result.summary_generated is False

    @pytest.mark.asyncio
    async def test_tokens_saved_calculation(self, session_with_events):
        """Calculates tokens saved correctly."""
        condenser = SlidingWindowCondenser()
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=3)

        result = await condenser.condense(view, config)
        
        assert result.tokens_saved > 0
    
    def test_name_property(self):
        """Has correct name."""
        condenser = SlidingWindowCondenser()
        assert condenser.name == "sliding_window"


# =============================================================================
# Test SummarizingCondenser
# =============================================================================


class TestSummarizingCondenser:
    """Tests for SummarizingCondenser."""

    @pytest.mark.asyncio
    async def test_creates_summary(self, session_with_events):
        """Creates a summary of condensed events."""

        async def mock_summarize(text: str, existing: Optional[str] = None) -> str:
            return "Summary: Customer inquired about order #12345 and requested address change."

        condenser = SummarizingCondenser(mock_summarize)
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=3)

        result = await condenser.condense(view, config)

        assert isinstance(result, CondensationResult)
        assert result.summary_generated is True
        assert result.view.summary is not None
        assert "12345" in result.view.summary

    @pytest.mark.asyncio
    async def test_simple_summarization_fallback(self, session_with_events):
        """Falls back to simple summarization without LLM."""
        condenser = SummarizingCondenser()  # No LLM function
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=3)

        result = await condenser.condense(view, config)

        assert isinstance(result, CondensationResult)
        # Simple summarization should produce some summary or empty string
        assert result.view.summary is not None or result.view.summary == ""

    @pytest.mark.asyncio
    async def test_extends_existing_summary(self, session_with_events):
        """Extends existing summary."""
        session_with_events.summary = "Previous: User contacted about shipping."
        view = CompactionView.from_session(session_with_events)

        async def mock_summarize(text: str, existing: Optional[str] = None) -> str:
            if existing:
                return f"{existing} Updated: Address change requested."
            return "New summary"

        condenser = SummarizingCondenser(mock_summarize)
        config = CompactionConfig(keep_recent=3)

        result = await condenser.condense(view, config)

        assert "Previous" in result.view.summary
        assert "Updated" in result.view.summary

    def test_name_property(self):
        """Has correct name."""
        condenser = SummarizingCondenser()
        assert condenser.name == "summarizing"


# =============================================================================
# Test ImportanceCondenser
# =============================================================================


class TestImportanceCondenser:
    """Tests for ImportanceCondenser."""

    def test_default_scorer_system_events(self):
        """Default scorer rates system events highly."""
        event = Event(
            event_id="1",
            type=EventType.SYSTEM,
            content="System initialized",
        )
        score = default_importance_scorer(event)
        assert score > 0.7

    def test_default_scorer_questions(self):
        """Default scorer boosts questions."""
        event = Event(
            event_id="1",
            type=EventType.USER,
            content="What is my order status?",
        )
        score = default_importance_scorer(event)

        non_question = Event(
            event_id="2",
            type=EventType.USER,
            content="Here is my order status",
        )
        non_question_score = default_importance_scorer(non_question)

        assert score > non_question_score

    def test_default_scorer_decisions(self):
        """Default scorer boosts decisions."""
        event = Event(
            event_id="1",
            type=EventType.USER,
            content="I decided to cancel the order",
        )
        score = default_importance_scorer(event)
        assert score > 0.6

    @pytest.mark.asyncio
    async def test_keeps_important_events(self, session_with_events):
        """Keeps events based on importance."""
        condenser = ImportanceCondenser()
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=5)

        result = await condenser.condense(view, config)

        assert isinstance(result, CondensationResult)
        kept_events = result.view.to_context_events()
        assert len(kept_events) <= 5

    @pytest.mark.asyncio
    async def test_custom_scorer(self, session_with_events):
        """Uses custom scoring function."""

        def custom_scorer(event: Event) -> float:
            if "order" in event.content.lower():
                return 1.0
            return 0.0

        condenser = ImportanceCondenser(scoring_func=custom_scorer)
        view = CompactionView.from_session(session_with_events)
        config = CompactionConfig(keep_recent=5)

        result = await condenser.condense(view, config)

        # Events with "order" should be kept
        kept_contents = [e.content for e in result.view.to_context_events()]
        order_events = [c for c in kept_contents if "order" in c.lower()]
        assert len(order_events) > 0

    def test_get_event_scores(self, session_with_events):
        """Can get scores for inspection."""
        condenser = ImportanceCondenser()
        scores = condenser.get_event_scores(session_with_events.events)

        assert len(scores) == len(session_with_events.events)
        for _preview, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_name_property(self):
        """Has correct name."""
        condenser = ImportanceCondenser()
        assert condenser.name == "importance"


# =============================================================================
# Test DefaultContextAssembler
# =============================================================================

class TestDefaultContextAssembler:
    """Tests for DefaultContextAssembler."""
    
    @pytest.mark.asyncio
    async def test_assemble_basic(self, base_session, sample_memories):
        """Assembles basic context."""
        assembler = DefaultContextAssembler()
        context = await assembler.assemble(
            session=base_session,
            current_query="What's my order status?",
            memories=sample_memories,
            system_instructions="You are a helpful assistant.",
        )
        
        assert context is not None
        assert len(context.sections) > 0
    
    @pytest.mark.asyncio
    async def test_includes_system_instructions(self, base_session, sample_memories):
        """Includes system instructions."""
        assembler = DefaultContextAssembler()
        context = await assembler.assemble(
            session=base_session,
            current_query="test",
            memories=[],
            system_instructions="You are a helpful assistant.",
        )
        
        system_sections = [s for s in context.sections if s.name == "system_instructions"]
        assert len(system_sections) == 1
        assert "helpful assistant" in system_sections[0].content
    
    @pytest.mark.asyncio
    async def test_includes_memories_bullet(self, base_session, sample_memories):
        """Includes memories in bullet format."""
        assembler = DefaultContextAssembler(memory_format="bullet")
        context = await assembler.assemble(
            session=base_session,
            current_query="test",
            memories=sample_memories,
            system_instructions="",
        )
        
        memory_sections = [s for s in context.sections if s.name == "memories"]
        assert len(memory_sections) == 1
        assert "•" in memory_sections[0].content
    
    @pytest.mark.asyncio
    async def test_includes_memories_prose(self, base_session, sample_memories):
        """Includes memories in prose format."""
        assembler = DefaultContextAssembler(memory_format="prose")
        context = await assembler.assemble(
            session=base_session,
            current_query="test",
            memories=sample_memories,
            system_instructions="",
        )
        
        memory_sections = [s for s in context.sections if s.name == "memories"]
        assert len(memory_sections) == 1
        assert "•" not in memory_sections[0].content

    @pytest.mark.asyncio
    async def test_includes_memories_json(self, base_session, sample_memories):
        """Includes memories in JSON format and uses correct MemoryItem fields."""
        assembler = DefaultContextAssembler(memory_format="json")
        context = await assembler.assemble(
            session=base_session,
            current_query="test",
            memories=sample_memories,
            system_instructions="",
        )

        memory_sections = [s for s in context.sections if s.name == "memories"]
        assert len(memory_sections) == 1
        assert '"type"' in memory_sections[0].content
        assert "semantic" in memory_sections[0].content.lower()
    
    @pytest.mark.asyncio
    async def test_includes_conversation_history(self, session_with_events, sample_memories):
        """Includes conversation history."""
        assembler = DefaultContextAssembler()
        context = await assembler.assemble(
            session=session_with_events,
            current_query="test",
            memories=[],
            system_instructions="",
        )
        
        history_sections = [s for s in context.sections if s.name == "conversation_history"]
        assert len(history_sections) == 1
        assert "User:" in history_sections[0].content
        assert "Assistant:" in history_sections[0].content
    
    @pytest.mark.asyncio
    async def test_includes_session_summary(self, base_session, sample_memories):
        """Includes session summary if present."""
        base_session.summary = "Previous conversation about orders."
        
        assembler = DefaultContextAssembler()
        context = await assembler.assemble(
            session=base_session,
            current_query="test",
            memories=[],
            system_instructions="",
        )
        
        summary_sections = [s for s in context.sections if s.name == "session_summary"]
        assert len(summary_sections) == 1
        assert "orders" in summary_sections[0].content
    
    @pytest.mark.asyncio
    async def test_adds_metadata(self, base_session, sample_memories):
        """Adds metadata to context."""
        assembler = DefaultContextAssembler()
        context = await assembler.assemble(
            session=base_session,
            current_query="test",
            memories=sample_memories,
            system_instructions="",
        )
        
        assert context.metadata["session_id"] == "test_session"
        assert context.metadata["user_id"] == "test_user"
        assert context.metadata["memory_count"] == 2
    
    @pytest.mark.asyncio
    async def test_fit_to_budget(self, session_with_events, sample_memories):
        """Fits context to token budget - verifies budget trimming occurs."""
        assembler = DefaultContextAssembler()
        
        # Create a large context that exceeds the budget
        context = await assembler.assemble(
            session=session_with_events,
            current_query="test",
            memories=sample_memories,
            system_instructions="System instructions " * 100,
            token_budget=100,  # Very small budget
        )
        
        # Verify that budget trimming was attempted
        assert context.metadata.get("budget_trimmed") is True
        # Sections should have been removed (originally would have 4 sections)
        assert len(context.sections) < 4

    @pytest.mark.asyncio
    async def test_fit_to_budget_preserves_expertise_fields(self, session_with_events, sample_memories):
        """Budget trimming must preserve expertise fields on Context."""
        assembler = DefaultContextAssembler()
        expertise_items = [
            ExpertiseItem(
                item_id="strat-00001",
                section=ExpertiseSection.STRATEGIES,
                content="Always validate input",
            )
        ]

        context = await assembler.assemble(
            session=session_with_events,
            current_query="test",
            memories=sample_memories,
            system_instructions="System instructions " * 100,
            token_budget=100,
            expertise_items=expertise_items,
            expertise_id="exp-1",
        )

        assert context.metadata.get("budget_trimmed") is True
        assert context.expertise_id == "exp-1"
        assert len(context.expertise_items) == 1
        assert context.expertise_items[0].item_id == "strat-00001"
    
    def test_name_property(self):
        """Has correct name."""
        assembler = DefaultContextAssembler()
        assert assembler.name == "default"


class TestMinimalContextAssembler:
    """Tests for MinimalContextAssembler."""
    
    @pytest.mark.asyncio
    async def test_assembles_minimal(self, session_with_events):
        """Assembles minimal context."""
        assembler = MinimalContextAssembler()
        context = await assembler.assemble(
            session=session_with_events,
            current_query="test",
            memories=[],  # Ignored
            system_instructions="Be helpful.",
        )
        
        # Should have only system and history
        section_names = [s.name for s in context.sections]
        assert len(section_names) <= 2
    
    @pytest.mark.asyncio
    async def test_keeps_recent_events_only(self, session_with_events):
        """Only keeps recent events."""
        assembler = MinimalContextAssembler()
        context = await assembler.assemble(
            session=session_with_events,
            current_query="test",
            memories=[],
            system_instructions="",
        )
        
        history = next((s for s in context.sections if s.name == "history"), None)
        if history:
            # Should not have all events
            assert history.content.count("user") < len(session_with_events.events)
    
    def test_name_property(self):
        """Has correct name."""
        assembler = MinimalContextAssembler()
        assert assembler.name == "minimal"


# =============================================================================
# Test Registry Integration
# =============================================================================

class TestCondenserRegistry:
    """Tests for condenser registry integration."""
    
    def test_sliding_window_registered(self):
        """SlidingWindowCondenser is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_condenser("sliding_window") is not None
    
    def test_summarizing_registered(self):
        """SummarizingCondenser is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_condenser("summarizing") is not None
    
    def test_importance_registered(self):
        """ImportanceCondenser is registered."""
        from ctxforge.engine.registry import registry
        assert registry.get_condenser("importance") is not None

    def test_compactor_apis_still_work(self):
        """Deprecated compactor registry aliases still resolve condensers."""
        from ctxforge.engine.registry import registry
        assert registry.get_compactor("sliding_window") is registry.get_condenser("sliding_window")

