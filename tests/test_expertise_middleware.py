"""
Tests for expertise middleware.

Tests the ExpertiseRetrievalMiddleware, ExpertiseEvolutionMiddleware,
and ExpertiseAuditMiddleware.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest
import pytest_asyncio

from ctxforge.core.expertise import (
    CompletedTurn,
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ReflectionResult,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.middleware.base import MiddlewareChain
from ctxforge.middleware.expertise import (
    ExpertiseAuditMiddleware,
    ExpertiseEvolutionMiddleware,
    ExpertiseRetrievalMiddleware,
)
from ctxforge.middleware.expertise.middleware import (
    CURATION_PLAN_KEY,
    EXPERTISE_ID_KEY,
    EXPERTISE_ITEMS_KEY,
    EXPERTISE_ITEMS_USED_KEY,
    REFLECTION_RESULT_KEY,
    TURN_OUTCOME_KEY,
)
from ctxforge.middleware.protocol import MiddlewareContext

# =============================================================================
# Mock Implementations
# =============================================================================


class MockExpertiseStore:
    """Mock expertise store for testing."""
    
    def __init__(self):
        self._store: Dict[str, Expertise] = {}
        self.save_count = 0
    
    async def get(self, expertise_id: str) -> Optional[Expertise]:
        return self._store.get(expertise_id)
    
    async def save(self, expertise: Expertise) -> None:
        self._store[expertise.expertise_id] = expertise
        self.save_count += 1
    
    async def delete(self, expertise_id: str) -> bool:
        if expertise_id in self._store:
            del self._store[expertise_id]
            return True
        return False
    
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Expertise]:
        return list(self._store.values())[offset:offset + limit]
    
    async def exists(self, expertise_id: str) -> bool:
        return expertise_id in self._store


class MockExpertiseRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, items_to_return: Optional[List[ExpertiseItem]] = None):
        self._items = items_to_return or []
        self.retrieve_count = 0
        self.last_query: Optional[str] = None
    
    async def retrieve(
        self,
        query: str,
        expertise_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[ExpertiseItem]:
        self.retrieve_count += 1
        self.last_query = query
        return self._items[:top_k]


class MockReflector:
    """Mock reflector for testing."""
    
    def __init__(
        self,
        feedback: Optional[Dict[str, UsageFeedback]] = None,
        confidence: float = 0.8,
        suggestions: bool = False,
    ):
        self._feedback = feedback or {}
        self._confidence = confidence
        self._suggestions = suggestions
        self.reflect_count = 0
    
    async def reflect(
        self,
        turn: CompletedTurn,
        items_used: List[ExpertiseItem],
        outcome: TurnOutcome,
    ) -> ReflectionResult:
        self.reflect_count += 1
        
        # Auto-generate feedback if not provided
        feedback = dict(self._feedback)
        if not feedback:
            for item in items_used:
                if outcome == TurnOutcome.SUCCESS:
                    feedback[item.item_id] = UsageFeedback.HELPFUL
                elif outcome == TurnOutcome.FAILURE:
                    feedback[item.item_id] = UsageFeedback.HARMFUL
        
        return ReflectionResult(
            item_feedback=feedback,
            confidence=self._confidence,
            suggested_additions=["new insight"] if self._suggestions else [],
            suggested_removals=[],
        )


class MockCurator:
    """Mock curator for testing."""
    
    def __init__(self, operations: Optional[List[CurationOp]] = None):
        self._operations = operations or []
        self.curate_count = 0
    
    async def curate(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> Tuple[Expertise, CurationPlan]:
        self.curate_count += 1
        
        plan = CurationPlan(
            operations=self._operations,
            reasoning="Mock curation",
        )
        
        return expertise, plan


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_expertise():
    """Create a sample expertise for testing."""
    expertise = Expertise(
        expertise_id="test-expertise",
        name="Test Expertise",
        domain="testing",
    )
    expertise.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="Always verify inputs before processing",
    )
    expertise.add_item(
        section=ExpertiseSection.COMMON_MISTAKES,
        content="Don't forget to handle edge cases",
    )
    return expertise


@pytest.fixture
def sample_items():
    """Create sample expertise items."""
    return [
        ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Always verify inputs",
        ),
        ExpertiseItem(
            item_id="mist-00001",
            section=ExpertiseSection.COMMON_MISTAKES,
            content="Handle edge cases",
        ),
    ]


@pytest_asyncio.fixture
async def mock_store(sample_expertise):
    """Create a mock store with sample expertise."""
    store = MockExpertiseStore()
    await store.save(sample_expertise)
    store.save_count = 0  # Reset count
    return store


@pytest.fixture
def mock_retriever(sample_items):
    """Create a mock retriever."""
    return MockExpertiseRetriever(items_to_return=sample_items)


@pytest.fixture
def mock_reflector():
    """Create a mock reflector."""
    return MockReflector()


@pytest.fixture
def mock_curator():
    """Create a mock curator."""
    return MockCurator()


# =============================================================================
# ExpertiseRetrievalMiddleware Tests
# =============================================================================


class TestExpertiseRetrievalMiddleware:
    """Tests for ExpertiseRetrievalMiddleware."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_retriever):
        """Test middleware initialization."""
        middleware = ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            default_expertise_id="default-exp",
            top_k=5,
            min_score=0.5,
        )
        
        assert middleware.name == "expertise_retrieval"
        assert middleware.enabled is True
    
    @pytest.mark.asyncio
    async def test_retrieve_items_with_expertise_id(self, mock_retriever, sample_items):
        """Test retrieving items when expertise ID is in context."""
        middleware = ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            top_k=10,
        )
        
        context = MiddlewareContext(user_input="How do I handle errors?")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        
        async def terminal(ctx):
            return ctx
        
        result = await middleware.process(context, terminal)
        
        assert mock_retriever.retrieve_count == 1
        assert mock_retriever.last_query == "How do I handle errors?"
        assert result.get_metadata(EXPERTISE_ITEMS_KEY) == sample_items
        assert result.has_flag("expertise_retrieved")
    
    @pytest.mark.asyncio
    async def test_use_default_expertise_id(self, mock_retriever):
        """Test using default expertise ID."""
        middleware = ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            default_expertise_id="default-exp",
        )
        
        context = MiddlewareContext(user_input="Test query")
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_retriever.retrieve_count == 1
    
    @pytest.mark.asyncio
    async def test_skip_without_expertise_id(self, mock_retriever):
        """Test skipping when no expertise ID is available."""
        middleware = ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
        )
        
        context = MiddlewareContext(user_input="Test query")
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_retriever.retrieve_count == 0
    
    @pytest.mark.asyncio
    async def test_disabled_middleware(self, mock_retriever):
        """Test that disabled middleware is skipped."""
        middleware = ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            default_expertise_id="test",
            enabled=False,
        )
        
        context = MiddlewareContext(user_input="Test query")
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_retriever.retrieve_count == 0
    
    @pytest.mark.asyncio
    async def test_uses_processed_input(self, mock_retriever):
        """Test that processed_input is used if available."""
        middleware = ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            default_expertise_id="test",
        )
        
        context = MiddlewareContext(user_input="Original input")
        context.processed_input = "Processed input"
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_retriever.last_query == "Processed input"


# =============================================================================
# ExpertiseEvolutionMiddleware Tests
# =============================================================================


class TestExpertiseEvolutionMiddleware:
    """Tests for ExpertiseEvolutionMiddleware."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_reflector, mock_store):
        """Test middleware initialization."""
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            evolve_on_success=True,
            evolve_on_failure=True,
        )
        
        assert middleware.name == "expertise_evolution"
        assert middleware.enabled is True
    
    @pytest.mark.asyncio
    async def test_evolve_on_success(self, mock_reflector, mock_store, sample_expertise):
        """Test evolution on successful turn."""
        # Set up feedback
        mock_reflector._feedback = {
            "strat-00001": UsageFeedback.HELPFUL,
        }
        
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            evolve_on_success=True,
        )
        
        context = MiddlewareContext(
            user_input="Test input",
            agent_response="Test response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        context.set_metadata(EXPERTISE_ITEMS_USED_KEY, ["strat-00001"])
        
        async def terminal(ctx):
            return ctx
        
        result = await middleware.process(context, terminal)
        
        assert mock_reflector.reflect_count == 1
        assert mock_store.save_count == 1
        assert result.has_flag("expertise_evolved")
    
    @pytest.mark.asyncio
    async def test_evolve_on_failure(self, mock_reflector, mock_store, sample_expertise):
        """Test evolution on failed turn."""
        mock_reflector._feedback = {
            "strat-00001": UsageFeedback.HARMFUL,
        }
        
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            evolve_on_failure=True,
        )
        
        context = MiddlewareContext(
            user_input="Test input",
            agent_response="Wrong response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.FAILURE)
        context.set_metadata(EXPERTISE_ITEMS_USED_KEY, ["strat-00001"])
        
        async def terminal(ctx):
            return ctx
        
        result = await middleware.process(context, terminal)
        
        assert mock_reflector.reflect_count == 1
        assert result.has_flag("expertise_evolved")
    
    @pytest.mark.asyncio
    async def test_skip_without_outcome(self, mock_reflector, mock_store):
        """Test skipping when no outcome is specified."""
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
        )
        
        context = MiddlewareContext(user_input="Test input")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        # No TURN_OUTCOME_KEY set
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_reflector.reflect_count == 0
    
    @pytest.mark.asyncio
    async def test_skip_success_when_disabled(self, mock_reflector, mock_store):
        """Test skipping success evolution when disabled."""
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            evolve_on_success=False,
        )
        
        context = MiddlewareContext(user_input="Test input")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_reflector.reflect_count == 0
    
    @pytest.mark.asyncio
    async def test_auto_curate(self, mock_reflector, mock_store, mock_curator, sample_expertise):
        """Test automatic curation when suggestions are present."""
        mock_reflector._suggestions = True
        mock_curator._operations = [
            CurationOp(
                type=CuratorOperation.ADD,
                section=ExpertiseSection.STRATEGIES,
                content="New strategy",
            )
        ]
        
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            curator=mock_curator,
            auto_curate=True,
        )
        
        context = MiddlewareContext(
            user_input="Test input",
            agent_response="Test response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        
        async def terminal(ctx):
            return ctx
        
        result = await middleware.process(context, terminal)
        
        assert mock_curator.curate_count == 1
        assert result.has_flag("expertise_curated")
        assert result.get_metadata(CURATION_PLAN_KEY) is not None
    
    @pytest.mark.asyncio
    async def test_confidence_threshold(self, mock_reflector, mock_store):
        """Test that low confidence reflections are ignored."""
        mock_reflector._confidence = 0.3
        
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            min_confidence=0.5,
        )
        
        context = MiddlewareContext(
            user_input="Test input",
            agent_response="Test response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        
        async def terminal(ctx):
            return ctx
        
        result = await middleware.process(context, terminal)
        
        assert mock_reflector.reflect_count == 1
        # Low confidence means expertise should not be saved
        assert mock_store.save_count == 0
        assert not result.has_flag("expertise_evolved")


# =============================================================================
# ExpertiseAuditMiddleware Tests
# =============================================================================


class TestExpertiseAuditMiddleware:
    """Tests for ExpertiseAuditMiddleware."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test middleware initialization."""
        middleware = ExpertiseAuditMiddleware(
            log_items_retrieved=True,
            log_items_used=True,
            log_reflection=True,
            log_curation=True,
        )
        
        assert middleware.name == "expertise_audit"
        assert middleware.enabled is True
        assert middleware.events == []
    
    @pytest.mark.asyncio
    async def test_log_retrieved_items(self, sample_items):
        """Test logging retrieved items."""
        middleware = ExpertiseAuditMiddleware()
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(EXPERTISE_ITEMS_KEY, sample_items)
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert len(middleware.events) == 1
        event = middleware.events[0]
        assert event["expertise_id"] == "test-expertise"
        assert event["items_retrieved"] == 2
        assert "strat-00001" in event["item_ids_retrieved"]
    
    @pytest.mark.asyncio
    async def test_log_used_items(self):
        """Test logging used items."""
        middleware = ExpertiseAuditMiddleware()
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(EXPERTISE_ITEMS_USED_KEY, ["strat-00001", "mist-00001"])
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        event = middleware.events[0]
        assert event["items_used"] == ["strat-00001", "mist-00001"]
    
    @pytest.mark.asyncio
    async def test_log_reflection(self):
        """Test logging reflection results."""
        middleware = ExpertiseAuditMiddleware()
        
        reflection = ReflectionResult(
            item_feedback={
                "strat-00001": UsageFeedback.HELPFUL,
                "mist-00001": UsageFeedback.HARMFUL,
            },
            confidence=0.9,
            suggested_additions=["new insight"],
        )
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(REFLECTION_RESULT_KEY, reflection)
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        event = middleware.events[0]
        assert "reflection" in event
        assert event["reflection"]["confidence"] == 0.9
        assert event["reflection"]["has_suggestions"] is True
    
    @pytest.mark.asyncio
    async def test_log_curation(self):
        """Test logging curation plans."""
        middleware = ExpertiseAuditMiddleware()
        
        plan = CurationPlan(
            operations=[
                CurationOp(
                    type=CuratorOperation.ADD,
                    section=ExpertiseSection.STRATEGIES,
                    content="New strategy",
                ),
            ],
            reasoning="Test curation",
        )
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(CURATION_PLAN_KEY, plan)
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        event = middleware.events[0]
        assert "curation" in event
        assert event["curation"]["operation_count"] == 1
    
    @pytest.mark.asyncio
    async def test_custom_handler(self):
        """Test custom event handler."""
        handled_events = []
        
        def handler(event):
            handled_events.append(event)
        
        middleware = ExpertiseAuditMiddleware(custom_handler=handler)
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert len(handled_events) == 1
    
    @pytest.mark.asyncio
    async def test_skip_without_expertise_id(self):
        """Test skipping when no expertise ID."""
        middleware = ExpertiseAuditMiddleware()
        
        context = MiddlewareContext(user_input="Test query")
        # No expertise ID
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert len(middleware.events) == 0
    
    @pytest.mark.asyncio
    async def test_clear_events(self, sample_items):
        """Test clearing events."""
        middleware = ExpertiseAuditMiddleware()
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        assert len(middleware.events) == 1
        
        middleware.clear_events()
        assert len(middleware.events) == 0


# =============================================================================
# Middleware Chain Integration Tests
# =============================================================================


class TestMiddlewareChainIntegration:
    """Tests for middleware chain integration."""
    
    @pytest.mark.asyncio
    async def test_full_chain(
        self,
        mock_retriever,
        mock_reflector,
        mock_store,
        mock_curator,
        sample_items,
    ):
        """Test full middleware chain with all expertise middleware."""
        # Set up reflector to return suggestions
        mock_reflector._suggestions = True
        mock_curator._operations = [
            CurationOp(
                type=CuratorOperation.ADD,
                section=ExpertiseSection.STRATEGIES,
                content="New insight",
            )
        ]
        
        # Build chain
        chain = MiddlewareChain()
        chain.add(ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            default_expertise_id="test-expertise",
        ))
        chain.add(ExpertiseAuditMiddleware())
        chain.add(ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
            curator=mock_curator,
        ))
        
        # Create context with outcome
        context = MiddlewareContext(
            user_input="How do I handle errors?",
            agent_response="Here's how to handle errors...",
            user_id="user123",
            session_id="session456",
        )
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        context.set_metadata(EXPERTISE_ITEMS_USED_KEY, ["strat-00001"])
        
        # Execute chain
        result = await chain.execute(context)
        
        assert result.success is True
        assert mock_retriever.retrieve_count == 1
        assert mock_reflector.reflect_count == 1
        assert mock_curator.curate_count == 1
        
        # Check flags
        assert result.context.has_flag("expertise_retrieved")
        assert result.context.has_flag("expertise_evolved")
        assert result.context.has_flag("expertise_curated")
    
    @pytest.mark.asyncio
    async def test_chain_with_disabled_middleware(self, mock_retriever, mock_reflector, mock_store):
        """Test chain with some middleware disabled."""
        chain = MiddlewareChain()
        chain.add(ExpertiseRetrievalMiddleware(
            retriever=mock_retriever,
            default_expertise_id="test-expertise",
            enabled=False,  # Disabled
        ))
        chain.add(ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
        ))
        
        context = MiddlewareContext(user_input="Test query")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        
        result = await chain.execute(context)
        
        assert result.success is True
        assert mock_retriever.retrieve_count == 0  # Was disabled
    
    @pytest.mark.asyncio
    async def test_chain_error_handling(self, mock_store):
        """Test that chain handles middleware errors gracefully."""
        class FailingReflector:
            async def reflect(self, *args, **kwargs):
                raise RuntimeError("Reflection failed!")
        
        chain = MiddlewareChain()
        chain.add(ExpertiseEvolutionMiddleware(
            reflector=FailingReflector(),
            expertise_store=mock_store,
        ))
        
        context = MiddlewareContext(
            user_input="Test query",
            agent_response="Response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        
        result = await chain.execute(context)
        
        # Chain should complete despite error
        assert result.success is True
        # Error should be recorded
        assert result.context.get_metadata("expertise_evolution_error") is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_missing_expertise(self, mock_reflector):
        """Test handling when expertise doesn't exist."""
        empty_store = MockExpertiseStore()
        
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=empty_store,
        )
        
        context = MiddlewareContext(
            user_input="Test",
            agent_response="Response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "nonexistent")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        # Should complete without error
        assert mock_reflector.reflect_count == 0
    
    @pytest.mark.asyncio
    async def test_empty_items_used(self, mock_reflector, mock_store, sample_expertise):
        """Test evolution with no items used."""
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
        )
        
        context = MiddlewareContext(
            user_input="Test",
            agent_response="Response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, TurnOutcome.SUCCESS)
        context.set_metadata(EXPERTISE_ITEMS_USED_KEY, [])  # Empty
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        # Should still call reflector
        assert mock_reflector.reflect_count == 1
    
    @pytest.mark.asyncio
    async def test_outcome_as_string(self, mock_reflector, mock_store, sample_expertise):
        """Test that outcome can be provided as string."""
        middleware = ExpertiseEvolutionMiddleware(
            reflector=mock_reflector,
            expertise_store=mock_store,
        )
        
        context = MiddlewareContext(
            user_input="Test",
            agent_response="Response",
        )
        context.set_metadata(EXPERTISE_ID_KEY, "test-expertise")
        context.set_metadata(TURN_OUTCOME_KEY, "success")  # String instead of enum
        
        async def terminal(ctx):
            return ctx
        
        await middleware.process(context, terminal)
        
        assert mock_reflector.reflect_count == 1

