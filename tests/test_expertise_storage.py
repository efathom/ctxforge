"""
Tests for expertise storage implementations.

Tests CRUD operations, search, and usage logging for all storage backends.
"""


import pytest

from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseUsageLog,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.storage import InMemoryExpertiseStore

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def store() -> InMemoryExpertiseStore:
    """Create a fresh in-memory store for each test."""
    return InMemoryExpertiseStore()


@pytest.fixture
def sample_expertise() -> Expertise:
    """Create a sample expertise for testing."""
    return Expertise(
        expertise_id="test-expertise-001",
        name="Customer Support Expert",
        domain="support",
        token_budget=50000,
    )


@pytest.fixture
def sample_items() -> list[ExpertiseItem]:
    """Create sample expertise items for testing."""
    return [
        ExpertiseItem(
            item_id="strat-00001",
            section=ExpertiseSection.STRATEGIES,
            content="Always greet the customer by name to build rapport",
        ),
        ExpertiseItem(
            item_id="strat-00002",
            section=ExpertiseSection.STRATEGIES,
            content="Use positive language even when delivering bad news",
        ),
        ExpertiseItem(
            item_id="formula-00001",
            section=ExpertiseSection.FORMULAS,
            content="Discount = base_price * discount_rate * loyalty_multiplier",
        ),
        ExpertiseItem(
            item_id="mistake-00001",
            section=ExpertiseSection.COMMON_MISTAKES,
            content="Never promise resolution times you cannot guarantee",
        ),
    ]


@pytest.fixture
def expertise_with_items(
    sample_expertise: Expertise,
    sample_items: list[ExpertiseItem],
) -> Expertise:
    """Create expertise with pre-populated items."""
    sample_expertise.items = sample_items.copy()
    sample_expertise.next_item_id = 5
    return sample_expertise


# =============================================================================
# Core CRUD Tests
# =============================================================================

class TestExpertiseStoreCRUD:
    """Test basic CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_save_and_load(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test saving and loading an expertise."""
        await store.save(sample_expertise)
        
        loaded = await store.load(sample_expertise.expertise_id)
        
        assert loaded is not None
        assert loaded.expertise_id == sample_expertise.expertise_id
        assert loaded.name == sample_expertise.name
        assert loaded.domain == sample_expertise.domain
        assert loaded.token_budget == sample_expertise.token_budget
    
    @pytest.mark.asyncio
    async def test_save_with_items(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test saving expertise with items."""
        await store.save(expertise_with_items)
        
        loaded = await store.load(expertise_with_items.expertise_id)
        
        assert loaded is not None
        assert len(loaded.items) == 4
        assert loaded.items[0].item_id == "strat-00001"
        assert loaded.items[0].section == ExpertiseSection.STRATEGIES
    
    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store: InMemoryExpertiseStore):
        """Test loading a non-existent expertise returns None."""
        loaded = await store.load("nonexistent-id")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_delete(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test deleting an expertise."""
        await store.save(sample_expertise)
        
        result = await store.delete(sample_expertise.expertise_id)
        
        assert result is True
        loaded = await store.load(sample_expertise.expertise_id)
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store: InMemoryExpertiseStore):
        """Test deleting a non-existent expertise returns False."""
        result = await store.delete("nonexistent-id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_update_expertise(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test updating an existing expertise."""
        await store.save(sample_expertise)
        
        # Modify and save again
        sample_expertise.name = "Updated Expert Name"
        sample_expertise.version = 2
        await store.save(sample_expertise)
        
        loaded = await store.load(sample_expertise.expertise_id)
        
        assert loaded is not None
        assert loaded.name == "Updated Expert Name"
        assert loaded.version == 2
    
    @pytest.mark.asyncio
    async def test_list_expertise(self, store: InMemoryExpertiseStore):
        """Test listing expertise."""
        expertise1 = Expertise(
            expertise_id="exp-001",
            name="Expert 1",
            domain="support",
        )
        expertise2 = Expertise(
            expertise_id="exp-002",
            name="Expert 2",
            domain="sales",
        )
        expertise3 = Expertise(
            expertise_id="exp-003",
            name="Expert 3",
            domain="support",
        )
        
        await store.save(expertise1)
        await store.save(expertise2)
        await store.save(expertise3)
        
        # List all
        all_expertise = await store.list_expertise()
        assert len(all_expertise) == 3
        
        # List by domain
        support_expertise = await store.list_expertise(domain="support")
        assert len(support_expertise) == 2
        assert all(e.domain == "support" for e in support_expertise)
    
    @pytest.mark.asyncio
    async def test_list_expertise_pagination(self, store: InMemoryExpertiseStore):
        """Test listing expertise with pagination."""
        for i in range(5):
            expertise = Expertise(
                expertise_id=f"exp-{i:03d}",
                name=f"Expert {i}",
            )
            await store.save(expertise)
        
        # First page
        page1 = await store.list_expertise(limit=2, offset=0)
        assert len(page1) == 2
        
        # Second page
        page2 = await store.list_expertise(limit=2, offset=2)
        assert len(page2) == 2
        
        # Third page (partial)
        page3 = await store.list_expertise(limit=2, offset=4)
        assert len(page3) == 1


# =============================================================================
# Item Management Tests
# =============================================================================

class TestExpertiseItemOperations:
    """Test individual item operations."""
    
    @pytest.mark.asyncio
    async def test_add_item(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
        sample_items: list[ExpertiseItem],
    ):
        """Test adding an item to an expertise."""
        await store.save(sample_expertise)
        
        item = sample_items[0]
        await store.add_item(sample_expertise.expertise_id, item)
        
        loaded = await store.load(sample_expertise.expertise_id)
        assert loaded is not None
        assert len(loaded.items) == 1
        assert loaded.items[0].item_id == item.item_id
    
    @pytest.mark.asyncio
    async def test_update_item(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test updating an existing item."""
        await store.save(expertise_with_items)
        
        # Update the first item
        item = expertise_with_items.items[0].model_copy()
        item.content = "Updated content for this strategy"
        item.helpful_count = 10
        
        await store.update_item(expertise_with_items.expertise_id, item)
        
        loaded = await store.get_item(
            expertise_with_items.expertise_id,
            item.item_id,
        )
        
        assert loaded is not None
        assert loaded.content == "Updated content for this strategy"
        assert loaded.helpful_count == 10
    
    @pytest.mark.asyncio
    async def test_remove_item(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test removing an item from an expertise."""
        await store.save(expertise_with_items)
        
        result = await store.remove_item(
            expertise_with_items.expertise_id,
            "strat-00001",
        )
        
        assert result is True
        
        loaded = await store.load(expertise_with_items.expertise_id)
        assert loaded is not None
        assert len(loaded.items) == 3
        assert all(item.item_id != "strat-00001" for item in loaded.items)
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_item(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test removing a non-existent item returns False."""
        await store.save(expertise_with_items)
        
        result = await store.remove_item(
            expertise_with_items.expertise_id,
            "nonexistent-item",
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_item(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test getting a single item by ID."""
        await store.save(expertise_with_items)
        
        item = await store.get_item(
            expertise_with_items.expertise_id,
            "formula-00001",
        )
        
        assert item is not None
        assert item.item_id == "formula-00001"
        assert item.section == ExpertiseSection.FORMULAS
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_item(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test getting a non-existent item returns None."""
        await store.save(expertise_with_items)
        
        item = await store.get_item(
            expertise_with_items.expertise_id,
            "nonexistent-item",
        )
        
        assert item is None
    
    @pytest.mark.asyncio
    async def test_get_items_by_section(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test getting items filtered by section."""
        await store.save(expertise_with_items)
        
        strategies = await store.get_items_by_section(
            expertise_with_items.expertise_id,
            ExpertiseSection.STRATEGIES,
        )
        
        assert len(strategies) == 2
        assert all(item.section == ExpertiseSection.STRATEGIES for item in strategies)
        
        formulas = await store.get_items_by_section(
            expertise_with_items.expertise_id,
            ExpertiseSection.FORMULAS,
        )
        
        assert len(formulas) == 1
        assert formulas[0].item_id == "formula-00001"


# =============================================================================
# Item Counts Tests
# =============================================================================

class TestItemCountUpdates:
    """Test helpful/harmful count updates."""
    
    @pytest.mark.asyncio
    async def test_update_item_counts_helpful(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test updating helpful count."""
        await store.save(expertise_with_items)
        
        await store.update_item_counts(
            expertise_with_items.expertise_id,
            "strat-00001",
            helpful_delta=5,
        )
        
        item = await store.get_item(
            expertise_with_items.expertise_id,
            "strat-00001",
        )
        
        assert item is not None
        assert item.helpful_count == 5
        assert item.harmful_count == 0
    
    @pytest.mark.asyncio
    async def test_update_item_counts_harmful(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test updating harmful count."""
        await store.save(expertise_with_items)
        
        await store.update_item_counts(
            expertise_with_items.expertise_id,
            "strat-00001",
            harmful_delta=3,
        )
        
        item = await store.get_item(
            expertise_with_items.expertise_id,
            "strat-00001",
        )
        
        assert item is not None
        assert item.helpful_count == 0
        assert item.harmful_count == 3
    
    @pytest.mark.asyncio
    async def test_update_item_counts_both(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test updating both counts."""
        await store.save(expertise_with_items)
        
        await store.update_item_counts(
            expertise_with_items.expertise_id,
            "strat-00001",
            helpful_delta=10,
            harmful_delta=2,
        )
        
        item = await store.get_item(
            expertise_with_items.expertise_id,
            "strat-00001",
        )
        
        assert item is not None
        assert item.helpful_count == 10
        assert item.harmful_count == 2
    
    @pytest.mark.asyncio
    async def test_update_item_counts_incremental(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test that count updates are incremental."""
        await store.save(expertise_with_items)
        
        # First update
        await store.update_item_counts(
            expertise_with_items.expertise_id,
            "strat-00001",
            helpful_delta=5,
        )
        
        # Second update
        await store.update_item_counts(
            expertise_with_items.expertise_id,
            "strat-00001",
            helpful_delta=3,
        )
        
        item = await store.get_item(
            expertise_with_items.expertise_id,
            "strat-00001",
        )
        
        assert item is not None
        assert item.helpful_count == 8


# =============================================================================
# Search Tests
# =============================================================================

class TestItemSearch:
    """Test item search functionality."""
    
    @pytest.mark.asyncio
    async def test_search_items_by_keyword(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test searching items by keyword."""
        await store.save(expertise_with_items)
        
        results = await store.search_items(
            expertise_with_items.expertise_id,
            "customer",
        )
        
        assert len(results) == 1
        assert results[0].item_id == "strat-00001"
    
    @pytest.mark.asyncio
    async def test_search_items_multiple_matches(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test search with multiple matches."""
        await store.save(expertise_with_items)
        
        # Both strategy items contain relevant words
        results = await store.search_items(
            expertise_with_items.expertise_id,
            "positive language",
        )
        
        # Should match the item about positive language
        assert len(results) >= 1
        assert any(item.item_id == "strat-00002" for item in results)
    
    @pytest.mark.asyncio
    async def test_search_items_no_match(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test search with no matches."""
        await store.save(expertise_with_items)
        
        results = await store.search_items(
            expertise_with_items.expertise_id,
            "xyznonexistent",
        )
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_items_limit(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test search respects limit parameter."""
        # Add many items with similar content
        for i in range(10):
            item = ExpertiseItem(
                item_id=f"item-{i:05d}",
                section=ExpertiseSection.STRATEGIES,
                content=f"Strategy about customer service approach {i}",
            )
            sample_expertise.items.append(item)
        
        await store.save(sample_expertise)
        
        results = await store.search_items(
            sample_expertise.expertise_id,
            "customer service",
            limit=3,
        )
        
        assert len(results) == 3


# =============================================================================
# Usage Logging Tests
# =============================================================================

class TestUsageLogging:
    """Test usage logging functionality."""
    
    @pytest.mark.asyncio
    async def test_log_usage(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test logging usage."""
        await store.save(expertise_with_items)
        
        log = ExpertiseUsageLog(
            log_id="log-001",
            session_id="session-001",
            expertise_id=expertise_with_items.expertise_id,
            items_used=["strat-00001", "strat-00002"],
            feedback={
                "strat-00001": UsageFeedback.HELPFUL,
                "strat-00002": UsageFeedback.NEUTRAL,
            },
            outcome=TurnOutcome.SUCCESS,
            context_summary="Customer asked about return policy",
        )
        
        await store.log_usage(log)
        
        stats = await store.get_usage_stats(expertise_with_items.expertise_id)
        
        assert stats["total_uses"] == 1
        assert "strat-00001" in stats["item_usage"]
        assert stats["item_usage"]["strat-00001"] == 1
    
    @pytest.mark.asyncio
    async def test_usage_stats_multiple_logs(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test usage stats with multiple logs."""
        await store.save(expertise_with_items)
        
        # Log 1
        log1 = ExpertiseUsageLog(
            log_id="log-001",
            session_id="session-001",
            expertise_id=expertise_with_items.expertise_id,
            items_used=["strat-00001"],
            feedback={"strat-00001": UsageFeedback.HELPFUL},
            outcome=TurnOutcome.SUCCESS,
        )
        await store.log_usage(log1)
        
        # Log 2
        log2 = ExpertiseUsageLog(
            log_id="log-002",
            session_id="session-002",
            expertise_id=expertise_with_items.expertise_id,
            items_used=["strat-00001", "strat-00002"],
            feedback={
                "strat-00001": UsageFeedback.HELPFUL,
                "strat-00002": UsageFeedback.HARMFUL,
            },
            outcome=TurnOutcome.FAILURE,
        )
        await store.log_usage(log2)
        
        stats = await store.get_usage_stats(expertise_with_items.expertise_id)
        
        assert stats["total_uses"] == 2
        assert stats["item_usage"]["strat-00001"] == 2
        assert stats["item_usage"]["strat-00002"] == 1
        assert stats["outcome_counts"]["success"] == 1
        assert stats["outcome_counts"]["failure"] == 1
    
    @pytest.mark.asyncio
    async def test_usage_stats_feedback_counts(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test feedback counts in usage stats."""
        await store.save(expertise_with_items)
        
        # Multiple logs with different feedback
        for i in range(3):
            log = ExpertiseUsageLog(
                log_id=f"log-{i:03d}",
                session_id=f"session-{i:03d}",
                expertise_id=expertise_with_items.expertise_id,
                items_used=["strat-00001"],
                feedback={"strat-00001": UsageFeedback.HELPFUL},
            )
            await store.log_usage(log)
        
        # One log with harmful feedback
        log_harmful = ExpertiseUsageLog(
            log_id="log-harmful",
            session_id="session-harmful",
            expertise_id=expertise_with_items.expertise_id,
            items_used=["strat-00001"],
            feedback={"strat-00001": UsageFeedback.HARMFUL},
        )
        await store.log_usage(log_harmful)
        
        stats = await store.get_usage_stats(expertise_with_items.expertise_id)
        
        assert stats["feedback_counts"]["strat-00001"]["helpful"] == 3
        assert stats["feedback_counts"]["strat-00001"]["harmful"] == 1


# =============================================================================
# Concurrency and Isolation Tests
# =============================================================================

class TestConcurrencyAndIsolation:
    """Test concurrency and data isolation."""
    
    @pytest.mark.asyncio
    async def test_data_isolation(self, store: InMemoryExpertiseStore):
        """Test that modifications don't affect loaded copies."""
        expertise = Expertise(
            expertise_id="isolation-test",
            name="Isolation Test",
        )
        expertise.items.append(
            ExpertiseItem(
                item_id="item-001",
                section=ExpertiseSection.STRATEGIES,
                content="Original content",
            )
        )
        
        await store.save(expertise)
        
        # Load expertise
        loaded = await store.load(expertise.expertise_id)
        assert loaded is not None
        
        # Modify the loaded copy
        loaded.name = "Modified Name"
        loaded.items[0].content = "Modified content"
        
        # Load again - should be unchanged
        loaded2 = await store.load(expertise.expertise_id)
        assert loaded2 is not None
        assert loaded2.name == "Isolation Test"
        assert loaded2.items[0].content == "Original content"
    
    @pytest.mark.asyncio
    async def test_clear_store(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test clearing all data."""
        await store.save(expertise_with_items)
        
        log = ExpertiseUsageLog(
            log_id="log-001",
            session_id="session-001",
            expertise_id=expertise_with_items.expertise_id,
            items_used=["strat-00001"],
        )
        await store.log_usage(log)
        
        await store.clear()
        
        loaded = await store.load(expertise_with_items.expertise_id)
        assert loaded is None
        
        stats = await store.get_usage_stats(expertise_with_items.expertise_id)
        assert stats["total_uses"] == 0
    
    @pytest.mark.asyncio
    async def test_delete_cascades_usage_logs(
        self,
        store: InMemoryExpertiseStore,
        expertise_with_items: Expertise,
    ):
        """Test that deleting expertise also deletes its usage logs."""
        await store.save(expertise_with_items)
        
        log = ExpertiseUsageLog(
            log_id="log-001",
            session_id="session-001",
            expertise_id=expertise_with_items.expertise_id,
            items_used=["strat-00001"],
        )
        await store.log_usage(log)
        
        # Verify log exists
        stats_before = await store.get_usage_stats(expertise_with_items.expertise_id)
        assert stats_before["total_uses"] == 1
        
        # Delete expertise
        await store.delete(expertise_with_items.expertise_id)
        
        # Verify logs are gone
        stats_after = await store.get_usage_stats(expertise_with_items.expertise_id)
        assert stats_after["total_uses"] == 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_expertise(self, store: InMemoryExpertiseStore):
        """Test handling of expertise with no items."""
        expertise = Expertise(
            expertise_id="empty-expertise",
            name="Empty Expert",
        )
        
        await store.save(expertise)
        
        loaded = await store.load(expertise.expertise_id)
        assert loaded is not None
        assert len(loaded.items) == 0
        
        items = await store.get_items_by_section(
            expertise.expertise_id,
            ExpertiseSection.STRATEGIES,
        )
        assert len(items) == 0
    
    @pytest.mark.asyncio
    async def test_item_with_special_characters(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test handling of items with special characters."""
        item = ExpertiseItem(
            item_id="special-001",
            section=ExpertiseSection.FORMULAS,
            content="Calculate: result = (a + b) * c / d - e % f",
        )
        sample_expertise.items.append(item)
        
        await store.save(sample_expertise)
        
        loaded = await store.get_item(sample_expertise.expertise_id, "special-001")
        assert loaded is not None
        assert loaded.content == "Calculate: result = (a + b) * c / d - e % f"
    
    @pytest.mark.asyncio
    async def test_item_with_unicode(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test handling of items with unicode content."""
        item = ExpertiseItem(
            item_id="unicode-001",
            section=ExpertiseSection.STRATEGIES,
            content="客户服务策略: 始终保持友好态度 🎯",
        )
        sample_expertise.items.append(item)
        
        await store.save(sample_expertise)
        
        loaded = await store.get_item(sample_expertise.expertise_id, "unicode-001")
        assert loaded is not None
        assert "客户服务策略" in loaded.content
        assert "🎯" in loaded.content
    
    @pytest.mark.asyncio
    async def test_inactive_items_excluded_from_section_search(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test that inactive items are excluded from section search."""
        active_item = ExpertiseItem(
            item_id="active-001",
            section=ExpertiseSection.STRATEGIES,
            content="Active strategy",
            is_active=True,
        )
        inactive_item = ExpertiseItem(
            item_id="inactive-001",
            section=ExpertiseSection.STRATEGIES,
            content="Inactive strategy",
            is_active=False,
        )
        
        sample_expertise.items = [active_item, inactive_item]
        await store.save(sample_expertise)
        
        items = await store.get_items_by_section(
            sample_expertise.expertise_id,
            ExpertiseSection.STRATEGIES,
        )
        
        assert len(items) == 1
        assert items[0].item_id == "active-001"
    
    @pytest.mark.asyncio
    async def test_get_usage_stats_no_logs(
        self,
        store: InMemoryExpertiseStore,
        sample_expertise: Expertise,
    ):
        """Test getting usage stats when there are no logs."""
        await store.save(sample_expertise)
        
        stats = await store.get_usage_stats(sample_expertise.expertise_id)
        
        assert stats["total_uses"] == 0
        assert stats["item_usage"] == {}
        assert stats["feedback_counts"] == {}
        assert stats["outcome_counts"] == {}

