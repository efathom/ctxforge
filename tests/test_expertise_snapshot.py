"""
Tests for Expertise Snapshot Service.

Tests versioning and diffing capabilities for expertise domains.
"""

from datetime import datetime, timezone

import pytest

from ctxforge.core.expertise import Expertise, ExpertiseSection
from ctxforge.engine.services.expertise_snapshot_service import (
    ChangeType,
    ExpertiseDiff,
    ExpertiseSnapshot,
    ExpertiseSnapshotService,
    InMemorySnapshotStore,
    ItemChange,
)


class TestChangeType:
    """Tests for ChangeType enum."""
    
    def test_change_types_exist(self):
        """Test all change types exist."""
        assert ChangeType.ADDED == "added"
        assert ChangeType.REMOVED == "removed"
        assert ChangeType.MODIFIED == "modified"
        assert ChangeType.UNCHANGED == "unchanged"


class TestItemChange:
    """Tests for ItemChange model."""
    
    def test_create_added_change(self):
        """Test creating an added item change."""
        change = ItemChange(
            change_type=ChangeType.ADDED,
            item_id="item-123",
            section="strategies",
            content="New strategy content",
        )
        
        assert change.change_type == ChangeType.ADDED
        assert change.item_id == "item-123"
        assert change.old_content is None
    
    def test_create_modified_change(self):
        """Test creating a modified item change."""
        change = ItemChange(
            change_type=ChangeType.MODIFIED,
            item_id="item-123",
            section="strategies",
            content="Updated content",
            old_content="Original content",
        )
        
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_content == "Original content"


class TestExpertiseDiff:
    """Tests for ExpertiseDiff model."""
    
    def test_create_diff(self):
        """Test creating a diff."""
        diff = ExpertiseDiff(
            from_version="1.0.0",
            to_version="1.1.0",
            items_added=2,
            items_removed=1,
            items_modified=3,
            items_unchanged=10,
        )
        
        assert diff.from_version == "1.0.0"
        assert diff.to_version == "1.1.0"
        assert diff.has_changes is True
    
    def test_no_changes(self):
        """Test diff with no changes."""
        diff = ExpertiseDiff(
            from_version="1.0.0",
            to_version="1.0.0",
            items_added=0,
            items_removed=0,
            items_modified=0,
            items_unchanged=10,
        )
        
        assert diff.has_changes is False
    
    def test_to_summary(self):
        """Test summary generation."""
        diff = ExpertiseDiff(
            from_version="1.0.0",
            to_version="1.1.0",
            items_added=2,
            items_removed=1,
            items_modified=3,
        )
        
        summary = diff.to_summary()
        
        assert "1.0.0" in summary
        assert "1.1.0" in summary
        assert "+2 added" in summary
        assert "-1 removed" in summary
        assert "~3 modified" in summary
    
    def test_to_changelog(self):
        """Test changelog generation."""
        diff = ExpertiseDiff(
            from_version="1.0.0",
            to_version="1.1.0",
            to_timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            items_added=1,
            changes=[
                ItemChange(
                    change_type=ChangeType.ADDED,
                    item_id="item-1",
                    section="strategies",
                    content="New strategy for handling edge cases",
                ),
            ],
        )
        
        changelog = diff.to_changelog()
        
        assert "## Changelog" in changelog
        assert "1.0.0" in changelog
        assert "1.1.0" in changelog
        assert "strategies" in changelog
        assert "+" in changelog


class TestExpertiseSnapshot:
    """Tests for ExpertiseSnapshot model."""
    
    @pytest.fixture
    def expertise(self):
        """Create a sample expertise."""
        exp = Expertise(
            expertise_id="test-expertise",
            name="Test Expertise",
            domain="testing",
        )
        exp.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Strategy 1",
        )
        exp.add_item(
            section=ExpertiseSection.HEURISTICS,
            content="Heuristic 1",
        )
        return exp
    
    def test_create_from_expertise(self, expertise):
        """Test creating a snapshot from expertise."""
        snapshot = ExpertiseSnapshot.from_expertise(
            expertise=expertise,
            version="1.0.0",
            created_by="user-123",
            description="Initial version",
        )
        
        assert snapshot.expertise_id == "test-expertise"
        assert snapshot.version == "1.0.0"
        assert snapshot.created_by == "user-123"
        assert snapshot.description == "Initial version"
        assert len(snapshot.items) == 2
        assert snapshot.content_hash != ""
    
    def test_content_hash_changes(self, expertise):
        """Test that content hash changes when items change."""
        snapshot1 = ExpertiseSnapshot.from_expertise(
            expertise=expertise,
            version="1.0.0",
        )
        
        expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Strategy 2",
        )
        
        snapshot2 = ExpertiseSnapshot.from_expertise(
            expertise=expertise,
            version="1.1.0",
        )
        
        assert snapshot1.content_hash != snapshot2.content_hash
    
    def test_to_expertise(self, expertise):
        """Test reconstructing expertise from snapshot."""
        snapshot = ExpertiseSnapshot.from_expertise(
            expertise=expertise,
            version="1.0.0",
        )
        
        restored = snapshot.to_expertise()
        
        assert restored.expertise_id == expertise.expertise_id
        assert len(restored.items) == len(expertise.items)


class TestInMemorySnapshotStore:
    """Tests for InMemorySnapshotStore."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store."""
        return InMemorySnapshotStore()
    
    @pytest.fixture
    def sample_snapshot(self):
        """Create a sample snapshot."""
        return ExpertiseSnapshot(
            expertise_id="test-expertise",
            version="1.0.0",
            items=[{"item_id": "1", "content": "Test"}],
            content_hash="abc123",
        )
    
    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_snapshot):
        """Test saving and retrieving a snapshot."""
        await store.save_snapshot(sample_snapshot)
        
        retrieved = await store.get_snapshot("test-expertise", "1.0.0")
        
        assert retrieved is not None
        assert retrieved.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test retrieving a non-existent snapshot."""
        result = await store.get_snapshot("nonexistent", "1.0.0")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_latest(self, store):
        """Test getting the latest snapshot."""
        snap1 = ExpertiseSnapshot(
            expertise_id="test",
            version="1.0.0",
            items=[],
            content_hash="1",
        )
        snap2 = ExpertiseSnapshot(
            expertise_id="test",
            version="1.1.0",
            items=[],
            content_hash="2",
        )
        
        await store.save_snapshot(snap1)
        await store.save_snapshot(snap2)
        
        latest = await store.get_latest_snapshot("test")
        
        assert latest is not None
        assert latest.version == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_list_versions(self, store):
        """Test listing versions."""
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            await store.save_snapshot(ExpertiseSnapshot(
                expertise_id="test",
                version=v,
                items=[],
                content_hash=v,
            ))
        
        versions = await store.list_versions("test")
        
        assert len(versions) == 3
        assert "1.0.0" in versions
        assert "2.0.0" in versions


class TestExpertiseSnapshotService:
    """Tests for ExpertiseSnapshotService."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store."""
        return InMemorySnapshotStore()
    
    @pytest.fixture
    def service(self, store):
        """Create a service."""
        return ExpertiseSnapshotService(store=store)
    
    @pytest.fixture
    def expertise(self):
        """Create a sample expertise."""
        exp = Expertise(
            expertise_id="test-expertise",
            name="Test",
            domain="testing",
        )
        exp.add_item(ExpertiseSection.STRATEGIES, "Strategy 1")
        return exp
    
    @pytest.mark.asyncio
    async def test_create_snapshot(self, service, expertise):
        """Test creating a snapshot."""
        snapshot = await service.create_snapshot(
            expertise=expertise,
            version="1.0.0",
            created_by="user-123",
            description="Initial version",
        )
        
        assert snapshot.version == "1.0.0"
        assert snapshot.created_by == "user-123"
    
    @pytest.mark.asyncio
    async def test_get_snapshot(self, service, expertise):
        """Test getting a snapshot."""
        await service.create_snapshot(expertise, "1.0.0")
        
        snapshot = await service.get_snapshot("test-expertise", "1.0.0")
        
        assert snapshot is not None
        assert snapshot.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_get_latest(self, service, expertise):
        """Test getting the latest snapshot."""
        await service.create_snapshot(expertise, "1.0.0")
        expertise.add_item(ExpertiseSection.HEURISTICS, "Heuristic 1")
        await service.create_snapshot(expertise, "1.1.0")
        
        latest = await service.get_latest("test-expertise")
        
        assert latest is not None
        assert latest.version == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_list_versions(self, service, expertise):
        """Test listing versions."""
        await service.create_snapshot(expertise, "1.0.0")
        await service.create_snapshot(expertise, "1.1.0")
        
        versions = await service.list_versions("test-expertise")
        
        assert len(versions) == 2
    
    @pytest.mark.asyncio
    async def test_diff_snapshots(self, service):
        """Test diffing two snapshots."""
        # Create two snapshots with different items
        snap1 = ExpertiseSnapshot(
            expertise_id="test",
            version="1.0.0",
            items=[
                {"item_id": "1", "section": "strategies", "content": "Original"},
                {"item_id": "2", "section": "heuristics", "content": "To remove"},
            ],
            content_hash="1",
        )
        
        snap2 = ExpertiseSnapshot(
            expertise_id="test",
            version="1.1.0",
            items=[
                {"item_id": "1", "section": "strategies", "content": "Modified"},
                {"item_id": "3", "section": "gotchas", "content": "New item"},
            ],
            content_hash="2",
        )
        
        diff = service.diff_snapshots(snap1, snap2)
        
        assert diff.items_added == 1  # Item 3
        assert diff.items_removed == 1  # Item 2
        assert diff.items_modified == 1  # Item 1
        assert diff.has_changes is True
    
    @pytest.mark.asyncio
    async def test_diff_versions(self, service, expertise):
        """Test diffing by version strings."""
        # Create version 1.0.0
        await service.create_snapshot(expertise, "1.0.0")
        
        # Add an item and create version 1.1.0
        expertise.add_item(ExpertiseSection.HEURISTICS, "New heuristic")
        await service.create_snapshot(expertise, "1.1.0")
        
        diff = await service.diff_versions("test-expertise", "1.0.0", "1.1.0")
        
        assert diff is not None
        assert diff.items_added == 1
    
    @pytest.mark.asyncio
    async def test_diff_versions_missing(self, service):
        """Test diffing when version doesn't exist."""
        diff = await service.diff_versions("test", "1.0.0", "2.0.0")
        assert diff is None
    
    @pytest.mark.asyncio
    async def test_diff_with_current(self, service, expertise):
        """Test diffing current state with a version."""
        await service.create_snapshot(expertise, "1.0.0")
        
        # Modify expertise
        expertise.add_item(ExpertiseSection.COMMON_MISTAKES, "New gotcha")
        
        diff = await service.diff_with_current(expertise, "1.0.0")
        
        assert diff is not None
        assert diff.from_version == "1.0.0"
        assert diff.to_version == "current"
        assert diff.items_added == 1
