"""
Tests for expertise snapshot stores.

Tests InMemory, FileBasedSnapshotStore, and database-backed stores.
Database tests require running PostgreSQL/MySQL servers.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from ctxforge.core.expertise import Expertise, ExpertiseItem, ExpertiseSection
from ctxforge.engine.services.expertise_snapshot_service import (
    ExpertiseSnapshot,
    ExpertiseSnapshotService,
    FileBasedSnapshotStore,
    InMemorySnapshotStore,
)

# Check for asyncpg (PostgreSQL)
try:
    import asyncpg  # noqa: F401
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Check for aiomysql (MySQL)
try:
    import aiomysql  # noqa: F401
    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False


@pytest.fixture
def sample_expertise() -> Expertise:
    """Create sample expertise for testing."""
    expertise = Expertise(
        expertise_id="test-expertise",
        name="Test Knowledge",
        domain="testing",
    )
    expertise.items = [
        ExpertiseItem(
            section=ExpertiseSection.STRATEGIES,
            content="Use pytest for testing",
        ),
        ExpertiseItem(
            section=ExpertiseSection.COMMON_MISTAKES,
            content="Forgetting to mock dependencies",
        ),
    ]
    return expertise


@pytest.fixture
def sample_snapshot() -> ExpertiseSnapshot:
    """Create a sample snapshot for testing."""
    return ExpertiseSnapshot(
        expertise_id="test-expertise",
        version="1.0.0",
        content_hash="abc123",
        items=[
            {"section": "strategies", "content": "Use pytest"},
            {"section": "common_mistakes", "content": "Forgetting mocks"},
        ],
        created_at=datetime.now(timezone.utc),
        created_by="test-user",
        description="Test snapshot",
    )


# ============================================================================
# InMemorySnapshotStore Tests
# ============================================================================

class TestInMemorySnapshotStore:
    """Tests for in-memory snapshot store."""

    @pytest.fixture
    def store(self):
        return InMemorySnapshotStore()

    @pytest.mark.asyncio
    async def test_save_and_get_snapshot(self, store, sample_snapshot):
        """Test saving and retrieving a snapshot."""
        await store.save_snapshot(sample_snapshot)

        loaded = await store.get_snapshot("test-expertise", "1.0.0")

        assert loaded is not None
        assert loaded.version == "1.0.0"
        assert len(loaded.items) == 2

    @pytest.mark.asyncio
    async def test_get_latest_snapshot(self, store, sample_snapshot):
        """Test getting the latest snapshot."""
        await store.save_snapshot(sample_snapshot)

        # Create a newer snapshot
        newer = ExpertiseSnapshot(
            expertise_id="test-expertise",
            version="1.1.0",
            content_hash="def456",
            item_count=3,
            items=[],
            created_at=datetime.now(timezone.utc),
        )
        await store.save_snapshot(newer)

        latest = await store.get_latest_snapshot("test-expertise")

        assert latest is not None
        assert latest.version == "1.1.0"

    @pytest.mark.asyncio
    async def test_list_versions(self, store, sample_snapshot):
        """Test listing versions."""
        await store.save_snapshot(sample_snapshot)

        newer = ExpertiseSnapshot(
            expertise_id="test-expertise",
            version="1.1.0",
            content_hash="def456",
            item_count=3,
            items=[],
            created_at=datetime.now(timezone.utc),
        )
        await store.save_snapshot(newer)

        versions = await store.list_versions("test-expertise")

        assert len(versions) == 2
        assert "1.0.0" in versions
        assert "1.1.0" in versions

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a nonexistent snapshot."""
        result = await store.get_snapshot("nonexistent", "1.0.0")
        assert result is None


# ============================================================================
# FileBasedSnapshotStore Tests
# ============================================================================

class TestFileBasedSnapshotStore:
    """Tests for file-based snapshot store."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def store(self, temp_dir):
        return FileBasedSnapshotStore(temp_dir)

    @pytest.mark.asyncio
    async def test_save_and_get_snapshot(self, store, sample_snapshot, temp_dir):
        """Test saving and retrieving a snapshot."""
        await store.save_snapshot(sample_snapshot)

        # Verify file exists
        expertise_dir = os.path.join(temp_dir, "test-expertise")
        assert os.path.exists(expertise_dir)
        assert os.path.exists(os.path.join(expertise_dir, "1.0.0.json"))

        loaded = await store.get_snapshot("test-expertise", "1.0.0")

        assert loaded is not None
        assert loaded.version == "1.0.0"
        assert len(loaded.items) == 2

    @pytest.mark.asyncio
    async def test_get_latest_snapshot(self, store, sample_snapshot):
        """Test getting the latest snapshot via latest.json."""
        await store.save_snapshot(sample_snapshot)

        latest = await store.get_latest_snapshot("test-expertise")

        assert latest is not None
        assert latest.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_versions(self, store, sample_snapshot):
        """Test listing versions."""
        await store.save_snapshot(sample_snapshot)

        newer = ExpertiseSnapshot(
            expertise_id="test-expertise",
            version="1.1.0",
            content_hash="def456",
            item_count=3,
            items=[],
            created_at=datetime.now(timezone.utc),
        )
        await store.save_snapshot(newer)

        versions = await store.list_versions("test-expertise")

        assert len(versions) == 2

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, store, sample_snapshot, temp_dir):
        """Test deleting a snapshot."""
        await store.save_snapshot(sample_snapshot)

        result = await store.delete_snapshot("test-expertise", "1.0.0")
        assert result is True

        # File should be gone
        snapshot_path = os.path.join(temp_dir, "test-expertise", "1.0.0.json")
        assert not os.path.exists(snapshot_path)

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a nonexistent snapshot."""
        result = await store.get_snapshot("nonexistent", "1.0.0")
        assert result is None


# ============================================================================
# Integration Test with ExpertiseSnapshotService
# ============================================================================

class TestSnapshotServiceWithFileStore:
    """Integration tests using FileBasedSnapshotStore."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def service(self, temp_dir):
        store = FileBasedSnapshotStore(temp_dir)
        return ExpertiseSnapshotService(store)

    @pytest.mark.asyncio
    async def test_create_and_diff_snapshots(self, service, sample_expertise):
        """Test creating snapshots and diffing them."""
        # Create initial snapshot
        snapshot1 = await service.create_snapshot(
            sample_expertise,
            version="1.0.0",
            description="Initial version",
        )
        assert snapshot1.version == "1.0.0"

        # Modify expertise
        sample_expertise.items.append(
            ExpertiseItem(
                section=ExpertiseSection.HEURISTICS,
                content="Always write tests first",
            )
        )

        # Create second snapshot
        snapshot2 = await service.create_snapshot(
            sample_expertise,
            version="1.1.0",
            description="Added TDD heuristic",
        )
        assert snapshot2.version == "1.1.0"
        assert len(snapshot2.items) == 3

        # Diff versions
        diff = await service.diff_versions(
            sample_expertise.expertise_id,
            "1.0.0",
            "1.1.0",
        )

        assert diff.has_changes
        assert diff.items_added == 1


# ============================================================================
# PostgreSQL Snapshot Store Tests
# ============================================================================

def get_postgres_config():
    """Get PostgreSQL configuration from environment."""
    from ctxforge.storage.connection import PostgresConfig
    return PostgresConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DATABASE", "ctxforge_test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


async def postgres_is_available() -> bool:
    """Check if PostgreSQL is available."""
    if not ASYNCPG_AVAILABLE:
        return False
    try:
        from ctxforge.storage.connection import PostgresConnectionManager
        config = get_postgres_config()
        manager = PostgresConnectionManager(config)
        await manager.connect()
        await manager.disconnect()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not ASYNCPG_AVAILABLE,
    reason="asyncpg not installed"
)
class TestPostgresSnapshotStore:
    """Tests for PostgreSQL snapshot store."""

    @pytest.fixture
    async def store(self):
        available = await postgres_is_available()
        if not available:
            pytest.skip("PostgreSQL not available")

        from ctxforge.storage.postgres.expertise_snapshot import PostgresSnapshotStore
        config = get_postgres_config()
        store = PostgresSnapshotStore(config, table_name="test_expertise_snapshots")
        await store.initialize()

        yield store

        # Cleanup
        await store._manager.execute("DROP TABLE IF EXISTS test_expertise_snapshots")
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_snapshot):
        """Test saving and retrieving a snapshot."""
        await store.save_snapshot(sample_snapshot)

        loaded = await store.get_snapshot("test-expertise", "1.0.0")

        assert loaded is not None
        assert loaded.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_versions(self, store, sample_snapshot):
        """Test listing versions."""
        await store.save_snapshot(sample_snapshot)

        versions = await store.list_versions("test-expertise")

        assert "1.0.0" in versions


# ============================================================================
# MySQL Snapshot Store Tests
# ============================================================================

def get_mysql_config():
    """Get MySQL configuration from environment."""
    from ctxforge.storage.connection import MySQLConfig
    return MySQLConfig(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "ctxforge_test"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
    )


async def mysql_is_available() -> bool:
    """Check if MySQL is available."""
    if not AIOMYSQL_AVAILABLE:
        return False
    try:
        from ctxforge.storage.connection import MySQLConnectionManager
        config = get_mysql_config()
        manager = MySQLConnectionManager(config)
        await manager.connect()
        await manager.disconnect()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not AIOMYSQL_AVAILABLE,
    reason="aiomysql not installed"
)
class TestMySQLSnapshotStore:
    """Tests for MySQL snapshot store."""

    @pytest.fixture
    async def store(self):
        available = await mysql_is_available()
        if not available:
            pytest.skip("MySQL not available")

        from ctxforge.storage.mysql.expertise_snapshot import MySQLSnapshotStore
        config = get_mysql_config()
        store = MySQLSnapshotStore(config, table_name="test_expertise_snapshots")
        await store.initialize()

        yield store

        # Cleanup
        await store._manager.execute("DROP TABLE IF EXISTS test_expertise_snapshots")
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_snapshot):
        """Test saving and retrieving a snapshot."""
        await store.save_snapshot(sample_snapshot)

        loaded = await store.get_snapshot("test-expertise", "1.0.0")

        assert loaded is not None
        assert loaded.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_latest(self, store, sample_snapshot):
        """Test getting the latest snapshot."""
        await store.save_snapshot(sample_snapshot)

        latest = await store.get_latest_snapshot("test-expertise")

        assert latest is not None
        assert latest.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_list_versions(self, store, sample_snapshot):
        """Test listing versions."""
        await store.save_snapshot(sample_snapshot)

        versions = await store.list_versions("test-expertise")

        assert "1.0.0" in versions

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_snapshot):
        """Test deleting a snapshot."""
        await store.save_snapshot(sample_snapshot)

        result = await store.delete_snapshot("test-expertise", "1.0.0")
        assert result is True

        loaded = await store.get_snapshot("test-expertise", "1.0.0")
        assert loaded is None
