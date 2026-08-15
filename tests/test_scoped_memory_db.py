"""
Tests for Database-backed Scoped Memory Stores.

These tests verify the SQL generation and basic structure of MySQL and PostgreSQL
scoped memory store implementations without requiring actual database connections.
"""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)


class TestMySQLScopedMemoryStore:
    """Tests for MySQLScopedMemoryStore."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock connection manager."""
        manager = MagicMock()
        manager.is_connected = False
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        manager.execute = AsyncMock(return_value=1)
        manager.fetchone = AsyncMock(return_value=None)
        manager.fetchall = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def store(self, mock_manager):
        """Create a store with mocked connection."""
        from ctxforge.storage.mysql.scoped_memory import MySQLScopedMemoryStore
        store = MySQLScopedMemoryStore(connection_manager=mock_manager)
        store._initialized = True
        return store

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory."""
        return ScopedMemory(
            id="mem-1",
            scope=MemoryScope.PROJECT,
            scope_id="proj-1",
            category=MemoryCategory.CONVENTION,
            key="code-style",
            content="Use 4 spaces",
            priority=5,
        )

    async def test_save_calls_execute(self, store, mock_manager, sample_memory):
        """Test that save calls execute with correct SQL."""
        await store.save(sample_memory)

        mock_manager.execute.assert_called_once()
        call_args = mock_manager.execute.call_args
        sql = call_args[0][0]

        assert "INSERT INTO" in sql
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "scoped_memories" in sql

    async def test_get_calls_fetchone(self, store, mock_manager):
        """Test that get calls fetchone."""
        await store.get(MemoryScope.GLOBAL, "user-1", "key-1")

        mock_manager.fetchone.assert_called_once()
        call_args = mock_manager.fetchone.call_args
        sql = call_args[0][0]

        assert "SELECT * FROM" in sql
        assert "scope = %s" in sql

    async def test_get_by_id(self, store, mock_manager):
        """Test get_by_id."""
        await store.get_by_id("mem-123")

        mock_manager.fetchone.assert_called_once()
        sql = mock_manager.fetchone.call_args[0][0]

        assert "WHERE id = %s" in sql

    async def test_list_by_scope(self, store, mock_manager):
        """Test list_by_scope."""
        await store.list_by_scope(MemoryScope.PROJECT, "proj-1")

        mock_manager.fetchall.assert_called_once()
        sql = mock_manager.fetchall.call_args[0][0]

        assert "ORDER BY priority DESC" in sql

    async def test_list_by_scope_with_category(self, store, mock_manager):
        """Test list_by_scope with category filter."""
        await store.list_by_scope(
            MemoryScope.PROJECT, "proj-1",
            category=MemoryCategory.CONVENTION
        )

        sql = mock_manager.fetchall.call_args[0][0]
        assert "category = %s" in sql

    async def test_query_builds_scope_conditions(self, store, mock_manager):
        """Test query builds correct scope conditions."""
        query = ScopedMemoryQuery(
            user_id="user-1",
            project_id="proj-1",
            session_id="sess-1",
        )

        await store.query(query)

        sql = mock_manager.fetchall.call_args[0][0]
        # Should have conditions for all three scopes
        assert "scope = %s AND scope_id = %s" in sql
        assert "ORDER BY scope_priority DESC" in sql

    async def test_delete(self, store, mock_manager):
        """Test delete."""
        mock_manager.execute.return_value = 1

        result = await store.delete(MemoryScope.PROJECT, "proj-1", "key-1")

        assert result is True
        sql = mock_manager.execute.call_args[0][0]
        assert "DELETE FROM" in sql

    async def test_delete_by_id(self, store, mock_manager):
        """Test delete_by_id."""
        mock_manager.execute.return_value = 1

        result = await store.delete_by_id("mem-123")

        assert result is True
        sql = mock_manager.execute.call_args[0][0]
        assert "WHERE id = %s" in sql

    async def test_count(self, store, mock_manager):
        """Test count."""
        mock_manager.fetchone.return_value = {"cnt": 5}

        result = await store.count(scope=MemoryScope.PROJECT)

        assert result == 5
        sql = mock_manager.fetchone.call_args[0][0]
        assert "SELECT COUNT(*)" in sql

    async def test_clear_with_scope(self, store, mock_manager):
        """Test clear with scope filter."""
        mock_manager.execute.return_value = 3

        result = await store.clear(scope=MemoryScope.SESSION)

        assert result == 3
        sql = mock_manager.execute.call_args[0][0]
        assert "DELETE FROM" in sql
        assert "scope = %s" in sql

    async def test_deserialize_memory(self, store):
        """Test memory deserialization."""
        row = {
            "id": "mem-1",
            "scope": "project",
            "scope_id": "proj-1",
            "category": "convention",
            "key": "code-style",
            "content": "Use tabs",
            "metadata": '{"source": "user"}',
            "priority": 10,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        memory = store._deserialize_memory(row)

        assert memory.id == "mem-1"
        assert memory.scope == MemoryScope.PROJECT
        assert memory.category == MemoryCategory.CONVENTION
        assert memory.metadata["source"] == "user"


class TestPostgresScopedMemoryStore:
    """Tests for PostgresScopedMemoryStore."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock connection manager."""
        manager = MagicMock()
        manager.is_connected = False
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        manager.execute = AsyncMock(return_value="INSERT 0 1")
        manager.fetchrow = AsyncMock(return_value=None)
        manager.fetch = AsyncMock(return_value=[])
        manager.fetchval = AsyncMock(return_value=0)
        return manager

    @pytest.fixture
    def store(self, mock_manager):
        """Create a store with mocked connection."""
        from ctxforge.storage.postgres.scoped_memory import PostgresScopedMemoryStore
        store = PostgresScopedMemoryStore(connection_manager=mock_manager)
        store._initialized = True
        return store

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory."""
        return ScopedMemory(
            id="mem-1",
            scope=MemoryScope.PROJECT,
            scope_id="proj-1",
            category=MemoryCategory.CONVENTION,
            key="code-style",
            content="Use 4 spaces",
            priority=5,
        )

    async def test_save_uses_postgres_syntax(self, store, mock_manager, sample_memory):
        """Test that save uses PostgreSQL parameter syntax."""
        await store.save(sample_memory)

        mock_manager.execute.assert_called()
        sql = mock_manager.execute.call_args[0][0]

        assert "INSERT INTO" in sql
        assert "ON CONFLICT" in sql  # PostgreSQL upsert syntax
        assert "$1" in sql  # PostgreSQL parameter syntax

    async def test_get_uses_postgres_params(self, store, mock_manager):
        """Test that get uses PostgreSQL parameters."""
        await store.get(MemoryScope.GLOBAL, "user-1", "key-1")

        mock_manager.fetchrow.assert_called_once()
        sql = mock_manager.fetchrow.call_args[0][0]

        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql

    async def test_query_uses_any_for_categories(self, store, mock_manager):
        """Test that query uses ANY for category arrays."""
        query = ScopedMemoryQuery(
            user_id="user-1",
            categories=[MemoryCategory.PREFERENCE, MemoryCategory.CONVENTION],
        )

        await store.query(query)

        sql = mock_manager.fetch.call_args[0][0]
        assert "= ANY($" in sql

    async def test_count_returns_value(self, store, mock_manager):
        """Test count returns fetchval result."""
        mock_manager.fetchval.return_value = 10

        result = await store.count()

        assert result == 10

    async def test_delete_checks_result(self, store, mock_manager):
        """Test delete checks command result."""
        mock_manager.execute.return_value = "DELETE 1"

        result = await store.delete(MemoryScope.PROJECT, "proj-1", "key-1")

        assert result is True

    async def test_delete_returns_false_when_not_found(self, store, mock_manager):
        """Test delete returns False when nothing deleted."""
        mock_manager.execute.return_value = "DELETE 0"

        result = await store.delete(MemoryScope.PROJECT, "proj-1", "nonexistent")

        assert result is False
