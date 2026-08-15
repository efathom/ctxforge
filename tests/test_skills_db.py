"""
Tests for Database-backed Skill Stores.

These tests verify the SQL generation and basic structure of MySQL and PostgreSQL
skill store implementations without requiring actual database connections.
"""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.skill import (
    Skill,
    SkillScope,
)


class TestMySQLSkillStore:
    """Tests for MySQLSkillStore."""

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
        from ctxforge.storage.mysql.skill import MySQLSkillStore
        store = MySQLSkillStore(connection_manager=mock_manager)
        store._initialized = True
        return store

    @pytest.fixture
    def sample_skill(self):
        """Create a sample skill."""
        return Skill(
            name="sql-optimize",
            description="Optimize SQL queries",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# SQL Optimization\n\n...",
            triggers=["slow query", "optimize"],
            prerequisites=["sql-basics"],
            allowed_tools=["explain"],
        )

    async def test_save_calls_execute(self, store, mock_manager, sample_skill):
        """Test that save calls execute with correct SQL."""
        await store.save(sample_skill)

        mock_manager.execute.assert_called_once()
        sql = mock_manager.execute.call_args[0][0]

        assert "INSERT INTO" in sql
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "skills" in sql

    async def test_get_calls_fetchone(self, store, mock_manager):
        """Test that get calls fetchone."""
        await store.get("sql-optimize", SkillScope.BASE, "system")

        mock_manager.fetchone.assert_called_once()
        sql = mock_manager.fetchone.call_args[0][0]

        assert "SELECT * FROM" in sql
        assert "name = %s" in sql

    async def test_get_metadata_selects_fewer_columns(self, store, mock_manager):
        """Test get_metadata selects only metadata columns."""
        await store.get_metadata("sql-optimize", SkillScope.BASE, "system")

        sql = mock_manager.fetchone.call_args[0][0]

        assert "name, description, scope, scope_id, triggers, version" in sql
        assert "content" not in sql.split("FROM")[0]

    async def test_list_metadata(self, store, mock_manager):
        """Test list_metadata."""
        await store.list_metadata(SkillScope.BASE, "system")

        mock_manager.fetchall.assert_called_once()
        sql = mock_manager.fetchall.call_args[0][0]

        assert "ORDER BY name ASC" in sql

    async def test_list_all_metadata_includes_scope_priority(self, store, mock_manager):
        """Test list_all_metadata includes scope priority."""
        await store.list_all_metadata(user_id="user-1", project_id="proj-1")

        sql = mock_manager.fetchall.call_args[0][0]

        assert "scope_priority" in sql
        assert "CASE scope" in sql

    async def test_delete(self, store, mock_manager):
        """Test delete."""
        mock_manager.execute.return_value = 1

        result = await store.delete("sql-optimize", SkillScope.BASE, "system")

        assert result is True
        sql = mock_manager.execute.call_args[0][0]
        assert "DELETE FROM" in sql

    async def test_count(self, store, mock_manager):
        """Test count."""
        mock_manager.fetchone.return_value = {"cnt": 5}

        result = await store.count(scope=SkillScope.BASE)

        assert result == 5

    async def test_clear_with_scope(self, store, mock_manager):
        """Test clear with scope filter."""
        mock_manager.execute.return_value = 3

        result = await store.clear(scope=SkillScope.USER, scope_id="user-1")

        assert result == 3
        sql = mock_manager.execute.call_args[0][0]
        assert "DELETE FROM" in sql

    async def test_deserialize_skill(self, store):
        """Test skill deserialization."""
        row = {
            "name": "test-skill",
            "description": "Test description",
            "scope": "base",
            "scope_id": "system",
            "content": "# Content",
            "triggers": '["trigger1", "trigger2"]',
            "prerequisites": '[]',
            "allowed_tools": '["tool1"]',
            "metadata": '{}',
            "version": "1.0",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        skill = store._deserialize_skill(row)

        assert skill.name == "test-skill"
        assert skill.scope == SkillScope.BASE
        assert skill.triggers == ["trigger1", "trigger2"]
        assert skill.allowed_tools == ["tool1"]

    async def test_search_by_trigger_returns_matches(self, store, mock_manager):
        """Test search_by_trigger."""
        # Mock list_all_metadata to return skills
        mock_manager.fetchall.return_value = [
            {
                "name": "sql-optimize",
                "description": "Optimize SQL",
                "scope": "base",
                "scope_id": "system",
                "triggers": '["slow query"]',
                "version": "1.0",
                "scope_priority": 0,
            }
        ]

        matches = await store.search_by_trigger("I have a slow query")

        assert len(matches) == 1
        assert matches[0].skill.name == "sql-optimize"


class TestPostgresSkillStore:
    """Tests for PostgresSkillStore."""

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
        from ctxforge.storage.postgres.skill import PostgresSkillStore
        store = PostgresSkillStore(connection_manager=mock_manager)
        store._initialized = True
        return store

    @pytest.fixture
    def sample_skill(self):
        """Create a sample skill."""
        return Skill(
            name="sql-optimize",
            description="Optimize SQL queries",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# SQL Optimization\n\n...",
            triggers=["slow query"],
        )

    async def test_save_uses_postgres_syntax(self, store, mock_manager, sample_skill):
        """Test that save uses PostgreSQL parameter syntax."""
        await store.save(sample_skill)

        mock_manager.execute.assert_called()
        sql = mock_manager.execute.call_args[0][0]

        assert "INSERT INTO" in sql
        assert "ON CONFLICT" in sql
        assert "$1" in sql

    async def test_get_uses_postgres_params(self, store, mock_manager):
        """Test that get uses PostgreSQL parameters."""
        await store.get("sql-optimize", SkillScope.BASE, "system")

        mock_manager.fetchrow.assert_called_once()
        sql = mock_manager.fetchrow.call_args[0][0]

        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql

    async def test_list_all_metadata_builds_conditions(self, store, mock_manager):
        """Test list_all_metadata builds scope conditions."""
        await store.list_all_metadata(user_id="user-1", project_id="proj-1")

        sql = mock_manager.fetch.call_args[0][0]

        assert "scope = 'base'" in sql
        assert "scope = 'user'" in sql
        assert "scope = 'project'" in sql

    async def test_count_returns_value(self, store, mock_manager):
        """Test count returns fetchval result."""
        mock_manager.fetchval.return_value = 10

        result = await store.count()

        assert result == 10

    async def test_delete_checks_result(self, store, mock_manager):
        """Test delete checks command result."""
        mock_manager.execute.return_value = "DELETE 1"

        result = await store.delete("sql-optimize", SkillScope.BASE, "system")

        assert result is True

    async def test_delete_returns_false_when_not_found(self, store, mock_manager):
        """Test delete returns False when nothing deleted."""
        mock_manager.execute.return_value = "DELETE 0"

        result = await store.delete("nonexistent", SkillScope.BASE, "system")

        assert result is False

    async def test_create_table_sql_has_jsonb(self, mock_manager):
        """Test that create table SQL uses JSONB."""
        from ctxforge.storage.postgres.skill import CREATE_SKILLS_TABLE

        assert "JSONB" in CREATE_SKILLS_TABLE
        assert "TIMESTAMPTZ" in CREATE_SKILLS_TABLE

    async def test_create_table_sql_has_fts_index(self, mock_manager):
        """Test that create table SQL has full-text search index."""
        from ctxforge.storage.postgres.skill import CREATE_SKILLS_TABLE

        assert "to_tsvector" in CREATE_SKILLS_TABLE
        assert "GIN" in CREATE_SKILLS_TABLE
