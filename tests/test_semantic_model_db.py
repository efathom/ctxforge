"""
Tests for database-backed semantic model stores.

These tests require running PostgreSQL and/or MySQL servers.
They are skipped if the databases are not available.

To run PostgreSQL tests:
1. Start PostgreSQL server
2. Create a test database: CREATE DATABASE ctxforge_test;
3. Set environment variables:
   - POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE, POSTGRES_USER, POSTGRES_PASSWORD
4. Run: pytest tests/test_semantic_model_db.py -v -k postgres

To run MySQL tests:
1. Start MySQL server
2. Create a test database: CREATE DATABASE ctxforge_test;
3. Set environment variables:
   - MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD
4. Run: pytest tests/test_semantic_model_db.py -v -k mysql
"""

import os

import pytest

from ctxforge.core.semantic_model import (
    EntityDefinition,
    RelationshipDefinition,
    SemanticModel,
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


# ============================================================================
# PostgreSQL Tests
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


@pytest.fixture
def sample_model() -> SemanticModel:
    """Create a sample semantic model for testing."""
    return SemanticModel(
        name="Test Model",
        description="A test semantic model",
        version="1.0",
        entities=[
            EntityDefinition(
                name="users",
                description="User accounts",
                use_cases=["Find user by ID", "List active users"],
                attributes=[{"name": "id", "type": "integer"}],
            ),
            EntityDefinition(
                name="orders",
                description="Customer orders",
                use_cases=["Find order by ID"],
            ),
        ],
        relationships=[
            RelationshipDefinition(
                name="user_orders",
                from_entity="users",
                to_entity="orders",
                description="User's orders",
                cardinality="one_to_many",
            ),
        ],
        global_rules=["Always validate input", "Log all queries"],
        common_gotchas=["Check for null values"],
    )


@pytest.mark.skipif(
    not ASYNCPG_AVAILABLE,
    reason="asyncpg not installed. Install with: pip install asyncpg"
)
class TestPostgresSemanticModelStore:
    """Tests for PostgreSQL semantic model store."""

    @pytest.fixture
    async def store(self):
        """Create a test store."""
        available = await postgres_is_available()
        if not available:
            pytest.skip("PostgreSQL not available")

        from ctxforge.storage.postgres.semantic_model import PostgresSemanticModelStore
        config = get_postgres_config()
        store = PostgresSemanticModelStore(config, table_name="test_semantic_models")
        await store.initialize()

        yield store

        # Cleanup: drop test table
        await store._manager.execute("DROP TABLE IF EXISTS test_semantic_models")
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_load(self, store, sample_model):
        """Test saving and loading a model."""
        await store.save("test-model", sample_model)

        loaded = await store.load("test-model")

        assert loaded is not None
        assert loaded.name == "Test Model"
        assert loaded.version == "1.0"
        assert len(loaded.entities) == 2
        assert loaded.entities[0].name == "users"
        assert len(loaded.relationships) == 1
        assert "Always validate input" in loaded.global_rules

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        """Test loading a nonexistent model."""
        result = await store.load("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_model(self, store, sample_model):
        """Test updating an existing model."""
        await store.save("update-test", sample_model)

        # Modify and save again
        updated = SemanticModel(
            name="Updated Model",
            version="2.0",
            entities=[],
        )
        await store.save("update-test", updated)

        loaded = await store.load("update-test")
        assert loaded.name == "Updated Model"
        assert loaded.version == "2.0"

    @pytest.mark.asyncio
    async def test_list_models(self, store, sample_model):
        """Test listing models."""
        await store.save("model-a", sample_model)
        await store.save("model-b", sample_model)

        models = await store.list_models()

        assert "model-a" in models
        assert "model-b" in models

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_model):
        """Test deleting a model."""
        await store.save("to-delete", sample_model)

        result = await store.delete("to-delete")
        assert result is True

        loaded = await store.load("to-delete")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a nonexistent model."""
        result = await store.delete("nonexistent")
        assert result is False


# ============================================================================
# MySQL Tests
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
    reason="aiomysql not installed. Install with: pip install aiomysql"
)
class TestMySQLSemanticModelStore:
    """Tests for MySQL semantic model store."""

    @pytest.fixture
    async def store(self):
        """Create a test store."""
        available = await mysql_is_available()
        if not available:
            pytest.skip("MySQL not available")

        from ctxforge.storage.mysql.semantic_model import MySQLSemanticModelStore
        config = get_mysql_config()
        store = MySQLSemanticModelStore(config, table_name="test_semantic_models")
        await store.initialize()

        yield store

        # Cleanup: drop test table
        await store._manager.execute("DROP TABLE IF EXISTS test_semantic_models")
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_load(self, store, sample_model):
        """Test saving and loading a model."""
        await store.save("test-model", sample_model)

        loaded = await store.load("test-model")

        assert loaded is not None
        assert loaded.name == "Test Model"
        assert loaded.version == "1.0"
        assert len(loaded.entities) == 2
        assert loaded.entities[0].name == "users"
        assert len(loaded.relationships) == 1
        assert "Always validate input" in loaded.global_rules

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        """Test loading a nonexistent model."""
        result = await store.load("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_model(self, store, sample_model):
        """Test updating an existing model."""
        await store.save("update-test", sample_model)

        # Modify and save again
        updated = SemanticModel(
            name="Updated Model",
            version="2.0",
            entities=[],
        )
        await store.save("update-test", updated)

        loaded = await store.load("update-test")
        assert loaded.name == "Updated Model"
        assert loaded.version == "2.0"

    @pytest.mark.asyncio
    async def test_list_models(self, store, sample_model):
        """Test listing models."""
        await store.save("model-a", sample_model)
        await store.save("model-b", sample_model)

        models = await store.list_models()

        assert "model-a" in models
        assert "model-b" in models

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_model):
        """Test deleting a model."""
        await store.save("to-delete", sample_model)

        result = await store.delete("to-delete")
        assert result is True

        loaded = await store.load("to-delete")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a nonexistent model."""
        result = await store.delete("nonexistent")
        assert result is False
