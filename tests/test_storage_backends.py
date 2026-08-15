"""
Tests for storage backend implementations.

These tests use mocks to test the storage logic without requiring
actual Redis or PostgreSQL connections.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.exceptions import ConcurrencyError
from ctxforge.core.memory import MemoryItem, MemoryQuery, MemoryType
from ctxforge.core.session import Session
from ctxforge.storage.connection import (
    PostgresConfig,
    PostgresConnectionManager,
    RedisConfig,
    RedisConnectionManager,
)

# =============================================================================
# Test Configuration Classes
# =============================================================================

class TestRedisConfig:
    """Tests for RedisConfig."""
    
    def test_default_values(self):
        """Default configuration values are set correctly."""
        config = RedisConfig()
        
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.ssl is False
        assert config.max_connections == 10
        assert config.session_prefix == "ctx:session:"
        assert config.memory_prefix == "ctx:memory:"
        assert config.session_ttl_seconds == 86400
    
    def test_custom_values(self):
        """Custom configuration values are applied."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            password="secret",
            ssl=True,
            session_prefix="myapp:sess:",
        )
        
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.ssl is True
        assert config.session_prefix == "myapp:sess:"
    
    def test_get_connection_kwargs(self):
        """Connection kwargs are generated correctly."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            password="secret",
        )
        
        kwargs = config.get_connection_kwargs()
        
        assert kwargs["host"] == "redis.example.com"
        assert kwargs["port"] == 6380
        assert kwargs["password"] == "secret"
        assert kwargs["decode_responses"] is True


class TestPostgresConfig:
    """Tests for PostgresConfig."""
    
    def test_default_values(self):
        """Default configuration values are set correctly."""
        config = PostgresConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "context_engine"
        assert config.user == "postgres"
        assert config.password is None
        assert config.ssl is False
        assert config.sessions_table == "sessions"
        assert config.memories_table == "memories"
    
    def test_get_dsn(self):
        """DSN string is generated correctly."""
        config = PostgresConfig(
            host="db.example.com",
            user="myuser",
            password="secret",
            database="mydb",
        )
        
        dsn = config.get_dsn()
        
        assert "myuser:secret@db.example.com" in dsn
        assert "mydb" in dsn
    
    def test_get_dsn_without_password(self):
        """DSN string works without password."""
        config = PostgresConfig()
        dsn = config.get_dsn()
        
        assert "postgres@localhost" in dsn
        assert ":@" not in dsn  # No empty password


# =============================================================================
# Test Connection Managers
# =============================================================================

class TestRedisConnectionManager:
    """Tests for RedisConnectionManager."""
    
    def test_init_with_default_config(self):
        """Initializes with default config."""
        manager = RedisConnectionManager()
        
        assert manager.config.host == "localhost"
        assert not manager.is_connected
    
    def test_init_with_custom_config(self):
        """Initializes with custom config."""
        config = RedisConfig(host="redis.example.com")
        manager = RedisConnectionManager(config)
        
        assert manager.config.host == "redis.example.com"
    
    def test_client_raises_when_not_connected(self):
        """Accessing client raises when not connected."""
        manager = RedisConnectionManager()
        
        with pytest.raises(RuntimeError, match="Not connected"):
            _ = manager.client


class TestPostgresConnectionManager:
    """Tests for PostgresConnectionManager."""
    
    def test_init_with_default_config(self):
        """Initializes with default config."""
        manager = PostgresConnectionManager()
        
        assert manager.config.host == "localhost"
        assert not manager.is_connected
    
    def test_pool_raises_when_not_connected(self):
        """Accessing pool raises when not connected."""
        manager = PostgresConnectionManager()
        
        with pytest.raises(RuntimeError, match="Not connected"):
            _ = manager.pool


# =============================================================================
# Test Redis Session Store
# =============================================================================

class TestRedisSessionStore:
    """Tests for RedisSessionStore."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = MagicMock()
        client.ping = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        client.delete = AsyncMock()
        client.exists = AsyncMock(return_value=0)
        client.expire = AsyncMock()
        client.zadd = AsyncMock()
        client.zrem = AsyncMock()
        client.zrevrange = AsyncMock(return_value=[])
        client.mget = AsyncMock(return_value=[])
        client.scan = AsyncMock(return_value=(0, []))
        client.aclose = AsyncMock()
        
        # Pipeline mock - needs to be a proper async context manager
        pipe = MagicMock()
        pipe.watch = AsyncMock()
        pipe.multi = MagicMock()
        pipe.set = MagicMock()
        pipe.zadd = MagicMock()
        pipe.delete = MagicMock()
        pipe.zrem = MagicMock()
        pipe.execute = AsyncMock()
        
        # Create async context manager for pipeline
        async def pipeline_aenter(*args, **kwargs):
            return pipe
        async def pipeline_aexit(*args, **kwargs):
            pass
        pipe.__aenter__ = pipeline_aenter
        pipe.__aexit__ = pipeline_aexit
        client.pipeline = MagicMock(return_value=pipe)
        
        return client
    
    @pytest.fixture
    def mock_connection_manager(self, mock_redis_client):
        """Create a mock connection manager."""
        manager = MagicMock(spec=RedisConnectionManager)
        manager.is_connected = True
        manager.client = mock_redis_client
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        return manager
    
    @pytest.fixture
    def redis_store(self, mock_connection_manager):
        """Create a Redis session store with mocked connection."""
        from ctxforge.storage.redis import RedisSessionStore
        
        store = RedisSessionStore(connection_manager=mock_connection_manager)
        return store
    
    @pytest.mark.asyncio
    async def test_load_creates_new_session(self, redis_store, mock_redis_client):
        """Load returns new session when not found."""
        mock_redis_client.get.return_value = None
        
        session = await redis_store.load("sess_123", "user_456")
        
        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
        assert session.version == 0
    
    @pytest.mark.asyncio
    async def test_load_returns_existing_session(self, redis_store, mock_redis_client):
        """Load returns existing session when found."""
        existing = Session(
            session_id="sess_123",
            user_id="user_456",
            version=5,
        )
        mock_redis_client.get.return_value = existing.model_dump_json()
        
        session = await redis_store.load("sess_123", "user_456")
        
        assert session.session_id == "sess_123"
        assert session.version == 5
        mock_redis_client.expire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_new_session(self, redis_store, mock_redis_client):
        """Save creates new session."""
        session = Session(session_id="sess_123", user_id="user_456")
        mock_redis_client.get.return_value = None
        
        await redis_store.save(session)
        
        assert session.version == 1  # Incremented
    
    @pytest.mark.asyncio
    async def test_save_raises_on_version_conflict(self, redis_store, mock_redis_client):
        """Save raises ConcurrencyError on version conflict."""
        session = Session(session_id="sess_123", user_id="user_456", version=1)
        
        # Current version in store is higher
        existing = Session(session_id="sess_123", user_id="user_456", version=5)
        
        # The save method calls client.get directly (outside the pipeline)
        mock_redis_client.get = AsyncMock(return_value=existing.model_dump_json())
        
        with pytest.raises(ConcurrencyError):
            await redis_store.save(session)
    
    @pytest.mark.asyncio
    async def test_delete_existing_session(self, redis_store, mock_redis_client):
        """Delete returns True for existing session."""
        existing = Session(session_id="sess_123", user_id="user_456")
        mock_redis_client.get.return_value = existing.model_dump_json()
        
        result = await redis_store.delete("sess_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, redis_store, mock_redis_client):
        """Delete returns False for nonexistent session."""
        mock_redis_client.get.return_value = None
        
        result = await redis_store.delete("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_returns_true(self, redis_store, mock_redis_client):
        """Exists returns True when session exists."""
        mock_redis_client.exists.return_value = 1
        
        result = await redis_store.exists("sess_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_returns_false(self, redis_store, mock_redis_client):
        """Exists returns False when session doesn't exist."""
        mock_redis_client.exists.return_value = 0
        
        result = await redis_store.exists("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, redis_store, mock_redis_client):
        """List sessions returns user's sessions."""
        session1 = Session(session_id="sess_1", user_id="user_456")
        session2 = Session(session_id="sess_2", user_id="user_456")
        
        mock_redis_client.zrevrange.return_value = ["sess_1", "sess_2"]
        mock_redis_client.mget.return_value = [
            session1.model_dump_json(),
            session2.model_dump_json(),
        ]
        
        sessions = await redis_store.list_sessions("user_456", limit=10)
        
        assert len(sessions) == 2
        assert sessions[0].session_id == "sess_1"


# =============================================================================
# Test Redis Memory Store
# =============================================================================

class TestRedisMemoryStore:
    """Tests for RedisMemoryStore."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        client.ping = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        client.delete = AsyncMock()
        client.exists = AsyncMock(return_value=0)
        client.zadd = AsyncMock()
        client.zrem = AsyncMock()
        client.zcard = AsyncMock(return_value=0)
        client.zrevrange = AsyncMock(return_value=[])
        client.mget = AsyncMock(return_value=[])
        client.sadd = AsyncMock()
        client.srem = AsyncMock()
        client.smembers = AsyncMock(return_value=set())
        client.scan = AsyncMock(return_value=(0, []))
        client.aclose = AsyncMock()
        
        # Pipeline mock
        pipe = AsyncMock()
        pipe.set = MagicMock()
        pipe.delete = MagicMock()
        pipe.zadd = MagicMock()
        pipe.zrem = MagicMock()
        pipe.sadd = MagicMock()
        pipe.srem = MagicMock()
        pipe.execute = AsyncMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__ = AsyncMock()
        client.pipeline = MagicMock(return_value=pipe)
        
        return client
    
    @pytest.fixture
    def mock_connection_manager(self, mock_redis_client):
        """Create a mock connection manager."""
        manager = MagicMock(spec=RedisConnectionManager)
        manager.is_connected = True
        manager.client = mock_redis_client
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        return manager
    
    @pytest.fixture
    def redis_memory_store(self, mock_connection_manager):
        """Create a Redis memory store with mocked connection."""
        from ctxforge.storage.redis import RedisMemoryStore
        
        store = RedisMemoryStore(connection_manager=mock_connection_manager)
        return store
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory item."""
        return MemoryItem(
            memory_id="mem_123",
            user_id="user_456",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
            confidence_score=0.9,
            tags=["preference", "ui"],
        )
    
    @pytest.mark.asyncio
    async def test_add_memory(self, redis_memory_store, mock_redis_client, sample_memory):
        """Add creates new memory with indexes."""
        result = await redis_memory_store.add(sample_memory)
        
        assert result == "mem_123"
        # Pipeline should have been used
        pipe = mock_redis_client.pipeline.return_value
        pipe.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_memory(self, redis_memory_store, mock_redis_client, sample_memory):
        """Get returns memory by ID."""
        mock_redis_client.get.return_value = sample_memory.model_dump_json()
        
        result = await redis_memory_store.get("mem_123")
        
        assert result is not None
        assert result.memory_id == "mem_123"
        assert result.content == "User prefers dark mode"
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, redis_memory_store, mock_redis_client):
        """Get returns None when not found."""
        mock_redis_client.get.return_value = None
        
        result = await redis_memory_store.get("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_memory(self, redis_memory_store, mock_redis_client, sample_memory):
        """Update modifies existing memory."""
        mock_redis_client.exists.return_value = True
        mock_redis_client.get.return_value = sample_memory.model_dump_json()
        
        sample_memory.content = "Updated content"
        result = await redis_memory_store.update(sample_memory)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, redis_memory_store, mock_redis_client, sample_memory):
        """Update returns False when memory not found."""
        mock_redis_client.exists.return_value = False
        
        result = await redis_memory_store.update(sample_memory)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, redis_memory_store, mock_redis_client, sample_memory):
        """Delete removes memory and updates indexes."""
        mock_redis_client.get.return_value = sample_memory.model_dump_json()
        
        result = await redis_memory_store.delete("mem_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, redis_memory_store, mock_redis_client):
        """Delete returns False when memory not found."""
        mock_redis_client.get.return_value = None
        
        result = await redis_memory_store.delete("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count(self, redis_memory_store, mock_redis_client):
        """Count returns number of memories for user."""
        mock_redis_client.zcard.return_value = 42
        
        result = await redis_memory_store.count("user_456")
        
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_search_returns_matching_memories(
        self, redis_memory_store, mock_redis_client, sample_memory
    ):
        """Search returns memories matching query."""
        mock_redis_client.zrevrange.return_value = ["mem_123"]
        mock_redis_client.mget.return_value = [sample_memory.model_dump_json()]
        
        query = MemoryQuery(
            user_id="user_456",
            query_text="dark mode",
            limit=10,
        )
        
        results = await redis_memory_store.search(query)
        
        assert len(results) == 1
        assert results[0].memory_id == "mem_123"


# =============================================================================
# Test PostgreSQL Session Store
# =============================================================================

class TestPostgresSessionStore:
    """Tests for PostgresSessionStore."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock PostgreSQL connection manager."""
        manager = MagicMock(spec=PostgresConnectionManager)
        manager.is_connected = True
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        manager.execute = AsyncMock()
        manager.fetch = AsyncMock(return_value=[])
        manager.fetchrow = AsyncMock(return_value=None)
        manager.fetchval = AsyncMock(return_value=None)
        return manager
    
    @pytest.fixture
    def postgres_store(self, mock_connection_manager):
        """Create a PostgreSQL session store with mocked connection."""
        from ctxforge.storage.postgres import PostgresSessionStore
        
        store = PostgresSessionStore(connection_manager=mock_connection_manager)
        store._initialized = True  # Skip table creation
        return store
    
    @pytest.mark.asyncio
    async def test_load_creates_new_session(self, postgres_store, mock_connection_manager):
        """Load returns new session when not found."""
        mock_connection_manager.fetchrow.return_value = None
        
        session = await postgres_store.load("sess_123", "user_456")
        
        assert session.session_id == "sess_123"
        assert session.user_id == "user_456"
    
    @pytest.mark.asyncio
    async def test_load_returns_existing_session(self, postgres_store, mock_connection_manager):
        """Load returns existing session when found."""
        existing = Session(
            session_id="sess_123",
            user_id="user_456",
            version=5,
        )
        mock_connection_manager.fetchrow.return_value = {
            "data": existing.model_dump_json()
        }
        
        session = await postgres_store.load("sess_123", "user_456")
        
        assert session.session_id == "sess_123"
        assert session.version == 5
    
    @pytest.mark.asyncio
    async def test_save_raises_on_version_conflict(self, postgres_store, mock_connection_manager):
        """Save raises ConcurrencyError on version conflict."""
        session = Session(session_id="sess_123", user_id="user_456", version=1)
        
        # Current version in DB is higher
        mock_connection_manager.fetchval.return_value = 5
        
        with pytest.raises(ConcurrencyError):
            await postgres_store.save(session)
    
    @pytest.mark.asyncio
    async def test_delete_existing_session(self, postgres_store, mock_connection_manager):
        """Delete returns True for existing session."""
        mock_connection_manager.fetchval.return_value = "sess_123"
        
        result = await postgres_store.delete("sess_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, postgres_store, mock_connection_manager):
        """Delete returns False for nonexistent session."""
        mock_connection_manager.fetchval.return_value = None
        
        result = await postgres_store.delete("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists(self, postgres_store, mock_connection_manager):
        """Exists checks for session presence."""
        mock_connection_manager.fetchval.return_value = 1
        
        result = await postgres_store.exists("sess_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, postgres_store, mock_connection_manager):
        """List sessions returns user's sessions."""
        session1 = Session(session_id="sess_1", user_id="user_456")
        session2 = Session(session_id="sess_2", user_id="user_456")
        
        mock_connection_manager.fetch.return_value = [
            {"data": session1.model_dump_json()},
            {"data": session2.model_dump_json()},
        ]
        
        sessions = await postgres_store.list_sessions("user_456")
        
        assert len(sessions) == 2


# =============================================================================
# Test PostgreSQL Memory Store
# =============================================================================

class TestPostgresMemoryStore:
    """Tests for PostgresMemoryStore."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock PostgreSQL connection manager."""
        manager = MagicMock(spec=PostgresConnectionManager)
        manager.is_connected = True
        manager.connect = AsyncMock()
        manager.disconnect = AsyncMock()
        manager.execute = AsyncMock()
        manager.fetch = AsyncMock(return_value=[])
        manager.fetchrow = AsyncMock(return_value=None)
        manager.fetchval = AsyncMock(return_value=None)
        return manager
    
    @pytest.fixture
    def postgres_memory_store(self, mock_connection_manager):
        """Create a PostgreSQL memory store with mocked connection."""
        from ctxforge.storage.postgres import PostgresMemoryStore
        
        store = PostgresMemoryStore(connection_manager=mock_connection_manager)
        store._initialized = True  # Skip table creation
        return store
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory item."""
        return MemoryItem(
            memory_id="mem_123",
            user_id="user_456",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
            confidence_score=0.9,
            tags=["preference", "ui"],
        )
    
    @pytest.mark.asyncio
    async def test_add_memory(
        self, postgres_memory_store, mock_connection_manager, sample_memory
    ):
        """Add creates new memory."""
        mock_connection_manager.fetchval.return_value = "mem_123"
        
        result = await postgres_memory_store.add(sample_memory)
        
        assert result == "mem_123"
    
    @pytest.mark.asyncio
    async def test_get_memory(
        self, postgres_memory_store, mock_connection_manager, sample_memory
    ):
        """Get returns memory by ID."""
        mock_connection_manager.fetchrow.return_value = {
            "data": sample_memory.model_dump_json()
        }
        
        result = await postgres_memory_store.get("mem_123")
        
        assert result is not None
        assert result.memory_id == "mem_123"
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, postgres_memory_store, mock_connection_manager):
        """Get returns None when not found."""
        mock_connection_manager.fetchrow.return_value = None
        
        result = await postgres_memory_store.get("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_memory(
        self, postgres_memory_store, mock_connection_manager, sample_memory
    ):
        """Update modifies existing memory."""
        mock_connection_manager.fetchval.return_value = "mem_123"
        
        result = await postgres_memory_store.update(sample_memory)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_update_memory_not_found(
        self, postgres_memory_store, mock_connection_manager, sample_memory
    ):
        """Update returns False when memory not found."""
        mock_connection_manager.fetchval.return_value = None
        
        result = await postgres_memory_store.update(sample_memory)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, postgres_memory_store, mock_connection_manager):
        """Delete removes memory."""
        mock_connection_manager.fetchval.return_value = "mem_123"
        
        result = await postgres_memory_store.delete("mem_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, postgres_memory_store, mock_connection_manager):
        """Delete returns False when memory not found."""
        mock_connection_manager.fetchval.return_value = None
        
        result = await postgres_memory_store.delete("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count(self, postgres_memory_store, mock_connection_manager):
        """Count returns number of memories for user."""
        mock_connection_manager.fetchval.return_value = 42
        
        result = await postgres_memory_store.count("user_456")
        
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_search(self, postgres_memory_store, mock_connection_manager, sample_memory):
        """Search returns matching memories."""
        mock_connection_manager.fetch.return_value = [
            {"data": sample_memory.model_dump_json()}
        ]
        
        query = MemoryQuery(
            user_id="user_456",
            query_text="dark mode",
            limit=10,
        )
        
        results = await postgres_memory_store.search(query)
        
        assert len(results) == 1
        assert results[0].memory_id == "mem_123"
    
    @pytest.mark.asyncio
    async def test_get_by_user(
        self, postgres_memory_store, mock_connection_manager, sample_memory
    ):
        """Get by user returns all user's memories."""
        mock_connection_manager.fetch.return_value = [
            {"data": sample_memory.model_dump_json()}
        ]
        
        results = await postgres_memory_store.get_by_user("user_456")
        
        assert len(results) == 1


# =============================================================================
# Test Registry Integration
# =============================================================================

class TestStorageBackendRegistry:
    """Tests for storage backend registry integration."""
    
    def test_redis_session_store_registered(self):
        """RedisSessionStore is registered."""
        from ctxforge.engine.registry import registry
        
        assert registry.get_session_store("redis") is not None
    
    def test_redis_memory_store_registered(self):
        """RedisMemoryStore is registered."""
        from ctxforge.engine.registry import registry
        
        assert registry.get_memory_store("redis") is not None
    
    def test_postgres_session_store_registered(self):
        """PostgresSessionStore is registered."""
        from ctxforge.engine.registry import registry
        
        assert registry.get_session_store("postgres") is not None
    
    def test_postgres_memory_store_registered(self):
        """PostgresMemoryStore is registered."""
        from ctxforge.engine.registry import registry
        
        assert registry.get_memory_store("postgres") is not None
    
    def test_list_session_stores(self):
        """All session stores are listed."""
        from ctxforge.engine.registry import registry
        
        stores = registry.list_session_stores()
        
        assert "memory" in stores
        assert "redis" in stores
        assert "postgres" in stores
    
    def test_list_memory_stores(self):
        """All memory stores are listed."""
        from ctxforge.engine.registry import registry
        
        stores = registry.list_memory_stores()
        
        assert "memory" in stores
        assert "redis" in stores
        assert "postgres" in stores

