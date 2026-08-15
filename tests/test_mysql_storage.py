"""
Tests for MySQL storage implementations.

These tests require a running MySQL server. They are skipped if MySQL is not available.

To run these tests:
1. Start MySQL server
2. Create a test database: CREATE DATABASE ctxforge_test;
3. Set environment variables:
   - MYSQL_HOST (default: localhost)
   - MYSQL_PORT (default: 3306)
   - MYSQL_DATABASE (default: ctxforge_test)
   - MYSQL_USER (default: root)
   - MYSQL_PASSWORD (default: empty)
4. Run: pytest tests/test_mysql_storage.py -v
"""

import asyncio
import os

import pytest

# Check if aiomysql is available
try:
    import aiomysql
    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False

from ctxforge.core.exceptions import ConcurrencyError
from ctxforge.core.expertise import Expertise, ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryItem, MemoryQuery, MemoryType
from ctxforge.core.session import Session
from ctxforge.storage.connection import MySQLConfig, MySQLConnectionManager
from ctxforge.storage.mysql import MySQLExpertiseStore, MySQLMemoryStore, MySQLSessionStore

# Skip all tests if aiomysql is not installed
pytestmark = pytest.mark.skipif(
    not AIOMYSQL_AVAILABLE,
    reason="aiomysql not installed. Install with: pip install aiomysql"
)


def get_test_config() -> MySQLConfig:
    """Get MySQL configuration from environment or defaults."""
    return MySQLConfig(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "ctxforge_test"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
    )


async def mysql_is_available() -> bool:
    """Check if MySQL is available and accessible."""
    if not AIOMYSQL_AVAILABLE:
        return False
    
    config = get_test_config()
    try:
        conn = await aiomysql.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password or "",
            db=config.database,
            connect_timeout=5,
        )
        await conn.ensure_closed()
        return True
    except Exception:
        return False


# Run check synchronously at import time
def check_mysql():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(mysql_is_available())


MYSQL_AVAILABLE = check_mysql() if AIOMYSQL_AVAILABLE else False


# Skip all tests if MySQL is not accessible
pytestmark = [
    pytest.mark.skipif(
        not AIOMYSQL_AVAILABLE,
        reason="aiomysql not installed"
    ),
    pytest.mark.skipif(
        not MYSQL_AVAILABLE,
        reason="MySQL server not accessible"
    ),
]


@pytest.fixture
def mysql_config() -> MySQLConfig:
    """Get MySQL test configuration."""
    return get_test_config()


@pytest.fixture
async def session_store(mysql_config):
    """Create and initialize a session store for testing."""
    store = MySQLSessionStore(mysql_config)
    await store.initialize()
    yield store
    await store.clear()
    await store.disconnect()


@pytest.fixture
async def memory_store(mysql_config):
    """Create and initialize a memory store for testing."""
    store = MySQLMemoryStore(mysql_config)
    await store.initialize()
    yield store
    await store.clear()
    await store.disconnect()


@pytest.fixture
async def expertise_store(mysql_config):
    """Create and initialize an expertise store for testing."""
    store = MySQLExpertiseStore(mysql_config)
    await store.initialize()
    yield store
    # Clean up - delete test expertise
    await store.close()


class TestMySQLConnectionManager:
    """Tests for MySQLConnectionManager."""
    
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, mysql_config):
        """Test connection lifecycle."""
        manager = MySQLConnectionManager(mysql_config)
        
        assert not manager.is_connected
        
        await manager.connect()
        assert manager.is_connected
        
        await manager.disconnect()
        assert not manager.is_connected
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mysql_config):
        """Test async context manager usage."""
        async with MySQLConnectionManager(mysql_config) as manager:
            assert manager.is_connected
            
            # Simple query
            result = await manager.fetchval("SELECT 1")
            assert result == 1
        
        assert not manager.is_connected
    
    @pytest.mark.asyncio
    async def test_execute_and_fetch(self, mysql_config):
        """Test execute and fetch operations."""
        async with MySQLConnectionManager(mysql_config) as manager:
            # Create temp table
            await manager.execute(
                "CREATE TEMPORARY TABLE test_table (id INT, name VARCHAR(50))"
            )
            
            # Insert
            await manager.execute(
                "INSERT INTO test_table VALUES (%s, %s)",
                (1, "test"),
            )
            
            # Fetch one
            row = await manager.fetchone("SELECT * FROM test_table WHERE id = %s", (1,))
            assert row["id"] == 1
            assert row["name"] == "test"
            
            # Fetch all
            await manager.execute("INSERT INTO test_table VALUES (%s, %s)", (2, "test2"))
            rows = await manager.fetchall("SELECT * FROM test_table ORDER BY id")
            assert len(rows) == 2
            
            # Fetch val
            count = await manager.fetchval("SELECT COUNT(*) FROM test_table")
            assert count == 2


class TestMySQLSessionStore:
    """Tests for MySQLSessionStore."""
    
    @pytest.mark.asyncio
    async def test_load_creates_new_session(self, session_store):
        """Test that load creates a new session if not found."""
        session = await session_store.load("mysql_sess_1", "user_1")
        
        assert session.session_id == "mysql_sess_1"
        assert session.user_id == "user_1"
        assert session.version == 0
    
    @pytest.mark.asyncio
    async def test_save_and_load(self, session_store):
        """Test saving and loading a session."""
        session = Session(session_id="mysql_sess_2", user_id="user_1")
        session.add_user_message("Hello MySQL")
        session.state.set("key", "value")
        
        await session_store.save(session)
        
        loaded = await session_store.load("mysql_sess_2", "user_1")
        
        assert loaded.session_id == "mysql_sess_2"
        assert len(loaded.events) == 1
        assert loaded.state.get("key") == "value"
        assert loaded.version == 1
    
    @pytest.mark.asyncio
    async def test_optimistic_locking(self, session_store):
        """Test optimistic locking prevents concurrent modifications."""
        session = Session(session_id="mysql_sess_3", user_id="user_1")
        await session_store.save(session)
        
        # Load two copies
        copy1 = await session_store.load("mysql_sess_3", "user_1")
        copy2 = await session_store.load("mysql_sess_3", "user_1")
        
        # Save first copy
        copy1.add_user_message("From copy 1")
        await session_store.save(copy1)
        
        # Try to save second copy - should fail
        copy2.add_user_message("From copy 2")
        with pytest.raises(ConcurrencyError):
            await session_store.save(copy2)
    
    @pytest.mark.asyncio
    async def test_delete(self, session_store):
        """Test deleting a session."""
        session = Session(session_id="mysql_sess_4", user_id="user_1")
        await session_store.save(session)
        
        assert await session_store.exists("mysql_sess_4") is True
        assert await session_store.delete("mysql_sess_4") is True
        assert await session_store.exists("mysql_sess_4") is False
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, session_store):
        """Test listing sessions for a user."""
        for i in range(5):
            session = Session(session_id=f"mysql_list_sess_{i}", user_id="list_user")
            await session_store.save(session)
        
        sessions = await session_store.list_sessions("list_user")
        assert len(sessions) == 5
        
        sessions = await session_store.list_sessions("list_user", limit=3)
        assert len(sessions) == 3


class TestMySQLMemoryStore:
    """Tests for MySQLMemoryStore."""
    
    @pytest.mark.asyncio
    async def test_add_and_get(self, memory_store):
        """Test adding and retrieving a memory."""
        memory = MemoryItem(
            user_id="mysql_user_1",
            content="User prefers dark theme",
            type=MemoryType.SEMANTIC,
        )
        
        memory_id = await memory_store.add(memory)
        retrieved = await memory_store.get(memory_id)
        
        assert retrieved is not None
        assert retrieved.content == "User prefers dark theme"
    
    @pytest.mark.asyncio
    async def test_search_by_query_text(self, memory_store):
        """Test searching by query text using full-text search."""
        await memory_store.add(MemoryItem(
            user_id="mysql_user_2",
            content="User is vegetarian and enjoys cooking healthy meals",
            type=MemoryType.SEMANTIC,
        ))
        await memory_store.add(MemoryItem(
            user_id="mysql_user_2",
            content="User prefers dark mode interface",
            type=MemoryType.SEMANTIC,
        ))
        
        results = await memory_store.search(MemoryQuery(
            user_id="mysql_user_2",
            query_text="vegetarian",
        ))
        
        assert len(results) >= 1
        assert "vegetarian" in results[0].content.lower()
    
    @pytest.mark.asyncio
    async def test_search_by_type(self, memory_store):
        """Test filtering by memory type."""
        await memory_store.add(MemoryItem(
            user_id="mysql_user_3",
            content="Semantic memory",
            type=MemoryType.SEMANTIC,
        ))
        await memory_store.add(MemoryItem(
            user_id="mysql_user_3",
            content="Episodic memory",
            type=MemoryType.EPISODIC,
        ))
        
        results = await memory_store.search(MemoryQuery(
            user_id="mysql_user_3",
            types=[MemoryType.SEMANTIC],
        ))
        
        assert len(results) == 1
        assert results[0].type == MemoryType.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_search_by_tags(self, memory_store):
        """Test filtering by tags using JSON_CONTAINS."""
        memory = MemoryItem(
            user_id="mysql_user_4",
            content="User likes pizza",
            type=MemoryType.SEMANTIC,
            tags=["food", "preference"],
        )
        await memory_store.add(memory)
        
        await memory_store.add(MemoryItem(
            user_id="mysql_user_4",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
            tags=["ui"],
        ))
        
        results = await memory_store.search(MemoryQuery(
            user_id="mysql_user_4",
            tags=["food"],
        ))
        
        assert len(results) == 1
        assert "pizza" in results[0].content
    
    @pytest.mark.asyncio
    async def test_update(self, memory_store):
        """Test updating a memory."""
        memory = MemoryItem(
            user_id="mysql_user_5",
            content="Original content",
            type=MemoryType.SEMANTIC,
        )
        memory_id = await memory_store.add(memory)
        
        memory.update_content("Updated content")
        result = await memory_store.update(memory)
        
        assert result is True
        
        retrieved = await memory_store.get(memory_id)
        assert retrieved.content == "Updated content"
    
    @pytest.mark.asyncio
    async def test_delete(self, memory_store):
        """Test deleting a memory."""
        memory = MemoryItem(
            user_id="mysql_user_6",
            content="To delete",
            type=MemoryType.SEMANTIC,
        )
        memory_id = await memory_store.add(memory)
        
        assert await memory_store.delete(memory_id) is True
        assert await memory_store.get(memory_id) is None
    
    @pytest.mark.asyncio
    async def test_count(self, memory_store):
        """Test counting memories."""
        for i in range(5):
            await memory_store.add(MemoryItem(
                user_id="mysql_user_7",
                content=f"Memory {i}",
                type=MemoryType.SEMANTIC,
            ))
        
        count = await memory_store.count("mysql_user_7")
        assert count == 5
    
    @pytest.mark.asyncio
    async def test_cosine_similarity(self, memory_store):
        """Test the static cosine similarity method."""
        # Unit vectors
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert memory_store._cosine_similarity(a, b) == pytest.approx(1.0)
        
        # Orthogonal vectors
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert memory_store._cosine_similarity(a, b) == pytest.approx(0.0)
        
        # 45-degree angle
        a = [1.0, 0.0]
        b = [0.707, 0.707]
        assert memory_store._cosine_similarity(a, b) == pytest.approx(0.707, rel=0.01)


class TestMySQLExpertiseStore:
    """Tests for MySQLExpertiseStore."""
    
    @pytest.mark.asyncio
    async def test_save_and_load(self, expertise_store):
        """Test saving and loading expertise."""
        expertise = Expertise(
            expertise_id="mysql_exp_1",
            name="Test Expertise",
            domain="testing",
            description="An expertise for testing MySQL storage",
        )
        
        # Add some items
        expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Always write tests first",
            source="test",
        )
        expertise.add_item(
            section=ExpertiseSection.COMMON_MISTAKES,
            content="Not cleaning up test data",
            source="test",
        )
        
        await expertise_store.save(expertise)
        
        loaded = await expertise_store.load("mysql_exp_1")
        
        assert loaded is not None
        assert loaded.name == "Test Expertise"
        assert loaded.domain == "testing"
        assert len(loaded.items) == 2
        
        # Clean up
        await expertise_store.delete("mysql_exp_1")
    
    @pytest.mark.asyncio
    async def test_add_and_update_item(self, expertise_store):
        """Test adding and updating individual items."""
        expertise = Expertise(
            expertise_id="mysql_exp_2",
            name="Item Test",
            domain="testing",
        )
        await expertise_store.save(expertise)
        
        # Add item
        item = ExpertiseItem(
            item_id="item_1",
            section=ExpertiseSection.STRATEGIES,
            content="Original content",
            source="test",
        )
        await expertise_store.add_item("mysql_exp_2", item)
        
        # Retrieve and verify
        retrieved = await expertise_store.get_item("mysql_exp_2", "item_1")
        assert retrieved is not None
        assert retrieved.content == "Original content"
        
        # Update
        item.content = "Updated content"
        item.helpful_count = 5
        await expertise_store.update_item("mysql_exp_2", item)
        
        updated = await expertise_store.get_item("mysql_exp_2", "item_1")
        assert updated.content == "Updated content"
        assert updated.helpful_count == 5
        
        # Clean up
        await expertise_store.delete("mysql_exp_2")
    
    @pytest.mark.asyncio
    async def test_search_items(self, expertise_store):
        """Test full-text search on items."""
        expertise = Expertise(
            expertise_id="mysql_exp_3",
            name="Search Test",
            domain="testing",
        )
        expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Use async await for database operations",
            source="test",
        )
        expertise.add_item(
            section=ExpertiseSection.HEURISTICS,
            content="Prefer connection pooling for performance",
            source="test",
        )
        await expertise_store.save(expertise)
        
        # Search
        results = await expertise_store.search_items("mysql_exp_3", "database")
        
        assert len(results) >= 1
        assert "database" in results[0].content.lower()
        
        # Clean up
        await expertise_store.delete("mysql_exp_3")
    
    @pytest.mark.asyncio
    async def test_update_item_counts(self, expertise_store):
        """Test updating helpful/harmful counts."""
        expertise = Expertise(
            expertise_id="mysql_exp_4",
            name="Counts Test",
            domain="testing",
        )
        expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Test strategy",
            source="test",
        )
        await expertise_store.save(expertise)
        
        # Get the item_id
        loaded = await expertise_store.load("mysql_exp_4")
        item_id = loaded.items[0].item_id
        
        # Update counts
        await expertise_store.update_item_counts(
            "mysql_exp_4",
            item_id,
            helpful_delta=3,
            harmful_delta=1,
        )
        
        item = await expertise_store.get_item("mysql_exp_4", item_id)
        assert item.helpful_count == 3
        assert item.harmful_count == 1
        
        # Clean up
        await expertise_store.delete("mysql_exp_4")
    
    @pytest.mark.asyncio
    async def test_list_expertise(self, expertise_store):
        """Test listing expertise by domain."""
        for i in range(3):
            expertise = Expertise(
                expertise_id=f"mysql_list_exp_{i}",
                name=f"List Test {i}",
                domain="list_domain",
            )
            await expertise_store.save(expertise)
        
        # Add one with different domain
        expertise = Expertise(
            expertise_id="mysql_list_exp_other",
            name="Other Domain",
            domain="other_domain",
        )
        await expertise_store.save(expertise)
        
        # List by domain
        results = await expertise_store.list_expertise(domain="list_domain")
        assert len(results) == 3
        
        # Clean up
        for i in range(3):
            await expertise_store.delete(f"mysql_list_exp_{i}")
        await expertise_store.delete("mysql_list_exp_other")
