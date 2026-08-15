"""
Tests for storage implementations.
"""


import pytest

from ctxforge.core.exceptions import ConcurrencyError
from ctxforge.core.memory import MemoryItem, MemoryQuery, MemoryType
from ctxforge.core.session import Session
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemorySessionStore()
    
    @pytest.mark.asyncio
    async def test_load_creates_new_session(self, store):
        """Test that load creates a new session if not found."""
        session = await store.load("sess_1", "user_1")
        
        assert session.session_id == "sess_1"
        assert session.user_id == "user_1"
        assert session.version == 0
    
    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        """Test saving and loading a session."""
        session = Session(session_id="sess_1", user_id="user_1")
        session.add_user_message("Hello")
        session.state.set("key", "value")
        
        await store.save(session)
        
        loaded = await store.load("sess_1", "user_1")
        
        assert loaded.session_id == "sess_1"
        assert len(loaded.events) == 1
        assert loaded.state.get("key") == "value"
        assert loaded.version == 1  # Incremented on save
    
    @pytest.mark.asyncio
    async def test_optimistic_locking(self, store):
        """Test optimistic locking prevents concurrent modifications."""
        session = Session(session_id="sess_1", user_id="user_1")
        await store.save(session)
        
        # Load two copies
        copy1 = await store.load("sess_1", "user_1")
        copy2 = await store.load("sess_1", "user_1")
        
        # Save first copy
        copy1.add_user_message("From copy 1")
        await store.save(copy1)
        
        # Try to save second copy - should fail
        copy2.add_user_message("From copy 2")
        with pytest.raises(ConcurrencyError):
            await store.save(copy2)
    
    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting a session."""
        session = Session(session_id="sess_1", user_id="user_1")
        await store.save(session)
        
        assert await store.exists("sess_1") is True
        assert await store.delete("sess_1") is True
        assert await store.exists("sess_1") is False
        assert await store.delete("sess_1") is False  # Already deleted
    
    @pytest.mark.asyncio
    async def test_exists(self, store):
        """Test checking if session exists."""
        assert await store.exists("sess_1") is False
        
        session = Session(session_id="sess_1", user_id="user_1")
        await store.save(session)
        
        assert await store.exists("sess_1") is True
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, store):
        """Test listing sessions for a user."""
        # Create sessions for different users
        for i in range(5):
            session = Session(session_id=f"sess_{i}", user_id="user_1")
            await store.save(session)
        
        session = Session(session_id="sess_other", user_id="user_2")
        await store.save(session)
        
        # List for user_1
        sessions = await store.list_sessions("user_1")
        assert len(sessions) == 5
        
        # List with limit
        sessions = await store.list_sessions("user_1", limit=3)
        assert len(sessions) == 3
        
        # List for user_2
        sessions = await store.list_sessions("user_2")
        assert len(sessions) == 1
    
    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all sessions."""
        for i in range(3):
            session = Session(session_id=f"sess_{i}", user_id="user_1")
            await store.save(session)
        
        await store.clear()
        
        assert await store.exists("sess_0") is False
        assert await store.exists("sess_1") is False
        assert await store.exists("sess_2") is False
    
    @pytest.mark.asyncio
    async def test_deep_copy_on_load(self, store):
        """Test that loaded sessions are deep copies."""
        session = Session(session_id="sess_1", user_id="user_1")
        session.state.set("items", [1, 2, 3])
        await store.save(session)
        
        loaded = await store.load("sess_1", "user_1")
        loaded.state.get("items").append(4)
        
        # Reload and verify original is unchanged
        reloaded = await store.load("sess_1", "user_1")
        assert reloaded.state.get("items") == [1, 2, 3]


class TestInMemoryMemoryStore:
    """Tests for InMemoryMemoryStore."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemoryMemoryStore()
    
    @pytest.mark.asyncio
    async def test_add_and_get(self, store):
        """Test adding and retrieving a memory."""
        memory = MemoryItem(
            user_id="user_1",
            content="User is vegetarian",
            type=MemoryType.SEMANTIC,
        )
        
        memory_id = await store.add(memory)
        
        retrieved = await store.get(memory_id)
        
        assert retrieved is not None
        assert retrieved.content == "User is vegetarian"
        assert retrieved.access_count == 1  # Access was recorded
    
    @pytest.mark.asyncio
    async def test_search_by_query_text(self, store):
        """Test searching by query text."""
        await store.add(MemoryItem(
            user_id="user_1",
            content="User is vegetarian and likes salads",
            type=MemoryType.SEMANTIC,
        ))
        await store.add(MemoryItem(
            user_id="user_1",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
        ))
        await store.add(MemoryItem(
            user_id="user_1",
            content="User traveled to Paris",
            type=MemoryType.EPISODIC,
        ))
        
        # Search for vegetarian
        results = await store.search(MemoryQuery(
            user_id="user_1",
            query_text="vegetarian food",
        ))
        
        assert len(results) >= 1
        assert "vegetarian" in results[0].content.lower()
    
    @pytest.mark.asyncio
    async def test_search_by_type(self, store):
        """Test filtering by memory type."""
        await store.add(MemoryItem(
            user_id="user_1",
            content="User is vegetarian",
            type=MemoryType.SEMANTIC,
        ))
        await store.add(MemoryItem(
            user_id="user_1",
            content="User went to Paris",
            type=MemoryType.EPISODIC,
        ))
        
        results = await store.search(MemoryQuery(
            user_id="user_1",
            types=[MemoryType.SEMANTIC],
        ))
        
        assert len(results) == 1
        assert results[0].type == MemoryType.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_search_by_tags(self, store):
        """Test filtering by tags."""
        memory = MemoryItem(
            user_id="user_1",
            content="User likes pizza",
            type=MemoryType.SEMANTIC,
            tags=["food", "preference"],
        )
        await store.add(memory)
        
        await store.add(MemoryItem(
            user_id="user_1",
            content="User prefers dark mode",
            type=MemoryType.SEMANTIC,
            tags=["ui"],
        ))
        
        results = await store.search(MemoryQuery(
            user_id="user_1",
            tags=["food"],
        ))
        
        assert len(results) == 1
        assert "pizza" in results[0].content
    
    @pytest.mark.asyncio
    async def test_search_by_confidence(self, store):
        """Test filtering by confidence."""
        await store.add(MemoryItem(
            user_id="user_1",
            content="Certain fact",
            type=MemoryType.SEMANTIC,
            confidence_score=1.0,
        ))
        await store.add(MemoryItem(
            user_id="user_1",
            content="Uncertain fact",
            type=MemoryType.SEMANTIC,
            confidence_score=0.3,
        ))
        
        results = await store.search(MemoryQuery(
            user_id="user_1",
            min_confidence=0.5,
        ))
        
        assert len(results) == 1
        assert results[0].content == "Certain fact"
    
    @pytest.mark.asyncio
    async def test_search_excludes_inactive(self, store):
        """Test that inactive memories are excluded by default."""
        memory = MemoryItem(
            user_id="user_1",
            content="Active memory",
            type=MemoryType.SEMANTIC,
        )
        await store.add(memory)
        
        inactive = MemoryItem(
            user_id="user_1",
            content="Inactive memory",
            type=MemoryType.SEMANTIC,
        )
        inactive.deactivate()
        await store.add(inactive)
        
        results = await store.search(MemoryQuery(user_id="user_1"))
        
        assert len(results) == 1
        assert results[0].content == "Active memory"
    
    @pytest.mark.asyncio
    async def test_search_respects_user_id(self, store):
        """Test that search only returns user's memories."""
        await store.add(MemoryItem(
            user_id="user_1",
            content="User 1 memory",
            type=MemoryType.SEMANTIC,
        ))
        await store.add(MemoryItem(
            user_id="user_2",
            content="User 2 memory",
            type=MemoryType.SEMANTIC,
        ))
        
        results = await store.search(MemoryQuery(user_id="user_1"))
        
        assert len(results) == 1
        assert results[0].content == "User 1 memory"
    
    @pytest.mark.asyncio
    async def test_update(self, store):
        """Test updating a memory."""
        memory = MemoryItem(
            user_id="user_1",
            content="Original",
            type=MemoryType.SEMANTIC,
        )
        memory_id = await store.add(memory)
        
        memory.update_content("Updated")
        result = await store.update(memory)
        
        assert result is True
        
        retrieved = await store.get(memory_id)
        assert retrieved.content == "Updated"
    
    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting a memory."""
        memory = MemoryItem(
            user_id="user_1",
            content="To delete",
            type=MemoryType.SEMANTIC,
        )
        memory_id = await store.add(memory)
        
        assert await store.delete(memory_id) is True
        assert await store.get(memory_id) is None
        assert await store.delete(memory_id) is False  # Already deleted
    
    @pytest.mark.asyncio
    async def test_get_by_user(self, store):
        """Test getting all memories for a user."""
        for i in range(5):
            await store.add(MemoryItem(
                user_id="user_1",
                content=f"Memory {i}",
                type=MemoryType.SEMANTIC,
            ))
        
        await store.add(MemoryItem(
            user_id="user_2",
            content="Other user",
            type=MemoryType.SEMANTIC,
        ))
        
        memories = await store.get_by_user("user_1")
        assert len(memories) == 5
        
        memories = await store.get_by_user("user_1", limit=3)
        assert len(memories) == 3
    
    @pytest.mark.asyncio
    async def test_count(self, store):
        """Test counting memories."""
        for i in range(5):
            await store.add(MemoryItem(
                user_id="user_1",
                content=f"Memory {i}",
                type=MemoryType.SEMANTIC,
            ))
        
        count = await store.count("user_1")
        assert count == 5
        
        count = await store.count("user_2")
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all memories."""
        await store.add(MemoryItem(
            user_id="user_1",
            content="Test",
            type=MemoryType.SEMANTIC,
        ))
        
        await store.clear()
        
        count = await store.count("user_1")
        assert count == 0

