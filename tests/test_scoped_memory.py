"""
Tests for Scoped Memory Models and Protocols.

This module tests the core scoped memory data structures
and storage protocol implementations.
"""
from datetime import datetime

import pytest

from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    MergedMemoryResult,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.scoped_memory import (
    ScopedMemoryAutoLearnMiddleware,
    ScopedMemoryMiddleware,
)
from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore


class TestMemoryScope:
    """Tests for MemoryScope enum."""

    def test_scope_values(self):
        """Test that all scope values are defined."""
        assert MemoryScope.GLOBAL.value == "global"
        assert MemoryScope.PROJECT.value == "project"
        assert MemoryScope.SESSION.value == "session"

    def test_scope_priority(self):
        """Test scope priority ordering."""
        assert MemoryScope.priority(MemoryScope.GLOBAL) == 0
        assert MemoryScope.priority(MemoryScope.PROJECT) == 1
        assert MemoryScope.priority(MemoryScope.SESSION) == 2

    def test_session_overrides_project(self):
        """Test that session has higher priority than project."""
        assert MemoryScope.priority(MemoryScope.SESSION) > \
               MemoryScope.priority(MemoryScope.PROJECT)

    def test_project_overrides_global(self):
        """Test that project has higher priority than global."""
        assert MemoryScope.priority(MemoryScope.PROJECT) > \
               MemoryScope.priority(MemoryScope.GLOBAL)


class TestMemoryCategory:
    """Tests for MemoryCategory enum."""

    def test_category_values(self):
        """Test that all category values are defined."""
        assert MemoryCategory.PREFERENCE.value == "preference"
        assert MemoryCategory.CONVENTION.value == "convention"
        assert MemoryCategory.ARCHITECTURE.value == "architecture"
        assert MemoryCategory.INSTRUCTION.value == "instruction"
        assert MemoryCategory.CONTEXT.value == "context"
        assert MemoryCategory.GOTCHA.value == "gotcha"

    def test_display_names(self):
        """Test display name generation."""
        assert MemoryCategory.get_display_name(MemoryCategory.PREFERENCE) == "Preferences"
        assert MemoryCategory.get_display_name(MemoryCategory.GOTCHA) == "Gotchas & Warnings"


class TestScopedMemory:
    """Tests for ScopedMemory dataclass."""

    def test_create_memory(self):
        """Test basic memory creation."""
        memory = ScopedMemory(
            id="mem-1",
            scope=MemoryScope.PROJECT,
            scope_id="project-123",
            category=MemoryCategory.CONVENTION,
            key="code-style",
            content="Use 4 spaces for indentation",
        )

        assert memory.id == "mem-1"
        assert memory.scope == MemoryScope.PROJECT
        assert memory.scope_id == "project-123"
        assert memory.category == MemoryCategory.CONVENTION
        assert memory.key == "code-style"
        assert memory.content == "Use 4 spaces for indentation"
        assert memory.priority == 0
        assert memory.metadata == {}

    def test_memory_with_priority(self):
        """Test memory with priority."""
        memory = ScopedMemory(
            id="mem-2",
            scope=MemoryScope.GLOBAL,
            scope_id="user-1",
            category=MemoryCategory.PREFERENCE,
            key="language",
            content="Prefer Python",
            priority=10,
        )

        assert memory.priority == 10

    def test_memory_with_metadata(self):
        """Test memory with metadata."""
        memory = ScopedMemory(
            id="mem-3",
            scope=MemoryScope.SESSION,
            scope_id="session-abc",
            category=MemoryCategory.CONTEXT,
            key="current-task",
            content="Working on API refactoring",
            metadata={"source": "user", "confidence": 0.95},
        )

        assert memory.metadata["source"] == "user"
        assert memory.metadata["confidence"] == 0.95

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now()
        memory = ScopedMemory(
            id="mem-4",
            scope=MemoryScope.PROJECT,
            scope_id="proj-1",
            category=MemoryCategory.ARCHITECTURE,
            key="db-choice",
            content="Use PostgreSQL",
            priority=5,
            metadata={"reason": "scalability"},
            created_at=now,
            updated_at=now,
        )

        data = memory.to_dict()

        assert data["id"] == "mem-4"
        assert data["scope"] == "project"
        assert data["scope_id"] == "proj-1"
        assert data["category"] == "architecture"
        assert data["key"] == "db-choice"
        assert data["content"] == "Use PostgreSQL"
        assert data["priority"] == 5
        assert data["metadata"]["reason"] == "scalability"
        assert data["created_at"] == now.isoformat()

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "mem-5",
            "scope": "global",
            "scope_id": "user-xyz",
            "category": "instruction",
            "key": "response-style",
            "content": "Be concise",
            "priority": 3,
            "metadata": {},
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T10:30:00",
        }

        memory = ScopedMemory.from_dict(data)

        assert memory.id == "mem-5"
        assert memory.scope == MemoryScope.GLOBAL
        assert memory.category == MemoryCategory.INSTRUCTION
        assert memory.priority == 3

    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {
            "id": "mem-6",
            "scope": "session",
            "scope_id": "sess-1",
            "category": "context",
            "key": "topic",
            "content": "Discussing auth",
        }

        memory = ScopedMemory.from_dict(data)

        assert memory.id == "mem-6"
        assert memory.scope == MemoryScope.SESSION
        assert memory.priority == 0
        assert memory.metadata == {}


class TestScopedMemoryQuery:
    """Tests for ScopedMemoryQuery dataclass."""

    def test_default_query(self):
        """Test query with default values."""
        query = ScopedMemoryQuery()

        assert query.user_id is None
        assert query.project_id is None
        assert query.session_id is None
        assert query.categories is None
        assert query.include_global is True
        assert query.include_project is True
        assert query.include_session is True

    def test_query_with_all_scopes(self):
        """Test query with all scope IDs."""
        query = ScopedMemoryQuery(
            user_id="user-1",
            project_id="proj-1",
            session_id="sess-1",
        )

        scope_ids = query.get_scope_ids()

        assert MemoryScope.GLOBAL in scope_ids
        assert MemoryScope.PROJECT in scope_ids
        assert MemoryScope.SESSION in scope_ids
        assert scope_ids[MemoryScope.GLOBAL] == "user-1"
        assert scope_ids[MemoryScope.PROJECT] == "proj-1"
        assert scope_ids[MemoryScope.SESSION] == "sess-1"

    def test_query_exclude_session(self):
        """Test query excluding session scope."""
        query = ScopedMemoryQuery(
            user_id="user-1",
            project_id="proj-1",
            session_id="sess-1",
            include_session=False,
        )

        scope_ids = query.get_scope_ids()

        assert MemoryScope.GLOBAL in scope_ids
        assert MemoryScope.PROJECT in scope_ids
        assert MemoryScope.SESSION not in scope_ids

    def test_query_with_categories(self):
        """Test query with category filter."""
        query = ScopedMemoryQuery(
            user_id="user-1",
            categories=[MemoryCategory.PREFERENCE, MemoryCategory.CONVENTION],
        )

        assert len(query.categories) == 2
        assert MemoryCategory.PREFERENCE in query.categories


class TestMergedMemoryResult:
    """Tests for MergedMemoryResult dataclass."""

    def test_empty_result(self):
        """Test empty merged result."""
        result = MergedMemoryResult(
            memories=[],
            scope_counts={},
            override_count=0,
        )

        assert result.total_count == 0
        assert result.by_category() == {}
        assert result.format_for_prompt() == ""

    def test_result_with_memories(self):
        """Test result with memories from multiple scopes."""
        memories = [
            ScopedMemory(
                id="1", scope=MemoryScope.GLOBAL, scope_id="u1",
                category=MemoryCategory.PREFERENCE, key="k1",
                content="Global pref"
            ),
            ScopedMemory(
                id="2", scope=MemoryScope.PROJECT, scope_id="p1",
                category=MemoryCategory.PREFERENCE, key="k2",
                content="Project pref"
            ),
            ScopedMemory(
                id="3", scope=MemoryScope.PROJECT, scope_id="p1",
                category=MemoryCategory.CONVENTION, key="k3",
                content="Code style"
            ),
        ]

        result = MergedMemoryResult(
            memories=memories,
            scope_counts={MemoryScope.GLOBAL: 1, MemoryScope.PROJECT: 2},
            override_count=0,
        )

        assert result.total_count == 3
        by_cat = result.by_category()
        assert len(by_cat[MemoryCategory.PREFERENCE]) == 2
        assert len(by_cat[MemoryCategory.CONVENTION]) == 1

    def test_format_for_prompt(self):
        """Test prompt formatting."""
        memories = [
            ScopedMemory(
                id="1", scope=MemoryScope.GLOBAL, scope_id="u1",
                category=MemoryCategory.PREFERENCE, key="k1",
                content="Use Python 3.10+", priority=5
            ),
            ScopedMemory(
                id="2", scope=MemoryScope.PROJECT, scope_id="p1",
                category=MemoryCategory.GOTCHA, key="k2",
                content="Avoid global state", priority=10
            ),
        ]

        result = MergedMemoryResult(
            memories=memories,
            scope_counts={MemoryScope.GLOBAL: 1, MemoryScope.PROJECT: 1},
            override_count=0,
        )

        formatted = result.format_for_prompt()

        assert "## Context & Memories" in formatted
        assert "### Preferences" in formatted
        assert "[GLOBAL] Use Python 3.10+" in formatted
        assert "### Gotchas & Warnings" in formatted
        assert "[PROJECT] Avoid global state" in formatted


# =============================================================================
# In-Memory Store Tests
# =============================================================================


class TestInMemoryScopedMemoryStore:
    """Tests for InMemoryScopedMemoryStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemoryScopedMemoryStore()

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing."""
        return ScopedMemory(
            id="mem-1",
            scope=MemoryScope.PROJECT,
            scope_id="proj-1",
            category=MemoryCategory.CONVENTION,
            key="code-style",
            content="Use 4 spaces for indentation",
        )

    async def test_initialize(self, store):
        """Test store initialization."""
        await store.initialize()
        # Should not raise

    async def test_save_and_get(self, store, sample_memory):
        """Test saving and retrieving a memory."""
        await store.save(sample_memory)

        retrieved = await store.get(
            MemoryScope.PROJECT, "proj-1", "code-style"
        )

        assert retrieved is not None
        assert retrieved.id == "mem-1"
        assert retrieved.content == "Use 4 spaces for indentation"

    async def test_get_nonexistent(self, store):
        """Test retrieving a non-existent memory."""
        result = await store.get(MemoryScope.GLOBAL, "user-1", "nonexistent")
        assert result is None

    async def test_get_by_id(self, store, sample_memory):
        """Test retrieving by ID."""
        await store.save(sample_memory)

        retrieved = await store.get_by_id("mem-1")

        assert retrieved is not None
        assert retrieved.key == "code-style"

    async def test_save_updates_existing(self, store, sample_memory):
        """Test that save updates if key exists."""
        await store.save(sample_memory)

        # Update the same key
        updated = ScopedMemory(
            id="mem-2",  # New ID
            scope=MemoryScope.PROJECT,
            scope_id="proj-1",
            category=MemoryCategory.CONVENTION,
            key="code-style",  # Same key
            content="Use tabs instead",
        )
        await store.save(updated)

        # Should have only one memory
        count = await store.count()
        assert count == 1

        # Should have the updated content
        retrieved = await store.get(MemoryScope.PROJECT, "proj-1", "code-style")
        assert retrieved.content == "Use tabs instead"
        assert retrieved.id == "mem-2"

    async def test_list_by_scope(self, store):
        """Test listing memories by scope."""
        # Add multiple memories
        await store.save(ScopedMemory(
            id="1", scope=MemoryScope.PROJECT, scope_id="proj-1",
            category=MemoryCategory.CONVENTION, key="k1", content="c1"
        ))
        await store.save(ScopedMemory(
            id="2", scope=MemoryScope.PROJECT, scope_id="proj-1",
            category=MemoryCategory.ARCHITECTURE, key="k2", content="c2"
        ))
        await store.save(ScopedMemory(
            id="3", scope=MemoryScope.PROJECT, scope_id="proj-2",
            category=MemoryCategory.CONVENTION, key="k3", content="c3"
        ))

        # List for proj-1
        memories = await store.list_by_scope(MemoryScope.PROJECT, "proj-1")
        assert len(memories) == 2

        # List with category filter
        memories = await store.list_by_scope(
            MemoryScope.PROJECT, "proj-1", MemoryCategory.CONVENTION
        )
        assert len(memories) == 1
        assert memories[0].key == "k1"

    async def test_query_across_scopes(self, store):
        """Test querying memories across scopes."""
        await store.save(ScopedMemory(
            id="1", scope=MemoryScope.GLOBAL, scope_id="user-1",
            category=MemoryCategory.PREFERENCE, key="lang", content="Python"
        ))
        await store.save(ScopedMemory(
            id="2", scope=MemoryScope.PROJECT, scope_id="proj-1",
            category=MemoryCategory.PREFERENCE, key="framework", content="FastAPI"
        ))
        await store.save(ScopedMemory(
            id="3", scope=MemoryScope.SESSION, scope_id="sess-1",
            category=MemoryCategory.CONTEXT, key="task", content="Refactoring"
        ))

        query = ScopedMemoryQuery(
            user_id="user-1",
            project_id="proj-1",
            session_id="sess-1",
        )

        results = await store.query(query)
        assert len(results) == 3

        # Results should be sorted by scope priority (session first)
        assert results[0].scope == MemoryScope.SESSION
        assert results[1].scope == MemoryScope.PROJECT
        assert results[2].scope == MemoryScope.GLOBAL

    async def test_query_with_category_filter(self, store):
        """Test querying with category filter."""
        await store.save(ScopedMemory(
            id="1", scope=MemoryScope.GLOBAL, scope_id="user-1",
            category=MemoryCategory.PREFERENCE, key="k1", content="c1"
        ))
        await store.save(ScopedMemory(
            id="2", scope=MemoryScope.GLOBAL, scope_id="user-1",
            category=MemoryCategory.GOTCHA, key="k2", content="c2"
        ))

        query = ScopedMemoryQuery(
            user_id="user-1",
            categories=[MemoryCategory.PREFERENCE],
        )

        results = await store.query(query)
        assert len(results) == 1
        assert results[0].category == MemoryCategory.PREFERENCE

    async def test_delete(self, store, sample_memory):
        """Test deleting a memory."""
        await store.save(sample_memory)

        deleted = await store.delete(
            MemoryScope.PROJECT, "proj-1", "code-style"
        )
        assert deleted is True

        # Should not exist
        retrieved = await store.get(
            MemoryScope.PROJECT, "proj-1", "code-style"
        )
        assert retrieved is None

    async def test_delete_nonexistent(self, store):
        """Test deleting a non-existent memory."""
        deleted = await store.delete(
            MemoryScope.GLOBAL, "user-1", "nonexistent"
        )
        assert deleted is False

    async def test_delete_by_id(self, store, sample_memory):
        """Test deleting by ID."""
        await store.save(sample_memory)

        deleted = await store.delete_by_id("mem-1")
        assert deleted is True

        # Both lookups should fail
        assert await store.get_by_id("mem-1") is None
        assert await store.get(
            MemoryScope.PROJECT, "proj-1", "code-style"
        ) is None

    async def test_count(self, store):
        """Test counting memories."""
        await store.save(ScopedMemory(
            id="1", scope=MemoryScope.GLOBAL, scope_id="user-1",
            category=MemoryCategory.PREFERENCE, key="k1", content="c1"
        ))
        await store.save(ScopedMemory(
            id="2", scope=MemoryScope.PROJECT, scope_id="proj-1",
            category=MemoryCategory.CONVENTION, key="k2", content="c2"
        ))

        # Total count
        assert await store.count() == 2

        # By scope
        assert await store.count(scope=MemoryScope.GLOBAL) == 1
        assert await store.count(scope=MemoryScope.PROJECT) == 1
        assert await store.count(scope=MemoryScope.SESSION) == 0

    async def test_clear_all(self, store):
        """Test clearing all memories."""
        await store.save(ScopedMemory(
            id="1", scope=MemoryScope.GLOBAL, scope_id="user-1",
            category=MemoryCategory.PREFERENCE, key="k1", content="c1"
        ))
        await store.save(ScopedMemory(
            id="2", scope=MemoryScope.PROJECT, scope_id="proj-1",
            category=MemoryCategory.CONVENTION, key="k2", content="c2"
        ))

        deleted_count = await store.clear()
        assert deleted_count == 2
        assert await store.count() == 0

    async def test_clear_by_scope(self, store):
        """Test clearing memories by scope."""
        await store.save(ScopedMemory(
            id="1", scope=MemoryScope.GLOBAL, scope_id="user-1",
            category=MemoryCategory.PREFERENCE, key="k1", content="c1"
        ))
        await store.save(ScopedMemory(
            id="2", scope=MemoryScope.PROJECT, scope_id="proj-1",
            category=MemoryCategory.CONVENTION, key="k2", content="c2"
        ))

        deleted_count = await store.clear(scope=MemoryScope.PROJECT)
        assert deleted_count == 1
        assert await store.count() == 1
        assert await store.count(scope=MemoryScope.GLOBAL) == 1


# =============================================================================
# Service Tests
# =============================================================================


class TestScopedMemoryService:
    """Tests for ScopedMemoryService."""

    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        store = InMemoryScopedMemoryStore()
        return ScopedMemoryService(store)

    async def test_initialize(self, service):
        """Test service initialization."""
        await service.initialize()
        # Should not raise

    async def test_save_global(self, service):
        """Test saving a global memory."""
        memory = await service.save_global(
            user_id="user-1",
            key="language",
            content="Prefer Python",
            category=MemoryCategory.PREFERENCE,
        )

        assert memory.scope == MemoryScope.GLOBAL
        assert memory.scope_id == "user-1"
        assert memory.content == "Prefer Python"

        # Retrieve it
        retrieved = await service.get(MemoryScope.GLOBAL, "user-1", "language")
        assert retrieved is not None
        assert retrieved.content == "Prefer Python"

    async def test_save_project(self, service):
        """Test saving a project memory."""
        memory = await service.save_project(
            project_id="proj-1",
            key="framework",
            content="Use FastAPI",
            category=MemoryCategory.ARCHITECTURE,
        )

        assert memory.scope == MemoryScope.PROJECT
        assert memory.scope_id == "proj-1"

    async def test_save_session(self, service):
        """Test saving a session memory."""
        memory = await service.save_session(
            session_id="sess-1",
            key="current-task",
            content="Refactoring API",
            category=MemoryCategory.CONTEXT,
        )

        assert memory.scope == MemoryScope.SESSION
        assert memory.scope_id == "sess-1"

    async def test_save_with_priority(self, service):
        """Test saving with priority."""
        memory = await service.save_global(
            user_id="user-1",
            key="important",
            content="High priority item",
            category=MemoryCategory.INSTRUCTION,
            priority=10,
        )

        assert memory.priority == 10

    async def test_get_merged_memories_single_scope(self, service):
        """Test getting merged memories from single scope."""
        await service.save_global(
            user_id="user-1", key="k1", content="c1",
            category=MemoryCategory.PREFERENCE
        )
        await service.save_global(
            user_id="user-1", key="k2", content="c2",
            category=MemoryCategory.CONVENTION
        )

        result = await service.get_merged_memories(user_id="user-1")

        assert result.total_count == 2
        assert result.override_count == 0
        assert MemoryScope.GLOBAL in result.scope_counts
        assert result.scope_counts[MemoryScope.GLOBAL] == 2

    async def test_get_merged_memories_multiple_scopes(self, service):
        """Test getting merged memories from multiple scopes."""
        await service.save_global(
            user_id="user-1", key="lang", content="Python",
            category=MemoryCategory.PREFERENCE
        )
        await service.save_project(
            project_id="proj-1", key="db", content="PostgreSQL",
            category=MemoryCategory.ARCHITECTURE
        )
        await service.save_session(
            session_id="sess-1", key="task", content="Refactoring",
            category=MemoryCategory.CONTEXT
        )

        result = await service.get_merged_memories(
            user_id="user-1",
            project_id="proj-1",
            session_id="sess-1"
        )

        assert result.total_count == 3
        assert MemoryScope.GLOBAL in result.scope_counts
        assert MemoryScope.PROJECT in result.scope_counts
        assert MemoryScope.SESSION in result.scope_counts

    async def test_get_merged_memories_with_override(self, service):
        """Test that higher scope overrides lower scope by key."""
        # Global setting
        await service.save_global(
            user_id="user-1", key="db", content="MySQL",
            category=MemoryCategory.ARCHITECTURE
        )
        # Project overrides
        await service.save_project(
            project_id="proj-1", key="db", content="PostgreSQL",
            category=MemoryCategory.ARCHITECTURE
        )

        result = await service.get_merged_memories(
            user_id="user-1",
            project_id="proj-1"
        )

        assert result.total_count == 1
        assert result.override_count == 1
        # Should have the project version
        assert result.memories[0].content == "PostgreSQL"
        assert result.memories[0].scope == MemoryScope.PROJECT

    async def test_get_merged_memories_session_overrides_all(self, service):
        """Test that session scope overrides both project and global."""
        await service.save_global(
            user_id="user-1", key="mode", content="production",
            category=MemoryCategory.CONTEXT
        )
        await service.save_project(
            project_id="proj-1", key="mode", content="staging",
            category=MemoryCategory.CONTEXT
        )
        await service.save_session(
            session_id="sess-1", key="mode", content="development",
            category=MemoryCategory.CONTEXT
        )

        result = await service.get_merged_memories(
            user_id="user-1",
            project_id="proj-1",
            session_id="sess-1"
        )

        assert result.total_count == 1
        assert result.override_count == 2
        assert result.memories[0].content == "development"

    async def test_format_for_prompt(self, service):
        """Test formatting memories for prompt injection."""
        await service.save_global(
            user_id="user-1", key="style", content="Be concise",
            category=MemoryCategory.INSTRUCTION
        )
        await service.save_project(
            project_id="proj-1", key="warning", content="Avoid global state",
            category=MemoryCategory.GOTCHA
        )

        formatted = await service.format_for_prompt(
            user_id="user-1",
            project_id="proj-1"
        )

        assert "## Context & Memories" in formatted
        assert "### Instructions" in formatted
        assert "[GLOBAL] Be concise" in formatted
        assert "### Gotchas & Warnings" in formatted
        assert "[PROJECT] Avoid global state" in formatted

    async def test_clear_session(self, service):
        """Test clearing session memories."""
        await service.save_session(
            session_id="sess-1", key="k1", content="c1",
            category=MemoryCategory.CONTEXT
        )
        await service.save_session(
            session_id="sess-1", key="k2", content="c2",
            category=MemoryCategory.CONTEXT
        )
        await service.save_global(
            user_id="user-1", key="k3", content="c3",
            category=MemoryCategory.PREFERENCE
        )

        deleted = await service.clear_session("sess-1")

        assert deleted == 2
        # Global should remain
        assert await service.count() == 1

    async def test_delete_by_id(self, service):
        """Test deleting by ID."""
        memory = await service.save_global(
            user_id="user-1", key="k1", content="c1",
            category=MemoryCategory.PREFERENCE
        )

        deleted = await service.delete_by_id(memory.id)
        assert deleted is True

        # Should be gone
        assert await service.get_by_id(memory.id) is None


# =============================================================================
# Middleware Tests
# =============================================================================


class TestScopedMemoryMiddleware:
    """Tests for ScopedMemoryMiddleware."""

    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        store = InMemoryScopedMemoryStore()
        return ScopedMemoryService(store)

    @pytest.fixture
    def middleware(self, service):
        """Create middleware with the service."""
        return ScopedMemoryMiddleware(
            memory_service=service,
            user_id="user-1",
            project_id="proj-1",
        )

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        """Simple next function for testing."""
        return context

    async def test_middleware_name(self, middleware):
        """Test middleware has correct name."""
        assert middleware.name == "scoped_memory"

    async def test_middleware_enabled(self, middleware):
        """Test middleware enabled property."""
        assert middleware.enabled is True
        middleware.enabled = False
        assert middleware.enabled is False

    async def test_inject_memories(self, service, middleware):
        """Test that memories are injected as a context section."""
        # Add some memories
        await service.save_global(
            user_id="user-1", key="style", content="Be concise",
            category=MemoryCategory.INSTRUCTION
        )
        await service.save_project(
            project_id="proj-1", key="framework", content="Use FastAPI",
            category=MemoryCategory.ARCHITECTURE
        )

        # Create context
        context = MiddlewareContext(user_input="Hello")

        # Process
        result = await middleware._do_process(context, self._next_fn)

        # processed_input should remain clean
        assert result.processed_input == "Hello"
        # Memories should be in context_sections
        section_names = [s.name for s in result.context_sections]
        assert "scoped_memories" in section_names
        mem_section = next(s for s in result.context_sections if s.name == "scoped_memories")
        assert "Be concise" in mem_section.content
        assert "Use FastAPI" in mem_section.content
        assert result.has_flag("scoped_memories_injected")

    async def test_no_memories(self, service, middleware):
        """Test when no memories exist."""
        context = MiddlewareContext(user_input="Hello")

        result = await middleware._do_process(context, self._next_fn)

        # Should still have the original input
        assert result.processed_input == "Hello"
        assert not result.has_flag("scoped_memories_injected")

    async def test_session_memories_included(self, service, middleware):
        """Test that session memories are included as a context section."""
        await service.save_session(
            session_id="sess-1", key="task", content="Working on auth",
            category=MemoryCategory.CONTEXT
        )

        context = MiddlewareContext(
            user_input="Hello",
            session_id="sess-1",
        )

        result = await middleware._do_process(context, self._next_fn)

        # processed_input should remain clean
        assert result.processed_input == "Hello"
        # Memories should be in context_sections
        section_names = [s.name for s in result.context_sections]
        assert "scoped_memories" in section_names
        mem_section = next(s for s in result.context_sections if s.name == "scoped_memories")
        assert "Working on auth" in mem_section.content
        assert "[SESSION]" in mem_section.content

    async def test_disabled_middleware(self, service, middleware):
        """Test that disabled middleware is skipped."""
        await service.save_global(
            user_id="user-1", key="k1", content="c1",
            category=MemoryCategory.PREFERENCE
        )

        middleware.enabled = False
        context = MiddlewareContext(user_input="Hello")

        result = await middleware.process(context, self._next_fn)

        # Memories should NOT be injected when disabled
        assert result.processed_input == "Hello"
        assert len(result.context_sections) == 0

    async def test_section_priority(self, service):
        """Test that scoped memories section has correct priority."""
        middleware = ScopedMemoryMiddleware(
            memory_service=service,
            user_id="user-1",
        )

        await service.save_global(
            user_id="user-1", key="k1", content="Remember this",
            category=MemoryCategory.PREFERENCE
        )

        context = MiddlewareContext(user_input="Hello")
        result = await middleware._do_process(context, self._next_fn)

        assert len(result.context_sections) == 1
        section = result.context_sections[0]
        assert section.name == "scoped_memories"
        assert section.priority == 55
        assert section.is_required is False
        assert "Remember this" in section.content


class TestScopedMemoryAutoLearnMiddleware:
    """Tests for ScopedMemoryAutoLearnMiddleware."""

    @pytest.fixture
    def service(self):
        """Create a fresh service for each test."""
        store = InMemoryScopedMemoryStore()
        return ScopedMemoryService(store)

    @pytest.fixture
    def middleware(self, service):
        """Create middleware with the service."""
        return ScopedMemoryAutoLearnMiddleware(
            memory_service=service,
            user_id="user-1",
        )

    async def _next_fn(self, context: MiddlewareContext) -> MiddlewareContext:
        """Simple next function for testing."""
        return context

    async def test_middleware_name(self, middleware):
        """Test middleware has correct name."""
        assert middleware.name == "scoped_memory_auto_learn"

    async def test_extract_preference(self, service, middleware):
        """Test that preferences are extracted and saved."""
        context = MiddlewareContext(
            user_input="I prefer Python for backend development",
            session_id="sess-1",
        )

        await middleware._do_process(context, self._next_fn)

        # Check if memory was saved
        memories = await service.list_by_scope(
            MemoryScope.SESSION, "sess-1"
        )

        assert len(memories) == 1
        assert "I prefer Python" in memories[0].content

    async def test_no_extraction_without_pattern(self, service, middleware):
        """Test that non-preference text is not extracted."""
        context = MiddlewareContext(
            user_input="What is the weather today?",
            session_id="sess-1",
        )

        await middleware._do_process(context, self._next_fn)

        memories = await service.list_by_scope(
            MemoryScope.SESSION, "sess-1"
        )

        assert len(memories) == 0

    async def test_no_extraction_without_session(self, service, middleware):
        """Test that extraction is skipped without session_id."""
        context = MiddlewareContext(
            user_input="I prefer TypeScript",
            # No session_id
        )

        await middleware._do_process(context, self._next_fn)

        # No session memories should exist
        count = await service.count(scope=MemoryScope.SESSION)
        assert count == 0


class TestNewMemoryCategories:
    """Verify the new observation-related MemoryCategory values."""

    def test_new_categories_exist(self):
        """All new observation categories are defined."""
        assert MemoryCategory.DECISION.value == "decision"
        assert MemoryCategory.BUGFIX.value == "bugfix"
        assert MemoryCategory.DISCOVERY.value == "discovery"
        assert MemoryCategory.FEATURE.value == "feature"
        assert MemoryCategory.REFACTOR.value == "refactor"

    def test_new_categories_display_names(self):
        """New categories have correct display names."""
        assert MemoryCategory.get_display_name(MemoryCategory.DECISION) == "Decisions"
        assert MemoryCategory.get_display_name(MemoryCategory.BUGFIX) == "Bug Fixes"
        assert MemoryCategory.get_display_name(MemoryCategory.DISCOVERY) == "Discoveries"
        assert MemoryCategory.get_display_name(MemoryCategory.FEATURE) == "Features"
        assert MemoryCategory.get_display_name(MemoryCategory.REFACTOR) == "Refactoring"

    @pytest.mark.asyncio
    async def test_scoped_memory_with_new_categories(self):
        """ScopedMemory can be created with new categories and stored."""
        store = InMemoryScopedMemoryStore()
        svc = ScopedMemoryService(store)

        await svc.save(
            scope=MemoryScope.PROJECT,
            scope_id="proj-x",
            category=MemoryCategory.DECISION,
            key="use_redis",
            content="Decided to use Redis for caching",
        )
        await svc.save(
            scope=MemoryScope.PROJECT,
            scope_id="proj-x",
            category=MemoryCategory.BUGFIX,
            key="fix_auth",
            content="Fixed race condition in auth flow",
        )

        memories = await svc.list_by_scope(MemoryScope.PROJECT, "proj-x")
        assert len(memories) == 2
        categories = {m.category for m in memories}
        assert MemoryCategory.DECISION in categories
        assert MemoryCategory.BUGFIX in categories
