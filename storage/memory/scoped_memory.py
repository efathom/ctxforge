"""
In-memory scoped memory store implementation.

Suitable for testing and single-instance deployments.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.engine.registry import registry


@registry.register_scoped_memory_store("memory")
class InMemoryScopedMemoryStore:
    """
    In-memory scoped memory store.

    Memories are stored in a dictionary keyed by (scope, scope_id, key).
    Suitable for testing and single-instance deployments.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the store.

        Args:
            config: Optional configuration (unused for in-memory)
        """
        # Key: (scope, scope_id, key) -> ScopedMemory
        self._store: Dict[tuple, ScopedMemory] = {}
        # Index by ID for fast lookup
        self._by_id: Dict[str, ScopedMemory] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the store (no-op for in-memory)."""
        pass

    async def save(self, memory: ScopedMemory) -> None:
        """Save a scoped memory. Updates if key already exists in scope."""
        async with self._lock:
            key = (memory.scope, memory.scope_id, memory.key)

            # Update existing if found
            if key in self._store:
                existing = self._store[key]
                # Remove old ID mapping
                if existing.id in self._by_id:
                    del self._by_id[existing.id]

            self._store[key] = memory
            self._by_id[memory.id] = memory

    async def get(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> Optional[ScopedMemory]:
        """Get a specific memory by scope, scope_id, and key."""
        lookup_key = (scope, scope_id, key)
        return self._store.get(lookup_key)

    async def get_by_id(self, memory_id: str) -> Optional[ScopedMemory]:
        """Get a memory by its unique ID."""
        return self._by_id.get(memory_id)

    async def list_by_scope(
        self,
        scope: MemoryScope,
        scope_id: str,
        category: Optional[MemoryCategory] = None
    ) -> List[ScopedMemory]:
        """List all memories for a given scope and scope_id."""
        memories = [
            m for m in self._store.values()
            if m.scope == scope and m.scope_id == scope_id
        ]

        if category is not None:
            memories = [m for m in memories if m.category == category]

        # Sort by priority (descending), then by key
        memories.sort(key=lambda m: (-m.priority, m.key))
        return memories

    async def query(self, query: ScopedMemoryQuery) -> List[ScopedMemory]:
        """Query memories across multiple scopes based on query parameters."""
        results: List[ScopedMemory] = []
        scope_ids = query.get_scope_ids()

        for scope, scope_id in scope_ids.items():
            scope_memories = await self.list_by_scope(scope, scope_id)
            results.extend(scope_memories)

        # Filter by categories if specified
        if query.categories:
            results = [m for m in results if m.category in query.categories]

        # Sort by scope priority (higher scope wins), then priority, then key
        results.sort(key=lambda m: (
            -MemoryScope.priority(m.scope),
            -m.priority,
            m.key
        ))

        return results

    async def delete(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> bool:
        """Delete a specific memory. Returns True if deleted."""
        async with self._lock:
            lookup_key = (scope, scope_id, key)
            if lookup_key in self._store:
                memory = self._store.pop(lookup_key)
                if memory.id in self._by_id:
                    del self._by_id[memory.id]
                return True
            return False

    async def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory by its unique ID. Returns True if deleted."""
        async with self._lock:
            if memory_id in self._by_id:
                memory = self._by_id.pop(memory_id)
                lookup_key = (memory.scope, memory.scope_id, memory.key)
                if lookup_key in self._store:
                    del self._store[lookup_key]
                return True
            return False

    async def update(self, memory: ScopedMemory) -> None:
        """Update an existing memory."""
        memory.updated_at = datetime.now()
        await self.save(memory)

    async def count(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count memories, optionally filtered by scope."""
        if scope is None and scope_id is None:
            return len(self._store)

        count = 0
        for m in self._store.values():
            if scope is not None and m.scope != scope:
                continue
            if scope_id is not None and m.scope_id != scope_id:
                continue
            count += 1
        return count

    async def clear(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear memories, optionally filtered by scope. Returns count deleted."""
        async with self._lock:
            if scope is None and scope_id is None:
                count = len(self._store)
                self._store.clear()
                self._by_id.clear()
                return count

            # Find memories to delete
            to_delete = []
            for key, m in self._store.items():
                if scope is not None and m.scope != scope:
                    continue
                if scope_id is not None and m.scope_id != scope_id:
                    continue
                to_delete.append((key, m))

            # Delete them
            for key, m in to_delete:
                del self._store[key]
                if m.id in self._by_id:
                    del self._by_id[m.id]

            return len(to_delete)
