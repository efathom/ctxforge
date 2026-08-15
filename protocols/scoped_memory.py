"""
Protocol for Scoped Memory Storage.

Defines the interface for scoped memory stores that support
hierarchical memory with GLOBAL, PROJECT, and SESSION scopes.
"""
from typing import List, Optional, Protocol, runtime_checkable

from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)


@runtime_checkable
class IScopedMemoryStore(Protocol):
    """Protocol for scoped memory storage."""

    async def initialize(self) -> None:
        """Initialize the store (create tables, etc.)."""
        ...

    async def save(self, memory: ScopedMemory) -> None:
        """Save a scoped memory. Updates if key already exists in scope."""
        ...

    async def get(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> Optional[ScopedMemory]:
        """Get a specific memory by scope, scope_id, and key."""
        ...

    async def get_by_id(self, memory_id: str) -> Optional[ScopedMemory]:
        """Get a memory by its unique ID."""
        ...

    async def list_by_scope(
        self,
        scope: MemoryScope,
        scope_id: str,
        category: Optional[MemoryCategory] = None
    ) -> List[ScopedMemory]:
        """List all memories for a given scope and scope_id."""
        ...

    async def query(self, query: ScopedMemoryQuery) -> List[ScopedMemory]:
        """Query memories across multiple scopes based on query parameters."""
        ...

    async def delete(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> bool:
        """Delete a specific memory. Returns True if deleted."""
        ...

    async def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory by its unique ID. Returns True if deleted."""
        ...

    async def update(self, memory: ScopedMemory) -> None:
        """Update an existing memory."""
        ...

    async def count(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count memories, optionally filtered by scope."""
        ...

    async def clear(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear memories, optionally filtered by scope. Returns count deleted."""
        ...
