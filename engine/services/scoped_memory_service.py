"""
Scoped Memory Service.

This service manages hierarchical scoped memories with
GLOBAL, PROJECT, and SESSION namespaces.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    MergedMemoryResult,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.protocols.scoped_memory import IScopedMemoryStore

logger = logging.getLogger(__name__)


class ScopedMemoryService:
    """
    Service for managing hierarchical scoped memories.

    Provides convenient methods for saving memories at different scopes
    and retrieving merged memories with proper hierarchy resolution.
    """

    def __init__(self, store: IScopedMemoryStore):
        """
        Initialize the service.

        Args:
            store: The storage backend for scoped memories
        """
        self._store = store

    async def initialize(self) -> None:
        """Initialize the underlying store."""
        await self._store.initialize()

    # =========================================================================
    # Save Methods (convenience wrappers for each scope)
    # =========================================================================

    async def save_global(
        self,
        user_id: str,
        key: str,
        content: str,
        category: MemoryCategory,
        priority: int = 0,
        metadata: Optional[Dict] = None,
    ) -> ScopedMemory:
        """
        Save a global (user-level) memory.

        Args:
            user_id: The user ID (scope_id for global scope)
            key: Unique key within the scope
            content: The memory content
            category: Memory category
            priority: Priority for ordering (higher = first)
            metadata: Optional metadata

        Returns:
            The saved ScopedMemory
        """
        memory = ScopedMemory(
            id=str(uuid.uuid4()),
            scope=MemoryScope.GLOBAL,
            scope_id=user_id,
            category=category,
            key=key,
            content=content,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await self._store.save(memory)
        logger.debug(f"Saved global memory: {key} for user {user_id}")
        return memory

    async def save_project(
        self,
        project_id: str,
        key: str,
        content: str,
        category: MemoryCategory,
        priority: int = 0,
        metadata: Optional[Dict] = None,
    ) -> ScopedMemory:
        """
        Save a project-level memory.

        Args:
            project_id: The project ID (scope_id for project scope)
            key: Unique key within the scope
            content: The memory content
            category: Memory category
            priority: Priority for ordering (higher = first)
            metadata: Optional metadata

        Returns:
            The saved ScopedMemory
        """
        memory = ScopedMemory(
            id=str(uuid.uuid4()),
            scope=MemoryScope.PROJECT,
            scope_id=project_id,
            category=category,
            key=key,
            content=content,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await self._store.save(memory)
        logger.debug(f"Saved project memory: {key} for project {project_id}")
        return memory

    async def save_session(
        self,
        session_id: str,
        key: str,
        content: str,
        category: MemoryCategory,
        priority: int = 0,
        metadata: Optional[Dict] = None,
    ) -> ScopedMemory:
        """
        Save a session-level memory.

        Args:
            session_id: The session ID (scope_id for session scope)
            key: Unique key within the scope
            content: The memory content
            category: Memory category
            priority: Priority for ordering (higher = first)
            metadata: Optional metadata

        Returns:
            The saved ScopedMemory
        """
        memory = ScopedMemory(
            id=str(uuid.uuid4()),
            scope=MemoryScope.SESSION,
            scope_id=session_id,
            category=category,
            key=key,
            content=content,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await self._store.save(memory)
        logger.debug(f"Saved session memory: {key} for session {session_id}")
        return memory

    async def save(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str,
        content: str,
        category: MemoryCategory,
        priority: int = 0,
        metadata: Optional[Dict] = None,
    ) -> ScopedMemory:
        """
        Save a memory at any scope.

        Args:
            scope: The memory scope (GLOBAL, PROJECT, or SESSION)
            scope_id: The ID for the scope (user_id, project_id, or session_id)
            key: Unique key within the scope
            content: The memory content
            category: Memory category
            priority: Priority for ordering (higher = first)
            metadata: Optional metadata

        Returns:
            The saved ScopedMemory
        """
        memory = ScopedMemory(
            id=str(uuid.uuid4()),
            scope=scope,
            scope_id=scope_id,
            category=category,
            key=key,
            content=content,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await self._store.save(memory)
        logger.debug(f"Saved {scope.value} memory: {key}")
        return memory

    # =========================================================================
    # Retrieval Methods
    # =========================================================================

    async def get(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> Optional[ScopedMemory]:
        """Get a specific memory by scope, scope_id, and key."""
        return await self._store.get(scope, scope_id, key)

    async def get_by_id(self, memory_id: str) -> Optional[ScopedMemory]:
        """Get a memory by its unique ID."""
        return await self._store.get_by_id(memory_id)

    async def list_by_scope(
        self,
        scope: MemoryScope,
        scope_id: str,
        category: Optional[MemoryCategory] = None
    ) -> List[ScopedMemory]:
        """List all memories for a given scope and scope_id."""
        return await self._store.list_by_scope(scope, scope_id, category)

    async def get_merged_memories(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        categories: Optional[List[MemoryCategory]] = None
    ) -> MergedMemoryResult:
        """
        Get memories merged by hierarchy (session > project > global).

        Memories with the same key are deduplicated, with higher-scope
        memories taking precedence (SESSION overrides PROJECT overrides GLOBAL).

        Args:
            user_id: The user ID for global scope
            project_id: Optional project ID for project scope
            session_id: Optional session ID for session scope
            categories: Optional list of categories to filter

        Returns:
            MergedMemoryResult with deduplicated memories
        """
        query = ScopedMemoryQuery(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            categories=categories,
        )

        # Get all matching memories
        all_memories = await self._store.query(query)

        # Deduplicate by key (higher scope wins)
        # Since query results are sorted by scope priority, we just take first
        memories_by_key: Dict[str, ScopedMemory] = {}
        scope_counts: Dict[MemoryScope, int] = {}
        override_count = 0

        for memory in all_memories:
            if memory.key not in memories_by_key:
                memories_by_key[memory.key] = memory
                scope_counts[memory.scope] = scope_counts.get(memory.scope, 0) + 1
            else:
                # This memory was overridden by a higher-scope one
                override_count += 1

        # Get final list sorted by priority then key
        final_memories = list(memories_by_key.values())
        final_memories.sort(key=lambda m: (-m.priority, m.key))

        return MergedMemoryResult(
            memories=final_memories,
            scope_counts=scope_counts,
            override_count=override_count,
        )

    async def format_for_prompt(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        categories: Optional[List[MemoryCategory]] = None
    ) -> str:
        """
        Format merged memories as prompt injection text.

        Args:
            user_id: The user ID for global scope
            project_id: Optional project ID for project scope
            session_id: Optional session ID for session scope
            categories: Optional list of categories to filter

        Returns:
            Formatted string for prompt injection
        """
        result = await self.get_merged_memories(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            categories=categories,
        )
        return result.format_for_prompt()

    # =========================================================================
    # Deletion Methods
    # =========================================================================

    async def delete(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> bool:
        """Delete a specific memory."""
        return await self._store.delete(scope, scope_id, key)

    async def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory by its unique ID."""
        return await self._store.delete_by_id(memory_id)

    async def clear_session(self, session_id: str) -> int:
        """
        Clear all session memories (useful at session end).

        Args:
            session_id: The session ID to clear

        Returns:
            Number of memories deleted
        """
        return await self._store.clear(
            scope=MemoryScope.SESSION,
            scope_id=session_id
        )

    async def clear_project(self, project_id: str) -> int:
        """
        Clear all project memories.

        Args:
            project_id: The project ID to clear

        Returns:
            Number of memories deleted
        """
        return await self._store.clear(
            scope=MemoryScope.PROJECT,
            scope_id=project_id
        )

    # =========================================================================
    # Stats Methods
    # =========================================================================

    async def count(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count memories, optionally filtered by scope."""
        return await self._store.count(scope, scope_id)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def save_observations(self, observations: List[ScopedMemory]) -> int:
        """Save a batch of observation-derived scoped memories.

        Args:
            observations: Pre-built ``ScopedMemory`` instances.

        Returns:
            Number of memories successfully saved.
        """
        saved = 0
        for obs in observations:
            try:
                await self._store.save(obs)
                saved += 1
            except Exception as exc:
                logger.warning("Failed to save observation %s: %s", obs.id, exc)
        return saved
