"""
In-memory memory store implementation.

Suitable for testing and small-scale deployments.
"""

import asyncio
from typing import Dict, List, Optional

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import IMemoryStore


@registry.register_memory_store("memory")
class InMemoryMemoryStore(IMemoryStore):
    """
    In-memory memory store.
    
    Memories are stored in a list with simple keyword-based search.
    Suitable for testing and small-scale deployments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the store.
        
        Args:
            config: Optional configuration (unused for in-memory)
        """
        self._store: List[MemoryItem] = []
        self._lock = asyncio.Lock()
    
    async def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """
        Search for memories.
        
        Uses simple keyword matching for relevance scoring.
        In production, use vector similarity search.
        """
        # Filter by user
        candidates = [
            m for m in self._store
            if m.user_id == query.user_id
            and m.is_active
            and not m.is_expired()
        ]
        
        # Filter by types if specified
        if query.types:
            candidates = [m for m in candidates if m.type in query.types]
        
        # Filter by tags if specified
        if query.tags:
            candidates = [
                m for m in candidates
                if any(t in m.tags for t in query.tags)
            ]
        
        # Filter by confidence
        candidates = [
            m for m in candidates
            if m.confidence_score >= query.min_confidence
        ]
        
        # Score by query text similarity (simple keyword overlap)
        if query.query_text:
            query_words = set(query.query_text.lower().split())
            scored = []
            
            for mem in candidates:
                content_words = set(mem.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0 or not query_words:
                    scored.append((overlap, mem))
            
            # Sort by score (descending), then by recency
            scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
            candidates = [item[1] for item in scored]
        
        # Apply limit and offset
        results = candidates[query.offset:query.offset + query.limit]
        
        # Record access
        for mem in results:
            mem.record_access()
        
        return results
    
    async def add(self, item: MemoryItem) -> str:
        """Add a new memory."""
        async with self._lock:
            self._store.append(item)
            return item.memory_id
    
    async def update(self, item: MemoryItem) -> bool:
        """Update an existing memory."""
        async with self._lock:
            for i, mem in enumerate(self._store):
                if mem.memory_id == item.memory_id:
                    self._store[i] = item
                    return True
            return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        async with self._lock:
            for i, mem in enumerate(self._store):
                if mem.memory_id == memory_id:
                    self._store.pop(i)
                    return True
            return False
    
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory by ID."""
        for mem in self._store:
            if mem.memory_id == memory_id:
                mem.record_access()
                return mem
        return None
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """Get all memories for a user."""
        memories = [
            m for m in self._store
            if m.user_id == user_id
            and (include_inactive or m.is_active)
        ]
        
        # Sort by created_at descending
        memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return memories[:limit]
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        return len([
            m for m in self._store
            if m.user_id == user_id and m.is_active
        ])
    
    async def keyword_search(
        self,
        user_id: str,
        keywords: List[str],
        limit: int = 10,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> List[MemoryItem]:
        """Search memories by keyword overlap on the ``keywords`` field."""
        kw_set = {k.lower() for k in keywords}
        candidates = [
            m for m in self._store
            if m.user_id == user_id and m.is_active and not m.is_expired()
        ]

        # Apply structured metadata filters
        if filters:
            for field_name, values in filters.items():
                val_set = {v.lower() for v in values}
                candidates = [
                    m for m in candidates
                    if val_set & {v.lower() for v in getattr(m, field_name, [])}
                ]

        # Score by keyword overlap
        scored = []
        for m in candidates:
            mem_kw = {k.lower() for k in m.keywords}
            overlap = len(kw_set & mem_kw)
            if overlap > 0:
                scored.append((overlap, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    async def clear(self) -> None:
        """Clear all memories (for testing)."""
        async with self._lock:
            self._store.clear()

