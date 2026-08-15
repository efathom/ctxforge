"""
Deduplicating Memory Store Wrapper.

A decorator that adds deduplication to any IMemoryStore implementation.
"""

from typing import List, Optional

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.extraction.consolidation.deduplicator import DeduplicationConsolidator
from ctxforge.protocols.storage import IMemoryStore


class DeduplicatingMemoryStore(IMemoryStore):
    """
    A memory store wrapper that automatically deduplicates on add.
    
    Uses the decorator pattern to wrap any IMemoryStore and add
    deduplication logic without modifying the underlying store.
    
    Example:
        base_store = InMemoryMemoryStore()
        dedup_store = DeduplicatingMemoryStore(base_store)
        
        # Duplicates are automatically filtered
        await dedup_store.add(memory1)
        await dedup_store.add(memory1_duplicate)  # Skipped
        
        # With custom threshold
        consolidator = DeduplicationConsolidator(similarity_threshold=0.9)
        dedup_store = DeduplicatingMemoryStore(base_store, consolidator)
    """
    
    def __init__(
        self,
        store: IMemoryStore,
        consolidator: Optional[DeduplicationConsolidator] = None,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize the deduplicating wrapper.
        
        Args:
            store: The underlying memory store to wrap
            consolidator: Optional custom deduplication consolidator
            similarity_threshold: Threshold for duplicate detection (0.0-1.0)
                                  Used only if consolidator is not provided
        """
        self._store = store
        self._consolidator = consolidator or DeduplicationConsolidator(
            similarity_threshold=similarity_threshold,
            keep_strategy="confidence",
        )
    
    async def add(self, item: MemoryItem) -> str:
        """
        Add a memory, checking for duplicates first.
        
        Returns:
            The memory_id if stored, empty string if duplicate was skipped
        """
        # Get existing memories for this user to check for duplicates
        existing = await self._store.search(
            MemoryQuery(user_id=item.user_id, limit=1000)
        )
        
        # Check if this is a duplicate
        non_duplicates = await self._consolidator.consolidate([item], existing)
        
        if non_duplicates:
            stored_item = non_duplicates[0]
            
            # Check if this is an update to an existing memory
            if stored_item.metadata.get("is_update"):
                # Update the existing memory instead of adding
                existing_id = stored_item.metadata.get("updates_memory_id")
                if existing_id:
                    stored_item.metadata.pop("is_update", None)
                    stored_item.metadata.pop("updates_memory_id", None)
                    # Update the memory_id to match the existing one
                    stored_item.memory_id = existing_id
                    await self._store.update(stored_item)
                    return existing_id
            
            return await self._store.add(stored_item)
        
        # Duplicate was found and skipped
        return ""
    
    async def add_batch(self, items: List[MemoryItem]) -> List[str]:
        """
        Add multiple memories with deduplication.
        
        Deduplicates within the batch and against existing memories.
        
        Returns:
            List of memory_ids for stored items (empty strings for skipped)
        """
        if not items:
            return []
        
        # All items should be for the same user for batch dedup
        user_id = items[0].user_id
        existing = await self._store.search(
            MemoryQuery(user_id=user_id, limit=1000)
        )
        
        # Consolidate all items at once
        non_duplicates = await self._consolidator.consolidate(items, existing)
        
        # Track which original items were kept
        kept_ids = {item.memory_id for item in non_duplicates}
        
        results = []
        for item in items:
            if item.memory_id in kept_ids:
                # Find the (possibly merged) version
                stored_item = next(
                    (nd for nd in non_duplicates if nd.memory_id == item.memory_id),
                    item
                )
                
                if stored_item.metadata.get("is_update"):
                    existing_id = stored_item.metadata.get("updates_memory_id")
                    if existing_id:
                        stored_item.metadata.pop("is_update", None)
                        stored_item.metadata.pop("updates_memory_id", None)
                        stored_item.memory_id = existing_id
                        await self._store.update(stored_item)
                        results.append(existing_id)
                        continue
                
                result = await self._store.add(stored_item)
                results.append(result)
            else:
                results.append("")  # Duplicate, skipped
        
        return results
    
    # Delegate all other methods to the underlying store
    
    async def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search for memories."""
        return await self._store.search(query)
    
    async def update(self, item: MemoryItem) -> bool:
        """Update an existing memory."""
        return await self._store.update(item)
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        return await self._store.delete(memory_id)
    
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a specific memory by ID."""
        return await self._store.get(memory_id)
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """Get all memories for a user."""
        # Forward protocol-compatible args to underlying store
        try:
            return await self._store.get_by_user(
                user_id,
                limit=limit,
                include_inactive=include_inactive,
            )
        except TypeError:
            # Backward-compatible path for older stores that used (user_id, limit, offset)
            return await self._store.get_by_user(user_id, limit, 0)
    
    async def clear_user(self, user_id: str) -> int:
        """Clear all memories for a user."""
        deleted = 0
        while True:
            items = await self._store.get_by_user(user_id, limit=1000)
            if not items:
                break
            for mem in items:
                if mem.memory_id and await self._store.delete(mem.memory_id):
                    deleted += 1
            if len(items) < 1000:
                break
        return deleted
    
    async def count(self, user_id: Optional[str] = None) -> int:
        """Count memories."""
        return await self._store.count(user_id)

