"""
Deduplication Consolidator.

Removes duplicate memories based on content similarity.
"""

from typing import Awaitable, Callable, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.extraction.consolidation.base import BaseConsolidator
from ctxforge.utils.hashing import compute_content_hash
from ctxforge.utils.similarity import ISimilarityCalculator


class DeduplicationConsolidator(BaseConsolidator):
    """
    Consolidator that removes duplicate memories.
    
    When duplicates are found:
    - Keeps the memory with higher confidence
    - If confidence is equal, keeps the more recent one
    - Combines tags and metadata from duplicates
    
    Example:
        consolidator = DeduplicationConsolidator(similarity_threshold=0.85)
        
        # Filter out duplicates from new memories
        to_store = await consolidator.consolidate(
            new_items=new_memories,
            existing_items=existing_memories,
        )
        
        # With custom similarity calculator
        from ctxforge.extraction.similarity import LevenshteinSimilarityCalculator
        calculator = LevenshteinSimilarityCalculator()
        consolidator = DeduplicationConsolidator(similarity_calculator=calculator)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_func: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        keep_strategy: str = "confidence",  # confidence, recency, merge
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Threshold for duplicate detection
            embedding_func: Optional embedding function
            keep_strategy: Strategy for handling duplicates:
                - "confidence": Keep higher confidence
                - "recency": Keep more recent
                - "merge": Merge all duplicates
            similarity_calculator: Calculator for text similarity
        """
        super().__init__(similarity_threshold, embedding_func, similarity_calculator)
        self._keep_strategy = keep_strategy
    
    @property
    def name(self) -> str:
        """The name of this consolidator."""
        return f"deduplication:{self._keep_strategy}"
    
    async def consolidate(
        self,
        new_items: List[MemoryItem],
        existing_items: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Consolidate new items, removing duplicates.
        
        Checks new items against each other and against existing items.
        
        Args:
            new_items: New items to add
            existing_items: Existing items in the store
            
        Returns:
            Non-duplicate items to store
        """
        if not new_items:
            return []
        
        result = []
        
        # First, deduplicate within new items
        deduplicated_new = await self._deduplicate_list(new_items)

        # Fast O(1) hash-based dedup against existing items
        existing_hashes = {}
        for ex in existing_items:
            h = ex.metadata.get("content_hash")
            if h:
                existing_hashes[h] = ex

        hash_passed = []
        for new_item in deduplicated_new:
            h = new_item.metadata.get("content_hash")
            if not h:
                h = compute_content_hash(new_item.content, new_item.type.value)
                new_item.metadata["content_hash"] = h
            if h in existing_hashes:
                existing_hashes[h].record_access()
                continue
            hash_passed.append(new_item)

        # Then check against existing items
        for new_item in hash_passed:
            duplicates = await self.find_duplicates(
                new_item,
                existing_items,
                self._similarity_threshold,
            )
            
            if duplicates:
                # Handle based on strategy
                if self._keep_strategy == "merge":
                    # Merge with existing duplicates
                    all_items = [new_item] + duplicates
                    merged = await self.merge_memories(all_items)
                    # Mark as update to existing
                    merged.metadata["is_update"] = True
                    merged.metadata["updates_memory_id"] = duplicates[0].memory_id
                    result.append(merged)
                elif self._keep_strategy == "recency":
                    # Keep if new item is more recent
                    most_recent_existing = max(
                        duplicates,
                        key=lambda m: m.updated_at or m.created_at,
                    )
                    new_time = new_item.updated_at or new_item.created_at
                    existing_time = most_recent_existing.updated_at or most_recent_existing.created_at
                    
                    if new_time > existing_time:
                        # New is more recent, update
                        new_item.metadata["is_update"] = True
                        new_item.metadata["updates_memory_id"] = most_recent_existing.memory_id
                        result.append(new_item)
                    # else: skip, existing is more recent
                else:  # confidence (default)
                    # Keep if new item has higher confidence
                    highest_existing = max(
                        duplicates,
                        key=lambda m: m.confidence_score,
                    )
                    
                    if new_item.confidence_score > highest_existing.confidence_score:
                        # New has higher confidence, update
                        new_item.metadata["is_update"] = True
                        new_item.metadata["updates_memory_id"] = highest_existing.memory_id
                        result.append(new_item)
                    # else: skip, existing has higher confidence
            else:
                # No duplicates, add new item
                result.append(new_item)
        
        return result
    
    async def _deduplicate_list(
        self,
        items: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Deduplicate within a list of items.
        
        Args:
            items: Items to deduplicate
            
        Returns:
            Deduplicated list
        """
        if len(items) <= 1:
            return items
        
        result = []
        
        for item in items:
            duplicates = await self.find_duplicates(
                item,
                result,
                self._similarity_threshold,
            )
            
            if duplicates:
                # Handle duplicate
                if self._keep_strategy == "merge":
                    # Find and update the existing item
                    idx = result.index(duplicates[0])
                    merged = await self.merge_memories([result[idx], item])
                    result[idx] = merged
                elif self._keep_strategy == "recency":
                    item_time = item.updated_at or item.created_at
                    existing_time = duplicates[0].updated_at or duplicates[0].created_at
                    if item_time > existing_time:
                        idx = result.index(duplicates[0])
                        result[idx] = item
                else:  # confidence
                    if item.confidence_score > duplicates[0].confidence_score:
                        idx = result.index(duplicates[0])
                        result[idx] = item
            else:
                result.append(item)
        
        return result
    
    async def find_exact_duplicates(
        self,
        item: MemoryItem,
        candidates: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Find exact content duplicates (stricter than similarity).
        
        Args:
            item: The item to check
            candidates: Potential duplicate candidates
            
        Returns:
            Exact duplicates
        """
        content_normalized = item.content.lower().strip()
        
        return [
            c for c in candidates
            if c.content.lower().strip() == content_normalized
        ]
