"""
Merging Consolidator.

Intelligently merges similar memories into comprehensive ones.
More aggressive than deduplication, combines related facts.
"""

import re
from typing import Awaitable, Callable, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.extraction.consolidation.base import BaseConsolidator
from ctxforge.utils.similarity import ISimilarityCalculator


class MergingConsolidator(BaseConsolidator):
    """
    Consolidator that merges related memories.
    
    Groups similar memories together and combines them into
    more comprehensive statements. Useful for building up
    detailed user profiles from partial information.
    
    Example:
        consolidator = MergingConsolidator()
        
        # Merge related memories
        # Input: ["User likes coffee", "User prefers dark roast"]
        # Output: ["User likes coffee, preferring dark roast"]
        
        merged = await consolidator.consolidate(
            new_items=new_memories,
            existing_items=existing_memories,
        )
        
        # With custom similarity calculator
        from ctxforge.extraction.similarity import LevenshteinSimilarityCalculator
        calculator = LevenshteinSimilarityCalculator()
        consolidator = MergingConsolidator(similarity_calculator=calculator)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.70,
        merge_threshold: float = 0.60,
        embedding_func: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        llm_merge_func: Optional[Callable[[List[str]], Awaitable[str]]] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the merger.
        
        Args:
            similarity_threshold: Threshold for duplicate detection
            merge_threshold: Threshold for merging related items
            embedding_func: Optional embedding function
            llm_merge_func: Optional LLM function for intelligent merging
            similarity_calculator: Calculator for text similarity
        """
        super().__init__(similarity_threshold, embedding_func, similarity_calculator)
        self._merge_threshold = merge_threshold
        self._llm_merge_func = llm_merge_func
    
    @property
    def name(self) -> str:
        """The name of this consolidator."""
        return "merging"
    
    async def consolidate(
        self,
        new_items: List[MemoryItem],
        existing_items: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Consolidate by merging related memories.
        
        Groups memories by topic/type and merges similar ones.
        
        Args:
            new_items: New items to add
            existing_items: Existing items in the store
            
        Returns:
            Consolidated items to store
        """
        if not new_items:
            return []
        
        result = []
        
        # Group new items by type and topic
        groups = await self._group_by_similarity(new_items)
        
        for group in groups:
            if len(group) == 1:
                # Single item, check against existing
                item = group[0]
                related_existing = await self._find_related(item, existing_items)
                
                if related_existing:
                    # Merge with existing
                    merged = await self._merge_with_existing(item, related_existing)
                    merged.metadata["is_update"] = True
                    merged.metadata["updates_memory_id"] = related_existing[0].memory_id
                    result.append(merged)
                else:
                    result.append(item)
            else:
                # Merge group together first
                merged_group = await self._merge_group(group)
                
                # Then check against existing
                related_existing = await self._find_related(merged_group, existing_items)
                
                if related_existing:
                    merged = await self._merge_with_existing(merged_group, related_existing)
                    merged.metadata["is_update"] = True
                    merged.metadata["updates_memory_id"] = related_existing[0].memory_id
                    result.append(merged)
                else:
                    result.append(merged_group)
        
        return result
    
    async def _group_by_similarity(
        self,
        items: List[MemoryItem],
    ) -> List[List[MemoryItem]]:
        """
        Group items by similarity.
        
        Args:
            items: Items to group
            
        Returns:
            List of groups
        """
        if not items:
            return []
        
        groups = []
        used = set()
        
        for i, item in enumerate(items):
            if i in used:
                continue
            
            group = [item]
            used.add(i)
            
            for j, other in enumerate(items):
                if j in used or j == i:
                    continue
                
                # Check if similar enough to group
                similarity = await self._calculate_similarity(item, other)
                
                if similarity >= self._merge_threshold:
                    # Also check type compatibility
                    if item.type == other.type:
                        group.append(other)
                        used.add(j)
            
            groups.append(group)
        
        return groups
    
    async def _find_related(
        self,
        item: MemoryItem,
        candidates: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Find related memories that could be merged.
        
        More permissive than duplicate detection.
        
        Args:
            item: The item to match
            candidates: Potential related candidates
            
        Returns:
            Related memories
        """
        related = []
        
        for candidate in candidates:
            # Must be same type
            if candidate.type != item.type:
                continue
            
            similarity = await self._calculate_similarity(item, candidate)
            
            if similarity >= self._merge_threshold:
                related.append(candidate)
        
        return related
    
    async def _merge_group(
        self,
        group: List[MemoryItem],
    ) -> MemoryItem:
        """
        Merge a group of similar memories.
        
        Args:
            group: Memories to merge
            
        Returns:
            Merged memory
        """
        if len(group) == 1:
            return group[0]
        
        # If we have an LLM merge function, use it
        if self._llm_merge_func:
            return await self._llm_merge(group)
        
        # Otherwise use rule-based merging
        return await self._rule_based_merge(group)
    
    async def _merge_with_existing(
        self,
        new_item: MemoryItem,
        existing_items: List[MemoryItem],
    ) -> MemoryItem:
        """
        Merge a new item with existing related items.
        
        Args:
            new_item: New memory
            existing_items: Related existing memories
            
        Returns:
            Merged memory
        """
        all_items = [new_item] + existing_items
        
        if self._llm_merge_func:
            return await self._llm_merge(all_items)
        
        return await self._rule_based_merge(all_items)
    
    async def _rule_based_merge(
        self,
        items: List[MemoryItem],
    ) -> MemoryItem:
        """
        Merge using rule-based approach.
        
        Combines content, takes highest confidence, merges metadata.
        
        Args:
            items: Items to merge
            
        Returns:
            Merged memory
        """
        # Sort by confidence (highest first)
        sorted_items = sorted(
            items,
            key=lambda m: (m.confidence_score, len(m.content)),
            reverse=True,
        )
        
        base = sorted_items[0]
        
        # Combine content if significantly different
        combined_content = base.content
        for item in sorted_items[1:]:
            # Only add if it contains new information
            if not self._content_subset(item.content, combined_content):
                # Extract unique part
                unique_part = self._extract_unique_info(item.content, combined_content)
                if unique_part:
                    combined_content = f"{combined_content} Additionally, {unique_part.lower()}"
        
        # Create merged memory
        merged = MemoryItem(
            user_id=base.user_id,
            content=combined_content,
            type=base.type,
            source=base.source,
            confidence_score=base.confidence_score,
            tags=list(set(tag for item in items for tag in item.tags)),
            metadata={
                "merged_from": [item.memory_id for item in items],
                "merge_type": "rule_based",
            },
        )
        
        # Keep the original ID if updating
        if len(items) == 2 and items[1].memory_id:
            merged.memory_id = items[1].memory_id
        
        return merged
    
    async def _llm_merge(
        self,
        items: List[MemoryItem],
    ) -> MemoryItem:
        """
        Merge using LLM for intelligent combination.
        
        Args:
            items: Items to merge
            
        Returns:
            Merged memory
        """
        contents = [item.content for item in items]
        
        # Call LLM to merge
        merged_content = await self._llm_merge_func(contents)
        
        # Use highest confidence item as base
        base = max(items, key=lambda m: m.confidence_score)
        
        return MemoryItem(
            user_id=base.user_id,
            content=merged_content,
            type=base.type,
            source=base.source,
            confidence_score=base.confidence_score,
            tags=list(set(tag for item in items for tag in item.tags)),
            metadata={
                "merged_from": [item.memory_id for item in items],
                "merge_type": "llm",
            },
        )
    
    def _content_subset(self, content1: str, content2: str) -> bool:
        """
        Check if content1 is essentially a subset of content2.
        
        Args:
            content1: Content to check
            content2: Content to check against
            
        Returns:
            True if content1 is mostly contained in content2
        """
        # Simple word overlap check
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'user', 'that'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1:
            return True
        
        overlap = len(words1 & words2) / len(words1)
        return overlap > 0.8
    
    def _extract_unique_info(self, content: str, base_content: str) -> str:
        """
        Extract unique information from content not in base.
        
        Args:
            content: Content to extract from
            base_content: Base content to compare against
            
        Returns:
            Unique information string
        """
        # Simple approach: find words not in base
        words_content = set(content.lower().split())
        words_base = set(base_content.lower().split())
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'user', 'that', 'and', 'or'}
        
        unique_words = words_content - words_base - stop_words
        
        if not unique_words:
            return ""
        
        # Find the sentence or phrase containing unique words
        # Simple: just return the unique part
        for word in unique_words:
            # Find word in original content and extract phrase
            pattern = rf'\b\w*{re.escape(word)}\w*(?:\s+\w+){{0,3}}'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return ""
