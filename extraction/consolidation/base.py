"""
Base Consolidator abstract class.

Provides common functionality for memory consolidation.
"""

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.protocols.extractor import IConsolidator
from ctxforge.utils.math import cosine_similarity
from ctxforge.utils.similarity import (
    ISimilarityCalculator,
    TextSimilarityCalculator,
)


class BaseConsolidator(IConsolidator, ABC):
    """
    Abstract base class for memory consolidators.
    
    Provides common functionality for:
    - Duplicate detection
    - Similarity calculation
    - Memory comparison
    
    Subclasses must implement:
    - consolidate(): Main consolidation logic
    - name property
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_func: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the consolidator.
        
        Args:
            similarity_threshold: Threshold for considering items similar
            embedding_func: Optional async function to generate embeddings
            similarity_calculator: Calculator for text similarity (uses TextSimilarityCalculator if not provided)
        """
        self._similarity_threshold = similarity_threshold
        self._embedding_func = embedding_func
        self._similarity_calculator = similarity_calculator or TextSimilarityCalculator()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this consolidator."""
        ...
    
    @property
    def similarity_calculator(self) -> ISimilarityCalculator:
        """The similarity calculator being used."""
        return self._similarity_calculator
    
    @similarity_calculator.setter
    def similarity_calculator(self, calculator: ISimilarityCalculator) -> None:
        """Set a new similarity calculator."""
        self._similarity_calculator = calculator
    
    @abstractmethod
    async def consolidate(
        self,
        new_items: List[MemoryItem],
        existing_items: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Consolidate new items with existing ones.
        
        Args:
            new_items: New items to add
            existing_items: Existing items in the store
            
        Returns:
            Items to store (new + updated existing)
        """
        ...
    
    async def find_duplicates(
        self,
        item: MemoryItem,
        candidates: List[MemoryItem],
        threshold: Optional[float] = None,
    ) -> List[MemoryItem]:
        """
        Find potential duplicates of an item.
        
        Uses embedding similarity if available, falls back to
        text similarity.
        
        Args:
            item: The item to check
            candidates: Potential duplicate candidates
            threshold: Similarity threshold (uses default if not provided)
            
        Returns:
            List of potential duplicates
        """
        if not candidates:
            return []
        
        threshold = threshold or self._similarity_threshold
        duplicates = []
        
        for candidate in candidates:
            similarity = await self._calculate_similarity(item, candidate)
            
            if similarity >= threshold:
                duplicates.append(candidate)
        
        return duplicates
    
    async def _calculate_similarity(
        self,
        item1: MemoryItem,
        item2: MemoryItem,
    ) -> float:
        """
        Calculate similarity between two memory items.
        
        Uses embeddings if available, falls back to configured calculator.
        
        Args:
            item1: First memory item
            item2: Second memory item
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Try embedding-based similarity first
        if item1.embedding and item2.embedding:
            similarity = cosine_similarity(item1.embedding, item2.embedding)
            # Clamp to [0, 1] range (cosine similarity can be [-1, 1])
            return max(0.0, min(1.0, similarity))
        
        # If we have an embedding function and items lack embeddings,
        # we could generate them, but that's expensive
        # For now, fall back to text similarity using configured calculator
        return self._similarity_calculator.calculate(item1.content, item2.content)
    
    async def merge_memories(
        self,
        memories: List[MemoryItem],
    ) -> MemoryItem:
        """
        Merge multiple memories into one.
        
        Default implementation keeps the most recent/confident
        and combines metadata. Subclasses can override for
        more sophisticated merging.
        
        Args:
            memories: Memories to merge
            
        Returns:
            The merged memory
        """
        if not memories:
            raise ValueError("Cannot merge empty list of memories")
        
        if len(memories) == 1:
            return memories[0]
        
        # Sort by confidence and recency
        sorted_memories = sorted(
            memories,
            key=lambda m: (m.confidence_score, m.updated_at or m.created_at),
            reverse=True,
        )
        
        # Use the best one as base
        base = sorted_memories[0]
        
        # Combine tags from all
        combined_tags = set(base.tags)
        for memory in sorted_memories[1:]:
            combined_tags.update(memory.tags)
        
        # Combine related memory IDs
        combined_related = set(base.related_memory_ids)
        for memory in sorted_memories[1:]:
            combined_related.update(memory.related_memory_ids)
            # Also add the merged memory's ID as related
            if memory.memory_id != base.memory_id:
                combined_related.add(memory.memory_id)
        
        # Update base with combined data
        base.tags = list(combined_tags)
        base.related_memory_ids = list(combined_related)
        
        # Update access count
        total_access = sum(m.access_count for m in memories)
        base.access_count = total_access
        
        # Add merge metadata
        base.metadata["merged_from"] = [m.memory_id for m in memories if m.memory_id != base.memory_id]
        base.metadata["merge_count"] = len(memories)
        
        return base
    
    def _items_conflict(
        self,
        item1: MemoryItem,
        item2: MemoryItem,
    ) -> bool:
        """
        Check if two items potentially conflict.
        
        Simple heuristic: same type and similar topics but
        different values.
        
        Args:
            item1: First item
            item2: Second item
            
        Returns:
            True if items may conflict
        """
        # Different types rarely conflict
        if item1.type != item2.type:
            return False
        
        # Check for contradictory patterns
        # e.g., "likes X" vs "dislikes X"
        content1 = item1.content.lower()
        content2 = item2.content.lower()
        
        # Simple contradiction detection
        positive = ["likes", "loves", "prefers", "enjoys", "always"]
        negative = ["dislikes", "hates", "avoids", "never"]
        
        has_positive_1 = any(p in content1 for p in positive)
        has_negative_1 = any(n in content1 for n in negative)
        has_positive_2 = any(p in content2 for p in positive)
        has_negative_2 = any(n in content2 for n in negative)
        
        # Conflict if one positive and one negative about similar topic
        if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
            # Check if they're about the same thing
            similarity = self._similarity_calculator.calculate(content1, content2)
            if similarity > 0.5:
                return True
        
        return False
