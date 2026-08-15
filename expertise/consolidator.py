"""
Expertise Consolidator Implementation.

Provides deduplication and merging capabilities for expertise items,
using the existing consolidation infrastructure from extraction.

Inspired by ACE framework's BulletpointAnalyzer.
"""

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    SimilarGroup,
)
from ctxforge.utils.math import cosine_similarity
from ctxforge.utils.similarity import (
    ISimilarityCalculator,
    TextSimilarityCalculator,
)


class ExpertiseConsolidator:
    """
    Consolidates expertise items through deduplication and merging.
    
    Uses similarity calculation to identify groups of similar items
    and provides methods to merge them into single items, preserving
    usage statistics.
    
    Follows patterns from extraction/consolidation/ module.
    
    Example:
        >>> consolidator = ExpertiseConsolidator()
        >>> groups = await consolidator.find_similar_groups(items, threshold=0.85)
        >>> for group in groups:
        ...     merged = await consolidator.merge_group(group)
    """
    
    def __init__(
        self,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
        embedding_func: Optional[Callable[[str], Awaitable[List[float]]]] = None,
        llm_merge_func: Optional[Callable[[List[str]], Awaitable[str]]] = None,
        default_threshold: float = 0.85,
    ):
        """
        Initialize the consolidator.
        
        Args:
            similarity_calculator: Calculator for text similarity
            embedding_func: Optional async function to generate embeddings
            llm_merge_func: Optional async function to merge content via LLM
            default_threshold: Default similarity threshold for grouping
        """
        self._similarity_calculator = similarity_calculator or TextSimilarityCalculator()
        self._embedding_func = embedding_func
        self._llm_merge_func = llm_merge_func
        self._default_threshold = default_threshold
    
    @property
    def name(self) -> str:
        """The name of this consolidator."""
        return "expertise-consolidator"
    
    async def find_similar_groups(
        self,
        items: List[ExpertiseItem],
        threshold: Optional[float] = None,
        only_active: bool = True,
    ) -> List[SimilarGroup]:
        """
        Find groups of similar expertise items.
        
        Args:
            items: Items to analyze
            threshold: Similarity threshold (0.0-1.0)
            only_active: Only consider active items
            
        Returns:
            List of SimilarGroup objects
        """
        threshold = threshold or self._default_threshold
        
        # Filter to active items if requested
        if only_active:
            items = [i for i in items if i.is_active]
        
        if len(items) < 2:
            return []
        
        # Calculate pairwise similarities
        similarity_matrix = await self._build_similarity_matrix(items)
        
        # Find groups using union-find approach
        groups = self._find_groups_from_matrix(items, similarity_matrix, threshold)
        
        return groups
    
    async def _build_similarity_matrix(
        self,
        items: List[ExpertiseItem],
    ) -> List[List[float]]:
        """Build a similarity matrix for all items."""
        n = len(items)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            matrix[i][i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                similarity = await self._calculate_similarity(items[i], items[j])
                matrix[i][j] = similarity
                matrix[j][i] = similarity
        
        return matrix
    
    async def _calculate_similarity(
        self,
        item1: ExpertiseItem,
        item2: ExpertiseItem,
    ) -> float:
        """
        Calculate similarity between two items.
        
        Uses embeddings if available, falls back to text similarity.
        """
        # Try embedding-based similarity first
        if item1.embedding and item2.embedding:
            similarity = cosine_similarity(item1.embedding, item2.embedding)
            return max(0.0, min(1.0, similarity))
        
        # Try to generate embeddings if function is available
        if self._embedding_func:
            try:
                emb1 = await self._embedding_func(item1.content)
                emb2 = await self._embedding_func(item2.content)
                similarity = cosine_similarity(emb1, emb2)
                return max(0.0, min(1.0, similarity))
            except Exception:
                pass  # Fall back to text similarity
        
        # Fall back to text similarity
        return self._similarity_calculator.calculate(item1.content, item2.content)
    
    def _find_groups_from_matrix(
        self,
        items: List[ExpertiseItem],
        similarity_matrix: List[List[float]],
        threshold: float,
    ) -> List[SimilarGroup]:
        """Find groups of similar items from similarity matrix."""
        n = len(items)
        visited = set()
        groups = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Find all items similar to i
            similar_indices = [i]
            similar_scores = [1.0]  # Self-similarity
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if similarity_matrix[i][j] >= threshold:
                    similar_indices.append(j)
                    similar_scores.append(similarity_matrix[i][j])
            
            # Only create group if there are multiple items
            if len(similar_indices) > 1:
                group_items = [items[idx] for idx in similar_indices]
                groups.append(SimilarGroup(
                    items=group_items,
                    similarity_scores=similar_scores,
                ))
                visited.update(similar_indices)
        
        return groups
    
    async def merge_group(
        self,
        group: SimilarGroup,
        section: Optional[ExpertiseSection] = None,
    ) -> ExpertiseItem:
        """
        Merge a group of similar items into one.
        
        Combines usage counts and creates merged content.
        
        Args:
            group: The group to merge
            section: Section for the merged item (uses first item's section if not specified)
            
        Returns:
            Merged ExpertiseItem
        """
        if group.item_count == 0:
            raise ValueError("Cannot merge empty group")
        
        if group.item_count == 1:
            return group.items[0]
        
        # Use section from first item if not specified
        if section is None:
            section = group.primary_item.section
        
        # Sum usage counts
        total_helpful = group.total_helpful
        total_harmful = group.total_harmful
        
        # Merge content
        if self._llm_merge_func:
            # Use LLM to intelligently merge
            try:
                contents = [item.content for item in group.items]
                merged_content = await self._llm_merge_func(contents)
            except Exception:
                # Fall back to simple merge
                merged_content = self._simple_merge_content(group.items)
        else:
            merged_content = self._simple_merge_content(group.items)
        
        # Create merged item with first item's ID
        merged = ExpertiseItem(
            item_id=group.primary_item.item_id,
            section=section,
            content=merged_content,
            helpful_count=total_helpful,
            harmful_count=total_harmful,
            source="consolidator:merge",
            is_active=True,
            metadata={
                "merged_from": group.item_ids,
                "merge_count": group.item_count,
            },
            created_at=group.primary_item.created_at,
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        
        return merged
    
    def _simple_merge_content(self, items: List[ExpertiseItem]) -> str:
        """
        Simple content merging without LLM.
        
        Takes the longest/most detailed content.
        """
        # Return the longest content (usually most detailed)
        return max(items, key=lambda i: len(i.content)).content
    
    async def consolidate_expertise(
        self,
        expertise: Expertise,
        threshold: Optional[float] = None,
        apply_changes: bool = True,
    ) -> Dict[str, Any]:
        """
        Consolidate an entire expertise by merging similar items.
        
        Args:
            expertise: The expertise to consolidate
            threshold: Similarity threshold
            apply_changes: If True, modify expertise in place
            
        Returns:
            Dictionary with consolidation statistics
        """
        threshold = threshold or self._default_threshold
        
        # Find similar groups
        groups = await self.find_similar_groups(
            expertise.active_items,
            threshold=threshold,
        )
        
        if not groups:
            return {
                "groups_found": 0,
                "items_merged": 0,
                "items_removed": 0,
            }
        
        items_merged = 0
        items_removed = 0
        
        if apply_changes:
            for group in groups:
                # Create merged item
                merged = await self.merge_group(group)
                
                # Deactivate original items
                for item in group.items:
                    if item.item_id != merged.item_id:
                        item.deactivate()
                        items_removed += 1
                
                # Update the primary item with merged content
                primary = expertise.get_item(merged.item_id)
                if primary:
                    primary.content = merged.content
                    primary.helpful_count = merged.helpful_count
                    primary.harmful_count = merged.harmful_count
                    primary.metadata = merged.metadata
                    primary.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                
                items_merged += 1
            
            expertise.increment_version()
        
        return {
            "groups_found": len(groups),
            "items_merged": items_merged,
            "items_removed": items_removed,
        }
    
    async def find_duplicates(
        self,
        item: ExpertiseItem,
        candidates: List[ExpertiseItem],
        threshold: Optional[float] = None,
    ) -> List[ExpertiseItem]:
        """
        Find potential duplicates of an item.
        
        Args:
            item: The item to check
            candidates: Potential duplicate candidates
            threshold: Similarity threshold
            
        Returns:
            List of potential duplicates
        """
        threshold = threshold or self._default_threshold
        duplicates = []
        
        for candidate in candidates:
            if candidate.item_id == item.item_id:
                continue
            
            similarity = await self._calculate_similarity(item, candidate)
            if similarity >= threshold:
                duplicates.append(candidate)
        
        return duplicates
    
    async def is_duplicate(
        self,
        item: ExpertiseItem,
        candidates: List[ExpertiseItem],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if an item is a duplicate of any candidate.
        
        Args:
            item: The item to check
            candidates: Existing items to compare against
            threshold: Similarity threshold
            
        Returns:
            True if item is a duplicate
        """
        duplicates = await self.find_duplicates(item, candidates, threshold)
        return len(duplicates) > 0

