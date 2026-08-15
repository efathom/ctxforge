"""
Knowledge type filtering for retrieval.
"""

from typing import List, Optional, Set

from ctxforge.core.knowledge_types import KnowledgeType


class KnowledgeTypeFilter:
    """
    Filter retrieved items by knowledge type.
    
    Allows prioritizing certain types (e.g., rules and gotchas first)
    or filtering to specific types for focused retrieval.
    
    Example usage:
    ```python
    filter = KnowledgeTypeFilter(
        include_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA],
        priority_types=[KnowledgeType.RULE],
    )
    
    # Check if item should be included
    if filter.should_include(item.knowledge_type):
        # Apply priority boost
        score += filter.get_priority_boost(item.knowledge_type)
    ```
    """
    
    def __init__(
        self,
        include_types: Optional[List[KnowledgeType]] = None,
        exclude_types: Optional[List[KnowledgeType]] = None,
        priority_types: Optional[List[KnowledgeType]] = None,
        entity_filter: Optional[str] = None,
    ):
        """
        Initialize the filter.
        
        Args:
            include_types: Only include these types (None = all types)
            exclude_types: Exclude these types
            priority_types: Types to boost in ranking
            entity_filter: Filter by entity applicability
        """
        self._include: Optional[Set[KnowledgeType]] = (
            set(include_types) if include_types else None
        )
        self._exclude: Set[KnowledgeType] = (
            set(exclude_types) if exclude_types else set()
        )
        self._priority = priority_types or [
            KnowledgeType.RULE,
            KnowledgeType.GOTCHA,
            KnowledgeType.CONSTRAINT,
        ]
        self._entity = entity_filter
    
    @property
    def include_types(self) -> Optional[Set[KnowledgeType]]:
        """Get the set of types to include."""
        return self._include
    
    @property
    def exclude_types(self) -> Set[KnowledgeType]:
        """Get the set of types to exclude."""
        return self._exclude
    
    @property
    def priority_types(self) -> List[KnowledgeType]:
        """Get the priority types list."""
        return self._priority
    
    @property
    def entity_filter(self) -> Optional[str]:
        """Get the entity filter."""
        return self._entity
    
    def should_include(self, knowledge_type: KnowledgeType) -> bool:
        """
        Check if a knowledge type should be included.
        
        Args:
            knowledge_type: The type to check
            
        Returns:
            True if the type should be included
        """
        if knowledge_type in self._exclude:
            return False
        if self._include is not None:
            return knowledge_type in self._include
        return True
    
    def get_priority_boost(self, knowledge_type: KnowledgeType) -> float:
        """
        Get score boost for priority types.
        
        Args:
            knowledge_type: The type to get boost for
            
        Returns:
            Score boost (0.0 to 0.2) based on priority position
        """
        if knowledge_type in self._priority:
            idx = self._priority.index(knowledge_type)
            # First priority type gets 0.2 boost, decreasing by 0.05 for each subsequent
            return max(0.0, 0.2 - (idx * 0.05))
        return 0.0
    
    def filter_items(
        self,
        items: List,
        type_getter=None,
    ) -> List:
        """
        Filter a list of items by knowledge type.
        
        Args:
            items: List of items to filter
            type_getter: Function to get knowledge type from item.
                         Defaults to item.knowledge_type or item.metadata.get("knowledge_type")
                         
        Returns:
            Filtered list of items
        """
        if type_getter is None:
            def type_getter(item):
                if hasattr(item, 'knowledge_type') and item.knowledge_type:
                    kt = item.knowledge_type
                    if isinstance(kt, str):
                        try:
                            return KnowledgeType(kt)
                        except ValueError:
                            return KnowledgeType.INSIGHT
                    return kt
                if hasattr(item, 'metadata') and item.metadata:
                    kt = item.metadata.get('knowledge_type')
                    if kt:
                        if isinstance(kt, str):
                            try:
                                return KnowledgeType(kt)
                            except ValueError:
                                return KnowledgeType.INSIGHT
                        return kt
                return KnowledgeType.INSIGHT
        
        return [
            item for item in items
            if self.should_include(type_getter(item))
        ]
    
    def sort_by_priority(
        self,
        items: List,
        type_getter=None,
        score_getter=None,
    ) -> List:
        """
        Sort items by knowledge type priority and score.
        
        Args:
            items: List of items to sort
            type_getter: Function to get knowledge type from item
            score_getter: Function to get base score from item
            
        Returns:
            Sorted list with priority types first
        """
        if type_getter is None:
            def type_getter(item):
                if hasattr(item, 'knowledge_type') and item.knowledge_type:
                    kt = item.knowledge_type
                    if isinstance(kt, str):
                        try:
                            return KnowledgeType(kt)
                        except ValueError:
                            return KnowledgeType.INSIGHT
                    return kt
                return KnowledgeType.INSIGHT
        
        if score_getter is None:
            def score_getter(item):
                if hasattr(item, 'score'):
                    return item.score
                return 0.0
        
        def sort_key(item):
            kt = type_getter(item)
            base_score = score_getter(item)
            boost = self.get_priority_boost(kt)
            return -(base_score + boost)  # Negative for descending sort
        
        return sorted(items, key=sort_key)
