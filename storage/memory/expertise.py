"""
In-memory expertise store implementation.

Suitable for testing and single-instance deployments.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseUsageLog,
)
from ctxforge.engine.registry import registry
from ctxforge.protocols.expertise import IExpertiseStore


@registry.register_expertise_store("memory")
class InMemoryExpertiseStore(IExpertiseStore):
    """
    In-memory expertise store.
    
    Expertise and items are stored in dictionaries with simple search.
    Suitable for testing and single-instance deployments.
    
    Example:
        store = InMemoryExpertiseStore()
        
        expertise = Expertise(expertise_id="test", name="Test Expertise")
        await store.save(expertise)
        
        loaded = await store.load("test")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the store.
        
        Args:
            config: Optional configuration (unused for in-memory)
        """
        self._expertise: Dict[str, Expertise] = {}
        self._usage_logs: List[ExpertiseUsageLog] = []
        self._lock = asyncio.Lock()
    
    async def save(self, expertise: Expertise) -> None:
        """
        Save or update an expertise knowledge base.
        
        Args:
            expertise: The expertise to save
        """
        async with self._lock:
            expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            # Store a deep copy to prevent mutation
            self._expertise[expertise.expertise_id] = expertise.model_copy(deep=True)
    
    async def load(self, expertise_id: str) -> Optional[Expertise]:
        """
        Load an expertise by ID.
        
        Args:
            expertise_id: Unique identifier of the expertise
            
        Returns:
            The expertise if found, None otherwise
        """
        expertise = self._expertise.get(expertise_id)
        if expertise:
            return expertise.model_copy(deep=True)
        return None
    
    async def delete(self, expertise_id: str) -> bool:
        """
        Delete an expertise knowledge base.
        
        Args:
            expertise_id: ID of the expertise to delete
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if expertise_id in self._expertise:
                del self._expertise[expertise_id]
                # Also delete usage logs for this expertise
                self._usage_logs = [
                    log for log in self._usage_logs
                    if log.expertise_id != expertise_id
                ]
                return True
            return False
    
    async def list_expertise(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Expertise]:
        """
        List expertise knowledge bases.
        
        Args:
            domain: Optional domain filter
            limit: Maximum number to return
            offset: Offset for pagination
            
        Returns:
            List of expertise matching criteria
        """
        results = list(self._expertise.values())
        
        # Filter by domain if specified
        if domain:
            results = [e for e in results if e.domain == domain]
        
        # Sort by updated_at descending
        results.sort(key=lambda e: e.updated_at, reverse=True)
        
        # Apply pagination
        return [e.model_copy(deep=True) for e in results[offset:offset + limit]]
    
    async def add_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """
        Add an item to an expertise.
        
        Args:
            expertise_id: ID of the expertise
            item: Item to add
        """
        async with self._lock:
            expertise = self._expertise.get(expertise_id)
            if expertise:
                expertise.items.append(item.model_copy(deep=True))
                expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    async def update_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """
        Update an existing item.
        
        Args:
            expertise_id: ID of the expertise
            item: Updated item (matched by item_id)
        """
        async with self._lock:
            expertise = self._expertise.get(expertise_id)
            if expertise:
                for i, existing in enumerate(expertise.items):
                    if existing.item_id == item.item_id:
                        expertise.items[i] = item.model_copy(deep=True)
                        expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        break
    
    async def remove_item(self, expertise_id: str, item_id: str) -> bool:
        """
        Remove an item from an expertise.
        
        Args:
            expertise_id: ID of the expertise
            item_id: ID of the item to remove
            
        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            expertise = self._expertise.get(expertise_id)
            if expertise:
                for i, item in enumerate(expertise.items):
                    if item.item_id == item_id:
                        expertise.items.pop(i)
                        expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        return True
            return False
    
    async def get_item(
        self,
        expertise_id: str,
        item_id: str,
    ) -> Optional[ExpertiseItem]:
        """
        Get a single item by ID.
        
        Args:
            expertise_id: ID of the expertise
            item_id: ID of the item
            
        Returns:
            The item if found, None otherwise
        """
        expertise = self._expertise.get(expertise_id)
        if expertise:
            for item in expertise.items:
                if item.item_id == item_id:
                    return item.model_copy(deep=True)
        return None
    
    async def get_items_by_section(
        self,
        expertise_id: str,
        section: ExpertiseSection,
    ) -> List[ExpertiseItem]:
        """
        Get all items in a section.
        
        Args:
            expertise_id: ID of the expertise
            section: Section to filter by
            
        Returns:
            List of items in the section
        """
        expertise = self._expertise.get(expertise_id)
        if expertise:
            return [
                item.model_copy(deep=True)
                for item in expertise.items
                if item.section == section and item.is_active
            ]
        return []
    
    async def update_item_counts(
        self,
        expertise_id: str,
        item_id: str,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
    ) -> None:
        """
        Update helpful/harmful counts for an item.
        
        Args:
            expertise_id: ID of the expertise
            item_id: ID of the item
            helpful_delta: Amount to add to helpful count
            harmful_delta: Amount to add to harmful count
        """
        async with self._lock:
            expertise = self._expertise.get(expertise_id)
            if expertise:
                for item in expertise.items:
                    if item.item_id == item_id:
                        item.helpful_count += helpful_delta
                        item.harmful_count += harmful_delta
                        item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        break
    
    async def search_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """
        Search items by text content.
        
        Uses simple keyword matching.
        
        Args:
            expertise_id: ID of the expertise
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching items
        """
        expertise = self._expertise.get(expertise_id)
        if not expertise:
            return []
        
        query_words = set(query.lower().split())
        scored_items = []
        
        for item in expertise.items:
            if not item.is_active:
                continue
            
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0:
                scored_items.append((overlap, item))
        
        # Sort by score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        return [
            item.model_copy(deep=True)
            for _, item in scored_items[:limit]
        ]
    
    async def log_usage(self, log: ExpertiseUsageLog) -> None:
        """
        Log expertise usage in a turn.
        
        Args:
            log: Usage log entry
        """
        async with self._lock:
            self._usage_logs.append(log.model_copy(deep=True))
    
    async def get_usage_stats(self, expertise_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for an expertise.
        
        Args:
            expertise_id: ID of the expertise
            
        Returns:
            Dictionary with usage statistics
        """
        logs = [log for log in self._usage_logs if log.expertise_id == expertise_id]
        
        total_uses = len(logs)
        item_usage: Dict[str, int] = {}
        feedback_counts: Dict[str, Dict[str, int]] = {}
        outcome_counts: Dict[str, int] = {}
        
        for log in logs:
            # Count item usage
            for item_id in log.items_used:
                item_usage[item_id] = item_usage.get(item_id, 0) + 1
            
            # Count feedback
            for item_id, feedback in log.feedback.items():
                if item_id not in feedback_counts:
                    feedback_counts[item_id] = {"helpful": 0, "harmful": 0, "neutral": 0}
                feedback_counts[item_id][feedback.value] += 1
            
            # Count outcomes
            if log.outcome:
                outcome_counts[log.outcome.value] = outcome_counts.get(log.outcome.value, 0) + 1
        
        return {
            "total_uses": total_uses,
            "item_usage": item_usage,
            "feedback_counts": feedback_counts,
            "outcome_counts": outcome_counts,
        }
    
    async def clear(self) -> None:
        """Clear all data (for testing)."""
        async with self._lock:
            self._expertise.clear()
            self._usage_logs.clear()

