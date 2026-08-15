"""
Redis expertise store implementation.

Provides high-performance expertise storage with sorted sets and JSON serialization.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.exceptions import StorageError
from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseUsageLog,
)
from ctxforge.engine.registry import registry
from ctxforge.protocols.expertise import IExpertiseStore
from ctxforge.storage.connection import RedisConfig, RedisConnectionManager


@registry.register_expertise_store("redis")
class RedisExpertiseStore(IExpertiseStore):
    """
    Redis-based expertise store.
    
    Features:
    - JSON storage with sorted set indexes
    - Section-based indexing for fast filtering
    - Usage log storage with time-series support
    - Configurable TTL support
    
    Example:
        from ctxforge.storage.connection import RedisConfig, RedisConnectionManager
        
        config = RedisConfig(host="localhost", port=6379)
        manager = RedisConnectionManager(config)
        await manager.connect()
        
        store = RedisExpertiseStore(config=config, connection_manager=manager)
        
        expertise = Expertise(expertise_id="test", name="Test")
        await store.save(expertise)
    """
    
    # Key prefixes
    EXPERTISE_PREFIX = "expertise:"
    ITEM_PREFIX = "expertise:item:"
    USAGE_PREFIX = "expertise:usage:"
    INDEX_PREFIX = "expertise:index:"
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        connection_manager: Optional[RedisConnectionManager] = None,
    ):
        """
        Initialize the Redis expertise store.
        
        Args:
            config: Redis configuration
            connection_manager: Optional pre-existing connection manager
        """
        self.config = config or RedisConfig()
        self._manager = connection_manager or RedisConnectionManager(self.config)
        self._owns_connection = connection_manager is None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self._manager.is_connected:
            await self._manager.connect()
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._owns_connection and self._manager.is_connected:
            await self._manager.disconnect()

    async def initialize(self) -> None:
        """Lifecycle alias for ctxforge-managed initialization."""
        await self.connect()

    async def close(self) -> None:
        """Lifecycle alias for ctxforge-managed teardown."""
        await self.disconnect()
    
    def _expertise_key(self, expertise_id: str) -> str:
        """Get the Redis key for an expertise."""
        return f"{self.EXPERTISE_PREFIX}{expertise_id}"
    
    def _item_key(self, expertise_id: str, item_id: str) -> str:
        """Get the Redis key for an item."""
        return f"{self.ITEM_PREFIX}{expertise_id}:{item_id}"
    
    def _items_index_key(self, expertise_id: str) -> str:
        """Get the key for expertise's item index."""
        return f"{self.INDEX_PREFIX}items:{expertise_id}"
    
    def _section_index_key(self, expertise_id: str, section: ExpertiseSection) -> str:
        """Get the key for section-based index."""
        return f"{self.INDEX_PREFIX}section:{expertise_id}:{section.value}"
    
    def _usage_log_key(self, log_id: str) -> str:
        """Get the Redis key for a usage log."""
        return f"{self.USAGE_PREFIX}{log_id}"
    
    def _usage_index_key(self, expertise_id: str) -> str:
        """Get the key for expertise's usage log index."""
        return f"{self.INDEX_PREFIX}usage:{expertise_id}"
    
    def _all_expertise_key(self) -> str:
        """Get the key for all expertise index."""
        return f"{self.INDEX_PREFIX}all"
    
    def _domain_index_key(self, domain: str) -> str:
        """Get the key for domain-based index."""
        return f"{self.INDEX_PREFIX}domain:{domain}"
    
    def _serialize_expertise(self, expertise: Expertise) -> str:
        """Serialize expertise metadata (without items) to JSON."""
        data = {
            "expertise_id": expertise.expertise_id,
            "name": expertise.name,
            "domain": expertise.domain,
            "version": expertise.version,
            "token_budget": expertise.token_budget,
            "next_item_id": expertise.next_item_id,
            "metadata": expertise.metadata,
            "created_at": expertise.created_at.isoformat(),
            "updated_at": expertise.updated_at.isoformat(),
        }
        return json.dumps(data)
    
    def _deserialize_expertise(self, data: str) -> Expertise:
        """Deserialize expertise from JSON."""
        obj = json.loads(data)
        return Expertise(
            expertise_id=obj["expertise_id"],
            name=obj["name"],
            domain=obj.get("domain"),
            version=obj["version"],
            token_budget=obj["token_budget"],
            next_item_id=obj["next_item_id"],
            metadata=obj.get("metadata", {}),
            items=[],  # Items loaded separately
            created_at=datetime.fromisoformat(obj["created_at"]),
            updated_at=datetime.fromisoformat(obj["updated_at"]),
        )
    
    def _serialize_item(self, item: ExpertiseItem) -> str:
        """Serialize an item to JSON."""
        return item.model_dump_json()
    
    def _deserialize_item(self, data: str) -> ExpertiseItem:
        """Deserialize an item from JSON."""
        return ExpertiseItem.model_validate_json(data)
    
    def _serialize_usage_log(self, log: ExpertiseUsageLog) -> str:
        """Serialize a usage log to JSON."""
        return log.model_dump_json()
    
    def _deserialize_usage_log(self, data: str) -> ExpertiseUsageLog:
        """Deserialize a usage log from JSON."""
        return ExpertiseUsageLog.model_validate_json(data)
    
    async def save(self, expertise: Expertise) -> None:
        """Save or update an expertise knowledge base."""
        await self.connect()
        client = self._manager.client
        
        try:
            expertise_key = self._expertise_key(expertise.expertise_id)
            items_index_key = self._items_index_key(expertise.expertise_id)
            all_expertise_key = self._all_expertise_key()
            
            async with client.pipeline() as pipe:
                # Save expertise metadata
                pipe.set(expertise_key, self._serialize_expertise(expertise))
                
                # Add to all expertise index
                pipe.zadd(
                    all_expertise_key,
                    {expertise.expertise_id: expertise.updated_at.timestamp()},
                )
                
                # Add to domain index if domain is set
                if expertise.domain:
                    domain_key = self._domain_index_key(expertise.domain)
                    pipe.zadd(
                        domain_key,
                        {expertise.expertise_id: expertise.updated_at.timestamp()},
                    )
                
                await pipe.execute()
            
            # Clear existing items and re-add
            # First, get existing item IDs to delete
            existing_item_ids = await client.smembers(items_index_key)
            
            if existing_item_ids:
                async with client.pipeline() as pipe:
                    for item_id in existing_item_ids:
                        item_key = self._item_key(expertise.expertise_id, item_id)
                        pipe.delete(item_key)
                    pipe.delete(items_index_key)
                    
                    # Delete section indexes
                    for section in ExpertiseSection:
                        section_key = self._section_index_key(expertise.expertise_id, section)
                        pipe.delete(section_key)
                    
                    await pipe.execute()
            
            # Add new items
            for item in expertise.items:
                await self._store_item(expertise.expertise_id, item)
        
        except Exception as e:
            raise StorageError(f"Failed to save expertise: {e}") from e
    
    async def _store_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """Store a single item with all indexes."""
        client = self._manager.client
        
        item_key = self._item_key(expertise_id, item.item_id)
        items_index_key = self._items_index_key(expertise_id)
        section_key = self._section_index_key(expertise_id, item.section)
        
        async with client.pipeline() as pipe:
            # Store the item
            pipe.set(item_key, self._serialize_item(item))
            
            # Add to items index with creation timestamp
            pipe.sadd(items_index_key, item.item_id)
            
            # Add to section index
            pipe.sadd(section_key, item.item_id)
            
            await pipe.execute()
    
    async def load(self, expertise_id: str) -> Optional[Expertise]:
        """Load an expertise by ID."""
        await self.connect()
        client = self._manager.client
        
        try:
            expertise_key = self._expertise_key(expertise_id)
            items_index_key = self._items_index_key(expertise_id)
            
            # Load expertise metadata
            data = await client.get(expertise_key)
            if not data:
                return None
            
            expertise = self._deserialize_expertise(data)
            
            # Load all items
            item_ids = await client.smembers(items_index_key)
            
            if item_ids:
                item_keys = [self._item_key(expertise_id, iid) for iid in item_ids]
                item_data_list = await client.mget(item_keys)
                
                items = []
                for item_data in item_data_list:
                    if item_data:
                        items.append(self._deserialize_item(item_data))
                
                # Sort by created_at
                items.sort(key=lambda i: i.created_at)
                expertise.items = items
            
            return expertise
        
        except Exception as e:
            raise StorageError(f"Failed to load expertise: {e}") from e
    
    async def delete(self, expertise_id: str) -> bool:
        """Delete an expertise knowledge base."""
        await self.connect()
        client = self._manager.client
        
        try:
            expertise_key = self._expertise_key(expertise_id)
            items_index_key = self._items_index_key(expertise_id)
            usage_index_key = self._usage_index_key(expertise_id)
            all_expertise_key = self._all_expertise_key()
            
            # Check if exists
            if not await client.exists(expertise_key):
                return False
            
            # Get expertise for domain info
            data = await client.get(expertise_key)
            expertise = self._deserialize_expertise(data) if data else None
            
            # Get all item IDs to delete
            item_ids = await client.smembers(items_index_key)
            
            # Get all usage log IDs to delete
            usage_log_ids = await client.zrange(usage_index_key, 0, -1)
            
            async with client.pipeline() as pipe:
                # Delete expertise metadata
                pipe.delete(expertise_key)
                
                # Delete from all expertise index
                pipe.zrem(all_expertise_key, expertise_id)
                
                # Delete from domain index
                if expertise and expertise.domain:
                    domain_key = self._domain_index_key(expertise.domain)
                    pipe.zrem(domain_key, expertise_id)
                
                # Delete items
                for item_id in item_ids:
                    item_key = self._item_key(expertise_id, item_id)
                    pipe.delete(item_key)
                
                # Delete items index
                pipe.delete(items_index_key)
                
                # Delete section indexes
                for section in ExpertiseSection:
                    section_key = self._section_index_key(expertise_id, section)
                    pipe.delete(section_key)
                
                # Delete usage logs
                for log_id in usage_log_ids:
                    log_key = self._usage_log_key(log_id)
                    pipe.delete(log_key)
                
                # Delete usage index
                pipe.delete(usage_index_key)
                
                await pipe.execute()
            
            return True
        
        except Exception as e:
            raise StorageError(f"Failed to delete expertise: {e}") from e
    
    async def list_expertise(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Expertise]:
        """List expertise knowledge bases."""
        await self.connect()
        client = self._manager.client
        
        try:
            if domain:
                index_key = self._domain_index_key(domain)
            else:
                index_key = self._all_expertise_key()
            
            # Get expertise IDs sorted by update time (descending)
            expertise_ids = await client.zrevrange(
                index_key,
                offset,
                offset + limit - 1,
            )
            
            if not expertise_ids:
                return []
            
            # Load each expertise
            results = []
            for expertise_id in expertise_ids:
                expertise = await self.load(expertise_id)
                if expertise:
                    results.append(expertise)
            
            return results
        
        except Exception as e:
            raise StorageError(f"Failed to list expertise: {e}") from e
    
    async def add_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """Add an item to an expertise."""
        await self.connect()
        
        try:
            await self._store_item(expertise_id, item)
            
            # Update expertise timestamp
            await self._update_expertise_timestamp(expertise_id)
        
        except Exception as e:
            raise StorageError(f"Failed to add item: {e}") from e
    
    async def _update_expertise_timestamp(self, expertise_id: str) -> None:
        """Update the expertise's updated_at timestamp."""
        client = self._manager.client
        expertise_key = self._expertise_key(expertise_id)
        
        data = await client.get(expertise_key)
        if data:
            expertise = self._deserialize_expertise(data)
            expertise.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await client.set(expertise_key, self._serialize_expertise(expertise))
    
    async def update_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """Update an existing item."""
        await self.connect()
        client = self._manager.client
        
        try:
            item_key = self._item_key(expertise_id, item.item_id)
            
            # Get old item to update section index if needed
            old_data = await client.get(item_key)
            if old_data:
                old_item = self._deserialize_item(old_data)
                
                # Update section index if section changed
                if old_item.section != item.section:
                    old_section_key = self._section_index_key(expertise_id, old_item.section)
                    new_section_key = self._section_index_key(expertise_id, item.section)
                    
                    async with client.pipeline() as pipe:
                        pipe.srem(old_section_key, item.item_id)
                        pipe.sadd(new_section_key, item.item_id)
                        await pipe.execute()
            
            # Update the item
            item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await client.set(item_key, self._serialize_item(item))
            
            # Update expertise timestamp
            await self._update_expertise_timestamp(expertise_id)
        
        except Exception as e:
            raise StorageError(f"Failed to update item: {e}") from e
    
    async def remove_item(self, expertise_id: str, item_id: str) -> bool:
        """Remove an item from an expertise."""
        await self.connect()
        client = self._manager.client
        
        try:
            item_key = self._item_key(expertise_id, item_id)
            items_index_key = self._items_index_key(expertise_id)
            
            # Get item to find section
            data = await client.get(item_key)
            if not data:
                return False
            
            item = self._deserialize_item(data)
            section_key = self._section_index_key(expertise_id, item.section)
            
            async with client.pipeline() as pipe:
                pipe.delete(item_key)
                pipe.srem(items_index_key, item_id)
                pipe.srem(section_key, item_id)
                await pipe.execute()
            
            # Update expertise timestamp
            await self._update_expertise_timestamp(expertise_id)
            
            return True
        
        except Exception as e:
            raise StorageError(f"Failed to remove item: {e}") from e
    
    async def get_item(
        self,
        expertise_id: str,
        item_id: str,
    ) -> Optional[ExpertiseItem]:
        """Get a single item by ID."""
        await self.connect()
        client = self._manager.client
        
        try:
            item_key = self._item_key(expertise_id, item_id)
            data = await client.get(item_key)
            
            if not data:
                return None
            
            return self._deserialize_item(data)
        
        except Exception as e:
            raise StorageError(f"Failed to get item: {e}") from e
    
    async def get_items_by_section(
        self,
        expertise_id: str,
        section: ExpertiseSection,
    ) -> List[ExpertiseItem]:
        """Get all items in a section."""
        await self.connect()
        client = self._manager.client
        
        try:
            section_key = self._section_index_key(expertise_id, section)
            
            # Get item IDs in this section
            item_ids = await client.smembers(section_key)
            
            if not item_ids:
                return []
            
            # Fetch items
            item_keys = [self._item_key(expertise_id, iid) for iid in item_ids]
            item_data_list = await client.mget(item_keys)
            
            items = []
            for item_data in item_data_list:
                if item_data:
                    item = self._deserialize_item(item_data)
                    if item.is_active:
                        items.append(item)
            
            # Sort by created_at
            items.sort(key=lambda i: i.created_at)
            
            return items
        
        except Exception as e:
            raise StorageError(f"Failed to get items by section: {e}") from e
    
    async def update_item_counts(
        self,
        expertise_id: str,
        item_id: str,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
    ) -> None:
        """Update helpful/harmful counts for an item."""
        await self.connect()
        client = self._manager.client
        
        try:
            item_key = self._item_key(expertise_id, item_id)
            
            data = await client.get(item_key)
            if data:
                item = self._deserialize_item(data)
                item.helpful_count += helpful_delta
                item.harmful_count += harmful_delta
                item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                
                await client.set(item_key, self._serialize_item(item))
                await self._update_expertise_timestamp(expertise_id)
        
        except Exception as e:
            raise StorageError(f"Failed to update item counts: {e}") from e
    
    async def search_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """Search items by text content using keyword matching."""
        await self.connect()
        client = self._manager.client
        
        try:
            items_index_key = self._items_index_key(expertise_id)
            
            # Get all item IDs
            item_ids = await client.smembers(items_index_key)
            
            if not item_ids:
                return []
            
            # Fetch all items
            item_keys = [self._item_key(expertise_id, iid) for iid in item_ids]
            item_data_list = await client.mget(item_keys)
            
            # Score by keyword overlap
            query_words = set(query.lower().split())
            scored_items = []
            
            for item_data in item_data_list:
                if item_data:
                    item = self._deserialize_item(item_data)
                    if not item.is_active:
                        continue
                    
                    content_words = set(item.content.lower().split())
                    overlap = len(query_words & content_words)
                    
                    if overlap > 0:
                        scored_items.append((overlap, item))
            
            # Sort by score descending
            scored_items.sort(key=lambda x: x[0], reverse=True)
            
            return [item for _, item in scored_items[:limit]]
        
        except Exception as e:
            raise StorageError(f"Failed to search items: {e}") from e
    
    async def log_usage(self, log: ExpertiseUsageLog) -> None:
        """Log expertise usage in a turn."""
        await self.connect()
        client = self._manager.client
        
        try:
            log_key = self._usage_log_key(log.log_id)
            usage_index_key = self._usage_index_key(log.expertise_id)
            
            async with client.pipeline() as pipe:
                # Store the log
                pipe.set(log_key, self._serialize_usage_log(log))
                
                # Add to usage index with timestamp
                pipe.zadd(
                    usage_index_key,
                    {log.log_id: log.timestamp.timestamp()},
                )
                
                await pipe.execute()
        
        except Exception as e:
            raise StorageError(f"Failed to log usage: {e}") from e
    
    async def get_usage_stats(self, expertise_id: str) -> Dict[str, Any]:
        """Get usage statistics for an expertise."""
        await self.connect()
        client = self._manager.client
        
        try:
            usage_index_key = self._usage_index_key(expertise_id)
            
            # Get all usage log IDs
            log_ids = await client.zrange(usage_index_key, 0, -1)
            
            if not log_ids:
                return {
                    "total_uses": 0,
                    "item_usage": {},
                    "feedback_counts": {},
                    "outcome_counts": {},
                }
            
            # Fetch all logs
            log_keys = [self._usage_log_key(lid) for lid in log_ids]
            log_data_list = await client.mget(log_keys)
            
            total_uses = 0
            item_usage: Dict[str, int] = {}
            feedback_counts: Dict[str, Dict[str, int]] = {}
            outcome_counts: Dict[str, int] = {}
            
            for log_data in log_data_list:
                if log_data:
                    log = self._deserialize_usage_log(log_data)
                    total_uses += 1
                    
                    # Count item usage
                    for item_id in log.items_used:
                        item_usage[item_id] = item_usage.get(item_id, 0) + 1
                    
                    # Count feedback
                    for item_id, feedback in log.feedback.items():
                        if item_id not in feedback_counts:
                            feedback_counts[item_id] = {
                                "helpful": 0,
                                "harmful": 0,
                                "neutral": 0,
                            }
                        feedback_counts[item_id][feedback.value] += 1
                    
                    # Count outcomes
                    if log.outcome:
                        outcome_counts[log.outcome.value] = (
                            outcome_counts.get(log.outcome.value, 0) + 1
                        )
            
            return {
                "total_uses": total_uses,
                "item_usage": item_usage,
                "feedback_counts": feedback_counts,
                "outcome_counts": outcome_counts,
            }
        
        except Exception as e:
            raise StorageError(f"Failed to get usage stats: {e}") from e
    
    async def clear(self) -> None:
        """Clear all expertise data (for testing)."""
        await self.connect()
        client = self._manager.client
        
        # Delete all keys with our prefixes
        prefixes = [
            self.EXPERTISE_PREFIX,
            self.ITEM_PREFIX,
            self.USAGE_PREFIX,
            self.INDEX_PREFIX,
        ]
        
        for prefix in prefixes:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=f"{prefix}*", count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break

