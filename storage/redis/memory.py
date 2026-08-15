"""
Redis memory store implementation.

Provides high-performance memory storage with indexing and keyword-based search.
"""

from typing import List, Optional

from ctxforge.core.memory import MemoryItem, MemoryQuery, MemoryType
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.storage.connection import RedisConfig, RedisConnectionManager


@registry.register_memory_store("redis")
class RedisMemoryStore(IMemoryStore):
    """
    Redis-based memory store.
    
    Features:
    - JSON storage with hash-based indexing
    - Keyword-based search using Redis SCAN
    - Filtering by type, tags, and confidence
    - Sorted sets for user memory indexes
    
    Note: For vector similarity search, use a dedicated vector store
    (Pinecone, ChromaDB, etc.) or Redis Stack with RediSearch.
    """
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        connection_manager: Optional[RedisConnectionManager] = None,
    ):
        """
        Initialize the Redis memory store.
        
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
    
    def _memory_key(self, memory_id: str) -> str:
        """Get the Redis key for a memory."""
        return f"{self.config.memory_prefix}{memory_id}"
    
    def _user_memories_key(self, user_id: str) -> str:
        """Get the Redis key for user's memory index."""
        return f"{self.config.memory_prefix}user:{user_id}"
    
    def _type_index_key(self, user_id: str, memory_type: MemoryType) -> str:
        """Get the Redis key for type-based index."""
        return f"{self.config.memory_prefix}type:{user_id}:{memory_type.value}"
    
    def _tag_index_key(self, user_id: str, tag: str) -> str:
        """Get the Redis key for tag-based index."""
        return f"{self.config.memory_prefix}tag:{user_id}:{tag}"
    
    def _serialize_memory(self, item: MemoryItem) -> str:
        """Serialize a memory to JSON."""
        return item.model_dump_json()
    
    def _deserialize_memory(self, data: str) -> MemoryItem:
        """Deserialize a memory from JSON."""
        return MemoryItem.model_validate_json(data)
    
    async def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search for memories."""
        await self.connect()
        client = self._manager.client
        
        # Get candidate memory IDs
        user_memories_key = self._user_memories_key(query.user_id)
        
        # Start with all user memories
        all_memory_ids = await client.zrevrange(user_memories_key, 0, -1)
        
        if not all_memory_ids:
            return []
        
        # Filter by type if specified
        if query.types:
            type_sets = [
                self._type_index_key(query.user_id, t)
                for t in query.types
            ]
            type_memory_ids = set()
            for key in type_sets:
                members = await client.smembers(key)
                type_memory_ids.update(members)
            all_memory_ids = [
                mid for mid in all_memory_ids
                if mid in type_memory_ids
            ]
        
        # Filter by tags if specified
        if query.tags:
            tag_sets = [
                self._tag_index_key(query.user_id, tag)
                for tag in query.tags
            ]
            tag_memory_ids = set()
            for key in tag_sets:
                members = await client.smembers(key)
                tag_memory_ids.update(members)
            all_memory_ids = [
                mid for mid in all_memory_ids
                if mid in tag_memory_ids
            ]
        
        # Fetch memories
        keys = [self._memory_key(mid) for mid in all_memory_ids]
        if not keys:
            return []
        
        data_list = await client.mget(keys)
        
        memories = []
        for data in data_list:
            if data:
                mem = self._deserialize_memory(data)
                
                # Apply additional filters
                if not mem.is_active:
                    continue
                if mem.is_expired():
                    continue
                if mem.confidence_score < query.min_confidence:
                    continue
                
                memories.append(mem)
        
        # Score by query text similarity if provided
        if query.query_text:
            query_words = set(query.query_text.lower().split())
            scored = []
            
            for mem in memories:
                content_words = set(mem.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0 or not query_words:
                    scored.append((overlap, mem))
            
            scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
            memories = [item[1] for item in scored]
        
        # Apply limit and offset
        results = memories[query.offset:query.offset + query.limit]
        
        # Record access
        for mem in results:
            mem.record_access()
            # Update access count in Redis (fire-and-forget)
            await client.set(
                self._memory_key(mem.memory_id),
                self._serialize_memory(mem),
            )
        
        return results
    
    async def add(self, item: MemoryItem) -> str:
        """Add a new memory."""
        await self.connect()
        client = self._manager.client
        
        key = self._memory_key(item.memory_id)
        user_key = self._user_memories_key(item.user_id)
        type_key = self._type_index_key(item.user_id, item.type)
        
        async with client.pipeline() as pipe:
            # Store the memory
            pipe.set(key, self._serialize_memory(item))
            
            # Add to user index with timestamp score
            pipe.zadd(user_key, {item.memory_id: item.created_at.timestamp()})
            
            # Add to type index
            pipe.sadd(type_key, item.memory_id)
            
            # Add to tag indexes
            for tag in item.tags:
                tag_key = self._tag_index_key(item.user_id, tag)
                pipe.sadd(tag_key, item.memory_id)
            
            await pipe.execute()
        
        return item.memory_id
    
    async def update(self, item: MemoryItem) -> bool:
        """Update an existing memory."""
        await self.connect()
        client = self._manager.client
        
        key = self._memory_key(item.memory_id)
        
        # Check if exists
        if not await client.exists(key):
            return False
        
        # Get old item to update indexes
        old_data = await client.get(key)
        if old_data:
            old_item = self._deserialize_memory(old_data)
            
            # Remove from old type index if changed
            if old_item.type != item.type:
                old_type_key = self._type_index_key(item.user_id, old_item.type)
                new_type_key = self._type_index_key(item.user_id, item.type)
                await client.srem(old_type_key, item.memory_id)
                await client.sadd(new_type_key, item.memory_id)
            
            # Update tag indexes
            old_tags = set(old_item.tags)
            new_tags = set(item.tags)
            
            # Remove old tags
            for tag in old_tags - new_tags:
                tag_key = self._tag_index_key(item.user_id, tag)
                await client.srem(tag_key, item.memory_id)
            
            # Add new tags
            for tag in new_tags - old_tags:
                tag_key = self._tag_index_key(item.user_id, tag)
                await client.sadd(tag_key, item.memory_id)
        
        # Update the memory
        await client.set(key, self._serialize_memory(item))
        
        return True
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        await self.connect()
        client = self._manager.client
        
        key = self._memory_key(memory_id)
        
        # Get memory first to update indexes
        data = await client.get(key)
        if not data:
            return False
        
        item = self._deserialize_memory(data)
        user_key = self._user_memories_key(item.user_id)
        type_key = self._type_index_key(item.user_id, item.type)
        
        async with client.pipeline() as pipe:
            # Remove memory
            pipe.delete(key)
            
            # Remove from user index
            pipe.zrem(user_key, memory_id)
            
            # Remove from type index
            pipe.srem(type_key, memory_id)
            
            # Remove from tag indexes
            for tag in item.tags:
                tag_key = self._tag_index_key(item.user_id, tag)
                pipe.srem(tag_key, memory_id)
            
            await pipe.execute()
        
        return True
    
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory by ID."""
        await self.connect()
        client = self._manager.client
        
        key = self._memory_key(memory_id)
        data = await client.get(key)
        
        if not data:
            return None
        
        item = self._deserialize_memory(data)
        item.record_access()
        
        # Update access info
        await client.set(key, self._serialize_memory(item))
        
        return item
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """Get all memories for a user."""
        await self.connect()
        client = self._manager.client
        
        user_key = self._user_memories_key(user_id)
        
        # Get memory IDs sorted by timestamp (descending)
        memory_ids = await client.zrevrange(user_key, 0, limit - 1)
        
        if not memory_ids:
            return []
        
        # Fetch all memories
        keys = [self._memory_key(mid) for mid in memory_ids]
        data_list = await client.mget(keys)
        
        memories = []
        for data in data_list:
            if data:
                mem = self._deserialize_memory(data)
                if include_inactive or mem.is_active:
                    memories.append(mem)
        
        return memories[:limit]
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        await self.connect()
        client = self._manager.client
        
        user_key = self._user_memories_key(user_id)
        return await client.zcard(user_key)
    
    async def clear(self) -> None:
        """Clear all memories (for testing)."""
        await self.connect()
        client = self._manager.client
        
        # Find all memory keys
        pattern = f"{self.config.memory_prefix}*"
        cursor = 0
        
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

