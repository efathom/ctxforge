"""
PostgreSQL memory store implementation.

Provides persistent memory storage with JSONB, full-text search, and vector support.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.exceptions import StorageError
from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.storage.connection import PostgresConfig, PostgresConnectionManager

# SQL statements for table creation
CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    memory_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL DEFAULT 0.5,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    tags TEXT[] NOT NULL DEFAULT '{{}}',
    data JSONB NOT NULL,
    embedding FLOAT[],
    headline VARCHAR(150),
    subtitle VARCHAR(300),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_user_id ON {table_name}(user_id);
CREATE INDEX IF NOT EXISTS idx_{table_name}_type ON {table_name}(memory_type);
CREATE INDEX IF NOT EXISTS idx_{table_name}_is_active ON {table_name}(is_active);
CREATE INDEX IF NOT EXISTS idx_{table_name}_tags ON {table_name} USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at ON {table_name}(created_at);
CREATE INDEX IF NOT EXISTS idx_{table_name}_headline ON {table_name}(headline);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_{table_name}_content_fts 
ON {table_name} USING GIN(to_tsvector('english', content));
"""


@registry.register_memory_store("postgres")
class PostgresMemoryStore(IMemoryStore):
    """
    PostgreSQL-based memory store.
    
    Features:
    - JSONB storage for flexible memory data
    - Full-text search using PostgreSQL's tsvector
    - Array-based tag indexing with GIN
    - Optional vector storage for embeddings (requires pgvector extension)
    """
    
    def __init__(
        self,
        config: Optional[PostgresConfig] = None,
        connection_manager: Optional[PostgresConnectionManager] = None,
    ):
        """
        Initialize the PostgreSQL memory store.
        
        Args:
            config: PostgreSQL configuration
            connection_manager: Optional pre-existing connection manager
        """
        self.config = config or PostgresConfig()
        self._manager = connection_manager or PostgresConnectionManager(self.config)
        self._owns_connection = connection_manager is None
        self._initialized = False
    
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        if not self._manager.is_connected:
            await self._manager.connect()
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._owns_connection and self._manager.is_connected:
            await self._manager.disconnect()

    async def close(self) -> None:
        """Lifecycle alias for ctxforge-managed teardown."""
        await self.disconnect()
    
    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return
        
        await self.connect()
        
        sql = CREATE_MEMORIES_TABLE.format(table_name=self.config.memories_table)
        await self._manager.execute(sql)
        
        # Try to add headline/subtitle columns if they don't exist (migration)
        await self._migrate_headline_columns()
        
        self._initialized = True
    
    async def _migrate_headline_columns(self) -> None:
        """Add headline and subtitle columns if they don't exist (migration)."""
        try:
            # Check if headline column exists
            check_sql = """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = $1 AND column_name = 'headline'
            """
            result = await self._manager.fetchrow(
                check_sql, self.config.memories_table
            )
            
            if not result:
                # Add headline column using IF NOT EXISTS for safety
                await self._manager.execute(
                    f"ALTER TABLE {self.config.memories_table} "
                    "ADD COLUMN IF NOT EXISTS headline VARCHAR(150)"
                )
                # Add subtitle column
                await self._manager.execute(
                    f"ALTER TABLE {self.config.memories_table} "
                    "ADD COLUMN IF NOT EXISTS subtitle VARCHAR(300)"
                )
                # Add index on headline
                await self._manager.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.config.memories_table}_headline "
                    f"ON {self.config.memories_table}(headline)"
                )
        except Exception:
            # Columns may already exist or other transient error - ignore
            pass
    
    def _serialize_memory(self, item: MemoryItem) -> Dict[str, Any]:
        """Serialize memory to dict for JSONB storage."""
        return json.loads(item.model_dump_json())
    
    def _deserialize_memory(self, row: Any) -> MemoryItem:
        """Deserialize memory from database row."""
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return MemoryItem.model_validate(data)
    
    async def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search for memories using full-text search."""
        await self.initialize()
        
        # Build dynamic query
        conditions = ["user_id = $1", "is_active = TRUE"]
        params: List[Any] = [query.user_id]
        param_idx = 2
        
        # Filter by types
        if query.types:
            type_values = [t.value for t in query.types]
            conditions.append(f"memory_type = ANY(${param_idx})")
            params.append(type_values)
            param_idx += 1
        
        # Filter by tags
        if query.tags:
            conditions.append(f"tags && ${param_idx}")  # Array overlap
            params.append(query.tags)
            param_idx += 1
        
        # Filter by confidence
        conditions.append(f"confidence_score >= ${param_idx}")
        params.append(query.min_confidence)
        param_idx += 1
        
        # Filter expired
        conditions.append(f"(expires_at IS NULL OR expires_at > ${param_idx})")
        params.append(datetime.now(timezone.utc))
        param_idx += 1
        
        # Full-text search
        order_by = "created_at DESC"
        if query.query_text:
            conditions.append(
                f"to_tsvector('english', content) @@ plainto_tsquery('english', ${param_idx})"
            )
            params.append(query.query_text)
            param_idx += 1
            
            # Order by relevance
            order_by = (
                f"ts_rank(to_tsvector('english', content), "
                f"plainto_tsquery('english', ${param_idx})) DESC, created_at DESC"
            )
            params.append(query.query_text)
            param_idx += 1
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([query.limit, query.offset])
        
        rows = await self._manager.fetch(sql, *params)
        
        memories = [self._deserialize_memory(row) for row in rows]
        
        # Record access
        for mem in memories:
            mem.record_access()
        
        return memories
    
    async def add(self, item: MemoryItem) -> str:
        """Add a new memory."""
        await self.initialize()
        
        query = f"""
            INSERT INTO {self.config.memories_table}
            (memory_id, user_id, content, memory_type, confidence_score, 
             is_active, tags, data, embedding, headline, subtitle,
             created_at, updated_at, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING memory_id
        """
        
        try:
            result = await self._manager.fetchval(
                query,
                item.memory_id,
                item.user_id,
                item.content,
                item.type.value,
                item.confidence_score,
                item.is_active,
                item.tags,
                json.dumps(self._serialize_memory(item)),
                item.embedding,
                item.headline,
                item.subtitle,
                item.created_at,
                item.updated_at,
                item.expires_at,
            )
            return result
        except Exception as e:
            raise StorageError(f"Failed to add memory: {e}") from e
    
    async def update(self, item: MemoryItem) -> bool:
        """Update an existing memory."""
        await self.initialize()
        
        query = f"""
            UPDATE {self.config.memories_table}
            SET content = $2,
                memory_type = $3,
                confidence_score = $4,
                is_active = $5,
                tags = $6,
                data = $7,
                embedding = $8,
                headline = $9,
                subtitle = $10,
                updated_at = $11,
                expires_at = $12
            WHERE memory_id = $1
            RETURNING memory_id
        """
        
        result = await self._manager.fetchval(
            query,
            item.memory_id,
            item.content,
            item.type.value,
            item.confidence_score,
            item.is_active,
            item.tags,
            json.dumps(self._serialize_memory(item)),
            item.embedding,
            item.headline,
            item.subtitle,
            datetime.now(timezone.utc),
            item.expires_at,
        )
        
        return result is not None
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        await self.initialize()
        
        query = f"""
            DELETE FROM {self.config.memories_table}
            WHERE memory_id = $1
            RETURNING memory_id
        """
        
        result = await self._manager.fetchval(query, memory_id)
        return result is not None
    
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory by ID."""
        await self.initialize()
        
        query = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE memory_id = $1
        """
        
        row = await self._manager.fetchrow(query, memory_id)
        
        if not row:
            return None
        
        item = self._deserialize_memory(row)
        item.record_access()
        
        return item
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """Get all memories for a user."""
        await self.initialize()
        
        conditions = ["user_id = $1"]
        if not include_inactive:
            conditions.append("is_active = TRUE")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT $2
        """
        
        rows = await self._manager.fetch(query, user_id, limit)
        
        return [self._deserialize_memory(row) for row in rows]
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        await self.initialize()
        
        query = f"""
            SELECT COUNT(*) FROM {self.config.memories_table}
            WHERE user_id = $1 AND is_active = TRUE
        """
        
        return await self._manager.fetchval(query, user_id)
    
    async def clear(self) -> None:
        """Clear all memories (for testing)."""
        await self.initialize()
        
        query = f"TRUNCATE TABLE {self.config.memories_table}"
        await self._manager.execute(query)
    
    async def keyword_search(
        self,
        user_id: str,
        keywords: List[str],
        limit: int = 10,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> List[MemoryItem]:
        """Search memories by keyword overlap on the ``keywords`` field.

        Loads candidate memories for the user and scores them by how many
        of the requested keywords appear in the memory's ``keywords`` list.
        Optionally filters by structured metadata (persons, locations, topics).
        """
        await self.initialize()

        query = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE user_id = $1
              AND is_active = TRUE
              AND (expires_at IS NULL OR expires_at > $2)
        """

        rows = await self._manager.fetch(
            query, user_id, datetime.now(timezone.utc)
        )

        if not rows:
            return []

        candidates = [self._deserialize_memory(row) for row in rows]

        kw_set = {k.lower() for k in keywords}

        if filters:
            for field_name, values in filters.items():
                val_set = {v.lower() for v in values}
                candidates = [
                    m for m in candidates
                    if val_set & {v.lower() for v in getattr(m, field_name, [])}
                ]

        scored = []
        for m in candidates:
            mem_kw = {k.lower() for k in m.keywords}
            overlap = len(kw_set & mem_kw)
            if overlap > 0:
                scored.append((overlap, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    async def search_by_embedding(
        self,
        user_id: str,
        embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> List[MemoryItem]:
        """
        Search memories by embedding similarity.
        
        Note: Requires pgvector extension to be installed.
        
        Args:
            user_id: User to search memories for
            embedding: Query embedding vector
            limit: Maximum results to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of memories ordered by similarity
        """
        await self.initialize()
        
        # This uses pgvector's cosine distance operator
        # Requires: CREATE EXTENSION vector;
        query = f"""
            SELECT data, 1 - (embedding <=> $2::vector) as similarity
            FROM {self.config.memories_table}
            WHERE user_id = $1 
              AND is_active = TRUE
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> $2::vector) >= $3
            ORDER BY embedding <=> $2::vector
            LIMIT $4
        """
        
        try:
            rows = await self._manager.fetch(
                query, user_id, embedding, min_similarity, limit
            )
            return [self._deserialize_memory(row) for row in rows]
        except Exception as e:
            # pgvector not installed, fall back to no results
            if "operator does not exist" in str(e):
                return []
            raise

