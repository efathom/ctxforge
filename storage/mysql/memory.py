"""
MySQL memory store implementation.

Provides persistent memory storage with JSON and full-text search support.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.exceptions import StorageError
from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.storage.connection import MySQLConfig, MySQLConnectionManager
from ctxforge.utils.math import cosine_similarity

# SQL statements for table creation
CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    memory_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL DEFAULT 0.5,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    tags JSON NOT NULL,
    data JSON NOT NULL,
    embedding JSON,
    headline VARCHAR(150),
    subtitle VARCHAR(300),
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    expires_at DATETIME(6),
    INDEX idx_user_id (user_id),
    INDEX idx_type (memory_type),
    INDEX idx_is_active (is_active),
    INDEX idx_created_at (created_at),
    INDEX idx_headline (headline(100)),
    FULLTEXT INDEX idx_content_fts (content)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# Migration to add headline/subtitle columns to existing tables
ADD_HEADLINE_COLUMNS = """
ALTER TABLE {table_name}
ADD COLUMN IF NOT EXISTS headline VARCHAR(150),
ADD COLUMN IF NOT EXISTS subtitle VARCHAR(300),
ADD INDEX IF NOT EXISTS idx_headline (headline(100))
"""


@registry.register_memory_store("mysql")
class MySQLMemoryStore(IMemoryStore):
    """
    MySQL-based memory store.
    
    Features:
    - JSON storage for flexible memory data
    - Full-text search using MySQL's FULLTEXT index
    - JSON array for tags (queried via JSON_CONTAINS)
    - JSON array for embeddings (application-level similarity)
    
    Note: Unlike PostgreSQL with pgvector, MySQL doesn't have native
    vector similarity search. Embedding search falls back to loading
    candidates and computing similarity in Python.
    
    Example:
        from ctxforge.storage.connection import MySQLConfig
        
        config = MySQLConfig(host="localhost", database="ctxforge")
        store = MySQLMemoryStore(config)
        await store.initialize()
    """
    
    def __init__(
        self,
        config: Optional[MySQLConfig] = None,
        connection_manager: Optional[MySQLConnectionManager] = None,
    ):
        """
        Initialize the MySQL memory store.
        
        Args:
            config: MySQL configuration
            connection_manager: Optional pre-existing connection manager
        """
        self.config = config or MySQLConfig()
        self._manager = connection_manager or MySQLConnectionManager(self.config)
        self._owns_connection = connection_manager is None
        self._initialized = False
    
    async def connect(self) -> None:
        """Connect to MySQL."""
        if not self._manager.is_connected:
            await self._manager.connect()
    
    async def disconnect(self) -> None:
        """Disconnect from MySQL."""
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
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = %s AND COLUMN_NAME = 'headline'
            """
            result = await self._manager.fetchone(
                check_sql, (self.config.memories_table,)
            )
            
            if not result:
                # Add headline column
                await self._manager.execute(
                    f"ALTER TABLE {self.config.memories_table} "
                    "ADD COLUMN headline VARCHAR(150)"
                )
                # Add subtitle column
                await self._manager.execute(
                    f"ALTER TABLE {self.config.memories_table} "
                    "ADD COLUMN subtitle VARCHAR(300)"
                )
                # Add index on headline
                await self._manager.execute(
                    f"ALTER TABLE {self.config.memories_table} "
                    "ADD INDEX idx_headline (headline(100))"
                )
        except Exception:
            # Ignore migration errors (columns may already exist)
            pass
    
    def _serialize_memory(self, item: MemoryItem) -> Dict[str, Any]:
        """Serialize memory to dict for JSON storage."""
        return json.loads(item.model_dump_json())
    
    def _deserialize_memory(self, row: Dict[str, Any]) -> MemoryItem:
        """Deserialize memory from database row."""
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return MemoryItem.model_validate(data)
    
    async def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search for memories using full-text search."""
        await self.initialize()
        
        # Build dynamic query
        conditions = ["user_id = %s", "is_active = TRUE"]
        params: List[Any] = [query.user_id]
        
        # Filter by types
        if query.types:
            type_values = [t.value for t in query.types]
            placeholders = ", ".join(["%s"] * len(type_values))
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(type_values)
        
        # Filter by tags (using JSON_CONTAINS)
        if query.tags:
            for tag in query.tags:
                conditions.append("JSON_CONTAINS(tags, %s)")
                params.append(json.dumps(tag))
        
        # Filter by confidence
        conditions.append("confidence_score >= %s")
        params.append(query.min_confidence)
        
        # Filter expired
        conditions.append("(expires_at IS NULL OR expires_at > %s)")
        params.append(datetime.now(timezone.utc))
        
        # Full-text search
        order_by = "created_at DESC"
        if query.query_text:
            # Use MATCH...AGAINST for full-text search
            # Note: MATCH returns 0 for short words, so we also add a LIKE fallback
            conditions.append(
                "(MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) OR content LIKE %s)"
            )
            params.append(query.query_text)
            params.append(f"%{query.query_text}%")
            
            # Order by relevance
            order_by = (
                "MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) DESC, "
                "created_at DESC"
            )
            params.append(query.query_text)
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT %s OFFSET %s
        """
        params.extend([query.limit, query.offset])
        
        rows = await self._manager.fetchall(sql, tuple(params))
        
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            await self._manager.execute(
                query,
                (
                    item.memory_id,
                    item.user_id,
                    item.content,
                    item.type.value,
                    item.confidence_score,
                    item.is_active,
                    json.dumps(item.tags),
                    json.dumps(self._serialize_memory(item)),
                    json.dumps(item.embedding) if item.embedding else None,
                    item.headline,
                    item.subtitle,
                    item.created_at,
                    item.updated_at,
                    item.expires_at,
                ),
            )
            return item.memory_id
        except Exception as e:
            raise StorageError(f"Failed to add memory: {e}") from e
    
    async def save(self, item: MemoryItem) -> str:
        """Save a memory (alias for add with upsert behavior)."""
        await self.initialize()
        
        query = f"""
            INSERT INTO {self.config.memories_table}
            (memory_id, user_id, content, memory_type, confidence_score, 
             is_active, tags, data, embedding, headline, subtitle,
             created_at, updated_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                content = VALUES(content),
                memory_type = VALUES(memory_type),
                confidence_score = VALUES(confidence_score),
                is_active = VALUES(is_active),
                tags = VALUES(tags),
                data = VALUES(data),
                embedding = VALUES(embedding),
                headline = VALUES(headline),
                subtitle = VALUES(subtitle),
                updated_at = VALUES(updated_at),
                expires_at = VALUES(expires_at)
        """
        
        try:
            await self._manager.execute(
                query,
                (
                    item.memory_id,
                    item.user_id,
                    item.content,
                    item.type.value,
                    item.confidence_score,
                    item.is_active,
                    json.dumps(item.tags),
                    json.dumps(self._serialize_memory(item)),
                    json.dumps(item.embedding) if item.embedding else None,
                    item.headline,
                    item.subtitle,
                    item.created_at,
                    item.updated_at,
                    item.expires_at,
                ),
            )
            return item.memory_id
        except Exception as e:
            raise StorageError(f"Failed to save memory: {e}") from e
    
    async def update(self, item: MemoryItem) -> bool:
        """Update an existing memory."""
        await self.initialize()
        
        query = f"""
            UPDATE {self.config.memories_table}
            SET content = %s,
                memory_type = %s,
                confidence_score = %s,
                is_active = %s,
                tags = %s,
                data = %s,
                embedding = %s,
                headline = %s,
                subtitle = %s,
                updated_at = %s,
                expires_at = %s
            WHERE memory_id = %s
        """
        
        affected = await self._manager.execute(
            query,
            (
                item.content,
                item.type.value,
                item.confidence_score,
                item.is_active,
                json.dumps(item.tags),
                json.dumps(self._serialize_memory(item)),
                json.dumps(item.embedding) if item.embedding else None,
                item.headline,
                item.subtitle,
                datetime.now(timezone.utc),
                item.expires_at,
                item.memory_id,
            ),
        )
        
        return affected > 0
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        await self.initialize()
        
        query = f"""
            DELETE FROM {self.config.memories_table}
            WHERE memory_id = %s
        """
        
        affected = await self._manager.execute(query, (memory_id,))
        return affected > 0
    
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory by ID."""
        await self.initialize()
        
        query = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE memory_id = %s
        """
        
        row = await self._manager.fetchone(query, (memory_id,))
        
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
        
        conditions = ["user_id = %s"]
        params: List[Any] = [user_id]
        
        if not include_inactive:
            conditions.append("is_active = TRUE")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT data FROM {self.config.memories_table}
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """
        params.append(limit)
        
        rows = await self._manager.fetchall(query, tuple(params))
        
        return [self._deserialize_memory(row) for row in rows]
    
    async def count(self, user_id: str) -> int:
        """Count memories for a user."""
        await self.initialize()
        
        query = f"""
            SELECT COUNT(*) FROM {self.config.memories_table}
            WHERE user_id = %s AND is_active = TRUE
        """
        
        result = await self._manager.fetchval(query, (user_id,))
        return result or 0
    
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
            WHERE user_id = %s
              AND is_active = TRUE
              AND (expires_at IS NULL OR expires_at > %s)
        """

        rows = await self._manager.fetchall(
            query, (user_id, datetime.now(timezone.utc))
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
        
        Note: MySQL doesn't have native vector similarity like pgvector.
        This implementation loads candidates with embeddings and computes
        cosine similarity in Python.
        
        For production use with large datasets, consider:
        - Using a dedicated vector store (Pinecone, Qdrant, etc.)
        - Adding a vector search extension if available
        
        Args:
            user_id: User to search memories for
            embedding: Query embedding vector
            limit: Maximum results to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of memories ordered by similarity
        """
        await self.initialize()
        
        # Load all memories with embeddings for this user
        query = f"""
            SELECT data, embedding FROM {self.config.memories_table}
            WHERE user_id = %s 
              AND is_active = TRUE
              AND embedding IS NOT NULL
        """
        
        rows = await self._manager.fetchall(query, (user_id,))
        
        if not rows:
            return []
        
        # Compute similarities in Python
        results = []
        for row in rows:
            stored_embedding = row.get("embedding")
            if isinstance(stored_embedding, str):
                stored_embedding = json.loads(stored_embedding)
            
            if not stored_embedding:
                continue
            
            # Cosine similarity
            similarity = cosine_similarity(embedding, stored_embedding)
            
            if similarity >= min_similarity:
                memory = self._deserialize_memory(row)
                results.append((similarity, memory))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in results[:limit]]
