"""
MySQL scoped memory store implementation.

Provides persistent storage for hierarchical scoped memories.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.exceptions import StorageError
from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.engine.registry import registry
from ctxforge.storage.connection import MySQLConfig, MySQLConnectionManager

# SQL statements for table creation
CREATE_SCOPED_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id VARCHAR(64) PRIMARY KEY,
    scope VARCHAR(20) NOT NULL,
    scope_id VARCHAR(128) NOT NULL,
    category VARCHAR(32) NOT NULL,
    `key` VARCHAR(256) NOT NULL,
    content TEXT NOT NULL,
    metadata JSON,
    priority INT DEFAULT 0,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    UNIQUE KEY uk_scope_key (scope, scope_id, `key`),
    INDEX idx_scope_lookup (scope, scope_id),
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


@registry.register_scoped_memory_store("mysql")
class MySQLScopedMemoryStore:
    """
    MySQL-based scoped memory store.

    Features:
    - Unique key constraint per (scope, scope_id, key)
    - JSON metadata storage
    - Indexed by scope and category
    """

    def __init__(
        self,
        config: Optional[MySQLConfig] = None,
        connection_manager: Optional[MySQLConnectionManager] = None,
        table_name: str = "scoped_memories",
    ):
        """
        Initialize the MySQL scoped memory store.

        Args:
            config: MySQL configuration
            connection_manager: Optional pre-existing connection manager
            table_name: Name of the table to use
        """
        self.config = config or MySQLConfig()
        self._manager = connection_manager or MySQLConnectionManager(self.config)
        self._owns_connection = connection_manager is None
        self._table_name = table_name
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

        sql = CREATE_SCOPED_MEMORIES_TABLE.format(table_name=self._table_name)
        await self._manager.execute(sql)
        self._initialized = True

    def _serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to string."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def _deserialize_memory(self, row: Dict[str, Any]) -> ScopedMemory:
        """Deserialize memory from database row."""
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        created_at = row["created_at"]
        updated_at = row["updated_at"]

        return ScopedMemory(
            id=row["id"],
            scope=MemoryScope(row["scope"]),
            scope_id=row["scope_id"],
            category=MemoryCategory(row["category"]),
            key=row["key"],
            content=row["content"],
            metadata=metadata,
            priority=row.get("priority", 0),
            created_at=(created_at if isinstance(created_at, datetime)
                        else datetime.fromisoformat(str(created_at))),
            updated_at=(updated_at if isinstance(updated_at, datetime)
                        else datetime.fromisoformat(str(updated_at))),
        )

    async def save(self, memory: ScopedMemory) -> None:
        """Save a scoped memory. Updates if key already exists in scope."""
        await self.initialize()

        query = f"""
            INSERT INTO {self._table_name}
            (id, scope, scope_id, category, `key`, content, metadata, priority,
             created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                id = VALUES(id),
                category = VALUES(category),
                content = VALUES(content),
                metadata = VALUES(metadata),
                priority = VALUES(priority),
                updated_at = VALUES(updated_at)
        """

        try:
            await self._manager.execute(
                query,
                (
                    memory.id,
                    memory.scope.value,
                    memory.scope_id,
                    memory.category.value,
                    memory.key,
                    memory.content,
                    json.dumps(memory.metadata),
                    memory.priority,
                    memory.created_at,
                    memory.updated_at,
                ),
            )
        except Exception as e:
            raise StorageError(f"Failed to save scoped memory: {e}") from e

    async def get(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> Optional[ScopedMemory]:
        """Get a specific memory by scope, scope_id, and key."""
        await self.initialize()

        query = f"""
            SELECT * FROM {self._table_name}
            WHERE scope = %s AND scope_id = %s AND `key` = %s
        """

        row = await self._manager.fetchone(query, (scope.value, scope_id, key))

        if not row:
            return None

        return self._deserialize_memory(row)

    async def get_by_id(self, memory_id: str) -> Optional[ScopedMemory]:
        """Get a memory by its unique ID."""
        await self.initialize()

        query = f"SELECT * FROM {self._table_name} WHERE id = %s"
        row = await self._manager.fetchone(query, (memory_id,))

        if not row:
            return None

        return self._deserialize_memory(row)

    async def list_by_scope(
        self,
        scope: MemoryScope,
        scope_id: str,
        category: Optional[MemoryCategory] = None
    ) -> List[ScopedMemory]:
        """List all memories for a given scope and scope_id."""
        await self.initialize()

        conditions = ["scope = %s", "scope_id = %s"]
        params: List[Any] = [scope.value, scope_id]

        if category is not None:
            conditions.append("category = %s")
            params.append(category.value)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT * FROM {self._table_name}
            WHERE {where_clause}
            ORDER BY priority DESC, `key` ASC
        """

        rows = await self._manager.fetchall(query, tuple(params))
        return [self._deserialize_memory(row) for row in rows]

    async def query(self, query_obj: ScopedMemoryQuery) -> List[ScopedMemory]:
        """Query memories across multiple scopes based on query parameters."""
        await self.initialize()

        scope_ids = query_obj.get_scope_ids()
        if not scope_ids:
            return []

        # Build OR conditions for each scope
        scope_conditions = []
        params: List[Any] = []

        for scope, sid in scope_ids.items():
            scope_conditions.append("(scope = %s AND scope_id = %s)")
            params.extend([scope.value, sid])

        where_parts = [f"({' OR '.join(scope_conditions)})"]

        # Category filter
        if query_obj.categories:
            category_placeholders = ", ".join(["%s"] * len(query_obj.categories))
            where_parts.append(f"category IN ({category_placeholders})")
            params.extend([c.value for c in query_obj.categories])

        where_clause = " AND ".join(where_parts)

        # Order by scope priority (session=2 > project=1 > global=0), then priority, then key
        query = f"""
            SELECT *,
                CASE scope 
                    WHEN 'session' THEN 2 
                    WHEN 'project' THEN 1 
                    ELSE 0 
                END as scope_priority
            FROM {self._table_name}
            WHERE {where_clause}
            ORDER BY scope_priority DESC, priority DESC, `key` ASC
        """

        rows = await self._manager.fetchall(query, tuple(params))
        return [self._deserialize_memory(row) for row in rows]

    async def delete(
        self,
        scope: MemoryScope,
        scope_id: str,
        key: str
    ) -> bool:
        """Delete a specific memory. Returns True if deleted."""
        await self.initialize()

        query = f"""
            DELETE FROM {self._table_name}
            WHERE scope = %s AND scope_id = %s AND `key` = %s
        """

        affected = await self._manager.execute(
            query, (scope.value, scope_id, key)
        )
        return affected > 0

    async def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory by its unique ID. Returns True if deleted."""
        await self.initialize()

        query = f"DELETE FROM {self._table_name} WHERE id = %s"
        affected = await self._manager.execute(query, (memory_id,))
        return affected > 0

    async def update(self, memory: ScopedMemory) -> None:
        """Update an existing memory."""
        memory.updated_at = datetime.now(timezone.utc)
        await self.save(memory)

    async def count(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count memories, optionally filtered by scope."""
        await self.initialize()

        conditions = []
        params: List[Any] = []

        if scope is not None:
            conditions.append("scope = %s")
            params.append(scope.value)

        if scope_id is not None:
            conditions.append("scope_id = %s")
            params.append(scope_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"SELECT COUNT(*) as cnt FROM {self._table_name} WHERE {where_clause}"
        row = await self._manager.fetchone(query, tuple(params))

        return row["cnt"] if row else 0

    async def clear(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear memories, optionally filtered by scope. Returns count deleted."""
        await self.initialize()

        conditions = []
        params: List[Any] = []

        if scope is not None:
            conditions.append("scope = %s")
            params.append(scope.value)

        if scope_id is not None:
            conditions.append("scope_id = %s")
            params.append(scope_id)

        if conditions:
            where_clause = " AND ".join(conditions)
            query = f"DELETE FROM {self._table_name} WHERE {where_clause}"
        else:
            # Get count before truncate
            count = await self.count()
            query = f"TRUNCATE TABLE {self._table_name}"
            await self._manager.execute(query)
            return count

        affected = await self._manager.execute(query, tuple(params))
        return affected
