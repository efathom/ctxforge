"""
PostgreSQL scoped memory store implementation.

Provides persistent storage for hierarchical scoped memories.
"""
import json
from datetime import datetime, timezone
from typing import Any, List, Optional

from ctxforge.core.exceptions import StorageError
from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.engine.registry import registry
from ctxforge.storage.connection import PostgresConfig, PostgresConnectionManager

# SQL statements for table creation
CREATE_SCOPED_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id VARCHAR(64) PRIMARY KEY,
    scope VARCHAR(20) NOT NULL,
    scope_id VARCHAR(128) NOT NULL,
    category VARCHAR(32) NOT NULL,
    key VARCHAR(256) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{{}}',
    priority INT DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(scope, scope_id, key)
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_scope_lookup ON {table_name}(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_{table_name}_category ON {table_name}(category);
"""


@registry.register_scoped_memory_store("postgres")
class PostgresScopedMemoryStore:
    """
    PostgreSQL-based scoped memory store.

    Features:
    - Unique key constraint per (scope, scope_id, key)
    - JSONB metadata storage
    - Indexed by scope and category
    """

    def __init__(
        self,
        config: Optional[PostgresConfig] = None,
        connection_manager: Optional[PostgresConnectionManager] = None,
        table_name: str = "scoped_memories",
    ):
        """
        Initialize the PostgreSQL scoped memory store.

        Args:
            config: PostgreSQL configuration
            connection_manager: Optional pre-existing connection manager
            table_name: Name of the table to use
        """
        self.config = config or PostgresConfig()
        self._manager = connection_manager or PostgresConnectionManager(self.config)
        self._owns_connection = connection_manager is None
        self._table_name = table_name
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

        sql = CREATE_SCOPED_MEMORIES_TABLE.format(table_name=self._table_name)
        await self._manager.execute(sql)
        self._initialized = True

    def _deserialize_memory(self, row: Any) -> ScopedMemory:
        """Deserialize memory from database row."""
        # asyncpg returns Record objects, access by key
        metadata = row["metadata"] or {}
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
            priority=row["priority"] or 0,
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
            (id, scope, scope_id, category, key, content, metadata, priority,
             created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (scope, scope_id, key) DO UPDATE SET
                id = EXCLUDED.id,
                category = EXCLUDED.category,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                priority = EXCLUDED.priority,
                updated_at = EXCLUDED.updated_at
        """

        try:
            await self._manager.execute(
                query,
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
            WHERE scope = $1 AND scope_id = $2 AND key = $3
        """

        row = await self._manager.fetchrow(query, scope.value, scope_id, key)

        if not row:
            return None

        return self._deserialize_memory(row)

    async def get_by_id(self, memory_id: str) -> Optional[ScopedMemory]:
        """Get a memory by its unique ID."""
        await self.initialize()

        query = f"SELECT * FROM {self._table_name} WHERE id = $1"
        row = await self._manager.fetchrow(query, memory_id)

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

        if category is not None:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE scope = $1 AND scope_id = $2 AND category = $3
                ORDER BY priority DESC, key ASC
            """
            rows = await self._manager.fetch(query, scope.value, scope_id, category.value)
        else:
            query = f"""
                SELECT * FROM {self._table_name}
                WHERE scope = $1 AND scope_id = $2
                ORDER BY priority DESC, key ASC
            """
            rows = await self._manager.fetch(query, scope.value, scope_id)

        return [self._deserialize_memory(row) for row in rows]

    async def query(self, query_obj: ScopedMemoryQuery) -> List[ScopedMemory]:
        """Query memories across multiple scopes based on query parameters."""
        await self.initialize()

        scope_ids = query_obj.get_scope_ids()
        if not scope_ids:
            return []

        # Build dynamic query with parameters
        conditions = []
        params: List[Any] = []
        param_idx = 1

        # Build scope conditions
        scope_parts = []
        for scope, sid in scope_ids.items():
            scope_parts.append(f"(scope = ${param_idx} AND scope_id = ${param_idx + 1})")
            params.extend([scope.value, sid])
            param_idx += 2

        conditions.append(f"({' OR '.join(scope_parts)})")

        # Category filter
        if query_obj.categories:
            category_values = [c.value for c in query_obj.categories]
            conditions.append(f"category = ANY(${param_idx})")
            params.append(category_values)
            param_idx += 1

        where_clause = " AND ".join(conditions)

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
            ORDER BY scope_priority DESC, priority DESC, key ASC
        """

        rows = await self._manager.fetch(query, *params)
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
            WHERE scope = $1 AND scope_id = $2 AND key = $3
        """

        result = await self._manager.execute(query, scope.value, scope_id, key)
        # asyncpg returns "DELETE n" string
        return "DELETE 0" not in str(result)

    async def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory by its unique ID. Returns True if deleted."""
        await self.initialize()

        query = f"DELETE FROM {self._table_name} WHERE id = $1"
        result = await self._manager.execute(query, memory_id)
        return "DELETE 0" not in str(result)

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
        param_idx = 1

        if scope is not None:
            conditions.append(f"scope = ${param_idx}")
            params.append(scope.value)
            param_idx += 1

        if scope_id is not None:
            conditions.append(f"scope_id = ${param_idx}")
            params.append(scope_id)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"SELECT COUNT(*) FROM {self._table_name} WHERE {where_clause}"
        result = await self._manager.fetchval(query, *params)

        return result or 0

    async def clear(
        self,
        scope: Optional[MemoryScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear memories, optionally filtered by scope. Returns count deleted."""
        await self.initialize()

        # First get count
        count = await self.count(scope, scope_id)

        conditions = []
        params: List[Any] = []
        param_idx = 1

        if scope is not None:
            conditions.append(f"scope = ${param_idx}")
            params.append(scope.value)
            param_idx += 1

        if scope_id is not None:
            conditions.append(f"scope_id = ${param_idx}")
            params.append(scope_id)
            param_idx += 1

        if conditions:
            where_clause = " AND ".join(conditions)
            query = f"DELETE FROM {self._table_name} WHERE {where_clause}"
            await self._manager.execute(query, *params)
        else:
            query = f"TRUNCATE TABLE {self._table_name}"
            await self._manager.execute(query)

        return count
