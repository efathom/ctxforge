"""
PostgreSQL expertise store implementation.

Provides persistent expertise storage with JSONB, full-text search, and GIN-indexed operations.
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
from ctxforge.storage.connection import PostgresConfig, PostgresConnectionManager


@registry.register_expertise_store("postgres")
class PostgresExpertiseStore(IExpertiseStore):
    """
    PostgreSQL-based expertise store.
    
    Uses the same PostgresConnectionManager as other PostgreSQL stores.
    Stores expertise and items in separate tables with full-text search support.
    
    Example:
        from ctxforge.storage.connection import PostgresConfig
        
        config = PostgresConfig(host="localhost", database="ctxforge")
        store = PostgresExpertiseStore(config)
        await store.initialize()
        
        expertise = Expertise(expertise_id="test", name="Test")
        await store.save(expertise)
    """
    
    def __init__(
        self,
        config: Optional[PostgresConfig] = None,
        connection_manager: Optional[PostgresConnectionManager] = None,
    ):
        """
        Initialize the PostgreSQL expertise store.
        
        Args:
            config: PostgreSQL configuration
            connection_manager: Optional pre-existing connection manager
        """
        self.config = config or PostgresConfig()
        self._conn_manager = connection_manager or PostgresConnectionManager(self.config)
        self._owns_connection = connection_manager is None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the store and create tables if needed."""
        if self._initialized:
            return
        
        if not self._conn_manager.is_connected:
            await self._conn_manager.connect()
        
        await self._create_tables()
        self._initialized = True
    
    async def _create_tables(self) -> None:
        """Create the required database tables if they don't exist."""
        async with self._conn_manager.acquire() as conn:
            # Create expertise table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS expertise (
                    expertise_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT,
                    description TEXT DEFAULT '',
                    version INTEGER DEFAULT 1,
                    token_budget INTEGER,
                    next_item_id INTEGER DEFAULT 1,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Migration: add description column if missing
            try:
                await conn.execute(
                    "ALTER TABLE expertise ADD COLUMN IF NOT EXISTS description TEXT DEFAULT ''"
                )
            except Exception:
                pass
            
            # Create expertise_items table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS expertise_items (
                    item_id TEXT NOT NULL,
                    expertise_id TEXT NOT NULL REFERENCES expertise(expertise_id) ON DELETE CASCADE,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    helpful_count INTEGER DEFAULT 0,
                    harmful_count INTEGER DEFAULT 0,
                    source TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    embedding FLOAT8[],
                    search_vector TSVECTOR,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Migration / safety: ensure PK is (expertise_id, item_id) rather than global item_id.
            # Older versions used item_id as the PK, which causes collisions across different expertises.
            try:
                await conn.execute("ALTER TABLE expertise_items DROP CONSTRAINT expertise_items_pkey")
            except Exception:
                # Constraint may not exist / may already be correct
                pass
            try:
                await conn.execute("ALTER TABLE expertise_items ADD PRIMARY KEY (expertise_id, item_id)")
            except Exception:
                # Already has the correct PK (or table not created yet)
                pass
            
            # Create expertise_usage_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS expertise_usage_logs (
                    log_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    expertise_id TEXT NOT NULL,
                    items_used TEXT[],
                    feedback JSONB,
                    outcome TEXT,
                    context_summary TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expertise_items_expertise_id 
                ON expertise_items(expertise_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expertise_items_section 
                ON expertise_items(expertise_id, section)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expertise_items_search 
                ON expertise_items USING GIN(search_vector)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expertise_usage_logs_expertise 
                ON expertise_usage_logs(expertise_id)
            """)
    
    async def close(self) -> None:
        """Close the connection if we own it."""
        if self._owns_connection and self._conn_manager:
            await self._conn_manager.disconnect()
    
    async def save(self, expertise: Expertise) -> None:
        """
        Save or update an expertise knowledge base.
        
        Uses upsert to handle both insert and update.
        Also syncs all items to the expertise_items table.
        """
        try:
            async with self._conn_manager.acquire() as conn:
                async with conn.transaction():
                    # Upsert expertise
                    await conn.execute(
                        """
                        INSERT INTO expertise (
                            expertise_id, name, domain, description, version,
                            token_budget, next_item_id, metadata, created_at,
                            updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (expertise_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            domain = EXCLUDED.domain,
                            description = EXCLUDED.description,
                            version = EXCLUDED.version,
                            token_budget = EXCLUDED.token_budget,
                            next_item_id = EXCLUDED.next_item_id,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at
                        """,
                        expertise.expertise_id,
                        expertise.name,
                        expertise.domain,
                        expertise.description,
                        expertise.version,
                        expertise.token_budget,
                        expertise.next_item_id,
                        json.dumps(expertise.metadata),
                        expertise.created_at,
                        datetime.now(timezone.utc).replace(tzinfo=None),
                    )
                    
                    # Sync items - delete existing and re-insert
                    await conn.execute(
                        "DELETE FROM expertise_items WHERE expertise_id = $1",
                        expertise.expertise_id,
                    )
                    
                    for item in expertise.items:
                        await self._insert_item(conn, expertise.expertise_id, item)
        
        except Exception as e:
            raise StorageError(f"Failed to save expertise: {e}") from e
    
    async def _insert_item(self, conn, expertise_id: str, item: ExpertiseItem) -> None:
        """Insert a single item."""
        await conn.execute(
            """
            INSERT INTO expertise_items (
                item_id, expertise_id, section, content,
                helpful_count, harmful_count, source, is_active,
                embedding, search_vector, metadata, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9,
                to_tsvector('english', $4), $10, $11, $12
            )
            """,
            item.item_id,
            expertise_id,
            item.section.value,
            item.content,
            item.helpful_count,
            item.harmful_count,
            item.source,
            item.is_active,
            item.embedding,
            json.dumps(item.metadata),
            item.created_at,
            item.updated_at,
        )
    
    async def load(self, expertise_id: str) -> Optional[Expertise]:
        """Load an expertise by ID."""
        try:
            async with self._conn_manager.acquire() as conn:
                # Load expertise
                row = await conn.fetchrow(
                    """
                    SELECT expertise_id, name, domain, description, version,
                           token_budget, next_item_id, metadata, created_at,
                           updated_at
                    FROM expertise
                    WHERE expertise_id = $1
                    """,
                    expertise_id,
                )
                
                if not row:
                    return None
                
                # Load items
                item_rows = await conn.fetch(
                    """
                    SELECT item_id, section, content, helpful_count, harmful_count,
                           source, is_active, embedding, metadata, created_at, updated_at
                    FROM expertise_items
                    WHERE expertise_id = $1
                    ORDER BY created_at
                    """,
                    expertise_id,
                )
                
                items = [
                    ExpertiseItem(
                        item_id=r["item_id"],
                        section=ExpertiseSection(r["section"]),
                        content=r["content"],
                        helpful_count=r["helpful_count"],
                        harmful_count=r["harmful_count"],
                        source=r["source"],
                        is_active=r["is_active"],
                        embedding=list(r["embedding"]) if r["embedding"] else None,
                        metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                        created_at=r["created_at"],
                        updated_at=r["updated_at"],
                    )
                    for r in item_rows
                ]
                
                return Expertise(
                    expertise_id=row["expertise_id"],
                    name=row["name"],
                    domain=row["domain"],
                    description=row.get("description") or "",
                    version=row["version"],
                    token_budget=row["token_budget"],
                    next_item_id=row["next_item_id"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    items=items,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        
        except Exception as e:
            raise StorageError(f"Failed to load expertise: {e}") from e
    
    async def delete(self, expertise_id: str) -> bool:
        """Delete an expertise knowledge base."""
        try:
            async with self._conn_manager.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM expertise WHERE expertise_id = $1",
                    expertise_id,
                )
                return "DELETE 1" in result
        
        except Exception as e:
            raise StorageError(f"Failed to delete expertise: {e}") from e
    
    async def list_expertise(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Expertise]:
        """List expertise knowledge bases."""
        try:
            async with self._conn_manager.acquire() as conn:
                if domain:
                    rows = await conn.fetch(
                        """
                        SELECT expertise_id FROM expertise
                        WHERE domain = $1
                        ORDER BY updated_at DESC
                        LIMIT $2 OFFSET $3
                        """,
                        domain,
                        limit,
                        offset,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT expertise_id FROM expertise
                        ORDER BY updated_at DESC
                        LIMIT $1 OFFSET $2
                        """,
                        limit,
                        offset,
                    )
                
                # Load each expertise fully
                results = []
                for row in rows:
                    expertise = await self.load(row["expertise_id"])
                    if expertise:
                        results.append(expertise)
                
                return results
        
        except Exception as e:
            raise StorageError(f"Failed to list expertise: {e}") from e
    
    async def add_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """Add an item to an expertise."""
        try:
            async with self._conn_manager.acquire() as conn:
                await self._insert_item(conn, expertise_id, item)
                
                # Update expertise timestamp
                await conn.execute(
                    "UPDATE expertise SET updated_at = $1 WHERE expertise_id = $2",
                    datetime.now(timezone.utc).replace(tzinfo=None),
                    expertise_id,
                )
        
        except Exception as e:
            raise StorageError(f"Failed to add item: {e}") from e
    
    async def update_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """Update an existing item."""
        try:
            async with self._conn_manager.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE expertise_items SET
                        section = $1,
                        content = $2,
                        helpful_count = $3,
                        harmful_count = $4,
                        source = $5,
                        is_active = $6,
                        embedding = $7,
                        metadata = $8,
                        updated_at = $9,
                        search_vector = to_tsvector('english', $2)
                    WHERE item_id = $10 AND expertise_id = $11
                    """,
                    item.section.value,
                    item.content,
                    item.helpful_count,
                    item.harmful_count,
                    item.source,
                    item.is_active,
                    item.embedding,
                    json.dumps(item.metadata),
                    datetime.now(timezone.utc).replace(tzinfo=None),
                    item.item_id,
                    expertise_id,
                )
                
                # Update expertise timestamp
                await conn.execute(
                    "UPDATE expertise SET updated_at = $1 WHERE expertise_id = $2",
                    datetime.now(timezone.utc).replace(tzinfo=None),
                    expertise_id,
                )
        
        except Exception as e:
            raise StorageError(f"Failed to update item: {e}") from e
    
    async def remove_item(self, expertise_id: str, item_id: str) -> bool:
        """Remove an item from an expertise."""
        try:
            async with self._conn_manager.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM expertise_items WHERE item_id = $1 AND expertise_id = $2",
                    item_id,
                    expertise_id,
                )
                
                if "DELETE 1" in result:
                    await conn.execute(
                        "UPDATE expertise SET updated_at = $1 WHERE expertise_id = $2",
                        datetime.now(timezone.utc).replace(tzinfo=None),
                        expertise_id,
                    )
                    return True
                return False
        
        except Exception as e:
            raise StorageError(f"Failed to remove item: {e}") from e
    
    async def get_item(
        self,
        expertise_id: str,
        item_id: str,
    ) -> Optional[ExpertiseItem]:
        """Get a single item by ID."""
        try:
            async with self._conn_manager.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT item_id, section, content, helpful_count, harmful_count,
                           source, is_active, embedding, metadata, created_at, updated_at
                    FROM expertise_items
                    WHERE item_id = $1 AND expertise_id = $2
                    """,
                    item_id,
                    expertise_id,
                )
                
                if not row:
                    return None
                
                return ExpertiseItem(
                    item_id=row["item_id"],
                    section=ExpertiseSection(row["section"]),
                    content=row["content"],
                    helpful_count=row["helpful_count"],
                    harmful_count=row["harmful_count"],
                    source=row["source"],
                    is_active=row["is_active"],
                    embedding=list(row["embedding"]) if row["embedding"] else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        
        except Exception as e:
            raise StorageError(f"Failed to get item: {e}") from e
    
    async def get_items_by_section(
        self,
        expertise_id: str,
        section: ExpertiseSection,
    ) -> List[ExpertiseItem]:
        """Get all items in a section."""
        try:
            async with self._conn_manager.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT item_id, section, content, helpful_count, harmful_count,
                           source, is_active, embedding, metadata, created_at, updated_at
                    FROM expertise_items
                    WHERE expertise_id = $1 AND section = $2 AND is_active = TRUE
                    ORDER BY created_at
                    """,
                    expertise_id,
                    section.value,
                )
                
                return [
                    ExpertiseItem(
                        item_id=row["item_id"],
                        section=ExpertiseSection(row["section"]),
                        content=row["content"],
                        helpful_count=row["helpful_count"],
                        harmful_count=row["harmful_count"],
                        source=row["source"],
                        is_active=row["is_active"],
                        embedding=list(row["embedding"]) if row["embedding"] else None,
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]
        
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
        try:
            async with self._conn_manager.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE expertise_items SET
                        helpful_count = helpful_count + $1,
                        harmful_count = harmful_count + $2,
                        updated_at = $3
                    WHERE item_id = $4 AND expertise_id = $5
                    """,
                    helpful_delta,
                    harmful_delta,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                    item_id,
                    expertise_id,
                )
                
                await conn.execute(
                    "UPDATE expertise SET updated_at = $1 WHERE expertise_id = $2",
                    datetime.now(timezone.utc).replace(tzinfo=None),
                    expertise_id,
                )
        
        except Exception as e:
            raise StorageError(f"Failed to update item counts: {e}") from e
    
    async def search_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """Search items by text content using PostgreSQL full-text search."""
        try:
            async with self._conn_manager.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT item_id, section, content, helpful_count, harmful_count,
                           source, is_active, embedding, metadata, created_at, updated_at,
                           ts_rank(search_vector, plainto_tsquery('english', $1)) as rank
                    FROM expertise_items
                    WHERE expertise_id = $2
                      AND is_active = TRUE
                      AND search_vector @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $3
                    """,
                    query,
                    expertise_id,
                    limit,
                )
                
                return [
                    ExpertiseItem(
                        item_id=row["item_id"],
                        section=ExpertiseSection(row["section"]),
                        content=row["content"],
                        helpful_count=row["helpful_count"],
                        harmful_count=row["harmful_count"],
                        source=row["source"],
                        is_active=row["is_active"],
                        embedding=list(row["embedding"]) if row["embedding"] else None,
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]
        
        except Exception as e:
            raise StorageError(f"Failed to search items: {e}") from e
    
    async def log_usage(self, log: ExpertiseUsageLog) -> None:
        """Log expertise usage in a turn."""
        try:
            async with self._conn_manager.acquire() as conn:
                # Convert feedback to JSON
                feedback_json = {
                    item_id: feedback.value
                    for item_id, feedback in log.feedback.items()
                }
                
                await conn.execute(
                    """
                    INSERT INTO expertise_usage_logs (
                        log_id, session_id, expertise_id, items_used,
                        feedback, outcome, context_summary, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    log.log_id,
                    log.session_id,
                    log.expertise_id,
                    log.items_used,
                    json.dumps(feedback_json),
                    log.outcome.value if log.outcome else None,
                    log.context_summary,
                    log.timestamp,
                )
        
        except Exception as e:
            raise StorageError(f"Failed to log usage: {e}") from e
    
    async def get_usage_stats(self, expertise_id: str) -> Dict[str, Any]:
        """Get usage statistics for an expertise."""
        try:
            async with self._conn_manager.acquire() as conn:
                # Get total uses
                total_row = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM expertise_usage_logs WHERE expertise_id = $1",
                    expertise_id,
                )
                total_uses = total_row["count"] if total_row else 0
                
                # Get outcome counts
                outcome_rows = await conn.fetch(
                    """
                    SELECT outcome, COUNT(*) as count
                    FROM expertise_usage_logs
                    WHERE expertise_id = $1 AND outcome IS NOT NULL
                    GROUP BY outcome
                    """,
                    expertise_id,
                )
                outcome_counts = {row["outcome"]: row["count"] for row in outcome_rows}
                
                # Get item usage from items_used array
                item_rows = await conn.fetch(
                    """
                    SELECT unnest(items_used) as item_id, COUNT(*) as count
                    FROM expertise_usage_logs
                    WHERE expertise_id = $1
                    GROUP BY item_id
                    """,
                    expertise_id,
                )
                item_usage = {row["item_id"]: row["count"] for row in item_rows}
                
                return {
                    "total_uses": total_uses,
                    "item_usage": item_usage,
                    "outcome_counts": outcome_counts,
                    "feedback_counts": {},  # Would need to parse JSONB for this
                }
        
        except Exception as e:
            raise StorageError(f"Failed to get usage stats: {e}") from e

