"""
MySQL expertise store implementation.

Provides persistent expertise storage with JSON and full-text search.
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
from ctxforge.storage.connection import MySQLConfig, MySQLConnectionManager

# SQL for creating expertise table
CREATE_EXPERTISE_TABLE = """
CREATE TABLE IF NOT EXISTS expertise (
    expertise_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    description TEXT,
    version INT DEFAULT 1,
    token_budget INT,
    next_item_id INT DEFAULT 1,
    metadata JSON,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# SQL for creating expertise_items table
CREATE_EXPERTISE_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS expertise_items (
    expertise_id VARCHAR(255) NOT NULL,
    item_id VARCHAR(255) NOT NULL,
    section VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    helpful_count INT DEFAULT 0,
    harmful_count INT DEFAULT 0,
    source VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    embedding JSON,
    metadata JSON,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (expertise_id, item_id),
    INDEX idx_section (expertise_id, section),
    FULLTEXT INDEX idx_content_fts (content),
    FOREIGN KEY (expertise_id) REFERENCES expertise(expertise_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# SQL for creating expertise_usage_logs table
CREATE_EXPERTISE_USAGE_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS expertise_usage_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    expertise_id VARCHAR(255) NOT NULL,
    items_used JSON,
    feedback JSON,
    outcome VARCHAR(50),
    context_summary TEXT,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    INDEX idx_expertise (expertise_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


@registry.register_expertise_store("mysql")
class MySQLExpertiseStore(IExpertiseStore):
    """
    MySQL-based expertise store.
    
    Uses the same MySQLConnectionManager as other MySQL stores.
    Stores expertise and items in separate tables with full-text search support.
    
    Example:
        from ctxforge.storage.connection import MySQLConfig
        
        config = MySQLConfig(host="localhost", database="ctxforge")
        store = MySQLExpertiseStore(config)
        await store.initialize()
        
        expertise = Expertise(expertise_id="test", name="Test")
        await store.save(expertise)
    """
    
    def __init__(
        self,
        config: Optional[MySQLConfig] = None,
        connection_manager: Optional[MySQLConnectionManager] = None,
    ):
        """
        Initialize the MySQL expertise store.
        
        Args:
            config: MySQL configuration
            connection_manager: Optional pre-existing connection manager
        """
        self.config = config or MySQLConfig()
        self._conn_manager = connection_manager or MySQLConnectionManager(self.config)
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
        # Create tables in order (expertise first due to foreign key)
        await self._conn_manager.execute(CREATE_EXPERTISE_TABLE)
        await self._conn_manager.execute(CREATE_EXPERTISE_ITEMS_TABLE)
        await self._conn_manager.execute(CREATE_EXPERTISE_USAGE_LOGS_TABLE)
    
    async def close(self) -> None:
        """Close the connection if we own it."""
        if self._owns_connection and self._conn_manager:
            await self._conn_manager.disconnect()
    
    async def save(self, expertise: Expertise) -> None:
        """
        Save or update an expertise knowledge base.
        
        Uses INSERT ... ON DUPLICATE KEY UPDATE for upsert.
        Also syncs all items to the expertise_items table.
        """
        await self.initialize()
        
        try:
            async with self._conn_manager.acquire() as conn:
                async with conn.cursor() as cur:
                    # Upsert expertise
                    await cur.execute(
                        """
                        INSERT INTO expertise (
                            expertise_id, name, domain, description, version, token_budget,
                            next_item_id, metadata, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            name = VALUES(name),
                            domain = VALUES(domain),
                            description = VALUES(description),
                            version = VALUES(version),
                            token_budget = VALUES(token_budget),
                            next_item_id = VALUES(next_item_id),
                            metadata = VALUES(metadata),
                            updated_at = VALUES(updated_at)
                        """,
                        (
                            expertise.expertise_id,
                            expertise.name,
                            expertise.domain,
                            expertise.description,
                            expertise.version,
                            expertise.token_budget,
                            expertise.next_item_id,
                            json.dumps(expertise.metadata),
                            expertise.created_at,
                            datetime.now(timezone.utc),
                        ),
                    )
                    
                    # Sync items - delete existing and re-insert
                    await cur.execute(
                        "DELETE FROM expertise_items WHERE expertise_id = %s",
                        (expertise.expertise_id,),
                    )
                    
                    for item in expertise.items:
                        await self._insert_item(cur, expertise.expertise_id, item)
                    
                    await conn.commit()
        
        except Exception as e:
            raise StorageError(f"Failed to save expertise: {e}") from e
    
    async def _insert_item(self, cursor, expertise_id: str, item: ExpertiseItem) -> None:
        """Insert a single item."""
        await cursor.execute(
            """
            INSERT INTO expertise_items (
                expertise_id, item_id, section, content,
                helpful_count, harmful_count, source, is_active,
                embedding, metadata, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                expertise_id,
                item.item_id,
                item.section.value,
                item.content,
                item.helpful_count,
                item.harmful_count,
                item.source,
                item.is_active,
                json.dumps(item.embedding) if item.embedding else None,
                json.dumps(item.metadata),
                item.created_at,
                item.updated_at,
            ),
        )
    
    async def load(self, expertise_id: str) -> Optional[Expertise]:
        """Load an expertise by ID."""
        await self.initialize()
        
        try:
            # Load expertise
            row = await self._conn_manager.fetchone(
                """
                SELECT expertise_id, name, domain, description, version, token_budget,
                       next_item_id, metadata, created_at, updated_at
                FROM expertise
                WHERE expertise_id = %s
                """,
                (expertise_id,),
            )
            
            if not row:
                return None
            
            # Load items
            item_rows = await self._conn_manager.fetchall(
                """
                SELECT item_id, section, content, helpful_count, harmful_count,
                       source, is_active, embedding, metadata, created_at, updated_at
                FROM expertise_items
                WHERE expertise_id = %s
                ORDER BY created_at
                """,
                (expertise_id,),
            )
            
            items = []
            for r in item_rows:
                embedding = r.get("embedding")
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                
                metadata = r.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                items.append(ExpertiseItem(
                    item_id=r["item_id"],
                    section=ExpertiseSection(r["section"]),
                    content=r["content"],
                    helpful_count=r["helpful_count"],
                    harmful_count=r["harmful_count"],
                    source=r["source"],
                    is_active=r["is_active"],
                    embedding=embedding,
                    metadata=metadata,
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                ))
            
            expertise_metadata = row.get("metadata", "{}")
            if isinstance(expertise_metadata, str):
                expertise_metadata = json.loads(expertise_metadata)
            
            return Expertise(
                expertise_id=row["expertise_id"],
                name=row["name"],
                domain=row["domain"],
                description=row.get("description") or "",
                version=row["version"],
                token_budget=row["token_budget"],
                next_item_id=row["next_item_id"],
                metadata=expertise_metadata,
                items=items,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        
        except Exception as e:
            raise StorageError(f"Failed to load expertise: {e}") from e
    
    async def delete(self, expertise_id: str) -> bool:
        """Delete an expertise knowledge base."""
        await self.initialize()
        
        try:
            affected = await self._conn_manager.execute(
                "DELETE FROM expertise WHERE expertise_id = %s",
                (expertise_id,),
            )
            return affected > 0
        
        except Exception as e:
            raise StorageError(f"Failed to delete expertise: {e}") from e
    
    async def list_expertise(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Expertise]:
        """List expertise knowledge bases."""
        await self.initialize()
        
        try:
            if domain:
                rows = await self._conn_manager.fetchall(
                    """
                    SELECT expertise_id FROM expertise
                    WHERE domain = %s
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (domain, limit, offset),
                )
            else:
                rows = await self._conn_manager.fetchall(
                    """
                    SELECT expertise_id FROM expertise
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
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
        await self.initialize()
        
        try:
            async with self._conn_manager.acquire() as conn:
                async with conn.cursor() as cur:
                    await self._insert_item(cur, expertise_id, item)
                    
                    # Update expertise timestamp
                    await cur.execute(
                        "UPDATE expertise SET updated_at = %s WHERE expertise_id = %s",
                        (datetime.now(timezone.utc), expertise_id),
                    )
                    await conn.commit()
        
        except Exception as e:
            raise StorageError(f"Failed to add item: {e}") from e
    
    async def update_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """Update an existing item."""
        await self.initialize()
        
        try:
            await self._conn_manager.execute(
                """
                UPDATE expertise_items SET
                    section = %s,
                    content = %s,
                    helpful_count = %s,
                    harmful_count = %s,
                    source = %s,
                    is_active = %s,
                    embedding = %s,
                    metadata = %s,
                    updated_at = %s
                WHERE item_id = %s AND expertise_id = %s
                """,
                (
                    item.section.value,
                    item.content,
                    item.helpful_count,
                    item.harmful_count,
                    item.source,
                    item.is_active,
                    json.dumps(item.embedding) if item.embedding else None,
                    json.dumps(item.metadata),
                    datetime.now(timezone.utc),
                    item.item_id,
                    expertise_id,
                ),
            )
            
            # Update expertise timestamp
            await self._conn_manager.execute(
                "UPDATE expertise SET updated_at = %s WHERE expertise_id = %s",
                (datetime.now(timezone.utc), expertise_id),
            )
        
        except Exception as e:
            raise StorageError(f"Failed to update item: {e}") from e
    
    async def remove_item(self, expertise_id: str, item_id: str) -> bool:
        """Remove an item from an expertise."""
        await self.initialize()
        
        try:
            affected = await self._conn_manager.execute(
                "DELETE FROM expertise_items WHERE item_id = %s AND expertise_id = %s",
                (item_id, expertise_id),
            )
            
            if affected > 0:
                await self._conn_manager.execute(
                    "UPDATE expertise SET updated_at = %s WHERE expertise_id = %s",
                    (datetime.now(timezone.utc), expertise_id),
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
        await self.initialize()
        
        try:
            row = await self._conn_manager.fetchone(
                """
                SELECT item_id, section, content, helpful_count, harmful_count,
                       source, is_active, embedding, metadata, created_at, updated_at
                FROM expertise_items
                WHERE item_id = %s AND expertise_id = %s
                """,
                (item_id, expertise_id),
            )
            
            if not row:
                return None
            
            embedding = row.get("embedding")
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            
            metadata = row.get("metadata", "{}")
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            return ExpertiseItem(
                item_id=row["item_id"],
                section=ExpertiseSection(row["section"]),
                content=row["content"],
                helpful_count=row["helpful_count"],
                harmful_count=row["harmful_count"],
                source=row["source"],
                is_active=row["is_active"],
                embedding=embedding,
                metadata=metadata,
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
        await self.initialize()
        
        try:
            rows = await self._conn_manager.fetchall(
                """
                SELECT item_id, section, content, helpful_count, harmful_count,
                       source, is_active, embedding, metadata, created_at, updated_at
                FROM expertise_items
                WHERE expertise_id = %s AND section = %s AND is_active = TRUE
                ORDER BY created_at
                """,
                (expertise_id, section.value),
            )
            
            items = []
            for r in rows:
                embedding = r.get("embedding")
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                
                metadata = r.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                items.append(ExpertiseItem(
                    item_id=r["item_id"],
                    section=ExpertiseSection(r["section"]),
                    content=r["content"],
                    helpful_count=r["helpful_count"],
                    harmful_count=r["harmful_count"],
                    source=r["source"],
                    is_active=r["is_active"],
                    embedding=embedding,
                    metadata=metadata,
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                ))
            
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
        await self.initialize()
        
        try:
            await self._conn_manager.execute(
                """
                UPDATE expertise_items SET
                    helpful_count = helpful_count + %s,
                    harmful_count = harmful_count + %s,
                    updated_at = %s
                WHERE item_id = %s AND expertise_id = %s
                """,
                (
                    helpful_delta,
                    harmful_delta,
                    datetime.now(timezone.utc),
                    item_id,
                    expertise_id,
                ),
            )
            
            await self._conn_manager.execute(
                "UPDATE expertise SET updated_at = %s WHERE expertise_id = %s",
                (datetime.now(timezone.utc), expertise_id),
            )
        
        except Exception as e:
            raise StorageError(f"Failed to update item counts: {e}") from e
    
    async def search_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """Search items by text content using MySQL full-text search."""
        await self.initialize()
        
        try:
            # Use MATCH...AGAINST for full-text search with LIKE fallback
            rows = await self._conn_manager.fetchall(
                """
                SELECT item_id, section, content, helpful_count, harmful_count,
                       source, is_active, embedding, metadata, created_at, updated_at,
                       MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance
                FROM expertise_items
                WHERE expertise_id = %s
                  AND is_active = TRUE
                  AND (MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) OR content LIKE %s)
                ORDER BY relevance DESC, helpful_count DESC
                LIMIT %s
                """,
                (query, expertise_id, query, f"%{query}%", limit),
            )
            
            items = []
            for r in rows:
                embedding = r.get("embedding")
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                
                metadata = r.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                items.append(ExpertiseItem(
                    item_id=r["item_id"],
                    section=ExpertiseSection(r["section"]),
                    content=r["content"],
                    helpful_count=r["helpful_count"],
                    harmful_count=r["harmful_count"],
                    source=r["source"],
                    is_active=r["is_active"],
                    embedding=embedding,
                    metadata=metadata,
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                ))
            
            return items
        
        except Exception as e:
            raise StorageError(f"Failed to search items: {e}") from e
    
    async def log_usage(self, log: ExpertiseUsageLog) -> None:
        """Log expertise usage in a turn."""
        await self.initialize()
        
        try:
            feedback_json = {
                item_id: feedback.value
                for item_id, feedback in log.feedback.items()
            }
            
            await self._conn_manager.execute(
                """
                INSERT INTO expertise_usage_logs (
                    log_id, session_id, expertise_id, items_used,
                    feedback, outcome, context_summary, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    log.log_id,
                    log.session_id,
                    log.expertise_id,
                    json.dumps(log.items_used),
                    json.dumps(feedback_json),
                    log.outcome.value if log.outcome else None,
                    log.context_summary,
                    log.timestamp,
                ),
            )
        
        except Exception as e:
            raise StorageError(f"Failed to log usage: {e}") from e
    
    async def get_usage_stats(self, expertise_id: str) -> Dict[str, Any]:
        """Get usage statistics for an expertise."""
        await self.initialize()
        
        try:
            # Get total uses
            total_uses = await self._conn_manager.fetchval(
                "SELECT COUNT(*) FROM expertise_usage_logs WHERE expertise_id = %s",
                (expertise_id,),
            ) or 0
            
            # Get outcome counts
            outcome_rows = await self._conn_manager.fetchall(
                """
                SELECT outcome, COUNT(*) as count
                FROM expertise_usage_logs
                WHERE expertise_id = %s AND outcome IS NOT NULL
                GROUP BY outcome
                """,
                (expertise_id,),
            )
            outcome_counts = {row["outcome"]: row["count"] for row in outcome_rows}
            
            return {
                "total_uses": total_uses,
                "item_usage": {},  # Would need JSON_EXTRACT to parse items_used
                "outcome_counts": outcome_counts,
                "feedback_counts": {},
            }
        
        except Exception as e:
            raise StorageError(f"Failed to get usage stats: {e}") from e
