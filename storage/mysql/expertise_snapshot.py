"""
MySQL expertise snapshot store implementation.

Provides persistent storage for expertise snapshots with version history.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional

from ctxforge.engine.services.expertise_snapshot_service import (
    ExpertiseSnapshot,
    ExpertiseSnapshotStore,
)
from ctxforge.storage.connection import MySQLConfig, MySQLConnectionManager

CREATE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    expertise_id VARCHAR(255) NOT NULL,
    version VARCHAR(100) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    item_count INT NOT NULL DEFAULT 0,
    items JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    description TEXT,
    metadata JSON DEFAULT NULL,
    UNIQUE KEY unique_expertise_version (expertise_id, version),
    INDEX idx_expertise_id (expertise_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


class MySQLSnapshotStore(ExpertiseSnapshotStore):
    """
    MySQL-based expertise snapshot store.

    Features:
    - Full version history with timestamps
    - JSON storage for snapshot items
    - Efficient querying by expertise_id
    - Unique constraint on (expertise_id, version)
    """

    def __init__(
        self,
        config: Optional[MySQLConfig] = None,
        connection_manager: Optional[MySQLConnectionManager] = None,
        table_name: str = "expertise_snapshots",
    ):
        """
        Initialize the MySQL snapshot store.

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

        sql = CREATE_SNAPSHOTS_TABLE.format(table_name=self._table_name)
        await self._manager.execute(sql)
        self._initialized = True

    async def save_snapshot(self, snapshot: ExpertiseSnapshot) -> None:
        """Save a snapshot."""
        await self.connect()

        # Format datetime for MySQL
        created_at_str = snapshot.created_at.strftime('%Y-%m-%d %H:%M:%S')

        sql = f"""
        INSERT INTO {self._table_name} (
            expertise_id, version, content_hash, item_count, items,
            created_at, created_by, description, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        AS new_values
        ON DUPLICATE KEY UPDATE
            content_hash = new_values.content_hash,
            item_count = new_values.item_count,
            items = new_values.items,
            created_at = new_values.created_at,
            created_by = new_values.created_by,
            description = new_values.description,
            metadata = new_values.metadata
        """

        await self._manager.execute(
            sql,
            (
                snapshot.expertise_id,
                snapshot.version,
                snapshot.content_hash,
                len(snapshot.items),  # Compute item_count from items
                json.dumps(snapshot.items, default=str),
                created_at_str,
                snapshot.created_by,
                snapshot.description,
                json.dumps(getattr(snapshot, 'metadata', {}))
                if getattr(snapshot, 'metadata', None) else None,
            ),
        )

    async def get_snapshot(
        self, expertise_id: str, version: str
    ) -> Optional[ExpertiseSnapshot]:
        """Get a specific snapshot by expertise ID and version."""
        await self.connect()

        sql = f"""
        SELECT expertise_id, version, content_hash, item_count, items,
               created_at, created_by, description, metadata
        FROM {self._table_name}
        WHERE expertise_id = %s AND version = %s
        """

        row = await self._manager.fetchone(sql, (expertise_id, version))

        if row is None:
            return None

        return self._row_to_snapshot(row)

    async def get_latest_snapshot(self, expertise_id: str) -> Optional[ExpertiseSnapshot]:
        """Get the most recent snapshot for an expertise."""
        await self.connect()

        sql = f"""
        SELECT expertise_id, version, content_hash, item_count, items,
               created_at, created_by, description, metadata
        FROM {self._table_name}
        WHERE expertise_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """

        row = await self._manager.fetchone(sql, (expertise_id,))

        if row is None:
            return None

        return self._row_to_snapshot(row)

    async def list_versions(self, expertise_id: str) -> List[str]:
        """List all versions for an expertise, sorted by creation time."""
        await self.connect()

        sql = f"""
        SELECT version FROM {self._table_name}
        WHERE expertise_id = %s
        ORDER BY created_at ASC
        """

        rows = await self._manager.fetchall(sql, (expertise_id,))

        return [row["version"] for row in rows]

    async def delete_snapshot(self, expertise_id: str, version: str) -> bool:
        """Delete a specific snapshot."""
        await self.connect()

        # Check if exists first
        check_sql = f"""
        SELECT id FROM {self._table_name}
        WHERE expertise_id = %s AND version = %s
        """
        row = await self._manager.fetchone(check_sql, (expertise_id, version))

        if row is None:
            return False

        sql = f"DELETE FROM {self._table_name} WHERE expertise_id = %s AND version = %s"
        await self._manager.execute(sql, (expertise_id, version))

        return True

    async def get_version_count(self, expertise_id: str) -> int:
        """Get the number of versions for an expertise."""
        await self.connect()

        sql = f"SELECT COUNT(*) as count FROM {self._table_name} WHERE expertise_id = %s"
        row = await self._manager.fetchone(sql, (expertise_id,))

        return row["count"] if row else 0

    def _row_to_snapshot(self, row: dict) -> ExpertiseSnapshot:
        """Convert a database row to an ExpertiseSnapshot."""
        items = row["items"]
        if isinstance(items, str):
            items = json.loads(items)

        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        return ExpertiseSnapshot(
            expertise_id=row["expertise_id"],
            version=row["version"],
            content_hash=row["content_hash"],
            items=items,
            created_at=created_at,
            created_by=row.get("created_by"),
            description=row.get("description", ""),
        )
