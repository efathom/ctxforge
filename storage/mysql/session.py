"""
MySQL session store implementation.

Provides persistent session storage with JSON, optimistic locking, and full ACID support.
"""

import json
from typing import Any, Dict, List, Optional

from ctxforge.core.exceptions import ConcurrencyError, StorageError
from ctxforge.core.session import Session
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import ISessionStore
from ctxforge.storage.connection import MySQLConfig, MySQLConnectionManager

# SQL statements for table creation
CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    version INT NOT NULL DEFAULT 1,
    state VARCHAR(50) NOT NULL DEFAULT 'active',
    data JSON NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_user_id (user_id),
    INDEX idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


@registry.register_session_store("mysql")
class MySQLSessionStore(ISessionStore):
    """
    MySQL-based session store.
    
    Features:
    - JSON storage for flexible session data
    - Optimistic locking with version checking
    - Indexed queries by user_id
    - Full ACID transaction support
    
    Example:
        from ctxforge.storage.connection import MySQLConfig
        
        config = MySQLConfig(host="localhost", database="ctxforge")
        store = MySQLSessionStore(config)
        await store.initialize()
        
        session = await store.load("sess-123", "user-456")
    """
    
    def __init__(
        self,
        config: Optional[MySQLConfig] = None,
        connection_manager: Optional[MySQLConnectionManager] = None,
    ):
        """
        Initialize the MySQL session store.
        
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
        
        sql = CREATE_SESSIONS_TABLE.format(table_name=self.config.sessions_table)
        await self._manager.execute(sql)
        self._initialized = True
    
    def _serialize_session(self, session: Session) -> Dict[str, Any]:
        """Serialize session to dict for JSON storage."""
        return json.loads(session.model_dump_json())
    
    def _deserialize_session(self, row: Dict[str, Any]) -> Session:
        """Deserialize session from database row."""
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return Session.model_validate(data)
    
    async def load(self, session_id: str, user_id: str) -> Session:
        """Load or create a session."""
        await self.initialize()
        
        query = f"""
            SELECT data FROM {self.config.sessions_table}
            WHERE session_id = %s
        """
        
        row = await self._manager.fetchone(query, (session_id,))
        
        if row:
            return self._deserialize_session(row)
        
        # Create new session
        return Session(session_id=session_id, user_id=user_id)
    
    async def save(self, session: Session) -> None:
        """Save a session with optimistic locking."""
        await self.initialize()
        
        # Check current version
        version_query = f"""
            SELECT version FROM {self.config.sessions_table}
            WHERE session_id = %s
        """
        
        current_version = await self._manager.fetchval(
            version_query, (session.session_id,)
        )
        
        if current_version is not None and current_version > session.version:
            raise ConcurrencyError(
                "Session was modified by another process",
                session_id=session.session_id,
                expected_version=session.version,
                actual_version=current_version,
            )
        
        # Increment version
        session.increment_version()
        
        # Upsert the session using ON DUPLICATE KEY UPDATE
        upsert_query = f"""
            INSERT INTO {self.config.sessions_table}
            (session_id, user_id, version, state, data, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                user_id = VALUES(user_id),
                version = VALUES(version),
                state = VALUES(state),
                data = VALUES(data),
                updated_at = VALUES(updated_at)
        """
        
        try:
            await self._manager.execute(
                upsert_query,
                (
                    session.session_id,
                    session.user_id,
                    session.version,
                    json.dumps(session.state.model_dump()),
                    json.dumps(self._serialize_session(session)),
                    session.created_at,
                    session.updated_at,
                ),
            )
        except Exception as e:
            raise StorageError(f"Failed to save session: {e}") from e
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        await self.initialize()
        
        query = f"""
            DELETE FROM {self.config.sessions_table}
            WHERE session_id = %s
        """
        
        affected = await self._manager.execute(query, (session_id,))
        return affected > 0
    
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        await self.initialize()
        
        query = f"""
            SELECT 1 FROM {self.config.sessions_table}
            WHERE session_id = %s
        """
        
        result = await self._manager.fetchval(query, (session_id,))
        return result is not None
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions for a user."""
        await self.initialize()
        
        query = f"""
            SELECT data FROM {self.config.sessions_table}
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
        """
        
        rows = await self._manager.fetchall(query, (user_id, limit, offset))
        
        return [self._deserialize_session(row) for row in rows]
    
    async def clear(self) -> None:
        """Clear all sessions (for testing)."""
        await self.initialize()
        
        query = f"TRUNCATE TABLE {self.config.sessions_table}"
        await self._manager.execute(query)
