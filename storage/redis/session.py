"""
Redis session store implementation.

Provides high-performance session storage with optimistic locking and TTL support.
"""

from typing import List, Optional

from ctxforge.core.exceptions import ConcurrencyError, StorageError
from ctxforge.core.session import Session
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import ISessionStore
from ctxforge.storage.connection import RedisConfig, RedisConnectionManager


@registry.register_session_store("redis")
class RedisSessionStore(ISessionStore):
    """
    Redis-based session store.
    
    Features:
    - Atomic operations with optimistic locking via WATCH/MULTI/EXEC
    - Configurable TTL for automatic session expiration
    - JSON serialization of session data
    - Namespace prefixing for multi-tenant support
    """
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        connection_manager: Optional[RedisConnectionManager] = None,
    ):
        """
        Initialize the Redis session store.
        
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
    
    def _session_key(self, session_id: str) -> str:
        """Get the Redis key for a session."""
        return f"{self.config.session_prefix}{session_id}"
    
    def _user_sessions_key(self, user_id: str) -> str:
        """Get the Redis key for user's session list."""
        return f"{self.config.session_prefix}user:{user_id}"
    
    def _serialize_session(self, session: Session) -> str:
        """Serialize a session to JSON."""
        return session.model_dump_json()
    
    def _deserialize_session(self, data: str) -> Session:
        """Deserialize a session from JSON."""
        return Session.model_validate_json(data)
    
    async def load(self, session_id: str, user_id: str) -> Session:
        """Load or create a session."""
        await self.connect()
        client = self._manager.client
        key = self._session_key(session_id)
        
        data = await client.get(key)
        
        if data:
            session = self._deserialize_session(data)
            # Refresh TTL on access
            await client.expire(key, self.config.session_ttl_seconds)
            return session
        
        # Create new session
        return Session(session_id=session_id, user_id=user_id)
    
    async def save(self, session: Session) -> None:
        """Save a session with optimistic locking."""
        await self.connect()
        client = self._manager.client
        key = self._session_key(session.session_id)
        user_sessions_key = self._user_sessions_key(session.user_id)
        
        # Watch the key for changes
        async with client.pipeline(transaction=True) as pipe:
            try:
                # Watch for concurrent modifications
                await pipe.watch(key)
                
                # Check current version
                current_data = await client.get(key)
                if current_data:
                    current = self._deserialize_session(current_data)
                    if current.version > session.version:
                        raise ConcurrencyError(
                            "Session was modified by another process",
                            session_id=session.session_id,
                            expected_version=session.version,
                            actual_version=current.version,
                        )
                
                # Increment version
                session.increment_version()
                
                # Start transaction
                pipe.multi()
                pipe.set(
                    key,
                    self._serialize_session(session),
                    ex=self.config.session_ttl_seconds,
                )
                # Add to user's session set with score as timestamp
                pipe.zadd(
                    user_sessions_key,
                    {session.session_id: session.updated_at.timestamp()},
                )
                
                await pipe.execute()
                
            except Exception as e:
                if "WATCH" in str(e) or isinstance(e, ConcurrencyError):
                    raise
                raise StorageError(f"Failed to save session: {e}") from e
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        await self.connect()
        client = self._manager.client
        key = self._session_key(session_id)
        
        # Get session first to find user_id
        data = await client.get(key)
        if not data:
            return False
        
        session = self._deserialize_session(data)
        user_sessions_key = self._user_sessions_key(session.user_id)
        
        # Delete session and remove from user set
        async with client.pipeline() as pipe:
            pipe.delete(key)
            pipe.zrem(user_sessions_key, session_id)
            await pipe.execute()
        
        return True
    
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        await self.connect()
        client = self._manager.client
        key = self._session_key(session_id)
        return await client.exists(key) > 0
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions for a user."""
        await self.connect()
        client = self._manager.client
        user_sessions_key = self._user_sessions_key(user_id)
        
        # Get session IDs sorted by timestamp (descending)
        session_ids = await client.zrevrange(
            user_sessions_key,
            offset,
            offset + limit - 1,
        )
        
        if not session_ids:
            return []
        
        # Fetch all sessions
        keys = [self._session_key(sid) for sid in session_ids]
        data_list = await client.mget(keys)
        
        sessions = []
        for data in data_list:
            if data:
                sessions.append(self._deserialize_session(data))
        
        return sessions
    
    async def clear(self) -> None:
        """Clear all sessions (for testing)."""
        await self.connect()
        client = self._manager.client
        
        # Find all session keys
        pattern = f"{self.config.session_prefix}*"
        cursor = 0
        
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

