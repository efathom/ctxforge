"""
In-memory session store implementation.

Suitable for testing and single-instance deployments.
"""

import asyncio
from typing import Dict, List, Optional

from ctxforge.core.exceptions import ConcurrencyError
from ctxforge.core.session import Session
from ctxforge.engine.registry import registry
from ctxforge.protocols.storage import ISessionStore


@registry.register_session_store("memory")
class InMemorySessionStore(ISessionStore):
    """
    In-memory session store.
    
    Sessions are stored in a dictionary with optimistic locking support.
    Suitable for single-instance deployments and testing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the store.
        
        Args:
            config: Optional configuration (unused for in-memory)
        """
        self._store: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
    
    async def load(self, session_id: str, user_id: str) -> Session:
        """Load or create a session."""
        async with self._lock:
            if session_id in self._store:
                # Return a deep copy to prevent mutation
                return self._store[session_id].model_copy(deep=True)
            
            # Create new session
            return Session(session_id=session_id, user_id=user_id)
    
    async def save(self, session: Session) -> None:
        """Save a session with optimistic locking."""
        async with self._lock:
            current = self._store.get(session.session_id)
            
            if current and current.version > session.version:
                raise ConcurrencyError(
                    "Session was modified by another process",
                    session_id=session.session_id,
                    expected_version=session.version,
                    actual_version=current.version,
                )
            
            # Increment version and save
            session.increment_version()
            self._store[session.session_id] = session.model_copy(deep=True)
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False
    
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._store
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions for a user."""
        user_sessions = [
            s.model_copy(deep=True)
            for s in self._store.values()
            if s.user_id == user_id
        ]
        
        # Sort by updated_at descending
        user_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        return user_sessions[offset:offset + limit]
    
    async def clear(self) -> None:
        """Clear all sessions (for testing)."""
        async with self._lock:
            self._store.clear()

