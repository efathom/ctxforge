from __future__ import annotations

"""
Session service.

This service owns session-store interactions and session lifecycle logic.
It exists to keep `ctxforge` thin and to consolidate session behaviors
behind a stable API.
"""

from typing import List

from ctxforge.core.exceptions import StorageError
from ctxforge.core.session import Session
from ctxforge.protocols.storage import ISessionStore


class SessionService:
    """Owns session store dependency and provides session CRUD operations."""

    def __init__(self, *, session_store: ISessionStore):
        self._store = session_store

    async def fetch(self, *, session_id: str, user_id: str) -> Session:
        """Load an existing session or raise a StorageError."""
        try:
            return await self._store.load(session_id, user_id)
        except Exception as e:
            raise StorageError(
                f"Failed to load session: {e}",
                operation="load_session",
            ) from e

    async def save(self, session: Session) -> None:
        await self._store.save(session)

    async def delete(self, *, session_id: str) -> bool:
        return await self._store.delete(session_id)

    async def list(self, *, user_id: str, limit: int = 10) -> List[Session]:
        return await self._store.list_sessions(user_id, limit)


