"""
Plugin fixture for lifecycle tests.

Registers a session store under the built-in "redis" backend name, but the store
is a pure in-memory stub. This allows testing EngineFactory lifecycle behavior
without requiring external services.
"""

from __future__ import annotations

from typing import List

from ctxforge.core.session import Session
from ctxforge.engine.registry import ComponentRegistry
from ctxforge.protocols.storage import ISessionStore


INITIALIZED = False
CLOSED = False


class LifecycleTestRedisSessionStore(ISessionStore):
    def __init__(self, config=None):
        # The real RedisSessionStore is constructed with a config object; the factory expects
        # this signature for session stores selected via `storage.session.backend`.
        self._config = config

    async def initialize(self) -> None:
        global INITIALIZED
        INITIALIZED = True

    async def close(self) -> None:
        global CLOSED
        CLOSED = True

    async def load(self, session_id: str, user_id: str) -> Session:
        return Session(session_id=session_id, user_id=user_id)

    async def save(self, session: Session) -> None:
        return None

    async def delete(self, session_id: str) -> bool:
        return True

    async def exists(self, session_id: str) -> bool:
        return False

    async def list_sessions(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Session]:
        return []


def register(registry: ComponentRegistry) -> None:
    registry.register_session_store("redis")(LifecycleTestRedisSessionStore)

