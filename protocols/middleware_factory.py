"""
Middleware factory protocol.

Enables dependency-aware middleware construction without hardcoding DI logic in EngineFactory.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from ctxforge.engine.deps import EngineDeps
from ctxforge.middleware.protocol import IMiddleware


@runtime_checkable
class IMiddlewareFactory(Protocol):
    def create(self, *, config: dict[str, Any], deps: EngineDeps) -> Optional[IMiddleware]:
        """Create a middleware instance or return None to skip."""

