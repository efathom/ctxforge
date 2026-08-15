"""
Lifecycle Protocol Interfaces.

ctxforge components are primarily duck-typed via Protocols. Some components also require
async setup/teardown (e.g., DB clients, networked stores).

Phase 1 introduces a minimal, optional lifecycle contract:
- `initialize()` for async setup
- `close()` for async teardown

Both methods are optional and discovered via duck-typing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IAsyncInitializable(Protocol):
    async def initialize(self) -> None:
        """Perform async initialization (connections, pools, etc.)."""


@runtime_checkable
class IAsyncClosable(Protocol):
    async def close(self) -> None:
        """Perform async teardown (disconnect, close pools, etc.)."""

