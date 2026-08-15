"""
Tool Bridge.

Mediates between synchronous skill script code and the async engine
tool layer. Uses a request/response queue pair so that a script running
in a worker thread can call engine tools without deadlocking.

::

    Script thread                      Engine event loop
    ────────────                       ──────────────────
    call_tool("search", {q: "x"})
      └─► put (name, args) on queue ──►  pick up from queue
                                          await tool_fn(name, args)
          wait on result_event  ◄──────  put result on result_queue
      ◄─► return result                   signal result_event
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Sentinel to signal the bridge consumer to stop.
_STOP = object()


@dataclass
class ToolCallRecord:
    """Record of a single tool call made through the bridge."""

    tool_name: str
    args: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "duration_sec": self.duration_sec,
        }


@dataclass
class ToolBridgeConfig:
    """Configuration for the tool bridge."""

    timeout_sec: float = 10.0
    blocked_tools: Set[str] = field(default_factory=lambda: {
        "save_skill", "execute_skill", "list_skills", "get_skill",
    })


class ToolBridge:
    """Queue-based bridge between synchronous script threads and the
    async engine tool layer.

    Usage::

        bridge = ToolBridge(tool_fn=my_async_tool_fn, config=cfg)
        call_tool = bridge.make_call_tool()

        # In the script thread:
        result = call_tool("search", {"q": "hello"})

        # The engine loop processes the request asynchronously and
        # returns the result to the script thread.
    """

    def __init__(
        self,
        *,
        tool_fn: Callable[..., Coroutine],
        config: Optional[ToolBridgeConfig] = None,
    ):
        self._tool_fn = tool_fn
        self._config = config or ToolBridgeConfig()
        self._records: List[ToolCallRecord] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def records(self) -> List[ToolCallRecord]:
        """All tool call records captured during this bridge's lifetime."""
        return list(self._records)

    def make_call_tool(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> Callable:
        """Create a synchronous ``call_tool`` callable for use in scripts.

        Args:
            loop: The event loop where async tool calls will be dispatched.

        Returns:
            A synchronous function ``call_tool(name, args) -> result``
            that can be called from a worker thread.
        """
        self._loop = loop

        def call_tool(name: str, args: Optional[Dict[str, Any]] = None) -> Any:
            if name in self._config.blocked_tools:
                raise RuntimeError(
                    f"Tool '{name}' is blocked inside skill scripts "
                    f"to prevent infinite recursion."
                )

            import time
            start = time.time()
            args = args or {}

            record = ToolCallRecord(tool_name=name, args=args)

            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._invoke_tool(name, args),
                    self._loop,
                )
                result = future.result(timeout=self._config.timeout_sec)
                record.result = result
                record.duration_sec = round(time.time() - start, 4)
                self._records.append(record)
                return result

            except TimeoutError:
                record.error = (
                    f"Tool call '{name}' timed out after "
                    f"{self._config.timeout_sec}s"
                )
                record.duration_sec = round(time.time() - start, 4)
                self._records.append(record)
                raise RuntimeError(record.error) from None

            except Exception as exc:
                record.error = str(exc)
                record.duration_sec = round(time.time() - start, 4)
                self._records.append(record)
                raise

        return call_tool

    async def _invoke_tool(
        self,
        name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Invoke the async tool function."""
        return await self._tool_fn(name, args)
