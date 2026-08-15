"""Tests for ToolBridge."""

import asyncio

import pytest

from ctxforge.engine.services.tool_bridge import (
    ToolBridge,
    ToolBridgeConfig,
)

# ---- helpers ---------------------------------------------------------------


async def _echo_tool(name: str, args: dict) -> dict:
    """A simple async tool that echoes back the call."""
    return {"tool": name, "args": args, "echo": True}


async def _slow_tool(name: str, args: dict) -> dict:
    """A tool that takes a long time."""
    await asyncio.sleep(10)
    return {"done": True}


async def _failing_tool(name: str, args: dict) -> dict:
    """A tool that always raises."""
    raise ValueError(f"Tool '{name}' exploded")


# ---- tests -----------------------------------------------------------------


class TestToolBridgeRoundTrip:

    @pytest.mark.asyncio
    async def test_call_tool_round_trip(self):
        bridge = ToolBridge(tool_fn=_echo_tool)
        loop = asyncio.get_running_loop()
        call_tool = bridge.make_call_tool(loop)

        # Run call_tool in a thread (simulating script execution)
        result = await loop.run_in_executor(
            None, call_tool, "search", {"q": "hello"},
        )
        assert result == {"tool": "search", "args": {"q": "hello"}, "echo": True}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self):
        bridge = ToolBridge(tool_fn=_echo_tool)
        loop = asyncio.get_running_loop()
        call_tool = bridge.make_call_tool(loop)

        def _multi_call():
            r1 = call_tool("tool-a", {"x": 1})
            r2 = call_tool("tool-b", {"y": 2})
            return r1, r2

        r1, r2 = await loop.run_in_executor(None, _multi_call)
        assert r1["tool"] == "tool-a"
        assert r2["tool"] == "tool-b"

    @pytest.mark.asyncio
    async def test_tool_call_records_captured(self):
        bridge = ToolBridge(tool_fn=_echo_tool)
        loop = asyncio.get_running_loop()
        call_tool = bridge.make_call_tool(loop)

        await loop.run_in_executor(None, call_tool, "search", {"q": "x"})
        await loop.run_in_executor(None, call_tool, "read", {"path": "y"})

        records = bridge.records
        assert len(records) == 2
        assert records[0].tool_name == "search"
        assert records[1].tool_name == "read"
        assert records[0].error is None
        assert records[0].duration_sec >= 0


class TestToolBridgeBlocking:

    @pytest.mark.asyncio
    async def test_blocked_tool_rejected(self):
        config = ToolBridgeConfig(
            blocked_tools={"execute_skill", "save_skill"},
        )
        bridge = ToolBridge(tool_fn=_echo_tool, config=config)
        loop = asyncio.get_running_loop()
        call_tool = bridge.make_call_tool(loop)

        def _call_blocked():
            call_tool("execute_skill", {})

        with pytest.raises(RuntimeError, match="blocked"):
            await loop.run_in_executor(None, _call_blocked)


class TestToolBridgeTimeout:

    @pytest.mark.asyncio
    async def test_tool_bridge_timeout(self):
        config = ToolBridgeConfig(timeout_sec=0.2)
        bridge = ToolBridge(tool_fn=_slow_tool, config=config)
        loop = asyncio.get_running_loop()
        call_tool = bridge.make_call_tool(loop)

        def _call_slow():
            call_tool("slow-tool", {})

        with pytest.raises(RuntimeError, match="timed out"):
            await loop.run_in_executor(None, _call_slow)

        # Record should capture the timeout error
        assert len(bridge.records) == 1
        assert bridge.records[0].error is not None


class TestToolBridgeFailure:

    @pytest.mark.asyncio
    async def test_tool_call_error_captured(self):
        bridge = ToolBridge(tool_fn=_failing_tool)
        loop = asyncio.get_running_loop()
        call_tool = bridge.make_call_tool(loop)

        def _call_fail():
            call_tool("bad-tool", {})

        with pytest.raises(ValueError, match="exploded"):
            await loop.run_in_executor(None, _call_fail)

        assert len(bridge.records) == 1
        assert "exploded" in bridge.records[0].error
