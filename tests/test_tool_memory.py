"""Tests for the TOOL memory type, ToolExecutionRecord, and related utilities."""


import pytest

from ctxforge.core.memory import (
    MemoryFactory,
    MemoryType,
    ToolExecutionRecord,
    add_tool_record,
    get_tool_statistics,
)


def test_memory_type_tool_value():
    assert MemoryType.TOOL.value == "tool"


def test_tool_execution_record_creation():
    record = ToolExecutionRecord(
        tool_name="search",
        input_params={"query": "hello"},
        output="result",
        success=True,
        time_cost=1.5,
        token_cost=100,
        quality_score=0.9,
    )
    assert record.tool_name == "search"
    assert record.success is True
    assert record.quality_score == 0.9


def test_tool_execution_record_model_dump():
    record = ToolExecutionRecord(
        tool_name="calc",
        input_params={"x": 1},
    )
    dumped = record.model_dump()
    assert "tool_name" in dumped
    assert "input_params" in dumped
    assert "timestamp" in dumped


def test_add_tool_record_on_tool_memory():
    memory = MemoryFactory.tool_memory(
        user_id="u1",
        tool_name="search",
        content="Search tool usage pattern",
    )
    record = ToolExecutionRecord(
        tool_name="search",
        input_params={"q": "test"},
        success=True,
        time_cost=0.5,
        quality_score=0.8,
    )
    add_tool_record(memory, record)
    assert len(memory.metadata["tool_records"]) == 1
    assert memory.access_count == 1


def test_add_tool_record_raises_on_non_tool_memory():
    memory = MemoryFactory.semantic_memory(user_id="u1", content="A fact")
    record = ToolExecutionRecord(tool_name="x", input_params={})
    with pytest.raises(ValueError, match="Expected TOOL memory"):
        add_tool_record(memory, record)


def test_get_tool_statistics_with_records():
    memory = MemoryFactory.tool_memory(
        user_id="u1", tool_name="calc", content="Calculator tool"
    )
    records = [
        ToolExecutionRecord(
            tool_name="calc", input_params={}, success=True,
            time_cost=1.0, quality_score=0.8,
        ),
        ToolExecutionRecord(
            tool_name="calc", input_params={}, success=False,
            time_cost=2.0, quality_score=0.4,
        ),
        ToolExecutionRecord(
            tool_name="calc", input_params={}, success=True,
            time_cost=1.5, quality_score=0.6,
        ),
    ]
    for r in records:
        add_tool_record(memory, r)

    stats = get_tool_statistics(memory)
    assert stats["count"] == 3
    assert abs(stats["success_rate"] - 2 / 3) < 1e-9
    assert abs(stats["avg_time_cost"] - 1.5) < 1e-9
    assert abs(stats["avg_score"] - 0.6) < 1e-9


def test_get_tool_statistics_empty():
    memory = MemoryFactory.tool_memory(
        user_id="u1", tool_name="empty", content="Empty tool"
    )
    stats = get_tool_statistics(memory)
    assert stats["count"] == 0
    assert stats["success_rate"] == 0.0
    assert stats["avg_time_cost"] == 0.0
    assert stats["avg_score"] == 0.0


def test_memory_factory_tool_memory():
    memory = MemoryFactory.tool_memory(
        user_id="u1",
        tool_name="search",
        content="Search tool pattern",
        tags=["extra"],
    )
    assert memory.type == MemoryType.TOOL
    assert "tool" in memory.tags
    assert "search" in memory.tags
    assert "extra" in memory.tags
    assert memory.metadata["tool_records"] == []
    assert memory.metadata["tool_name"] == "search"
