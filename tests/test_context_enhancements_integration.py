"""
Integration tests for context engine enhancements.

Tests the integration of:
1. Progressive Disclosure API
2. Timeline-Based Event Retrieval
3. Tool Output Compression

These tests verify that all components work together correctly.
"""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

from ctxforge.compaction.assembler import DefaultContextAssembler
from ctxforge.compression.tool_compressor import (
    CompressionStrategy,
    ToolOutputCompressor,
    compress_tool_event,
)
from ctxforge.config.base import (
    DynamicContextConfig,
    ProgressiveDisclosureConfig,
    TimelineConfig,
    ToolCompressionConfig,
)
from ctxforge.core.events import Event, EventMetadata, EventType
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.memory_index import DisclosureLevel, MemoryIndex, MemoryIndexEntry
from ctxforge.core.session import Session
from ctxforge.core.timeline import TimelineFilter  # noqa: F401 - used indirectly
from ctxforge.engine.services.timeline_service import TimelineService
from ctxforge.middleware.tool_compression import ToolCompressionMiddleware


def create_test_session_with_tool_outputs() -> Session:
    """Create a session with various event types including tool outputs."""
    session = Session(user_id="test-user")
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)

    # Turn 1: Weather query
    session.add_event(Event(
        type=EventType.USER,
        content="What's the weather?",
        timestamp=base_time,
    ))
    session.add_event(Event(
        type=EventType.TOOL_CALL,
        content="get_weather()",
        timestamp=base_time + timedelta(seconds=1),
        metadata=EventMetadata(tool_name="get_weather"),
    ))
    session.add_event(Event(
        type=EventType.TOOL_OUTPUT,
        content='{"temp": 72, "condition": "sunny", "humidity": 45}',
        timestamp=base_time + timedelta(seconds=2),
        metadata=EventMetadata(tool_name="get_weather"),
    ))
    session.add_event(Event(
        type=EventType.AGENT,
        content="It's 72°F and sunny with 45% humidity.",
        timestamp=base_time + timedelta(seconds=3),
    ))

    # Turn 2: File read (large output)
    session.add_event(Event(
        type=EventType.USER,
        content="Read my config file",
        timestamp=base_time + timedelta(minutes=5),
    ))
    session.add_event(Event(
        type=EventType.TOOL_CALL,
        content="read_file('config.yaml')",
        timestamp=base_time + timedelta(minutes=5, seconds=1),
        metadata=EventMetadata(tool_name="read_file"),
    ))
    # Large file content
    large_config = "# Configuration\n" + "\n".join([
        f"setting_{i}: value_{i}" for i in range(100)
    ])
    session.add_event(Event(
        type=EventType.TOOL_OUTPUT,
        content=large_config,
        timestamp=base_time + timedelta(minutes=5, seconds=2),
        metadata=EventMetadata(tool_name="read_file"),
    ))
    session.add_event(Event(
        type=EventType.AGENT,
        content="Here's your config file...",
        timestamp=base_time + timedelta(minutes=5, seconds=3),
    ))

    # Turn 3: Test run
    session.add_event(Event(
        type=EventType.USER,
        content="Run tests",
        timestamp=base_time + timedelta(minutes=10),
    ))
    session.add_event(Event(
        type=EventType.TOOL_CALL,
        content="run_terminal_cmd('pytest')",
        timestamp=base_time + timedelta(minutes=10, seconds=1),
        metadata=EventMetadata(tool_name="run_terminal_cmd"),
    ))
    session.add_event(Event(
        type=EventType.TOOL_OUTPUT,
        content="===== 50 passed in 3.5s =====",
        timestamp=base_time + timedelta(minutes=10, seconds=5),
        metadata=EventMetadata(tool_name="run_terminal_cmd"),
    ))
    session.add_event(Event(
        type=EventType.AGENT,
        content="All 50 tests passed!",
        timestamp=base_time + timedelta(minutes=10, seconds=6),
    ))

    return session


def create_test_memories() -> List[MemoryItem]:
    """Create test memories with headlines."""
    return [
        MemoryItem(
            memory_id="mem_1",
            user_id="test-user",
            content="User prefers dark mode in all applications.",
            type=MemoryType.SEMANTIC,
            headline="Prefers dark mode",
            subtitle="Always enables dark mode for comfort",
            confidence_score=0.95,
        ),
        MemoryItem(
            memory_id="mem_2",
            user_id="test-user",
            content="User is experienced with Python and uses pytest.",
            type=MemoryType.SEMANTIC,
            headline="Python developer",
            subtitle="Experienced with pytest for testing",
            confidence_score=0.90,
        ),
        MemoryItem(
            memory_id="mem_3",
            user_id="test-user",
            content="Working on a FastAPI backend project.",
            type=MemoryType.EPISODIC,
            headline="FastAPI project",
            subtitle="Backend service development",
            confidence_score=0.85,
        ),
    ]


class TestProgressiveDisclosureIntegration(unittest.TestCase):
    """Integration tests for Progressive Disclosure."""

    def setUp(self):
        """Set up test data."""
        self.memories = create_test_memories()

    def test_memory_index_from_memories(self):
        """MemoryIndex can be built from MemoryItems with headlines."""
        index = MemoryIndex()  # Don't pass total_memories, let add() handle it
        for memory in self.memories:
            entry = MemoryIndexEntry.from_memory(memory)
            index.add(entry)

        self.assertEqual(len(index.entries), 3)
        self.assertEqual(index.total_memories, 3)

        # Check entry has correct data
        entry = index.entries[0]
        self.assertEqual(entry.headline, "Prefers dark mode")
        self.assertEqual(entry.subtitle, "Always enables dark mode for comfort")

    def test_disclosure_levels_token_savings(self):
        """Different disclosure levels have expected token differences."""
        index = MemoryIndex(total_memories=len(self.memories))
        for memory in self.memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        headline_tokens = index.estimate_tokens(DisclosureLevel.HEADLINE)
        full_tokens = index.estimate_tokens(DisclosureLevel.FULL)

        # Headlines should be significantly smaller
        self.assertLess(headline_tokens, full_tokens)
        # At least 20% savings
        self.assertLess(headline_tokens, full_tokens * 0.8)

    def test_assembler_progressive_format(self):
        """DefaultContextAssembler supports progressive memory format."""
        assembler = DefaultContextAssembler(
            memory_format="progressive",
            use_progressive_disclosure=True,
        )

        formatted = assembler._format_memories(self.memories)

        # Progressive format includes memory content (truncated)
        self.assertIn("User prefers dark mode", formatted)
        # Should be structured output
        self.assertIn("User Context", formatted)


class TestTimelineIntegration(unittest.TestCase):
    """Integration tests for Timeline-Based Event Retrieval."""

    def setUp(self):
        """Set up test session."""
        self.session = create_test_session_with_tool_outputs()
        self.timeline_service = TimelineService()

    def test_session_has_timeline_methods(self):
        """Session has timeline query methods."""
        # Check methods exist
        self.assertTrue(hasattr(self.session, "query_timeline"))
        self.assertTrue(hasattr(self.session, "get_turns"))
        self.assertTrue(hasattr(self.session, "get_last_n_turn_events"))

    def test_get_turns_groups_correctly(self):
        """get_turns correctly groups events by USER events."""
        turns = self.session.get_turns()

        self.assertEqual(len(turns), 3)  # 3 user messages = 3 turns
        # Each turn starts with USER
        for turn in turns:
            self.assertEqual(turn[0].type, EventType.USER)

    def test_timeline_service_filters_tool_events(self):
        """TimelineService can exclude tool events."""
        result = self.timeline_service.get_user_assistant_exchanges(self.session)

        # No tool events
        for event in result.events:
            self.assertNotIn(event.type, [EventType.TOOL_CALL, EventType.TOOL_OUTPUT])

    def test_timeline_service_summarizes_activity(self):
        """TimelineService provides activity summary."""
        summary = self.timeline_service.summarize_activity(self.session)

        self.assertIn("event_counts", summary)
        self.assertIn("turn_count", summary)
        self.assertEqual(summary["turn_count"], 3)

    def test_last_n_turns_filters_correctly(self):
        """get_last_n_turns returns only recent turns."""
        events = self.session.get_last_n_turn_events(n=1)

        # Should have events from last turn only
        user_events = [e for e in events if e.type == EventType.USER]
        self.assertEqual(len(user_events), 1)
        self.assertIn("Run tests", user_events[0].content)


class TestToolCompressionIntegration(unittest.TestCase):
    """Integration tests for Tool Output Compression."""

    def setUp(self):
        """Set up compressor."""
        self.compressor = ToolOutputCompressor(
            max_output_chars=500,
            compression_threshold=100,
        )

    def test_compress_large_tool_output(self):
        """Large tool outputs are compressed."""
        large_content = "# File content\n" + "\n".join([
            f"line {i}: " + "x" * 50 for i in range(100)
        ])

        result = self.compressor.compress(large_content, tool_name="read_file")

        self.assertTrue(result.was_compressed)
        self.assertLess(len(result.compressed_content), len(large_content))
        self.assertGreater(result.tokens_saved, 0)

    def test_compress_json_output(self):
        """JSON outputs use key-value extraction."""
        json_content = json.dumps({
            "items": [{"id": i, "data": f"item_{i}" * 20} for i in range(20)],
            "total": 20,
        }, indent=2)

        result = self.compressor.compress(json_content, strategy=CompressionStrategy.KEY_VALUE)

        self.assertEqual(result.strategy_used, CompressionStrategy.KEY_VALUE)
        # Result should still be valid JSON (or truncated)
        self.assertIn("items", result.compressed_content)

    def test_compress_tool_event_adds_metadata(self):
        """compress_tool_event adds compression metadata to event."""
        large_content = "x" * 5000
        event = Event(
            type=EventType.TOOL_OUTPUT,
            content=large_content,
            timestamp=datetime.now(timezone.utc),
            metadata=EventMetadata(tool_name="read_file"),
        )

        compressed_event = compress_tool_event(event, self.compressor)

        self.assertTrue(compressed_event.metadata.compressed)
        self.assertIsNotNone(compressed_event.metadata.compression_strategy)
        self.assertIsNotNone(compressed_event.metadata.tokens_saved)

    def test_small_outputs_not_compressed(self):
        """Small tool outputs are not compressed."""
        small_content = "OK"

        result = self.compressor.compress(small_content)

        self.assertFalse(result.was_compressed)
        self.assertEqual(result.compressed_content, small_content)


class TestMiddlewareIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Tool Compression Middleware."""

    async def test_middleware_compresses_session_events(self):
        """ToolCompressionMiddleware compresses tool outputs in session."""
        session = create_test_session_with_tool_outputs()
        middleware = ToolCompressionMiddleware(
            max_output_chars=500,
            compression_threshold=100,
        )

        # Create mock context
        context = MagicMock()
        context.phase = "record"
        context.session = session
        context.get_metadata.return_value = "record"

        next_fn = AsyncMock(return_value=context)

        # Process
        await middleware._do_process(context, next_fn)

        # Check large tool output was compressed
        tool_outputs = [e for e in session.events if e.type == EventType.TOOL_OUTPUT]
        compressed_count = sum(
            1 for e in tool_outputs
            if e.metadata and e.metadata.compressed
        )
        self.assertGreater(compressed_count, 0)


class TestCombinedEnhancements(unittest.TestCase):
    """Test all enhancements working together."""

    def test_combined_token_savings(self):
        """Combined enhancements provide significant token savings."""
        session = create_test_session_with_tool_outputs()
        memories = create_test_memories()

        # Calculate original tokens
        event_tokens = sum(len(e.content) for e in session.events) // 4
        memory_tokens = sum(len(m.content) for m in memories) // 4
        original_total = event_tokens + memory_tokens

        # Apply timeline filtering (no tool events)
        filtered_events = [
            e for e in session.events
            if e.type in (EventType.USER, EventType.AGENT)
        ]
        filtered_event_tokens = sum(len(e.content) for e in filtered_events) // 4

        # Apply progressive disclosure (headlines only)
        index = MemoryIndex(total_memories=len(memories))
        for m in memories:
            index.add(MemoryIndexEntry.from_memory(m))
        headline_tokens = index.estimate_tokens(DisclosureLevel.HEADLINE)

        # Calculate savings
        optimized_total = filtered_event_tokens + headline_tokens
        savings_pct = (original_total - optimized_total) / original_total * 100

        # Should have meaningful savings
        self.assertGreater(savings_pct, 20)

    def test_all_configs_in_dynamic_context(self):
        """All enhancement configs are in DynamicContextConfig."""
        config = DynamicContextConfig()

        self.assertTrue(hasattr(config, "progressive_disclosure"))
        self.assertTrue(hasattr(config, "timeline"))
        self.assertTrue(hasattr(config, "tool_compression"))

        # Check they have expected types
        self.assertIsInstance(config.progressive_disclosure, ProgressiveDisclosureConfig)
        self.assertIsInstance(config.timeline, TimelineConfig)
        self.assertIsInstance(config.tool_compression, ToolCompressionConfig)


class TestConfigDefaults(unittest.TestCase):
    """Test configuration defaults are sensible."""

    def test_progressive_disclosure_defaults(self):
        """Progressive disclosure has sensible defaults."""
        config = ProgressiveDisclosureConfig()

        self.assertFalse(config.enabled)  # Opt-in
        self.assertEqual(config.max_headline_chars, 80)
        self.assertEqual(config.expand_top_n, 3)

    def test_timeline_defaults(self):
        """Timeline config has sensible defaults."""
        config = TimelineConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.default_time_range, "this_session")
        self.assertFalse(config.include_timestamps_in_history)

    def test_compression_defaults(self):
        """Compression config has sensible defaults."""
        config = ToolCompressionConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.max_output_chars, 2000)
        self.assertEqual(config.compression_threshold, 500)
        self.assertEqual(config.default_strategy, "auto")


if __name__ == "__main__":
    unittest.main()
