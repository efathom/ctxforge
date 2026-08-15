"""
Tests for tool output compression.

Tests CompressionStrategy, CompressionResult, ToolOutputCompressor,
compress_tool_event function, and ToolCompressionMiddleware.
"""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from ctxforge.compression.tool_compressor import (
    CompressionResult,
    CompressionStrategy,
    ToolOutputCompressor,
    compress_tool_event,
)
from ctxforge.config.base import ToolCompressionConfig
from ctxforge.core.events import Event, EventMetadata, EventType
from ctxforge.core.session import Session
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.tool_compression import (
    SelectiveCompressionMiddleware,
    ToolCompressionMiddleware,
)


def create_tool_output_event(
    content: str,
    tool_name: str = "test_tool",
) -> Event:
    """Helper to create tool output events."""
    return Event(
        type=EventType.TOOL_OUTPUT,
        content=content,
        timestamp=datetime.now(timezone.utc),
        metadata=EventMetadata(tool_name=tool_name),
    )


def create_long_content(chars: int = 5000) -> str:
    """Create long unique content for testing (no duplicates)."""
    import hashlib
    lines = []
    i = 0
    while len("\n".join(lines)) < chars:
        # Use hash to make each line unique
        unique_part = hashlib.md5(str(i).encode()).hexdigest()[:8]
        lines.append(f"Line {i}: content_{unique_part} with unique data here")
        i += 1
    return "\n".join(lines)


def create_json_content(items: int = 100) -> str:
    """Create JSON content for testing."""
    data = {
        "results": [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"This is a detailed description for item {i}",
                "metadata": {"key": f"value_{i}", "nested": {"a": i, "b": i * 2}},
            }
            for i in range(items)
        ],
        "total": items,
        "page": 1,
    }
    return json.dumps(data, indent=2)


def create_duplicate_content(unique_lines: int = 10, duplicates: int = 100) -> str:
    """Create content with duplicate lines (long enough to trigger compression)."""
    lines = []
    # Make lines longer to exceed threshold
    for i in range(unique_lines):
        lines.append(f"Unique line {i}: " + "x" * 50)
    # Add many duplicates
    for i in range(duplicates):
        lines.append(f"Unique line {i % unique_lines}: " + "x" * 50)
    return "\n".join(lines)


class TestCompressionStrategy(unittest.TestCase):
    """Tests for CompressionStrategy enum."""

    def test_enum_values(self):
        """Strategy enum has expected values."""
        self.assertEqual(CompressionStrategy.NONE.value, "none")
        self.assertEqual(CompressionStrategy.TRUNCATE.value, "truncate")
        self.assertEqual(CompressionStrategy.KEY_VALUE.value, "key_value")
        self.assertEqual(CompressionStrategy.SUMMARIZE.value, "summarize")
        self.assertEqual(CompressionStrategy.DEDUPE.value, "dedupe")

    def test_string_comparison(self):
        """Strategy can be compared as string."""
        self.assertEqual(CompressionStrategy.TRUNCATE, "truncate")


class TestCompressionResult(unittest.TestCase):
    """Tests for CompressionResult."""

    def test_compression_ratio_normal(self):
        """Compression ratio calculated correctly."""
        result = CompressionResult(
            original_content="a" * 1000,
            compressed_content="a" * 500,
            strategy_used=CompressionStrategy.TRUNCATE,
            tokens_saved=125,
        )
        self.assertEqual(result.compression_ratio, 0.5)

    def test_compression_ratio_empty(self):
        """Empty original returns 1.0."""
        result = CompressionResult(
            original_content="",
            compressed_content="",
            strategy_used=CompressionStrategy.NONE,
            tokens_saved=0,
        )
        self.assertEqual(result.compression_ratio, 1.0)

    def test_was_compressed_true(self):
        """was_compressed True when compression applied."""
        result = CompressionResult(
            original_content="test",
            compressed_content="t",
            strategy_used=CompressionStrategy.TRUNCATE,
            tokens_saved=1,
        )
        self.assertTrue(result.was_compressed)

    def test_was_compressed_false(self):
        """was_compressed False when no compression."""
        result = CompressionResult(
            original_content="test",
            compressed_content="test",
            strategy_used=CompressionStrategy.NONE,
            tokens_saved=0,
        )
        self.assertFalse(result.was_compressed)

    def test_chars_saved(self):
        """chars_saved calculated correctly."""
        result = CompressionResult(
            original_content="a" * 1000,
            compressed_content="a" * 300,
            strategy_used=CompressionStrategy.TRUNCATE,
            tokens_saved=175,
        )
        self.assertEqual(result.chars_saved, 700)


class TestToolOutputCompressor(unittest.TestCase):
    """Tests for ToolOutputCompressor."""

    def setUp(self):
        """Create compressor for tests."""
        self.compressor = ToolOutputCompressor(
            max_output_chars=2000,
            compression_threshold=500,
        )

    def test_short_content_not_compressed(self):
        """Content below threshold not compressed."""
        content = "Short content"
        result = self.compressor.compress(content)

        self.assertEqual(result.strategy_used, CompressionStrategy.NONE)
        self.assertEqual(result.compressed_content, content)
        self.assertEqual(result.tokens_saved, 0)

    def test_content_within_limit_not_compressed(self):
        """Content within max_chars not compressed."""
        content = "a" * 1500  # Below 2000 limit
        result = self.compressor.compress(content)

        self.assertEqual(result.strategy_used, CompressionStrategy.NONE)
        self.assertEqual(result.compressed_content, content)

    def test_truncate_long_content(self):
        """Long content truncated with markers."""
        content = create_long_content(5000)
        result = self.compressor.compress(content, strategy=CompressionStrategy.TRUNCATE)

        self.assertEqual(result.strategy_used, CompressionStrategy.TRUNCATE)
        self.assertIn("chars omitted", result.compressed_content)
        self.assertLess(len(result.compressed_content), len(content))
        self.assertGreater(result.tokens_saved, 0)

    def test_truncate_preserves_beginning_and_end(self):
        """Truncation keeps beginning and end."""
        content = "START_MARKER\n" + "x" * 5000 + "\nEND_MARKER"
        result = self.compressor.compress(content, strategy=CompressionStrategy.TRUNCATE)

        self.assertIn("START_MARKER", result.compressed_content)
        self.assertIn("END_MARKER", result.compressed_content)

    def test_key_value_json_content(self):
        """JSON content compressed with key extraction."""
        content = create_json_content(100)
        result = self.compressor.compress(content, strategy=CompressionStrategy.KEY_VALUE)

        self.assertEqual(result.strategy_used, CompressionStrategy.KEY_VALUE)
        self.assertLess(len(result.compressed_content), len(content))
        # Should still be valid JSON
        parsed = json.loads(result.compressed_content)
        self.assertIn("results", parsed)

    def test_key_value_truncates_long_arrays(self):
        """Long arrays truncated in JSON."""
        content = create_json_content(50)
        result = self.compressor.compress(content, strategy=CompressionStrategy.KEY_VALUE)

        self.assertIn("more...", result.compressed_content)

    def test_key_value_fallback_invalid_json(self):
        """Invalid JSON falls back to truncate."""
        content = "not json " * 1000
        result = self.compressor.compress(content, strategy=CompressionStrategy.KEY_VALUE)

        # Should still compress (fallback to truncate)
        self.assertIn("fallback", result.metadata)
        self.assertLess(len(result.compressed_content), len(content))

    def test_dedupe_removes_duplicates(self):
        """Duplicate lines removed."""
        content = create_duplicate_content(10, 50)
        result = self.compressor.compress(content, strategy=CompressionStrategy.DEDUPE)

        self.assertEqual(result.strategy_used, CompressionStrategy.DEDUPE)
        self.assertIn("duplicate lines removed", result.compressed_content)
        self.assertGreater(result.metadata.get("duplicates_removed", 0), 0)

    def test_auto_strategy_json(self):
        """Auto-detect selects KEY_VALUE for JSON."""
        content = create_json_content(50)
        result = self.compressor.compress(content)

        self.assertEqual(result.strategy_used, CompressionStrategy.KEY_VALUE)

    def test_auto_strategy_duplicates(self):
        """Auto-detect selects DEDUPE for duplicate content."""
        content = create_duplicate_content(5, 100)
        result = self.compressor.compress(content)

        self.assertEqual(result.strategy_used, CompressionStrategy.DEDUPE)

    def test_auto_strategy_default_truncate(self):
        """Auto-detect defaults to TRUNCATE."""
        content = create_long_content(5000)
        result = self.compressor.compress(content)

        self.assertEqual(result.strategy_used, CompressionStrategy.TRUNCATE)

    def test_tool_specific_strategy(self):
        """Tool-specific strategy used."""
        result = self.compressor.compress(
            create_long_content(5000),
            tool_name="read_file"
        )
        self.assertEqual(result.strategy_used, CompressionStrategy.TRUNCATE)

    def test_custom_tool_config(self):
        """Custom tool config respected."""
        compressor = ToolOutputCompressor(
            tool_configs={"my_tool": CompressionStrategy.DEDUPE}
        )
        # Even if content doesn't have duplicates, specified strategy used
        result = compressor.compress(create_long_content(5000), tool_name="my_tool")
        self.assertEqual(result.strategy_used, CompressionStrategy.DEDUPE)

    def test_set_tool_strategy(self):
        """Can set tool strategy dynamically."""
        self.compressor.set_tool_strategy("custom_tool", CompressionStrategy.KEY_VALUE)
        strategy = self.compressor.get_tool_strategy("custom_tool")
        self.assertEqual(strategy, CompressionStrategy.KEY_VALUE)


class TestCompressToolEvent(unittest.TestCase):
    """Tests for compress_tool_event function."""

    def setUp(self):
        """Create compressor and events."""
        self.compressor = ToolOutputCompressor(
            max_output_chars=2000,
            compression_threshold=500,
        )

    def test_compress_tool_output_event(self):
        """Tool output event compressed."""
        event = create_tool_output_event(create_long_content(5000))
        result = compress_tool_event(event, self.compressor)

        # Event ID is preserved, but content is different
        self.assertEqual(result.event_id, event.event_id)  # Same ID
        self.assertLess(len(result.content), len(event.content))
        self.assertTrue(result.metadata.compressed)
        self.assertIsNotNone(result.metadata.compression_strategy)

    def test_short_event_not_compressed(self):
        """Short event returned unchanged."""
        event = create_tool_output_event("short content")
        result = compress_tool_event(event, self.compressor)

        self.assertEqual(result, event)  # Same event

    def test_non_tool_output_not_compressed(self):
        """Non-tool-output events not compressed."""
        event = Event(
            type=EventType.USER,
            content=create_long_content(5000),
            timestamp=datetime.now(timezone.utc),
        )
        result = compress_tool_event(event, self.compressor)

        self.assertEqual(result, event)  # Unchanged

    def test_compression_metadata_added(self):
        """Compression metadata added to event."""
        event = create_tool_output_event(create_long_content(5000))
        result = compress_tool_event(event, self.compressor)

        self.assertTrue(result.metadata.compressed)
        self.assertIsNotNone(result.metadata.compression_strategy)
        self.assertIsNotNone(result.metadata.original_length)
        self.assertIsNotNone(result.metadata.tokens_saved)

    def test_custom_metadata_preserved(self):
        """Existing custom metadata preserved."""
        event = Event(
            type=EventType.TOOL_OUTPUT,
            content=create_long_content(5000),
            timestamp=datetime.now(timezone.utc),
            metadata=EventMetadata(
                tool_name="test_tool",
                custom={"existing_key": "existing_value"},
            ),
        )
        result = compress_tool_event(event, self.compressor)

        self.assertIn("existing_key", result.metadata.custom)
        self.assertEqual(result.metadata.custom["existing_key"], "existing_value")


class TestToolCompressionMiddleware(unittest.IsolatedAsyncioTestCase):
    """Tests for ToolCompressionMiddleware."""

    def setUp(self):
        """Create middleware for tests."""
        self.middleware = ToolCompressionMiddleware(
            max_output_chars=2000,
            compression_threshold=500,
        )

    def test_name_property(self):
        """Middleware has correct name."""
        self.assertEqual(self.middleware.name, "tool_compression")

    def test_initial_stats(self):
        """Initial stats are zero."""
        self.assertEqual(self.middleware.total_tokens_saved, 0)
        self.assertEqual(self.middleware.compression_count, 0)

    async def test_compresses_tool_outputs(self):
        """Middleware compresses tool outputs in session."""
        # Create session with tool output
        session = Session(user_id="test-user")
        session.add_event(create_tool_output_event(create_long_content(5000)))

        # Create context
        context = MagicMock(spec=MiddlewareContext)
        context.phase = "record"
        context.session = session
        context.get_metadata.return_value = "record"

        # Create next function
        next_fn = AsyncMock(return_value=context)

        # Process
        await self.middleware._do_process(context, next_fn)

        # Check compression
        self.assertGreater(self.middleware.total_tokens_saved, 0)
        self.assertEqual(self.middleware.compression_count, 1)

        # Check event was compressed
        compressed_event = session.events[0]
        self.assertTrue(compressed_event.metadata.compressed)

    async def test_skips_non_record_phase(self):
        """Middleware skips non-record phases."""
        context = MagicMock(spec=MiddlewareContext)
        context.phase = "query"
        context.get_metadata.return_value = "query"

        next_fn = AsyncMock(return_value=context)

        await self.middleware._do_process(context, next_fn)

        # Should just pass through
        next_fn.assert_called_once()
        self.assertEqual(self.middleware.compression_count, 0)

    async def test_skips_already_compressed(self):
        """Middleware skips already compressed events."""
        session = Session(user_id="test-user")
        # Create already compressed event
        event = Event(
            type=EventType.TOOL_OUTPUT,
            content=create_long_content(5000),
            timestamp=datetime.now(timezone.utc),
            metadata=EventMetadata(
                tool_name="test",
                compressed=True,
            ),
        )
        session.add_event(event)

        context = MagicMock(spec=MiddlewareContext)
        context.phase = "record"
        context.session = session
        context.get_metadata.return_value = "record"

        next_fn = AsyncMock(return_value=context)

        await self.middleware._do_process(context, next_fn)

        # Should not increment count
        self.assertEqual(self.middleware.compression_count, 0)

    def test_reset_stats(self):
        """Stats can be reset."""
        self.middleware._total_tokens_saved = 100
        self.middleware._compression_count = 5

        self.middleware.reset_stats()

        self.assertEqual(self.middleware.total_tokens_saved, 0)
        self.assertEqual(self.middleware.compression_count, 0)

    def test_get_stats(self):
        """get_stats returns correct dict."""
        self.middleware._total_tokens_saved = 500
        self.middleware._compression_count = 5

        stats = self.middleware.get_stats()

        self.assertEqual(stats["total_tokens_saved"], 500)
        self.assertEqual(stats["compression_count"], 5)
        self.assertEqual(stats["avg_tokens_per_compression"], 100.0)


class TestSelectiveCompressionMiddleware(unittest.IsolatedAsyncioTestCase):
    """Tests for SelectiveCompressionMiddleware."""

    def test_name_property(self):
        """Middleware has correct name."""
        middleware = SelectiveCompressionMiddleware()
        self.assertEqual(middleware.name, "selective_tool_compression")

    async def test_compresses_specified_tools(self):
        """Only specified tools compressed."""
        middleware = SelectiveCompressionMiddleware(
            tools_to_compress=["read_file"],
        )

        session = Session(user_id="test-user")
        session.add_event(create_tool_output_event(
            create_long_content(5000),
            tool_name="read_file"
        ))
        session.add_event(create_tool_output_event(
            create_long_content(5000),
            tool_name="other_tool"
        ))

        context = MagicMock(spec=MiddlewareContext)
        context.phase = "record"
        context.session = session
        context.get_metadata.return_value = "record"

        next_fn = AsyncMock(return_value=context)

        await middleware._do_process(context, next_fn)

        # First event (read_file) should be compressed
        self.assertTrue(session.events[0].metadata.compressed)
        # Second event (other_tool) should not be compressed
        self.assertFalse(session.events[1].metadata.compressed)

    async def test_compresses_all_when_none_specified(self):
        """All tools compressed when no list specified."""
        middleware = SelectiveCompressionMiddleware(tools_to_compress=None)

        session = Session(user_id="test-user")
        session.add_event(create_tool_output_event(
            create_long_content(5000),
            tool_name="any_tool"
        ))

        context = MagicMock(spec=MiddlewareContext)
        context.phase = "record"
        context.session = session
        context.get_metadata.return_value = "record"

        next_fn = AsyncMock(return_value=context)

        await middleware._do_process(context, next_fn)

        self.assertTrue(session.events[0].metadata.compressed)


class TestToolCompressionConfig(unittest.TestCase):
    """Tests for ToolCompressionConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = ToolCompressionConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.max_output_chars, 2000)
        self.assertEqual(config.compression_threshold, 500)
        self.assertEqual(config.default_strategy, "auto")
        self.assertEqual(config.tool_configs, {})

    def test_custom_values(self):
        """Config accepts custom values."""
        config = ToolCompressionConfig(
            enabled=False,
            max_output_chars=1000,
            compression_threshold=200,
            default_strategy="truncate",
            tool_configs={"read_file": "truncate"},
        )

        self.assertFalse(config.enabled)
        self.assertEqual(config.max_output_chars, 1000)
        self.assertEqual(config.compression_threshold, 200)
        self.assertEqual(config.default_strategy, "truncate")
        self.assertEqual(config.tool_configs["read_file"], "truncate")


class TestEventMetadataCompression(unittest.TestCase):
    """Tests for compression fields in EventMetadata."""

    def test_default_compression_fields(self):
        """Compression fields have correct defaults."""
        metadata = EventMetadata()

        self.assertFalse(metadata.compressed)
        self.assertIsNone(metadata.compression_strategy)
        self.assertIsNone(metadata.original_length)
        self.assertIsNone(metadata.tokens_saved)

    def test_set_compression_fields(self):
        """Compression fields can be set."""
        metadata = EventMetadata(
            compressed=True,
            compression_strategy="truncate",
            original_length=5000,
            tokens_saved=750,
        )

        self.assertTrue(metadata.compressed)
        self.assertEqual(metadata.compression_strategy, "truncate")
        self.assertEqual(metadata.original_length, 5000)
        self.assertEqual(metadata.tokens_saved, 750)


if __name__ == "__main__":
    unittest.main()
