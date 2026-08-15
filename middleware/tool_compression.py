"""
Tool Output Compression Middleware.

Compresses tool outputs during turn recording to save tokens.
"""
from __future__ import annotations

from typing import List, Optional

from ctxforge.compression.tool_compressor import (
    CompressionStrategy,
    ToolOutputCompressor,
    compress_tool_event,
)
from ctxforge.core.events import Event, EventType
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction


class ToolCompressionMiddleware(BaseMiddleware):
    """
    Middleware that compresses tool outputs during recording.

    This middleware runs in the 'record' phase and compresses
    tool output events before they are persisted.

    Example:
        >>> middleware = ToolCompressionMiddleware(
        ...     max_output_chars=2000,
        ...     compression_threshold=500,
        ... )
        >>> # Add to middleware chain
        >>> engine.add_middleware(middleware)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_output_chars: int = 2000,
        compression_threshold: int = 500,
        default_strategy: Optional[CompressionStrategy] = None,
    ):
        """
        Initialize the compression middleware.

        Args:
            enabled: Whether compression is enabled
            max_output_chars: Maximum characters for compressed output
            compression_threshold: Don't compress below this length
            default_strategy: Override strategy (None for auto-detection)
        """
        super().__init__(enabled=enabled)
        self._compressor = ToolOutputCompressor(
            max_output_chars=max_output_chars,
            compression_threshold=compression_threshold,
        )
        self._default_strategy = default_strategy
        self._total_tokens_saved = 0
        self._compression_count = 0

    @property
    def name(self) -> str:
        return "tool_compression"

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all compressions."""
        return self._total_tokens_saved

    @property
    def compression_count(self) -> int:
        """Number of events compressed."""
        return self._compression_count

    async def _do_process(
        self,
        context: MiddlewareContext,
        next_fn: NextFunction,
    ) -> MiddlewareContext:
        """Process and compress tool outputs."""
        # Only act during record phase
        phase = context.get_metadata("phase") or context.phase
        if phase not in ("record", "record_input_output"):
            return await next_fn(context)

        # Process session events if available
        session = context.session
        if session is None:
            return await next_fn(context)

        # Find and compress tool output events
        compressed_events: List[Event] = []
        tokens_saved = 0
        events_compressed = 0

        for event in session.events:
            if event.type == EventType.TOOL_OUTPUT:
                # Check if already compressed
                if event.metadata and event.metadata.compressed:
                    compressed_events.append(event)
                    continue

                # Compress the event
                new_event = compress_tool_event(event, self._compressor)

                if new_event is not event:
                    # Event was compressed
                    events_compressed += 1
                    if new_event.metadata and new_event.metadata.tokens_saved:
                        tokens_saved += new_event.metadata.tokens_saved

                compressed_events.append(new_event)
            else:
                compressed_events.append(event)

        # Update session events if any were compressed
        if events_compressed > 0:
            session.events = compressed_events
            self._total_tokens_saved += tokens_saved
            self._compression_count += events_compressed

            context.set_metadata("tool_compression_tokens_saved", tokens_saved)
            context.set_metadata("tool_compression_events_compressed", events_compressed)

        return await next_fn(context)

    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self._total_tokens_saved = 0
        self._compression_count = 0

    def get_stats(self) -> dict:
        """Get compression statistics."""
        return {
            "total_tokens_saved": self._total_tokens_saved,
            "compression_count": self._compression_count,
            "avg_tokens_per_compression": (
                self._total_tokens_saved / self._compression_count
                if self._compression_count > 0 else 0
            ),
        }


class SelectiveCompressionMiddleware(BaseMiddleware):
    """
    Middleware that compresses only specific tool outputs.

    Allows fine-grained control over which tools have their
    outputs compressed.
    """

    def __init__(
        self,
        enabled: bool = True,
        tools_to_compress: Optional[List[str]] = None,
        max_output_chars: int = 2000,
    ):
        """
        Initialize selective compression middleware.

        Args:
            enabled: Whether compression is enabled
            tools_to_compress: List of tool names to compress (None = all)
            max_output_chars: Maximum characters for compressed output
        """
        super().__init__(enabled=enabled)
        self._tools = set(tools_to_compress) if tools_to_compress else None
        self._compressor = ToolOutputCompressor(max_output_chars=max_output_chars)

    @property
    def name(self) -> str:
        return "selective_tool_compression"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next_fn: NextFunction,
    ) -> MiddlewareContext:
        """Process and selectively compress tool outputs."""
        phase = context.get_metadata("phase") or context.phase
        if phase not in ("record", "record_input_output"):
            return await next_fn(context)

        session = context.session
        if session is None:
            return await next_fn(context)

        compressed_events: List[Event] = []

        for event in session.events:
            if event.type == EventType.TOOL_OUTPUT:
                tool_name = event.metadata.tool_name if event.metadata else None

                # Check if this tool should be compressed
                should_compress = (
                    self._tools is None or
                    (tool_name and tool_name in self._tools)
                )

                if should_compress and not (event.metadata and event.metadata.compressed):
                    event = compress_tool_event(event, self._compressor)

            compressed_events.append(event)

        session.events = compressed_events
        return await next_fn(context)
