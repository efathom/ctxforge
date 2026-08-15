"""
Tool Output Compression Package.

Provides compression strategies for reducing token usage in tool outputs.
"""
from ctxforge.compression.tool_compressor import (
    CompressionResult,
    CompressionStrategy,
    ToolOutputCompressor,
    compress_tool_event,
)

__all__ = [
    "CompressionStrategy",
    "CompressionResult",
    "ToolOutputCompressor",
    "compress_tool_event",
]
