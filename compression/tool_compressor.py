"""
Tool Output Compression.

Compresses verbose tool outputs while preserving essential information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ctxforge.core.events import Event, EventMetadata, EventType


class CompressionStrategy(str, Enum):
    """Strategy for compressing tool output."""
    NONE = "none"
    TRUNCATE = "truncate"
    KEY_VALUE = "key_value"
    SUMMARIZE = "summarize"
    DEDUPE = "dedupe"


@dataclass
class CompressionResult:
    """Result of tool output compression."""
    original_content: str
    compressed_content: str
    strategy_used: CompressionStrategy
    tokens_saved: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """
        Calculate compression ratio.

        Returns 1.0 if original is empty, otherwise compressed/original.
        Lower is better (more compression).
        """
        orig = len(self.original_content)
        if orig == 0:
            return 1.0
        return len(self.compressed_content) / orig

    @property
    def was_compressed(self) -> bool:
        """Check if compression was actually applied."""
        return self.strategy_used != CompressionStrategy.NONE

    @property
    def chars_saved(self) -> int:
        """Calculate characters saved."""
        return len(self.original_content) - len(self.compressed_content)


class ToolOutputCompressor:
    """
    Compresses tool outputs using various strategies.

    Strategies:
    - TRUNCATE: Keep first and last N chars with omission marker
    - KEY_VALUE: Extract key-value pairs from structured data (JSON)
    - SUMMARIZE: LLM-based summarization (if provider available)
    - DEDUPE: Remove repeated patterns/lines

    Example:
        >>> compressor = ToolOutputCompressor(max_output_chars=1000)
        >>> result = compressor.compress(long_file_content, tool_name="read_file")
        >>> print(result.compressed_content)
        >>> print(f"Saved {result.tokens_saved} tokens")
    """

    # Default tool-to-strategy mappings
    DEFAULT_TOOL_CONFIGS: Dict[str, CompressionStrategy] = {
        "read_file": CompressionStrategy.TRUNCATE,
        "search": CompressionStrategy.KEY_VALUE,
        "codebase_search": CompressionStrategy.TRUNCATE,
        "list_dir": CompressionStrategy.DEDUPE,
        "grep": CompressionStrategy.TRUNCATE,
        "run_terminal_cmd": CompressionStrategy.TRUNCATE,
        "web_search": CompressionStrategy.KEY_VALUE,
    }

    def __init__(
        self,
        max_output_chars: int = 2000,
        max_json_depth: int = 2,
        compression_threshold: int = 500,
        llm_provider: Optional[Any] = None,
        tool_configs: Optional[Dict[str, CompressionStrategy]] = None,
    ):
        """
        Initialize the compressor.

        Args:
            max_output_chars: Maximum characters for compressed output
            max_json_depth: Maximum depth for JSON key extraction
            compression_threshold: Don't compress content shorter than this
            llm_provider: Optional LLM for summarization strategy
            tool_configs: Custom tool-to-strategy mappings
        """
        self._max_chars = max_output_chars
        self._max_depth = max_json_depth
        self._threshold = compression_threshold
        self._llm = llm_provider

        # Merge default configs with custom ones
        self._tool_configs = dict(self.DEFAULT_TOOL_CONFIGS)
        if tool_configs:
            self._tool_configs.update(tool_configs)

    def compress(
        self,
        content: str,
        tool_name: Optional[str] = None,
        strategy: Optional[CompressionStrategy] = None,
    ) -> CompressionResult:
        """
        Compress tool output content.

        Args:
            content: The tool output content to compress
            tool_name: Name of the tool (for strategy selection)
            strategy: Override strategy (optional)

        Returns:
            CompressionResult with compressed content and metadata
        """
        # Short content doesn't need compression
        if len(content) <= self._threshold:
            return CompressionResult(
                original_content=content,
                compressed_content=content,
                strategy_used=CompressionStrategy.NONE,
                tokens_saved=0,
                metadata={"reason": "below_threshold"},
            )

        # Content within max_chars also doesn't need compression
        if len(content) <= self._max_chars:
            return CompressionResult(
                original_content=content,
                compressed_content=content,
                strategy_used=CompressionStrategy.NONE,
                tokens_saved=0,
                metadata={"reason": "within_limit"},
            )

        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(content, tool_name)

        # Apply compression
        if strategy == CompressionStrategy.TRUNCATE:
            compressed, meta = self._compress_truncate(content)
        elif strategy == CompressionStrategy.KEY_VALUE:
            compressed, meta = self._compress_key_value(content)
        elif strategy == CompressionStrategy.DEDUPE:
            compressed, meta = self._compress_dedupe(content)
        elif strategy == CompressionStrategy.SUMMARIZE:
            # Summarize falls back to truncate if no LLM available
            if self._llm:
                compressed, meta = self._compress_summarize(content)
            else:
                compressed, meta = self._compress_truncate(content)
                meta["fallback"] = "no_llm_provider"
        else:
            compressed, meta = content[:self._max_chars], {}

        # Calculate tokens saved (estimate: ~4 chars per token)
        tokens_saved = (len(content) - len(compressed)) // 4

        return CompressionResult(
            original_content=content,
            compressed_content=compressed,
            strategy_used=strategy,
            tokens_saved=max(0, tokens_saved),
            metadata=meta,
        )

    def _select_strategy(
        self,
        content: str,
        tool_name: Optional[str],
    ) -> CompressionStrategy:
        """Select best compression strategy for content."""
        # Use tool-specific config if available
        if tool_name and tool_name in self._tool_configs:
            return self._tool_configs[tool_name]

        # Auto-detect based on content
        content_stripped = content.strip()

        # JSON detection
        if content_stripped.startswith('{') or content_stripped.startswith('['):
            try:
                json.loads(content)
                return CompressionStrategy.KEY_VALUE
            except (json.JSONDecodeError, ValueError):
                pass

        # Repeated lines detection
        lines = content.split('\n')
        if len(lines) > 20:
            unique_lines = set(line.strip() for line in lines if line.strip())
            if len(unique_lines) < len(lines) * 0.7:
                return CompressionStrategy.DEDUPE

        # Default to truncate
        return CompressionStrategy.TRUNCATE

    def _compress_truncate(
        self,
        content: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Truncate with context preservation.

        Keeps beginning (70%) and end (25%) of content with an
        omission marker in between.
        """
        if len(content) <= self._max_chars:
            return content, {}

        # Keep beginning and end
        keep_start = int(self._max_chars * 0.7)
        keep_end = int(self._max_chars * 0.25)

        start_part = content[:keep_start]
        end_part = content[-keep_end:] if keep_end > 0 else ""

        # Find clean break points (newlines)
        start_break = start_part.rfind('\n')
        if start_break > keep_start * 0.5:
            start_part = start_part[:start_break]

        if end_part:
            end_break = end_part.find('\n')
            if 0 < end_break < len(end_part) * 0.5:
                end_part = end_part[end_break + 1:]

        omitted = len(content) - len(start_part) - len(end_part)
        compressed = f"{start_part}\n\n... [{omitted} chars omitted] ...\n\n{end_part}"

        return compressed, {
            "omitted_chars": omitted,
            "original_length": len(content),
        }

    def _compress_key_value(
        self,
        content: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract key information from structured (JSON) content."""
        try:
            data = json.loads(content)
            extracted = self._extract_keys(data, depth=0)
            compressed = json.dumps(extracted, indent=2)

            # If still too long, truncate the result
            if len(compressed) > self._max_chars:
                return self._compress_truncate(compressed)

            return compressed, {
                "extraction_depth": self._max_depth,
                "original_length": len(content),
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback to truncate
            result, meta = self._compress_truncate(content)
            meta["fallback"] = "json_parse_failed"
            return result, meta

    def _extract_keys(self, obj: Any, depth: int) -> Any:
        """Recursively extract key information from nested structure."""
        if depth >= self._max_depth:
            if isinstance(obj, dict):
                return f"{{...{len(obj)} keys...}}"
            elif isinstance(obj, list):
                return f"[...{len(obj)} items...]"
            elif isinstance(obj, str) and len(obj) > 100:
                return obj[:97] + "..."
            return obj

        if isinstance(obj, dict):
            return {k: self._extract_keys(v, depth + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            if len(obj) > 5:
                first = self._extract_keys(obj[0], depth + 1)
                return [first, f"...{len(obj) - 1} more..."]
            return [self._extract_keys(item, depth + 1) for item in obj]
        elif isinstance(obj, str) and len(obj) > 200:
            return obj[:197] + "..."
        return obj

    def _compress_dedupe(
        self,
        content: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Remove duplicate lines and patterns."""
        lines = content.split('\n')

        seen: set = set()
        unique_lines: List[str] = []
        duplicates = 0

        for line in lines:
            # Normalize whitespace for comparison
            normalized = ' '.join(line.split())
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
            else:
                duplicates += 1

        compressed = '\n'.join(unique_lines)

        # Still might be too long
        if len(compressed) > self._max_chars:
            result, truncate_meta = self._compress_truncate(compressed)
            return result, {
                "duplicates_removed": duplicates,
                **truncate_meta,
            }

        if duplicates > 0:
            compressed += f"\n[{duplicates} duplicate lines removed]"

        return compressed, {
            "duplicates_removed": duplicates,
            "original_length": len(content),
        }

    def _compress_summarize(
        self,
        content: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Summarize content using LLM.

        Note: This is a placeholder for async LLM summarization.
        In practice, use the async version or fall back to truncate.
        """
        # For now, fall back to truncate since LLM calls are async
        result, meta = self._compress_truncate(content)
        meta["summarize_note"] = "LLM summarization requires async context"
        return result, meta

    def set_tool_strategy(
        self,
        tool_name: str,
        strategy: CompressionStrategy,
    ) -> None:
        """Set compression strategy for a specific tool."""
        self._tool_configs[tool_name] = strategy

    def get_tool_strategy(
        self,
        tool_name: str,
    ) -> Optional[CompressionStrategy]:
        """Get configured strategy for a tool."""
        return self._tool_configs.get(tool_name)


def compress_tool_event(
    event: Event,
    compressor: ToolOutputCompressor,
) -> Event:
    """
    Compress a tool output event's content.

    Returns a new Event with compressed content (Events are immutable).

    Args:
        event: The event to compress
        compressor: ToolOutputCompressor instance

    Returns:
        New Event with compressed content, or original if not compressed
    """
    if event.type != EventType.TOOL_OUTPUT:
        return event

    tool_name = event.metadata.tool_name if event.metadata else None
    result = compressor.compress(event.content, tool_name)

    if not result.was_compressed:
        return event

    # Build compression metadata
    compression_info = {
        "strategy": result.strategy_used.value,
        "tokens_saved": result.tokens_saved,
        "compression_ratio": round(result.compression_ratio, 3),
        "original_length": len(result.original_content),
    }
    compression_info.update(result.metadata)

    # Create new metadata with compression info
    existing_custom = event.metadata.custom if event.metadata else {}
    new_custom = {
        **existing_custom,
        "compression": compression_info,
    }

    new_metadata = EventMetadata(
        input_tokens=event.metadata.input_tokens if event.metadata else None,
        output_tokens=event.metadata.output_tokens if event.metadata else None,
        tool_name=event.metadata.tool_name if event.metadata else None,
        tool_args=event.metadata.tool_args if event.metadata else None,
        tool_result_type=event.metadata.tool_result_type if event.metadata else None,
        model=event.metadata.model if event.metadata else None,
        temperature=event.metadata.temperature if event.metadata else None,
        latency_ms=event.metadata.latency_ms if event.metadata else None,
        custom=new_custom,
        compressed=True,
        compression_strategy=result.strategy_used.value,
        original_length=len(result.original_content),
        tokens_saved=result.tokens_saved,
    )

    return Event(
        event_id=event.event_id,
        timestamp=event.timestamp,
        type=event.type,
        content=result.compressed_content,
        metadata=new_metadata,
        parent_id=event.parent_id,
        tags=list(event.tags) if event.tags else [],
    )
