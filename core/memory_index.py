"""
Progressive Disclosure Memory Index.

Provides compact memory representations with on-demand expansion.
Enables efficient context assembly by showing headlines first,
then expanding high-relevance memories based on token budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from ctxforge.core.memory import MemoryItem


class DisclosureLevel(str, Enum):
    """Level of detail for memory disclosure."""
    HEADLINE = "headline"    # 1-line summary (~20-30 tokens)
    SUMMARY = "summary"      # Headline + subtitle (~50-100 tokens)
    FULL = "full"           # Complete content


@dataclass
class MemoryIndexEntry:
    """
    Compact memory representation for progressive disclosure.

    Uses stored headline/subtitle from MemoryItem (LLM-generated
    and persisted for efficient retrieval).
    """
    memory_id: str
    memory_type: str  # MemoryType.value
    headline: str                    # ~20-30 tokens (stored in MemoryItem)
    subtitle: Optional[str] = None   # ~50-100 tokens (stored in MemoryItem)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    # Reference to full content (for expansion)
    _full_content: Optional[str] = None

    @classmethod
    def from_memory(
        cls,
        memory: "MemoryItem",
        headline: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> "MemoryIndexEntry":
        """
        Create index entry from a MemoryItem.

        Uses stored headline/subtitle from memory if available,
        otherwise uses provided values or falls back to truncated content.
        """
        # Get headline from memory or parameter, fallback to truncated content
        final_headline = headline
        if final_headline is None:
            final_headline = getattr(memory, 'headline', None)
        if final_headline is None:
            final_headline = memory.content[:80]
            if len(memory.content) > 80:
                final_headline += "..."

        # Get subtitle from memory or parameter
        final_subtitle = subtitle
        if final_subtitle is None:
            final_subtitle = getattr(memory, 'subtitle', None)

        return cls(
            memory_id=memory.memory_id,
            memory_type=memory.type.value,
            headline=final_headline,
            subtitle=final_subtitle,
            confidence=memory.confidence_score,
            created_at=memory.created_at,
            tags=list(memory.tags),
            _full_content=memory.content,
        )

    def to_prompt(self, level: DisclosureLevel = DisclosureLevel.HEADLINE) -> str:
        """Format entry for prompt inclusion at specified disclosure level."""
        if level == DisclosureLevel.HEADLINE:
            return f"• [{self.memory_type}] {self.headline}"
        elif level == DisclosureLevel.SUMMARY:
            content = self.subtitle or self.headline
            if self.subtitle and self.headline:
                content = f"{self.headline}: {self.subtitle}"
            return f"• [{self.memory_type}] {content}"
        else:  # FULL
            content = self._full_content or self.subtitle or self.headline
            return f"• [{self.memory_type}] {content}"

    def estimate_tokens(self, level: DisclosureLevel = DisclosureLevel.HEADLINE) -> int:
        """Estimate token count at specified disclosure level."""
        text = self.to_prompt(level)
        return len(text) // 4 + 1


@dataclass
class MemoryIndex:
    """
    Collection of memory index entries with progressive disclosure support.

    Enables efficient context assembly by showing headlines first,
    then expanding high-relevance memories.
    """
    entries: List[MemoryIndexEntry] = field(default_factory=list)
    total_memories: int = 0

    def add(self, entry: MemoryIndexEntry) -> None:
        """Add an entry to the index."""
        self.entries.append(entry)
        self.total_memories += 1

    def to_prompt(
        self,
        level: DisclosureLevel = DisclosureLevel.HEADLINE,
        expand_top_n: int = 0,
        max_entries: int = 20,
    ) -> str:
        """
        Format index for prompt inclusion.

        Args:
            level: Base disclosure level for all entries
            expand_top_n: Expand first N entries to FULL level
            max_entries: Maximum entries to include
        """
        if not self.entries:
            return ""

        lines = [f"User Context ({len(self.entries)} relevant memories):"]

        for i, entry in enumerate(self.entries[:max_entries]):
            if i < expand_top_n:
                lines.append(entry.to_prompt(DisclosureLevel.FULL))
            else:
                lines.append(entry.to_prompt(level))

        if self.total_memories > max_entries:
            lines.append(f"  ... and {self.total_memories - max_entries} more")

        return "\n".join(lines)

    def estimate_tokens(
        self,
        level: DisclosureLevel = DisclosureLevel.HEADLINE,
        expand_top_n: int = 0,
        max_entries: int = 20,
    ) -> int:
        """Estimate total tokens for prompt inclusion."""
        if not self.entries:
            return 0

        total = 10  # Header

        for i, entry in enumerate(self.entries[:max_entries]):
            if i < expand_top_n:
                total += entry.estimate_tokens(DisclosureLevel.FULL)
            else:
                total += entry.estimate_tokens(level)

        return total

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.entries)
