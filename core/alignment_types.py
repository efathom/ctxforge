"""
Core alignment type definitions.

These types are defined here to avoid circular dependencies between
the protocols and extraction modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AlignmentStatus(str, Enum):
    """Status of text alignment between extraction and source."""
    
    MATCH_EXACT = "exact"
    MATCH_FUZZY = "fuzzy"
    MATCH_PARTIAL = "partial"
    UNALIGNED = "unaligned"


@dataclass
class CharSpan:
    """Character-level span in source text."""
    
    start_pos: int
    end_pos: int
    
    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos
    
    def to_tuple(self) -> tuple[int, int]:
        return (self.start_pos, self.end_pos)
    
    @classmethod
    def from_tuple(cls, t: tuple[int, int]) -> "CharSpan":
        return cls(start_pos=t[0], end_pos=t[1])
    
    def overlaps(self, other: "CharSpan") -> bool:
        """Check if this span overlaps with another."""
        return not (self.end_pos <= other.start_pos or other.end_pos <= self.start_pos)
    
    def contains(self, pos: int) -> bool:
        """Check if a position is within this span."""
        return self.start_pos <= pos < self.end_pos


@dataclass
class TokenSpan:
    """Token-level span in tokenized text."""
    
    start_index: int
    end_index: int
    
    @property
    def length(self) -> int:
        return self.end_index - self.start_index


@dataclass
class AlignmentResult:
    """Result of aligning extracted text to source."""
    
    status: AlignmentStatus
    char_span: Optional[CharSpan] = None
    token_span: Optional[TokenSpan] = None
    matched_text: Optional[str] = None
    confidence: float = 1.0

