"""
Query Data Models.

Data models for query rewriting including results and reason tracking
for transformed queries.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RewriteReason(str, Enum):
    """Why the query was rewritten."""

    PRONOUN = "pronoun"  # Resolved pronouns (they, it, their)
    REFERENCE = "reference"  # Resolved references (that, those, the same)
    IMPLICIT = "implicit"  # Made implicit context explicit
    ELLIPSIS = "ellipsis"  # Completed elliptical expressions
    NO_CHANGE = "no_change"  # Query was already clear


class QueryRewriteResult(BaseModel):
    """Result of query rewriting."""

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    reason: RewriteReason
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    resolved_entities: List[str] = Field(default_factory=list)

    @classmethod
    def unchanged(cls, query: str) -> "QueryRewriteResult":
        """Create a result indicating no change was needed."""
        return cls(
            original_query=query,
            rewritten_query=query,
            was_rewritten=False,
            reason=RewriteReason.NO_CHANGE,
        )

    @classmethod
    def rewritten(
        cls,
        original: str,
        rewritten: str,
        reason: RewriteReason,
        confidence: float = 1.0,
        entities: Optional[List[str]] = None,
    ) -> "QueryRewriteResult":
        """Create a result indicating the query was rewritten."""
        return cls(
            original_query=original,
            rewritten_query=rewritten,
            was_rewritten=True,
            reason=reason,
            confidence=confidence,
            resolved_entities=entities or [],
        )
