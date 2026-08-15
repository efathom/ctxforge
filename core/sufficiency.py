"""
Sufficiency Data Models.

Data models for sufficiency judging including verdict, results,
and progressive retrieval tracking.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SufficiencyVerdict(str, Enum):
    """Whether retrieved content is sufficient to answer the query."""

    ENOUGH = "enough"  # Retrieved content can answer the query
    MORE = "more"  # Need to retrieve more content
    UNCERTAIN = "uncertain"  # Cannot determine sufficiency


class SufficiencyResult(BaseModel):
    """Result of sufficiency judgment."""

    verdict: SufficiencyVerdict
    reasoning: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    suggested_sources: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)
    iteration: int = Field(default=1, ge=1)

    @classmethod
    def enough(
        cls,
        reasoning: str = "Content is sufficient",
        confidence: float = 1.0,
    ) -> "SufficiencyResult":
        """Create a result indicating content is sufficient."""
        return cls(
            verdict=SufficiencyVerdict.ENOUGH,
            reasoning=reasoning,
            confidence=confidence,
        )

    @classmethod
    def need_more(
        cls,
        reasoning: str,
        missing_aspects: Optional[List[str]] = None,
        suggested_sources: Optional[List[str]] = None,
        confidence: float = 1.0,
    ) -> "SufficiencyResult":
        """Create a result indicating more content is needed."""
        return cls(
            verdict=SufficiencyVerdict.MORE,
            reasoning=reasoning,
            missing_aspects=missing_aspects or [],
            suggested_sources=suggested_sources or [],
            confidence=confidence,
        )

    @classmethod
    def uncertain(
        cls,
        reasoning: str = "Cannot determine sufficiency",
        confidence: float = 0.5,
    ) -> "SufficiencyResult":
        """Create a result indicating uncertainty."""
        return cls(
            verdict=SufficiencyVerdict.UNCERTAIN,
            reasoning=reasoning,
            confidence=confidence,
        )

    @property
    def is_sufficient(self) -> bool:
        """Check if content is sufficient."""
        return self.verdict == SufficiencyVerdict.ENOUGH

    @property
    def needs_more(self) -> bool:
        """Check if more content is needed."""
        return self.verdict == SufficiencyVerdict.MORE


class ProgressiveRetrievalStats(BaseModel):
    """Statistics for progressive retrieval."""

    total_iterations: int = 0
    initial_results: int = 0
    final_results: int = 0
    sources_tried: List[str] = Field(default_factory=list)
    final_verdict: Optional[SufficiencyVerdict] = None

    @property
    def results_added(self) -> int:
        """Number of results added through progressive retrieval."""
        return self.final_results - self.initial_results
