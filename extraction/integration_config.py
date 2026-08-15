"""Configuration for the multi-stage memory integration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ctxforge.core.memory import MemoryItem


@dataclass
class IntegrationConfig:
    """Configuration for the multi-stage memory integration pipeline."""

    enabled: bool = False
    detect_threshold: float = 0.5
    similarity_threshold: float = 0.80
    max_candidates_per_search: int = 5
    model: Optional[str] = None
    skip_detect_for_high_confidence: bool = True


@dataclass
class PreferenceEvolutionConfig:
    """Configuration for preference change detection and tracking."""

    enabled: bool = False
    contradiction_similarity_threshold: float = 0.70
    auto_supersede: bool = True
    track_history: bool = True
    importance_decay_on_supersede: float = 0.1


@dataclass
class SynthesizerConfig:
    """Configuration for memory narrative synthesis."""

    enabled: bool = False
    min_memories_to_synthesize: int = 3
    max_synthesis_tokens: int = 300
    model: Optional[str] = None


@dataclass
class PersonalizationMetricsConfig:
    """Configuration for personalization effectiveness metrics."""

    enabled: bool = False
    memory_hit_threshold: float = 0.5


@dataclass
class IntegrationResult:
    """Result of processing a single candidate through the integration pipeline."""

    memory_item: Optional["MemoryItem"] = None
    operation: str = "add"
    was_actionable: bool = True
    similarity_score: float = 0.0
    preference_changed: bool = False
    stage_metadata: dict = field(default_factory=dict)
