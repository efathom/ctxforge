"""
Core data structures for the ctxforge framework.

This module contains the fundamental data types used throughout the framework:
- Events: Immutable records of what happened
- Sessions: Working memory containers
- Memory: Long-term storage items
- Expertise: Evolving knowledge bases
- Context: Assembled prompt context
- Alignment Types: Source grounding types (shared across modules)
"""

from ctxforge.core.alignment_types import (
    AlignmentResult,
    AlignmentStatus,
    CharSpan,
    TokenSpan,
)
from ctxforge.core.categories import CategoryAssignment, MemoryCategory
from ctxforge.core.context import Context
from ctxforge.core.events import Event, EventType
from ctxforge.core.exceptions import (
    ConfigurationError,
    ContextEngineError,
    LLMError,
    StorageError,
    ValidationError,
)
from ctxforge.core.expertise import (
    CompletedTurn,
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseStats,
    ExpertiseUsageLog,
    ReflectionResult,
    SimilarGroup,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session, SessionState

__all__ = [
    # Events
    "Event",
    "EventType",
    # Session
    "Session",
    "SessionState",
    # Memory
    "MemoryItem",
    "MemoryType",
    # Context
    "Context",
    # Expertise
    "ExpertiseSection",
    "UsageFeedback",
    "TurnOutcome",
    "CuratorOperation",
    "ExpertiseItem",
    "Expertise",
    "CompletedTurn",
    "ExpertiseUsageLog",
    "ReflectionResult",
    "CurationOp",
    "CurationPlan",
    "ExpertiseStats",
    "SimilarGroup",
    # Exceptions
    "ContextEngineError",
    "ConfigurationError",
    "StorageError",
    "LLMError",
    "ValidationError",
    # Categories
    "MemoryCategory",
    "CategoryAssignment",
    # Alignment Types
    "AlignmentStatus",
    "CharSpan",
    "TokenSpan",
    "AlignmentResult",
]

