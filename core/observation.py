"""
Structured observation data models.

Observations are typed insights extracted from sessions (decisions, bug fixes,
discoveries, etc.) that can be stored as scoped memories for cross-session
knowledge retention.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ObservationType(str, Enum):
    """Types of structured observations."""

    DECISION = "decision"
    BUGFIX = "bugfix"
    DISCOVERY = "discovery"
    FEATURE = "feature"
    REFACTOR = "refactor"
    CHANGE = "change"


@dataclass
class Observation:
    """A single structured observation extracted from a session."""

    type: ObservationType
    summary: str
    detail: Optional[str] = None
    confidence: float = 0.8
    source_event_ids: List[str] = field(default_factory=list)


@dataclass
class SessionSummaryReport:
    """Structured summary of what happened in a session."""

    request: str = ""
    investigated: str = ""
    learned: str = ""
    completed: str = ""
    next_steps: str = ""
