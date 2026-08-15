"""
Compaction implementations.

Provides strategies for managing context window limits:
- SlidingWindowCondenser: Simple FIFO removal
- SummarizingCondenser: Summarizes old events
- ImportanceCondenser: Keeps important events based on scoring
- StructuredSummarizingCondenser: Structured summaries via LLM function calling
- CondenserPipeline: Chain multiple condensers together
- DefaultContextAssembler: Assembles context from components
- CompactionView: Immutable view for tracking forgotten events

All condensers implement the ICondenser protocol.
"""

from ctxforge.compaction.assembler import (
    DefaultContextAssembler,
    MinimalContextAssembler,
)
from ctxforge.compaction.base import BaseCondenser
from ctxforge.compaction.importance import (
    ImportanceCondenser,
    default_importance_scorer,
)
from ctxforge.compaction.pipeline import CondenserPipeline
from ctxforge.compaction.sliding_window import SlidingWindowCondenser
from ctxforge.compaction.structured_summary import (
    StructuredSummarizingCondenser,
    StructuredSummary,
)
from ctxforge.compaction.summarizing import SummarizingCondenser
from ctxforge.compaction.utils import (
    ScoringFunc,
    SummarizeFunc,
    estimate_event_tokens,
    estimate_tokens_simple,
)
from ctxforge.compaction.view import (
    CompactionView,
    CondensationResult,
    ICondenser,
    SessionLike,
)

__all__ = [
    # Base
    "BaseCondenser",
    # Condensers
    "SlidingWindowCondenser",
    "SummarizingCondenser",
    "ImportanceCondenser",
    "StructuredSummarizingCondenser",
    # Structured Summary
    "StructuredSummary",
    # Pipeline
    "CondenserPipeline",
    # Assemblers
    "DefaultContextAssembler",
    "MinimalContextAssembler",
    # View-based condensation
    "CompactionView",
    "CondensationResult",
    "ICondenser",
    "SessionLike",
    # Utilities
    "default_importance_scorer",
    "SummarizeFunc",
    "ScoringFunc",
    "estimate_tokens_simple",
    "estimate_event_tokens",
]
