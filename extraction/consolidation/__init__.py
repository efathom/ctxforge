"""
Memory Consolidation module.

Provides consolidators for merging, deduplicating, and managing
memories to maintain a clean and consistent memory store.

Available consolidators:
- DeduplicationConsolidator: Removes duplicate memories
- MergingConsolidator: Merges similar memories into comprehensive ones
- ConflictAwareConsolidator: Decides merge/add/ignore with contradiction handling
"""

from ctxforge.extraction.consolidation.base import BaseConsolidator
from ctxforge.extraction.consolidation.conflict_aware import (
    DEFAULT_CONTRADICTION_PROMPT,
    ConflictAwareConsolidator,
    ConsolidationAction,
    ConsolidationDecision,
)
from ctxforge.extraction.consolidation.deduplicator import DeduplicationConsolidator
from ctxforge.extraction.consolidation.merger import MergingConsolidator

__all__ = [
    "BaseConsolidator",
    "DeduplicationConsolidator",
    "MergingConsolidator",
    "ConflictAwareConsolidator",
    "ConsolidationAction",
    "ConsolidationDecision",
    "DEFAULT_CONTRADICTION_PROMPT",
]

