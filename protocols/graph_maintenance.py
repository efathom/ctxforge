"""
Graph maintenance protocols.

These interfaces represent optional maintenance steps that operate on top of stored graph
data (e.g., invalidating contradicted edges). They are separated from the core store
protocol to keep backends focused on persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode


@dataclass
class EdgeInvalidationPlan:
    """Decision output for invalidation: which edge IDs should be marked invalid."""

    invalidate_edge_ids: List[str]
    rationale: Optional[str] = None


class IGraphContradictionDetector(Protocol):
    """Decides which existing edges should be invalidated given a new edge."""

    async def detect_contradictions(
        self,
        *,
        scope_id: str,
        new_edge: GraphEdge,
        candidate_edges: List[GraphEdge],
        nodes: List[GraphNode],
        episodes: List[GraphEpisode],
        model: str | None = None,
    ) -> EdgeInvalidationPlan:
        ...


@dataclass
class EdgeTemporalInfo:
    """Temporal enrichment output for an edge."""

    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None


class IGraphEdgeTemporalExtractor(Protocol):
    """Extract/normalize temporal bounds for a graph edge based on episode text and reference time."""

    async def extract_temporal_info(
        self,
        *,
        scope_id: str,
        edge: GraphEdge,
        episodes: List[GraphEpisode],
        model: str | None = None,
    ) -> EdgeTemporalInfo:
        ...


