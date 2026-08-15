"""
Graph retrieval result types.

These types are designed to carry:
- the chosen query plan reason
- the selected subgraph (nodes/edges)
- a set of supporting "evidence" items (currently: episodes)
- reasoning paths discovered via multi-hop path mining
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass(frozen=True)
class GraphNodeHit:
    node_id: str
    label: str
    score: float
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class GraphEdgeHit:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    score: float
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class EvidenceItem:
    source: Literal["episode"]
    source_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class ReasoningPath:
    """An ordered sequence of node IDs forming a reasoning chain."""

    node_ids: List[str]
    edge_types: List[str]
    score: float = 0.0


@dataclass
class BridgeConnection:
    """A bridge node that connects two otherwise disconnected nodes."""

    source_node_id: str
    bridge_node_id: str
    target_node_id: str
    bridge_type: str = "inferred"


@dataclass
class GraphRetrievalResult:
    plan_mode: str
    plan_reason: str
    nodes: List[GraphNodeHit]
    edges: List[GraphEdgeHit]
    evidence: List[EvidenceItem]
    debug: Dict[str, Any]
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    bridge_connections: List[BridgeConnection] = field(default_factory=list)


