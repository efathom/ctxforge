"""
Personalized PageRank for graph retrieval.

Computes query-contextualized node importance by seeding teleportation
probabilities from query-to-node relevance scores and propagating
through the graph's edge structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from ctxforge.protocols.graph import GraphEdge, GraphNode

logger = logging.getLogger(__name__)


DEFAULT_NODE_TYPE_WEIGHTS: Dict[str, float] = {
    "Person": 1.0,
    "Organization": 1.0,
    "Location": 1.0,
    "Fact": 0.8,
    "Passage": 0.3,
}


@dataclass
class PPRConfig:
    """Configuration for Personalized PageRank retrieval."""

    enabled: bool = False
    damping: float = 0.5
    max_iterations: int = 50
    tolerance: float = 1e-6
    seed_top_k: int = 20
    result_top_k: int = 10
    node_type_weights: Dict[str, float] = field(default_factory=dict)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_seed_scores(
    query_embedding: List[float],
    nodes: List[GraphNode],
    top_k: int = 20,
) -> Dict[str, float]:
    """Compute query-to-node relevance scores using cosine similarity.

    Args:
        query_embedding: The query's embedding vector.
        nodes: Graph nodes to score (must have ``name_embedding``).
        top_k: Number of top-scoring nodes to return.

    Returns:
        Mapping of node_id to normalized relevance score.
    """
    scored: List[Tuple[str, float]] = []
    for node in nodes:
        if node.name_embedding is None:
            continue
        sim = _cosine_similarity(query_embedding, node.name_embedding)
        if sim > 0.0:
            scored.append((node.node_id, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    if not scored:
        return {}

    total = sum(s for _, s in scored)
    if total < 1e-12:
        return {}

    return {nid: s / total for nid, s in scored}


def _build_adjacency(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    node_type_weights: Dict[str, float],
    as_of: Optional[datetime] = None,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Set[str]]:
    """Build weighted adjacency list from nodes and edges.

    Edges that are temporally invalid (``invalid_at`` is set and before
    ``as_of``) are excluded.

    Returns:
        (adjacency dict, set of all node IDs in the graph).
    """
    node_ids: Set[str] = {n.node_id for n in nodes}
    node_label_map: Dict[str, str] = {}
    for n in nodes:
        if n.labels:
            node_label_map[n.node_id] = n.labels[0]

    adj: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in node_ids}

    for edge in edges:
        # Skip temporally invalid edges.
        if edge.invalid_at is not None:
            ref = as_of or datetime.now(timezone.utc).replace(tzinfo=None)
            if edge.invalid_at <= ref:
                continue

        src, tgt = edge.source_node_id, edge.target_node_id
        if src not in node_ids or tgt not in node_ids:
            continue

        # Determine edge weight using target node type.
        tgt_label = node_label_map.get(tgt, "")
        weight = node_type_weights.get(tgt_label, 1.0)

        adj.setdefault(src, []).append((tgt, weight))
        # Treat edges as bidirectional for PPR.
        src_label = node_label_map.get(src, "")
        reverse_weight = node_type_weights.get(src_label, 1.0)
        adj.setdefault(tgt, []).append((src, reverse_weight))

    return adj, node_ids


def personalized_pagerank(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    seed_scores: Dict[str, float],
    damping: float = 0.5,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    node_type_weights: Optional[Dict[str, float]] = None,
    as_of: Optional[datetime] = None,
) -> Dict[str, float]:
    """Compute Personalized PageRank over a subgraph.

    Args:
        nodes: Graph nodes to rank.
        edges: Graph edges (treated as bidirectional).
        seed_scores: Query-relevance scores for seed nodes (teleportation
            targets).  Values are normalised to sum to 1.0.
        damping: Probability of following an edge vs. teleporting back
            to a seed node.  0.5 gives balanced exploration.
        max_iterations: Maximum power-iteration steps.
        tolerance: L1 convergence threshold.
        node_type_weights: Per-label multipliers applied to edge transitions.
        as_of: Reference time for temporal edge filtering.

    Returns:
        Mapping of ``node_id`` → PPR score (higher = more query-relevant).
    """
    if not nodes or not seed_scores:
        return {}

    type_weights = node_type_weights or {}
    adj, all_ids = _build_adjacency(nodes, edges, type_weights, as_of)

    if not all_ids:
        return {}

    # Normalise reset vector.
    total_seed = sum(seed_scores.values())
    if total_seed < 1e-12:
        return {}
    reset: Dict[str, float] = {
        nid: seed_scores.get(nid, 0.0) / total_seed for nid in all_ids
    }

    # Initialise rank uniformly.
    n = len(all_ids)
    rank: Dict[str, float] = {nid: 1.0 / n for nid in all_ids}

    for _ in range(max_iterations):
        new_rank: Dict[str, float] = {}

        for nid in all_ids:
            # Contribution from neighbours.
            neighbour_sum = 0.0
            for src in all_ids:
                neighbours = adj.get(src, [])
                if not neighbours:
                    continue
                # Check if src has an edge to nid.
                total_weight = sum(w for _, w in neighbours)
                if total_weight < 1e-12:
                    continue
                for tgt, w in neighbours:
                    if tgt == nid:
                        neighbour_sum += rank[src] * (w / total_weight)

            new_rank[nid] = damping * neighbour_sum + (1.0 - damping) * reset.get(nid, 0.0)

        # Check convergence (L1 norm of change).
        diff = sum(abs(new_rank[nid] - rank[nid]) for nid in all_ids)
        rank = new_rank

        if diff < tolerance:
            break

    return rank
