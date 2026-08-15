"""Tests for Personalized PageRank graph retrieval."""

from ctxforge.graph.retrieval.pagerank import (
    PPRConfig,
    compute_seed_scores,
    personalized_pagerank,
)
from ctxforge.protocols.graph import GraphEdge, GraphNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, embedding: list[float] | None = None, label: str = "Entity") -> GraphNode:
    return GraphNode(
        node_id=nid,
        scope_id="test",
        name=nid,
        labels=[label],
        name_embedding=embedding,
    )


def _edge(eid: str, src: str, tgt: str, etype: str = "RELATED") -> GraphEdge:
    return GraphEdge(
        edge_id=eid,
        scope_id="test",
        source_node_id=src,
        target_node_id=tgt,
        edge_type=etype,
    )


# ---------------------------------------------------------------------------
# personalized_pagerank tests
# ---------------------------------------------------------------------------


def test_ppr_simple_chain():
    """A→B→C with seed at A; B and C should get decreasing scores."""
    nodes = [_node("A"), _node("B"), _node("C")]
    edges = [_edge("e1", "A", "B"), _edge("e2", "B", "C")]
    seeds = {"A": 1.0}

    scores = personalized_pagerank(nodes, edges, seeds, damping=0.5)

    assert scores["A"] > scores["B"]
    assert scores["B"] > scores["C"]
    assert all(s > 0 for s in scores.values())


def test_ppr_convergence():
    """Iteration stops within tolerance."""
    nodes = [_node("A"), _node("B")]
    edges = [_edge("e1", "A", "B")]
    seeds = {"A": 1.0}

    scores = personalized_pagerank(
        nodes, edges, seeds, damping=0.5, max_iterations=1000, tolerance=1e-10
    )

    assert len(scores) == 2
    assert all(s >= 0 for s in scores.values())


def test_ppr_seed_normalization():
    """Seeds are normalised to sum to 1.0 internally."""
    nodes = [_node("A"), _node("B")]
    edges = [_edge("e1", "A", "B")]
    seeds = {"A": 5.0, "B": 5.0}

    scores = personalized_pagerank(nodes, edges, seeds, damping=0.5)

    # Both seeded equally → roughly equal scores
    assert abs(scores["A"] - scores["B"]) < 0.1


def test_ppr_disconnected_nodes():
    """Unreachable nodes get only teleportation score."""
    nodes = [_node("A"), _node("B"), _node("C")]
    edges = [_edge("e1", "A", "B")]  # C is disconnected
    seeds = {"A": 1.0}

    scores = personalized_pagerank(nodes, edges, seeds, damping=0.8)

    # C gets no walk contribution, only (1-damping)*reset which is ~0 since C not seeded
    assert scores["C"] < scores["A"]
    assert scores["C"] < scores["B"]


def test_ppr_damping_effect():
    """Higher damping → more spread; lower damping → concentrated on seeds."""
    nodes = [_node("A"), _node("B"), _node("C")]
    edges = [_edge("e1", "A", "B"), _edge("e2", "B", "C")]
    seeds = {"A": 1.0}

    scores_low = personalized_pagerank(nodes, edges, seeds, damping=0.1)
    scores_high = personalized_pagerank(nodes, edges, seeds, damping=0.9)

    # Low damping: mostly teleport → seed A dominates
    assert scores_low["A"] > scores_high["A"]
    # High damping: more walk → remote node C gets more
    assert scores_high["C"] > scores_low["C"]


def test_ppr_temporal_filtering():
    """Invalidated edges are excluded from adjacency."""
    from datetime import datetime

    nodes = [_node("A"), _node("B")]
    edge = GraphEdge(
        edge_id="e1",
        scope_id="test",
        source_node_id="A",
        target_node_id="B",
        edge_type="RELATED",
        invalid_at=datetime(2020, 1, 1),
    )
    seeds = {"A": 1.0}

    scores = personalized_pagerank(
        nodes, [edge], seeds, damping=0.5, as_of=datetime(2025, 1, 1)
    )

    # Edge is invalid → no walk from A to B
    # B only gets teleportation (which is 0 since not seeded)
    assert scores.get("B", 0.0) < scores.get("A", 0.0)


def test_ppr_empty_graph():
    """Empty nodes/edges returns empty dict."""
    assert personalized_pagerank([], [], {"A": 1.0}) == {}
    assert personalized_pagerank([_node("A")], [], {}) == {}


def test_ppr_node_type_weights():
    """Node type weights affect score distribution.

    Use a hub topology so a node with multiple neighbours of different
    types actually sees its transition probabilities shift when weights
    change.  Hub H connects to both P (Passage) and E (Person).
    With default weights H splits evenly; with Passage weighted low,
    more mass flows to E than to P.
    """
    nodes = [
        _node("H", label="Person"),
        _node("P", label="Passage"),
        _node("E", label="Person"),
    ]
    edges = [_edge("e1", "H", "P"), _edge("e2", "H", "E")]
    seeds = {"H": 1.0}

    # Without weights (all default to 1.0)
    scores_default = personalized_pagerank(nodes, edges, seeds, damping=0.8)

    # With Passage weighted very low — H sends less mass to P
    scores_weighted = personalized_pagerank(
        nodes, edges, seeds, damping=0.8,
        node_type_weights={"Person": 1.0, "Passage": 0.05},
    )

    # Default: P and E get roughly equal scores (symmetric)
    # Weighted: P should get much less than E
    ratio_default = scores_default["P"] / scores_default["E"]
    ratio_weighted = scores_weighted["P"] / scores_weighted["E"]
    assert ratio_weighted < ratio_default


# ---------------------------------------------------------------------------
# compute_seed_scores tests
# ---------------------------------------------------------------------------


def test_compute_seed_scores_basic():
    """Nodes with embeddings close to query get higher scores."""
    query = [1.0, 0.0, 0.0]
    nodes = [
        _node("A", embedding=[1.0, 0.0, 0.0]),   # identical to query
        _node("B", embedding=[0.0, 1.0, 0.0]),   # orthogonal
        _node("C", embedding=[0.7, 0.7, 0.0]),   # partial match
    ]

    scores = compute_seed_scores(query, nodes, top_k=3)

    assert scores["A"] > scores["C"]
    assert "B" not in scores  # orthogonal → sim = 0


def test_compute_seed_scores_no_embeddings():
    """Nodes without embeddings are skipped."""
    query = [1.0, 0.0]
    nodes = [_node("A"), _node("B")]  # no embeddings

    scores = compute_seed_scores(query, nodes)
    assert scores == {}


def test_compute_seed_scores_top_k():
    """Only top_k nodes are returned."""
    query = [1.0, 0.0]
    nodes = [
        _node("A", embedding=[1.0, 0.0]),
        _node("B", embedding=[0.9, 0.1]),
        _node("C", embedding=[0.8, 0.2]),
    ]

    scores = compute_seed_scores(query, nodes, top_k=2)
    assert len(scores) == 2
    assert "A" in scores
    assert "B" in scores


def test_ppr_config_defaults():
    """PPRConfig has sensible defaults."""
    cfg = PPRConfig()
    assert cfg.enabled is False
    assert cfg.damping == 0.5
    assert cfg.seed_top_k == 20
    assert cfg.result_top_k == 10
