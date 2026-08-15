"""
Tests for Phase 5: Graph Bridge Discovery and Multi-Hop Path Mining.

Tests cover:
- Bridge discovery: connection checking, disconnected pair detection, bridge search
- Path mining: DFS path enumeration, deduplication, ranking, depth limits
- Path scoring: node scoring formula, budget enforcement
- Integration: GraphRetrievalResult with reasoning paths and bridge connections
- Rendering: reasoning paths and bridge connections in graph context output
"""

from datetime import datetime, timezone
from typing import List, Optional

import pytest

from ctxforge.config.base import GraphPathMiningConfig
from ctxforge.graph.retrieval.bridge_discovery import (
    _hours_between,
    _is_between,
    _node_entities,
    _node_keywords,
    check_connection,
    find_bridge_candidates,
)
from ctxforge.graph.retrieval.path_miner import (
    _build_adjacency,
    discover_reasoning_paths,
)
from ctxforge.graph.retrieval.path_scorer import (
    rank_and_limit_nodes,
    score_node,
)
from ctxforge.graph.retrieval.types import (
    BridgeConnection,
    GraphRetrievalResult,
    ReasoningPath,
)
from ctxforge.protocols.graph import GraphEdge, GraphNode

# =============================================================================
# Test Helpers
# =============================================================================


def make_node(
    node_id: str,
    name: str,
    labels: Optional[List[str]] = None,
    summary: Optional[str] = None,
    attributes: Optional[dict] = None,
    scope_id: str = "test-scope",
) -> GraphNode:
    """Create a GraphNode for testing."""
    return GraphNode(
        node_id=node_id,
        scope_id=scope_id,
        name=name,
        labels=labels or [],
        summary=summary,
        attributes=attributes or {},
    )


def make_edge(
    edge_id: str,
    source_node_id: str,
    target_node_id: str,
    edge_type: str = "RELATED_TO",
    scope_id: str = "test-scope",
) -> GraphEdge:
    """Create a GraphEdge for testing."""
    return GraphEdge(
        edge_id=edge_id,
        scope_id=scope_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
    )


@pytest.fixture
def default_config() -> GraphPathMiningConfig:
    return GraphPathMiningConfig(
        enabled=True,
        bridge_discovery_enabled=True,
        max_path_depth=3,
        max_paths=10,
        min_path_length=2,
        max_nodes=25,
        min_nodes=8,
        temporal_window_hours=168,
        bridge_search_top_k=5,
        bridge_proximity_hours=24.0,
        node_score_threshold_pct=0.10,
    )


# =============================================================================
# Bridge Discovery: Utility Functions
# =============================================================================


class TestBridgeDiscoveryUtils:
    """Tests for bridge discovery utility functions."""

    def test_hours_between_same_time(self):
        t = datetime(2024, 6, 15, 12, 0, 0)
        assert _hours_between(t, t) == 0.0

    def test_hours_between_one_hour(self):
        t1 = datetime(2024, 6, 15, 12, 0, 0)
        t2 = datetime(2024, 6, 15, 13, 0, 0)
        assert _hours_between(t1, t2) == pytest.approx(1.0)

    def test_hours_between_none(self):
        t = datetime(2024, 6, 15, 12, 0, 0)
        assert _hours_between(t, None) is None
        assert _hours_between(None, t) is None
        assert _hours_between(None, None) is None

    def test_is_between_true(self):
        t1 = datetime(2024, 6, 15, 10, 0, 0)
        t_mid = datetime(2024, 6, 15, 12, 0, 0)
        t2 = datetime(2024, 6, 15, 14, 0, 0)
        assert _is_between(t_mid, t1, t2) is True

    def test_is_between_false(self):
        t1 = datetime(2024, 6, 15, 10, 0, 0)
        t_out = datetime(2024, 6, 15, 16, 0, 0)
        t2 = datetime(2024, 6, 15, 14, 0, 0)
        assert _is_between(t_out, t1, t2) is False

    def test_is_between_reversed_order(self):
        """Should work regardless of which endpoint is earlier."""
        t1 = datetime(2024, 6, 15, 14, 0, 0)
        t_mid = datetime(2024, 6, 15, 12, 0, 0)
        t2 = datetime(2024, 6, 15, 10, 0, 0)
        assert _is_between(t_mid, t1, t2) is True

    def test_is_between_none_timestamp(self):
        t1 = datetime(2024, 6, 15, 10, 0, 0)
        t2 = datetime(2024, 6, 15, 14, 0, 0)
        assert _is_between(None, t1, t2) is False

    def test_node_entities_from_name_and_labels(self):
        node = make_node("n1", "Alice", labels=["Person"])
        entities = _node_entities(node)
        assert "Alice" in entities
        assert "Person" in entities

    def test_node_entities_from_attributes(self):
        node = make_node("n1", "Alice", attributes={"persons": ["Bob", "Charlie"]})
        entities = _node_entities(node)
        assert "Bob" in entities
        assert "Charlie" in entities

    def test_node_keywords_from_summary(self):
        node = make_node("n1", "Alice", summary="Alice visited Paris last summer")
        keywords = _node_keywords(node)
        assert "Alice" in keywords
        assert "Paris" in keywords


# =============================================================================
# Bridge Discovery: Connection Checking
# =============================================================================


class TestConnectionChecking:
    """Tests for check_connection between two nodes."""

    def test_connected_via_edge(self):
        n1 = make_node("n1", "Alice")
        n2 = make_node("n2", "Bob")
        edges = [make_edge("e1", "n1", "n2", "KNOWS")]
        assert check_connection(n1, n2, edges) == "edge_link"

    def test_connected_via_reverse_edge(self):
        n1 = make_node("n1", "Alice")
        n2 = make_node("n2", "Bob")
        edges = [make_edge("e1", "n2", "n1", "KNOWS")]
        assert check_connection(n1, n2, edges) == "edge_link"

    def test_connected_via_entity_overlap(self):
        n1 = make_node("n1", "Alice", labels=["Person"], attributes={"persons": ["Bob"]})
        n2 = make_node("n2", "Meeting", attributes={"persons": ["Bob"]})
        edges = []
        assert check_connection(n1, n2, edges) == "entity_link"

    def test_connected_via_temporal_proximity(self):
        t1 = datetime(2024, 6, 15, 12, 0, 0)
        t2 = datetime(2024, 6, 15, 14, 0, 0)  # 2 hours later
        n1 = make_node("n1", "Event1", attributes={"created_at": t1})
        n2 = make_node("n2", "Event2", attributes={"created_at": t2})
        edges = []
        assert check_connection(n1, n2, edges) == "temporal_flow"

    def test_disconnected_no_overlap(self):
        n1 = make_node("n1", "Alice", labels=["Person"])
        n2 = make_node("n2", "TechCorp", labels=["Organization"])
        edges = []
        assert check_connection(n1, n2, edges) is None

    def test_disconnected_temporal_too_far(self):
        t1 = datetime(2024, 6, 15, 12, 0, 0)
        t2 = datetime(2024, 6, 16, 12, 0, 0)  # 24 hours later
        n1 = make_node("n1", "Event1", attributes={"created_at": t1})
        n2 = make_node("n2", "Event2", attributes={"created_at": t2})
        edges = []
        assert check_connection(n1, n2, edges) is None

    def test_temporal_flow_hours_configurable(self):
        """Custom temporal_flow_hours should control the proximity threshold."""
        t1 = datetime(2024, 6, 15, 12, 0, 0)
        t2 = datetime(2024, 6, 15, 14, 0, 0)  # 2 hours later
        n1 = make_node("n1", "Event1", attributes={"created_at": t1})
        n2 = make_node("n2", "Event2", attributes={"created_at": t2})
        edges = []
        # Default (6h) -> connected
        assert check_connection(n1, n2, edges) == "temporal_flow"
        # Tighter threshold (1h) -> disconnected
        assert check_connection(n1, n2, edges, temporal_flow_hours=1.0) is None
        # Wider threshold (3h) -> connected
        assert check_connection(n1, n2, edges, temporal_flow_hours=3.0) == "temporal_flow"


# =============================================================================
# Bridge Discovery: Async Bridge Search
# =============================================================================


class TestBridgeCandidateSearch:
    """Tests for the async bridge candidate search."""

    @pytest.mark.asyncio
    async def test_no_bridges_when_disabled(self, default_config: GraphPathMiningConfig):
        config = default_config.model_copy(update={"bridge_discovery_enabled": False})
        nodes, connections = await find_bridge_candidates(
            scope_id="test",
            disconnected_pairs=[],
            graph_store=None,
            embedding_provider=None,
            config=config,
            existing_node_ids=set(),
        )
        assert nodes == []
        assert connections == []

    @pytest.mark.asyncio
    async def test_no_bridges_when_no_pairs(self, default_config: GraphPathMiningConfig):
        nodes, connections = await find_bridge_candidates(
            scope_id="test",
            disconnected_pairs=[],
            graph_store=None,
            embedding_provider=None,
            config=default_config,
            existing_node_ids=set(),
        )
        assert nodes == []
        assert connections == []


# =============================================================================
# Path Mining: DFS
# =============================================================================


class TestPathMining:
    """Tests for DFS-based path mining."""

    def test_empty_graph(self, default_config: GraphPathMiningConfig):
        paths = discover_reasoning_paths(nodes=[], edges=[], config=default_config)
        assert paths == []

    def test_no_edges(self, default_config: GraphPathMiningConfig):
        nodes = [make_node("n1", "Alice"), make_node("n2", "Bob")]
        paths = discover_reasoning_paths(nodes=nodes, edges=[], config=default_config)
        assert paths == []

    def test_simple_chain(self, default_config: GraphPathMiningConfig):
        """A -> B -> C should produce paths of length 2 and 3."""
        nodes = [
            make_node("n1", "Alice"),
            make_node("n2", "Bob"),
            make_node("n3", "Charlie"),
        ]
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e2", "n2", "n3", "KNOWS"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        assert len(paths) > 0

        # Should have at least one path of length 3 (A->B->C)
        long_paths = [p for p in paths if len(p.node_ids) == 3]
        assert len(long_paths) >= 1

    def test_max_depth_respected(self, default_config: GraphPathMiningConfig):
        """Paths should not exceed max_path_depth."""
        config = default_config.model_copy(update={"max_path_depth": 2})
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
            make_node("n4", "D"),
        ]
        edges = [
            make_edge("e1", "n1", "n2"),
            make_edge("e2", "n2", "n3"),
            make_edge("e3", "n3", "n4"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=config)
        for path in paths:
            assert len(path.node_ids) <= 2

    def test_min_path_length_respected(self, default_config: GraphPathMiningConfig):
        """All paths should have at least min_path_length nodes."""
        paths_config = default_config.model_copy(update={"min_path_length": 3})
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
        ]
        edges = [
            make_edge("e1", "n1", "n2"),
            make_edge("e2", "n2", "n3"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=paths_config)
        for path in paths:
            assert len(path.node_ids) >= 3

    def test_max_paths_cap(self, default_config: GraphPathMiningConfig):
        """Number of returned paths should not exceed max_paths."""
        config = default_config.model_copy(update={"max_paths": 2})
        nodes = [make_node(f"n{i}", f"Node{i}") for i in range(5)]
        edges = [
            make_edge(f"e{i}", f"n{i}", f"n{i+1}") for i in range(4)
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=config)
        assert len(paths) <= 2

    def test_deduplication(self, default_config: GraphPathMiningConfig):
        """Duplicate paths (same node sequence) should be removed."""
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
        ]
        # Two edges between same nodes (different types)
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e2", "n1", "n2", "WORKS_WITH"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        # Should be deduplicated by node_id tuple
        path_keys = set()
        for p in paths:
            key = tuple(p.node_ids)
            assert key not in path_keys, f"Duplicate path: {key}"
            path_keys.add(key)

    def test_bidirectional_traversal(self, default_config: GraphPathMiningConfig):
        """Edges should be traversable in both directions."""
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
        ]
        edges = [make_edge("e1", "n1", "n2")]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)

        # Should find paths starting from both n1 and n2
        starts = set()
        for p in paths:
            starts.add(p.node_ids[0])
        assert "n1" in starts
        assert "n2" in starts

    def test_cycle_handling(self, default_config: GraphPathMiningConfig):
        """DFS should handle cycles without infinite loops."""
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
        ]
        edges = [
            make_edge("e1", "n1", "n2"),
            make_edge("e2", "n2", "n3"),
            make_edge("e3", "n3", "n1"),  # cycle
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        # Should complete without hanging
        assert isinstance(paths, list)

    def test_edge_types_in_result(self, default_config: GraphPathMiningConfig):
        """ReasoningPath should contain edge_types."""
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
        ]
        edges = [make_edge("e1", "n1", "n2", "KNOWS")]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        assert len(paths) > 0
        for p in paths:
            assert len(p.edge_types) == len(p.node_ids) - 1

    def test_ranking_longer_paths_first(self, default_config: GraphPathMiningConfig):
        """Longer paths should appear before shorter ones in the result."""
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
        ]
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e2", "n2", "n3", "KNOWS"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        assert len(paths) >= 2
        # Verify descending length order
        for i in range(len(paths) - 1):
            assert len(paths[i].node_ids) >= len(paths[i + 1].node_ids)

    def test_ranking_timestamp_tiebreak(self, default_config: GraphPathMiningConfig):
        """Among paths of equal length, earlier timestamp should come first."""
        t_early = datetime(2024, 1, 1, 8, 0, 0)
        t_late = datetime(2024, 6, 15, 12, 0, 0)
        nodes = [
            make_node("n1", "A", attributes={"created_at": t_late}),
            make_node("n2", "B", attributes={"created_at": t_late}),
            make_node("n3", "C", attributes={"created_at": t_early}),
            make_node("n4", "D", attributes={"created_at": t_early}),
        ]
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e3", "n3", "n4", "KNOWS"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        # All paths are length 2; those touching n3/n4 (early) should precede n1/n2 (late)
        length_2_paths = [p for p in paths if len(p.node_ids) == 2]
        assert len(length_2_paths) >= 2
        # Find the first path that contains an early-timestamp node
        first_early_idx = None
        first_late_idx = None
        for idx, p in enumerate(length_2_paths):
            has_early = any(nid in ("n3", "n4") for nid in p.node_ids)
            has_late = all(nid in ("n1", "n2") for nid in p.node_ids)
            if has_early and first_early_idx is None:
                first_early_idx = idx
            if has_late and first_late_idx is None:
                first_late_idx = idx
        assert first_early_idx is not None
        assert first_late_idx is not None
        assert first_early_idx < first_late_idx

    def test_max_paths_capped_by_node_count(self, default_config: GraphPathMiningConfig):
        """max_paths is min(len(nodes), config.max_paths); few nodes should cap."""
        config = default_config.model_copy(update={"max_paths": 100})
        # Only 2 nodes -> cap is min(2, 100) = 2
        nodes = [make_node("n1", "A"), make_node("n2", "B")]
        edges = [make_edge("e1", "n1", "n2")]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=config)
        assert len(paths) <= 2

    def test_path_saved_at_exact_max_depth(self, default_config: GraphPathMiningConfig):
        """A path that reaches exactly max_depth should be saved."""
        config = default_config.model_copy(update={"max_path_depth": 3, "min_path_length": 2})
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
        ]
        edges = [
            make_edge("e1", "n1", "n2"),
            make_edge("e2", "n2", "n3"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=config)
        # There should be a path of exactly length 3 (max_depth)
        exact_max = [p for p in paths if len(p.node_ids) == 3]
        assert len(exact_max) >= 1

    def test_leaf_node_path_saved(self, default_config: GraphPathMiningConfig):
        """A path ending at a leaf (no unvisited neighbors) should be saved."""
        config = default_config.model_copy(update={"min_path_length": 2, "max_path_depth": 10})
        # n1 -> n2 -> n3 (n3 is a leaf)
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
        ]
        edges = [
            make_edge("e1", "n1", "n2"),
            make_edge("e2", "n2", "n3"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=config)
        # Starting from n1: n1->n2 (leaf at n2? no, n2 connects to n3)
        # n1->n2->n3 should be saved since n3 has no unvisited neighbors
        path_ids_list = [tuple(p.node_ids) for p in paths]
        assert ("n1", "n2", "n3") in path_ids_list

    def test_disconnected_components(self, default_config: GraphPathMiningConfig):
        """Paths should only contain nodes from the same connected component."""
        nodes = [
            make_node("n1", "A"),
            make_node("n2", "B"),
            make_node("n3", "C"),
            make_node("n4", "D"),
        ]
        # Two separate components: n1-n2 and n3-n4
        edges = [
            make_edge("e1", "n1", "n2"),
            make_edge("e2", "n3", "n4"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        for p in paths:
            ids = set(p.node_ids)
            # Each path must be entirely in one component
            in_comp1 = ids.issubset({"n1", "n2"})
            in_comp2 = ids.issubset({"n3", "n4"})
            assert in_comp1 or in_comp2, f"Path {p.node_ids} spans components"

    def test_single_node_self_loop(self, default_config: GraphPathMiningConfig):
        """A self-loop edge should not produce any paths (no revisiting)."""
        nodes = [make_node("n1", "A")]
        edges = [make_edge("e1", "n1", "n1", "SELF")]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        assert paths == []


# =============================================================================
# Path Mining: Adjacency Building
# =============================================================================


class TestAdjacencyBuilding:
    """Tests for adjacency list construction."""

    def test_basic_adjacency(self):
        nodes = [make_node("n1", "A"), make_node("n2", "B")]
        edges = [make_edge("e1", "n1", "n2")]
        adj, node_map = _build_adjacency(nodes, edges)
        assert "n2" in [nid for nid, _ in adj["n1"]]
        assert "n1" in [nid for nid, _ in adj["n2"]]  # bidirectional

    def test_ignores_edges_with_unknown_nodes(self):
        nodes = [make_node("n1", "A")]
        edges = [make_edge("e1", "n1", "n_unknown")]
        adj, _ = _build_adjacency(nodes, edges)
        assert len(adj.get("n1", [])) == 0


# =============================================================================
# Path Scorer: Node Scoring
# =============================================================================


class TestNodeScoring:
    """Tests for node scoring formula."""

    def test_target_entity_in_name(self):
        node = make_node("n1", "Alice", summary="Alice went to Paris")
        s = score_node(node, query_words={"alice", "paris"}, target_entity="Alice")
        assert s >= 100  # entity match bonus

    def test_target_entity_in_labels(self):
        node = make_node("n1", "Event", labels=["Alice"])
        s = score_node(node, query_words={"alice"}, target_entity="Alice")
        assert s >= 80

    def test_word_overlap_scoring(self):
        node = make_node("n1", "Meeting", summary="Alice met Bob at the conference")
        s = score_node(node, query_words={"alice", "bob", "conference"})
        # 3 overlapping words * 10 = 30
        assert s >= 30

    def test_keyword_overlap_boost(self):
        node = make_node("n1", "Alice", attributes={"keywords": ["coffee", "tea"]})
        s = score_node(node, query_words={"coffee"})
        # keyword overlap * 15
        assert s >= 15

    def test_recency_bonus(self):
        node = make_node("n1", "Event", attributes={"created_at": datetime.now(timezone.utc).replace(tzinfo=None)})
        s = score_node(node, query_words=set())
        assert s >= 5

    def test_zero_score_for_irrelevant_node(self):
        node = make_node("n1", "XYZ", summary="completely unrelated content")
        s = score_node(node, query_words={"alice", "paris"})
        assert s == 0.0


# =============================================================================
# Path Scorer: Budget Enforcement
# =============================================================================


class TestNodeBudget:
    """Tests for rank_and_limit_nodes budget enforcement."""

    def test_empty_nodes(self, default_config: GraphPathMiningConfig):
        result = rank_and_limit_nodes([], query="test", config=default_config)
        assert result == []

    def test_max_nodes_enforced(self, default_config: GraphPathMiningConfig):
        config = default_config.model_copy(update={"max_nodes": 3})
        nodes = [make_node(f"n{i}", f"Alice{i}", summary="Alice went to Paris") for i in range(10)]
        result = rank_and_limit_nodes(nodes, query="Alice Paris", config=config)
        assert len(result) <= 3

    def test_min_nodes_enforced(self, default_config: GraphPathMiningConfig):
        config = default_config.model_copy(update={"min_nodes": 5, "max_nodes": 25})
        # Create nodes where only 2 are relevant
        nodes = [
            make_node("n1", "Alice", summary="Alice went to Paris"),
            make_node("n2", "Bob", summary="Bob likes coffee"),
        ] + [
            make_node(f"n{i}", f"Irrelevant{i}") for i in range(3, 8)
        ]
        result = rank_and_limit_nodes(nodes, query="Alice Paris", config=config)
        assert len(result) >= min(5, len(nodes))

    def test_threshold_filtering(self, default_config: GraphPathMiningConfig):
        """Nodes below threshold should be filtered out (unless min_nodes requires them)."""
        config = default_config.model_copy(
            update={"node_score_threshold_pct": 0.50, "min_nodes": 0, "max_nodes": 100}
        )
        nodes = [
            make_node("n1", "Alice", summary="Alice went to Paris"),  # high score
            make_node("n2", "XYZ", summary="completely irrelevant"),  # zero score
        ]
        result = rank_and_limit_nodes(nodes, query="Alice Paris", config=config)
        # XYZ should be filtered out since its score is 0 (below 50% of max)
        result_ids = [n.node_id for n in result]
        assert "n1" in result_ids

    def test_sorted_by_score(self, default_config: GraphPathMiningConfig):
        """Returned nodes should be sorted by score (highest first)."""
        nodes = [
            make_node("n1", "XYZ"),
            make_node("n2", "Alice", summary="Alice went to Paris"),
        ]
        result = rank_and_limit_nodes(nodes, query="Alice Paris", config=default_config)
        if len(result) >= 2:
            # Alice node should come first
            assert result[0].name == "Alice"


# =============================================================================
# Result Types
# =============================================================================


class TestResultTypes:
    """Tests for Phase 5 result types."""

    def test_reasoning_path_creation(self):
        path = ReasoningPath(
            node_ids=["n1", "n2", "n3"],
            edge_types=["KNOWS", "WORKS_WITH"],
            score=0.85,
        )
        assert len(path.node_ids) == 3
        assert len(path.edge_types) == 2
        assert path.score == 0.85

    def test_bridge_connection_creation(self):
        bridge = BridgeConnection(
            source_node_id="n1",
            bridge_node_id="n_bridge",
            target_node_id="n2",
            bridge_type="inferred",
        )
        assert bridge.bridge_node_id == "n_bridge"
        assert bridge.bridge_type == "inferred"

    def test_graph_retrieval_result_with_paths(self):
        rr = GraphRetrievalResult(
            plan_mode="hybrid",
            plan_reason="test",
            nodes=[],
            edges=[],
            evidence=[],
            debug={},
            reasoning_paths=[
                ReasoningPath(node_ids=["n1", "n2"], edge_types=["KNOWS"]),
            ],
            bridge_connections=[
                BridgeConnection(
                    source_node_id="n1",
                    bridge_node_id="nb",
                    target_node_id="n2",
                ),
            ],
        )
        assert len(rr.reasoning_paths) == 1
        assert len(rr.bridge_connections) == 1

    def test_graph_retrieval_result_default_empty(self):
        rr = GraphRetrievalResult(
            plan_mode="local",
            plan_reason="test",
            nodes=[],
            edges=[],
            evidence=[],
            debug={},
        )
        assert rr.reasoning_paths == []
        assert rr.bridge_connections == []


# =============================================================================
# Config
# =============================================================================


class TestGraphPathMiningConfig:
    """Tests for GraphPathMiningConfig."""

    def test_defaults(self):
        config = GraphPathMiningConfig()
        assert config.enabled is False
        assert config.bridge_discovery_enabled is False
        assert config.max_path_depth == 3
        assert config.max_paths == 10
        assert config.min_path_length == 2
        assert config.max_nodes == 25
        assert config.min_nodes == 8
        assert config.temporal_flow_hours == 6.0
        assert config.temporal_window_hours == 168
        assert config.bridge_search_top_k == 5
        assert config.bridge_proximity_hours == 24.0
        assert config.node_score_threshold_pct == 0.10

    def test_custom_values(self):
        config = GraphPathMiningConfig(
            enabled=True,
            bridge_discovery_enabled=True,
            max_path_depth=5,
            max_paths=20,
            min_path_length=3,
        )
        assert config.enabled is True
        assert config.max_path_depth == 5
        assert config.max_paths == 20
        assert config.min_path_length == 3


# =============================================================================
# Integration: Full Pipeline
# =============================================================================


class TestFullPipeline:
    """Integration tests for the full bridge discovery + path mining pipeline."""

    def test_pipeline_with_connected_graph(self, default_config: GraphPathMiningConfig):
        """A fully connected graph should produce paths but no bridges."""
        nodes = [
            make_node("n1", "Alice", labels=["Person"]),
            make_node("n2", "Bob", labels=["Person"]),
            make_node("n3", "TechCorp", labels=["Organization"]),
        ]
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e2", "n2", "n3", "WORKS_FOR"),
        ]

        # Check connections
        conn_12 = check_connection(nodes[0], nodes[1], edges)
        conn_23 = check_connection(nodes[1], nodes[2], edges)
        assert conn_12 is not None
        assert conn_23 is not None

        # Mine paths
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)
        assert len(paths) > 0

        # Score and budget
        ranked = rank_and_limit_nodes(
            nodes, query="How are Alice and TechCorp connected?", config=default_config
        )
        assert len(ranked) > 0

    def test_pipeline_identifies_disconnected_pairs(self, default_config: GraphPathMiningConfig):
        """Disconnected nodes should be identified for bridge search."""
        nodes = [
            make_node("n1", "Alice", labels=["Person"]),
            make_node("n2", "TechCorp", labels=["Organization"]),
        ]
        edges = []  # no edges

        conn = check_connection(nodes[0], nodes[1], edges)
        assert conn is None  # disconnected

    def test_pipeline_paths_include_edge_types(self, default_config: GraphPathMiningConfig):
        """Paths should include the edge types traversed."""
        nodes = [
            make_node("n1", "Alice"),
            make_node("n2", "Bob"),
            make_node("n3", "TechCorp"),
        ]
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e2", "n2", "n3", "WORKS_FOR"),
        ]
        paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=default_config)

        # Find the 3-node path
        long_paths = [p for p in paths if len(p.node_ids) == 3]
        if long_paths:
            p = long_paths[0]
            assert len(p.edge_types) == 2
            assert all(et in ("KNOWS", "WORKS_FOR") for et in p.edge_types)

    def test_pipeline_star_topology(self, default_config: GraphPathMiningConfig):
        """Star topology: central node connected to many leaves."""
        center = make_node("center", "Hub")
        leaves = [make_node(f"leaf{i}", f"Leaf{i}") for i in range(5)]
        all_nodes = [center] + leaves
        edges = [make_edge(f"e{i}", "center", f"leaf{i}") for i in range(5)]

        paths = discover_reasoning_paths(nodes=all_nodes, edges=edges, config=default_config)
        assert len(paths) > 0

        # All paths should pass through center
        for p in paths:
            if len(p.node_ids) >= 3:
                assert "center" in p.node_ids
