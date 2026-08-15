"""
Tests for Phase 6 - Topology-Aware Context Serialization.

Covers:
- TopologySerializationConfig defaults and custom values
- LabeledFact, AnnotatedPath, BridgeSummary, TopologyView models
- TopologyAwareRenderer:
  - Label map building
  - Fact building (with/without timestamps, evidence, truncation)
  - Path building (with/without edge types, deduplication, capping)
  - Bridge summary building
  - Full render output
- GraphService integration: topology vs legacy mode dispatch
- RetrievalControllerService: merge helper, topology graph_section
- AssemblyService: graph_section_mode metadata tagging
"""

from typing import Any, Dict, List, Optional

import pytest

from ctxforge.compaction.topology_view import (
    AnnotatedPath,
    BridgeSummary,
    LabeledFact,
    TopologyAwareRenderer,
    TopologyView,
)
from ctxforge.config.base import TopologySerializationConfig
from ctxforge.graph.retrieval.types import (
    BridgeConnection,
    EvidenceItem,
    GraphEdgeHit,
    GraphNodeHit,
    GraphRetrievalResult,
    ReasoningPath,
)

# =============================================================================
# Helpers
# =============================================================================


def make_node(
    node_id: str,
    label: str,
    score: float = 1.0,
    attrs: Optional[Dict[str, Any]] = None,
) -> GraphNodeHit:
    return GraphNodeHit(
        node_id=node_id,
        label=label,
        score=score,
        attrs=attrs or {},
    )


def make_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    relation: str = "RELATED",
    score: float = 1.0,
    attrs: Optional[Dict[str, Any]] = None,
) -> GraphEdgeHit:
    return GraphEdgeHit(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        score=score,
        attrs=attrs or {},
    )


def make_evidence(
    source_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvidenceItem:
    return EvidenceItem(
        source="episode",
        source_id=source_id,
        content=content,
        score=1.0,
        metadata=metadata or {},
    )


def make_rr(
    nodes: Optional[List[GraphNodeHit]] = None,
    edges: Optional[List[GraphEdgeHit]] = None,
    evidence: Optional[List[EvidenceItem]] = None,
    reasoning_paths: Optional[List[ReasoningPath]] = None,
    bridge_connections: Optional[List[BridgeConnection]] = None,
) -> GraphRetrievalResult:
    return GraphRetrievalResult(
        plan_mode="hybrid",
        plan_reason="test",
        nodes=nodes or [],
        edges=edges or [],
        evidence=evidence or [],
        debug={},
        reasoning_paths=reasoning_paths or [],
        bridge_connections=bridge_connections or [],
    )


@pytest.fixture
def default_config() -> TopologySerializationConfig:
    return TopologySerializationConfig(enabled=True)


@pytest.fixture
def renderer(default_config: TopologySerializationConfig) -> TopologyAwareRenderer:
    return TopologyAwareRenderer(default_config)


# =============================================================================
# Config Tests
# =============================================================================


class TestTopologySerializationConfig:
    """Tests for TopologySerializationConfig."""

    def test_defaults(self):
        config = TopologySerializationConfig()
        assert config.enabled is False
        assert config.fact_label_prefix == "F"
        assert config.max_fact_content_chars == 300
        assert config.max_evidence_per_fact == 2
        assert config.max_evidence_chars == 200
        assert config.max_reasoning_paths == 10
        assert config.max_bridge_summaries == 5
        assert config.include_edge_types_in_paths is True
        assert config.include_timestamps is True

    def test_custom_values(self):
        config = TopologySerializationConfig(
            enabled=True,
            fact_label_prefix="N",
            max_fact_content_chars=150,
            max_evidence_per_fact=1,
            max_reasoning_paths=5,
            include_edge_types_in_paths=False,
            include_timestamps=False,
        )
        assert config.enabled is True
        assert config.fact_label_prefix == "N"
        assert config.max_fact_content_chars == 150
        assert config.max_evidence_per_fact == 1
        assert config.max_reasoning_paths == 5
        assert config.include_edge_types_in_paths is False
        assert config.include_timestamps is False


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for topology view data models."""

    def test_labeled_fact(self):
        fact = LabeledFact(
            label="F1",
            node_id="n1",
            entity_name="Alice",
            content="Alice met Bob",
            timestamp="2024-06-15",
            evidence=["[Evidence: 2024-06-15] Alice mentioned meeting Bob"],
        )
        assert fact.label == "F1"
        assert fact.node_id == "n1"
        assert fact.entity_name == "Alice"
        assert fact.content == "Alice met Bob"
        assert fact.timestamp == "2024-06-15"
        assert len(fact.evidence) == 1

    def test_labeled_fact_defaults(self):
        fact = LabeledFact(label="F1", node_id="n1", entity_name="A", content="test")
        assert fact.timestamp is None
        assert fact.evidence == []

    def test_annotated_path(self):
        path = AnnotatedPath(
            index=1,
            labels=["F1", "F2", "F3"],
            edge_types=["KNOWS", "WORKS_AT"],
            summary="F1 --[KNOWS]--> F2 --[WORKS_AT]--> F3",
        )
        assert path.index == 1
        assert len(path.labels) == 3
        assert len(path.edge_types) == 2
        assert "--[KNOWS]-->" in path.summary

    def test_bridge_summary(self):
        bridge = BridgeSummary(
            source_label="F1",
            bridge_label="F2",
            target_label="F3",
            source_name="Alice",
            bridge_name="Bob",
            target_name="TechCorp",
            bridge_type="inferred",
            description="Alice [F1] and TechCorp [F3] connected via Bob [F2] (inferred)",
        )
        assert bridge.source_name == "Alice"
        assert bridge.bridge_name == "Bob"
        assert "inferred" in bridge.description

    def test_topology_view(self):
        view = TopologyView()
        assert view.facts == []
        assert view.paths == []
        assert view.bridges == []

    def test_topology_view_with_data(self):
        view = TopologyView(
            facts=[LabeledFact(label="F1", node_id="n1", entity_name="A", content="test")],
            paths=[AnnotatedPath(index=1, labels=["F1"], edge_types=[], summary="F1")],
            bridges=[],
        )
        assert len(view.facts) == 1
        assert len(view.paths) == 1


# =============================================================================
# Renderer: Label Map Building
# =============================================================================


class TestLabelMapBuilding:
    """Tests for _build_label_maps."""

    def test_basic_label_map(self, renderer: TopologyAwareRenderer):
        nodes = [make_node("n1", "Alice"), make_node("n2", "Bob")]
        label_map, name_map = renderer._build_label_maps(nodes)
        assert label_map == {"n1": "F1", "n2": "F2"}
        assert name_map == {"n1": "Alice", "n2": "Bob"}

    def test_custom_prefix(self):
        config = TopologySerializationConfig(enabled=True, fact_label_prefix="N")
        r = TopologyAwareRenderer(config)
        nodes = [make_node("n1", "Alice")]
        label_map, _ = r._build_label_maps(nodes)
        assert label_map == {"n1": "N1"}

    def test_empty_nodes(self, renderer: TopologyAwareRenderer):
        label_map, name_map = renderer._build_label_maps([])
        assert label_map == {}
        assert name_map == {}


# =============================================================================
# Renderer: Fact Building
# =============================================================================


class TestFactBuilding:
    """Tests for _build_facts."""

    def test_basic_facts(self, renderer: TopologyAwareRenderer):
        nodes = [make_node("n1", "Alice", attrs={"summary": "Alice is a developer"})]
        label_map = {"n1": "F1"}
        facts = renderer._build_facts(nodes, [], [], label_map)
        assert len(facts) == 1
        assert facts[0].label == "F1"
        assert facts[0].entity_name == "Alice"
        assert "developer" in facts[0].content

    def test_fact_from_edge(self, renderer: TopologyAwareRenderer):
        nodes = [make_node("n1", "Alice"), make_node("n2", "Bob")]
        edges = [make_edge("e1", "n1", "n2", "KNOWS", attrs={"fact": "Alice knows Bob"})]
        label_map = {"n1": "F1", "n2": "F2"}
        facts = renderer._build_facts(nodes, edges, [], label_map)
        # n1 should have the edge fact
        f1 = next(f for f in facts if f.label == "F1")
        assert "Alice knows Bob" in f1.content

    def test_fact_with_timestamp(self, renderer: TopologyAwareRenderer):
        nodes = [make_node("n1", "Alice", attrs={"created_at": "2024-06-15"})]
        label_map = {"n1": "F1"}
        facts = renderer._build_facts(nodes, [], [], label_map)
        assert facts[0].timestamp == "2024-06-15"

    def test_fact_without_timestamp(self):
        config = TopologySerializationConfig(enabled=True, include_timestamps=False)
        r = TopologyAwareRenderer(config)
        nodes = [make_node("n1", "Alice", attrs={"created_at": "2024-06-15"})]
        label_map = {"n1": "F1"}
        facts = r._build_facts(nodes, [], [], label_map)
        assert facts[0].timestamp is None

    def test_fact_with_evidence(self, renderer: TopologyAwareRenderer):
        nodes = [make_node("n1", "Alice")]
        evidence = [make_evidence("n1", "Alice mentioned Bob", {"created_at": "2024-06-15"})]
        label_map = {"n1": "F1"}
        facts = renderer._build_facts(nodes, [], evidence, label_map)
        assert len(facts[0].evidence) == 1
        assert "Alice mentioned Bob" in facts[0].evidence[0]
        assert "2024-06-15" in facts[0].evidence[0]

    def test_evidence_capped(self):
        config = TopologySerializationConfig(enabled=True, max_evidence_per_fact=1)
        r = TopologyAwareRenderer(config)
        nodes = [make_node("n1", "Alice")]
        evidence = [
            make_evidence("n1", "Evidence 1"),
            make_evidence("n1", "Evidence 2"),
            make_evidence("n1", "Evidence 3"),
        ]
        label_map = {"n1": "F1"}
        facts = r._build_facts(nodes, [], evidence, label_map)
        assert len(facts[0].evidence) == 1

    def test_fact_content_truncated(self):
        config = TopologySerializationConfig(enabled=True, max_fact_content_chars=20)
        r = TopologyAwareRenderer(config)
        nodes = [make_node("n1", "Alice", attrs={"summary": "A very long summary that exceeds the limit"})]
        label_map = {"n1": "F1"}
        facts = r._build_facts(nodes, [], [], label_map)
        assert facts[0].content.endswith("...")
        assert len(facts[0].content) <= 23  # 20 + "..."

    def test_evidence_content_truncated(self):
        config = TopologySerializationConfig(enabled=True, max_evidence_chars=10)
        r = TopologyAwareRenderer(config)
        nodes = [make_node("n1", "Alice")]
        evidence = [make_evidence("n1", "A very long evidence string")]
        label_map = {"n1": "F1"}
        facts = r._build_facts(nodes, [], evidence, label_map)
        assert len(facts[0].evidence) == 1
        assert "..." in facts[0].evidence[0]

    def test_fact_fallback_to_label(self, renderer: TopologyAwareRenderer):
        """When no summary and no edge facts, content falls back to node label."""
        nodes = [make_node("n1", "Alice")]
        label_map = {"n1": "F1"}
        facts = renderer._build_facts(nodes, [], [], label_map)
        assert facts[0].content == "Alice"


# =============================================================================
# Renderer: Path Building
# =============================================================================


class TestPathBuilding:
    """Tests for _build_paths."""

    def test_basic_path(self, renderer: TopologyAwareRenderer):
        paths = [ReasoningPath(node_ids=["n1", "n2", "n3"], edge_types=["KNOWS", "WORKS_AT"])]
        label_map = {"n1": "F1", "n2": "F2", "n3": "F3"}
        edges = [
            make_edge("e1", "n1", "n2", "KNOWS"),
            make_edge("e2", "n2", "n3", "WORKS_AT"),
        ]
        result = renderer._build_paths(paths, label_map, edges)
        assert len(result) == 1
        assert result[0].labels == ["F1", "F2", "F3"]
        assert "--[KNOWS]-->" in result[0].summary
        assert "--[WORKS_AT]-->" in result[0].summary

    def test_path_without_edge_types(self):
        config = TopologySerializationConfig(enabled=True, include_edge_types_in_paths=False)
        r = TopologyAwareRenderer(config)
        paths = [ReasoningPath(node_ids=["n1", "n2"], edge_types=["KNOWS"])]
        label_map = {"n1": "F1", "n2": "F2"}
        result = r._build_paths(paths, label_map, [])
        assert len(result) == 1
        assert result[0].summary == "F1 -> F2"
        assert "--[" not in result[0].summary

    def test_path_deduplication(self, renderer: TopologyAwareRenderer):
        paths = [
            ReasoningPath(node_ids=["n1", "n2"], edge_types=["KNOWS"]),
            ReasoningPath(node_ids=["n1", "n2"], edge_types=["KNOWS"]),
        ]
        label_map = {"n1": "F1", "n2": "F2"}
        result = renderer._build_paths(paths, label_map, [])
        assert len(result) == 1

    def test_path_capping(self):
        config = TopologySerializationConfig(enabled=True, max_reasoning_paths=2)
        r = TopologyAwareRenderer(config)
        paths = [
            ReasoningPath(node_ids=[f"n{i}", f"n{i+1}"], edge_types=["REL"])
            for i in range(10)
        ]
        label_map = {f"n{i}": f"F{i}" for i in range(20)}
        result = r._build_paths(paths, label_map, [])
        assert len(result) <= 2

    def test_path_skips_single_node(self, renderer: TopologyAwareRenderer):
        """Paths with fewer than 2 resolvable labels should be skipped."""
        paths = [ReasoningPath(node_ids=["n1", "n_unknown"], edge_types=["REL"])]
        label_map = {"n1": "F1"}  # n_unknown not in map
        result = renderer._build_paths(paths, label_map, [])
        assert len(result) == 0

    def test_path_edge_type_from_path_fallback(self, renderer: TopologyAwareRenderer):
        """When edge not in edge map, fall back to path.edge_types."""
        paths = [ReasoningPath(node_ids=["n1", "n2"], edge_types=["CUSTOM"])]
        label_map = {"n1": "F1", "n2": "F2"}
        result = renderer._build_paths(paths, label_map, [])  # no edges
        assert len(result) == 1
        assert "--[CUSTOM]-->" in result[0].summary

    def test_path_sequential_indexing(self, renderer: TopologyAwareRenderer):
        paths = [
            ReasoningPath(node_ids=["n1", "n2"], edge_types=["A"]),
            ReasoningPath(node_ids=["n2", "n3"], edge_types=["B"]),
        ]
        label_map = {"n1": "F1", "n2": "F2", "n3": "F3"}
        result = renderer._build_paths(paths, label_map, [])
        assert result[0].index == 1
        assert result[1].index == 2


# =============================================================================
# Renderer: Bridge Building
# =============================================================================


class TestBridgeBuilding:
    """Tests for _build_bridges."""

    def test_basic_bridge(self, renderer: TopologyAwareRenderer):
        bridges = [BridgeConnection(source_node_id="n1", bridge_node_id="n2", target_node_id="n3")]
        label_map = {"n1": "F1", "n2": "F2", "n3": "F3"}
        name_map = {"n1": "Alice", "n2": "Bob", "n3": "TechCorp"}
        result = renderer._build_bridges(bridges, label_map, name_map)
        assert len(result) == 1
        assert result[0].source_name == "Alice"
        assert result[0].bridge_name == "Bob"
        assert result[0].target_name == "TechCorp"
        assert "Alice" in result[0].description
        assert "Bob" in result[0].description
        assert "TechCorp" in result[0].description

    def test_bridge_capping(self):
        config = TopologySerializationConfig(enabled=True, max_bridge_summaries=1)
        r = TopologyAwareRenderer(config)
        bridges = [
            BridgeConnection(source_node_id=f"s{i}", bridge_node_id=f"b{i}", target_node_id=f"t{i}")
            for i in range(5)
        ]
        label_map = {}
        name_map = {}
        result = r._build_bridges(bridges, label_map, name_map)
        assert len(result) == 1

    def test_bridge_unknown_nodes(self, renderer: TopologyAwareRenderer):
        """Unknown node IDs should be used as-is in labels/names."""
        bridges = [BridgeConnection(source_node_id="x1", bridge_node_id="x2", target_node_id="x3")]
        label_map = {}
        name_map = {}
        result = renderer._build_bridges(bridges, label_map, name_map)
        assert len(result) == 1
        assert "x1" in result[0].description
        assert "x2" in result[0].description


# =============================================================================
# Renderer: Full Render
# =============================================================================


class TestFullRender:
    """Tests for the complete render pipeline."""

    def test_render_empty_result(self, renderer: TopologyAwareRenderer):
        rr = make_rr()
        text = renderer.render(rr)
        assert text == ""

    def test_render_facts_only(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[make_node("n1", "Alice", attrs={"summary": "Alice is a developer"})],
        )
        text = renderer.render(rr)
        assert "[Facts from Graph]" in text
        assert "[F1]" in text
        assert "developer" in text
        assert "[Reasoning Paths]" not in text
        assert "[Bridge Connections" not in text

    def test_render_facts_with_timestamp(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[make_node("n1", "Alice", attrs={"summary": "Alice met Bob", "created_at": "2024-06-15"})],
        )
        text = renderer.render(rr)
        assert "[F1] 2024-06-15:" in text

    def test_render_facts_without_timestamp(self):
        config = TopologySerializationConfig(enabled=True, include_timestamps=False)
        r = TopologyAwareRenderer(config)
        rr = make_rr(
            nodes=[make_node("n1", "Alice", attrs={"summary": "Alice met Bob", "created_at": "2024-06-15"})],
        )
        text = r.render(rr)
        assert "[F1] Alice met Bob" in text
        assert "2024-06-15:" not in text

    def test_render_with_evidence(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[make_node("n1", "Alice")],
            evidence=[make_evidence("n1", "Alice mentioned meeting Bob", {"created_at": "2024-06-15"})],
        )
        text = renderer.render(rr)
        assert "[Evidence: 2024-06-15]" in text
        assert "Alice mentioned meeting Bob" in text

    def test_render_with_paths(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[make_node("n1", "Alice"), make_node("n2", "Bob")],
            edges=[make_edge("e1", "n1", "n2", "KNOWS")],
            reasoning_paths=[ReasoningPath(node_ids=["n1", "n2"], edge_types=["KNOWS"])],
        )
        text = renderer.render(rr)
        assert "[Reasoning Paths]" in text
        assert "F1 --[KNOWS]--> F2" in text

    def test_render_with_bridges(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[
                make_node("n1", "Alice"),
                make_node("n2", "Bob"),
                make_node("n3", "TechCorp"),
            ],
            bridge_connections=[
                BridgeConnection(source_node_id="n1", bridge_node_id="n2", target_node_id="n3"),
            ],
        )
        text = renderer.render(rr)
        assert "[Bridge Connections: 1 inferred links]" in text
        assert "Alice" in text
        assert "Bob" in text
        assert "TechCorp" in text

    def test_render_full(self, renderer: TopologyAwareRenderer):
        """Full render with facts, evidence, paths, and bridges."""
        rr = make_rr(
            nodes=[
                make_node("n1", "Alice", attrs={"summary": "Alice is a developer", "created_at": "2024-01-01"}),
                make_node("n2", "Bob", attrs={"summary": "Bob is a manager"}),
                make_node("n3", "TechCorp", attrs={"summary": "TechCorp is a company"}),
            ],
            edges=[
                make_edge("e1", "n1", "n2", "KNOWS", attrs={"fact": "Alice knows Bob"}),
                make_edge("e2", "n2", "n3", "WORKS_AT", attrs={"fact": "Bob works at TechCorp"}),
            ],
            evidence=[
                make_evidence("n1", "Alice said she met Bob at a conference", {"created_at": "2024-01-01"}),
            ],
            reasoning_paths=[
                ReasoningPath(node_ids=["n1", "n2", "n3"], edge_types=["KNOWS", "WORKS_AT"]),
            ],
            bridge_connections=[
                BridgeConnection(source_node_id="n1", bridge_node_id="n2", target_node_id="n3"),
            ],
        )
        text = renderer.render(rr)

        # All sections present
        assert "[Facts from Graph]" in text
        assert "[Reasoning Paths]" in text
        assert "[Bridge Connections:" in text

        # Facts use labels
        assert "[F1]" in text
        assert "[F2]" in text
        assert "[F3]" in text

        # Evidence inline
        assert "[Evidence:" in text

        # Path references labels with edge types
        assert "F1 --[KNOWS]--> F2 --[WORKS_AT]--> F3" in text

        # Bridge describes entities
        assert "Alice" in text
        assert "Bob" in text
        assert "TechCorp" in text

    def test_build_view_returns_topology_view(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[make_node("n1", "Alice")],
            reasoning_paths=[ReasoningPath(node_ids=["n1", "n2"], edge_types=["REL"])],
        )
        view = renderer.build_view(rr)
        assert isinstance(view, TopologyView)
        assert len(view.facts) == 1

    def test_render_view_matches_render(self, renderer: TopologyAwareRenderer):
        rr = make_rr(
            nodes=[make_node("n1", "Alice", attrs={"summary": "test"})],
        )
        view = renderer.build_view(rr)
        assert renderer.render(rr) == renderer.render_view(view)


# =============================================================================
# Renderer: Fact Rendering
# =============================================================================


class TestFactRendering:
    """Tests for _render_fact."""

    def test_render_fact_with_timestamp(self, renderer: TopologyAwareRenderer):
        fact = LabeledFact(label="F1", node_id="n1", entity_name="A", content="test", timestamp="2024-06-15")
        line = renderer._render_fact(fact)
        assert line == "[F1] 2024-06-15: test"

    def test_render_fact_without_timestamp(self, renderer: TopologyAwareRenderer):
        fact = LabeledFact(label="F1", node_id="n1", entity_name="A", content="test")
        line = renderer._render_fact(fact)
        assert line == "[F1] test"


# =============================================================================
# Integration: RetrievalControllerService._merge_graph_results
# =============================================================================


class TestMergeGraphResults:
    """Tests for the static _merge_graph_results helper."""

    def test_single_result_passthrough(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr = make_rr(nodes=[make_node("n1", "Alice")])
        merged = RetrievalControllerService._merge_graph_results([rr])
        assert merged is rr

    def test_merge_deduplicates_nodes(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr1 = make_rr(nodes=[make_node("n1", "Alice"), make_node("n2", "Bob")])
        rr2 = make_rr(nodes=[make_node("n2", "Bob"), make_node("n3", "Charlie")])
        merged = RetrievalControllerService._merge_graph_results([rr1, rr2])
        assert len(merged.nodes) == 3
        node_ids = [n.node_id for n in merged.nodes]
        assert node_ids == ["n1", "n2", "n3"]

    def test_merge_deduplicates_edges(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr1 = make_rr(edges=[make_edge("e1", "n1", "n2")])
        rr2 = make_rr(edges=[make_edge("e1", "n1", "n2"), make_edge("e2", "n2", "n3")])
        merged = RetrievalControllerService._merge_graph_results([rr1, rr2])
        assert len(merged.edges) == 2

    def test_merge_concatenates_evidence(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr1 = make_rr(evidence=[make_evidence("n1", "ev1")])
        rr2 = make_rr(evidence=[make_evidence("n2", "ev2")])
        merged = RetrievalControllerService._merge_graph_results([rr1, rr2])
        assert len(merged.evidence) == 2

    def test_merge_concatenates_paths(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr1 = make_rr(reasoning_paths=[ReasoningPath(node_ids=["n1", "n2"], edge_types=["A"])])
        rr2 = make_rr(reasoning_paths=[ReasoningPath(node_ids=["n3", "n4"], edge_types=["B"])])
        merged = RetrievalControllerService._merge_graph_results([rr1, rr2])
        assert len(merged.reasoning_paths) == 2

    def test_merge_concatenates_bridges(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr1 = make_rr(bridge_connections=[BridgeConnection(source_node_id="n1", bridge_node_id="n2", target_node_id="n3")])
        rr2 = make_rr(bridge_connections=[BridgeConnection(source_node_id="n4", bridge_node_id="n5", target_node_id="n6")])
        merged = RetrievalControllerService._merge_graph_results([rr1, rr2])
        assert len(merged.bridge_connections) == 2

    def test_merge_uses_first_plan(self):
        from ctxforge.engine.services.retrieval_controller_service import RetrievalControllerService

        rr1 = GraphRetrievalResult(
            plan_mode="local", plan_reason="first", nodes=[], edges=[], evidence=[], debug={},
        )
        rr2 = GraphRetrievalResult(
            plan_mode="global", plan_reason="second", nodes=[], edges=[], evidence=[], debug={},
        )
        merged = RetrievalControllerService._merge_graph_results([rr1, rr2])
        assert merged.plan_mode == "local"
        assert merged.plan_reason == "first"


# =============================================================================
# Integration: AssemblyService graph_section_mode
# =============================================================================


class TestAssemblyServiceMode:
    """Tests for graph_section_mode metadata tagging."""

    @pytest.mark.asyncio
    async def test_flat_mode_tagged(self):
        from unittest.mock import AsyncMock, MagicMock

        from ctxforge.config.base import EngineConfig
        from ctxforge.core.session import Session
        from ctxforge.engine.services.assembly_service import AssemblyService

        config = EngineConfig()
        mock_assembler = MagicMock()
        mock_context = MagicMock()
        mock_context.metadata = {}
        mock_context.add_section = MagicMock()
        mock_assembler.assemble = AsyncMock(return_value=mock_context)

        svc = AssemblyService(
            config=config,
            assembler_provider=lambda: mock_assembler,
            set_assembler=lambda x: None,
        )
        session = Session(session_id="s1", user_id="u1")
        ctx = await svc.assemble(
            session=session,
            current_query="test",
            memories=[],
            system_instructions="sys",
            token_budget=8000,
            include_history=True,
            max_history_events=10,
            graph_section="<FACTS>\n- test",
            graph_section_mode="flat",
        )
        assert ctx.metadata["graph_section_mode"] == "flat"

    @pytest.mark.asyncio
    async def test_topology_mode_tagged(self):
        from unittest.mock import AsyncMock, MagicMock

        from ctxforge.config.base import EngineConfig
        from ctxforge.core.session import Session
        from ctxforge.engine.services.assembly_service import AssemblyService

        config = EngineConfig()
        mock_assembler = MagicMock()
        mock_context = MagicMock()
        mock_context.metadata = {}
        mock_context.add_section = MagicMock()
        mock_assembler.assemble = AsyncMock(return_value=mock_context)

        svc = AssemblyService(
            config=config,
            assembler_provider=lambda: mock_assembler,
            set_assembler=lambda x: None,
        )
        session = Session(session_id="s1", user_id="u1")
        ctx = await svc.assemble(
            session=session,
            current_query="test",
            memories=[],
            system_instructions="sys",
            token_budget=8000,
            include_history=True,
            max_history_events=10,
            graph_section="[Facts from Graph]\n[F1] test",
            graph_section_mode="topology",
        )
        assert ctx.metadata["graph_section_mode"] == "topology"

    @pytest.mark.asyncio
    async def test_no_graph_section_no_mode(self):
        from unittest.mock import AsyncMock, MagicMock

        from ctxforge.config.base import EngineConfig
        from ctxforge.core.session import Session
        from ctxforge.engine.services.assembly_service import AssemblyService

        config = EngineConfig()
        mock_assembler = MagicMock()
        mock_context = MagicMock()
        mock_context.metadata = {}
        mock_context.add_section = MagicMock()
        mock_assembler.assemble = AsyncMock(return_value=mock_context)

        svc = AssemblyService(
            config=config,
            assembler_provider=lambda: mock_assembler,
            set_assembler=lambda x: None,
        )
        session = Session(session_id="s1", user_id="u1")
        ctx = await svc.assemble(
            session=session,
            current_query="test",
            memories=[],
            system_instructions="sys",
            token_budget=8000,
            include_history=True,
            max_history_events=10,
        )
        assert "graph_section_mode" not in ctx.metadata
