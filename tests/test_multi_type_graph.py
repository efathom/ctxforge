"""Tests for multi-type graph nodes (Passage, Fact) and their edges."""

from ctxforge.graph.default_ontology import GRAPH_ONTOLOGY
from ctxforge.graph.extraction.llm import LLMGraphExtractor, _stable_node_id
from ctxforge.graph.retrieval.pagerank import DEFAULT_NODE_TYPE_WEIGHTS
from ctxforge.graph.utils import format_graph_context
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode

# ---------------------------------------------------------------------------
# Ontology validation tests
# ---------------------------------------------------------------------------


def test_ontology_has_passage_and_fact():
    """Ontology defines Passage and Fact entity types."""
    assert GRAPH_ONTOLOGY.is_entity_type_known("Passage")
    assert GRAPH_ONTOLOGY.is_entity_type_known("Fact")


def test_ontology_same_as_edges():
    """SAME_AS edges are defined for entity pairs."""
    assert GRAPH_ONTOLOGY.is_edge_type_known("SAME_AS")
    assert GRAPH_ONTOLOGY.is_edge_allowed("SAME_AS", "Person", "Person")
    assert GRAPH_ONTOLOGY.is_edge_allowed("SAME_AS", "Organization", "Organization")


def test_ontology_mentions_edges():
    """MENTIONS edges connect Passage to entity types."""
    assert GRAPH_ONTOLOGY.is_edge_type_known("MENTIONS")
    assert GRAPH_ONTOLOGY.is_edge_allowed("MENTIONS", "Passage", "Person")
    assert GRAPH_ONTOLOGY.is_edge_allowed("MENTIONS", "Passage", "Organization")
    assert GRAPH_ONTOLOGY.is_edge_allowed("MENTIONS", "Passage", "Location")


def test_ontology_evidences_edges():
    """EVIDENCES edges connect Passage to Fact."""
    assert GRAPH_ONTOLOGY.is_edge_type_known("EVIDENCES")
    assert GRAPH_ONTOLOGY.is_edge_allowed("EVIDENCES", "Passage", "Fact")


def test_ontology_subject_object_of_edges():
    """SUBJECT_OF and OBJECT_OF edges connect entities to Facts."""
    for edge_type in ("SUBJECT_OF", "OBJECT_OF"):
        assert GRAPH_ONTOLOGY.is_edge_type_known(edge_type)
        assert GRAPH_ONTOLOGY.is_edge_allowed(edge_type, "Person", "Fact")
        assert GRAPH_ONTOLOGY.is_edge_allowed(edge_type, "Organization", "Fact")
        assert GRAPH_ONTOLOGY.is_edge_allowed(edge_type, "Location", "Fact")


# ---------------------------------------------------------------------------
# Passage and Fact node enrichment tests
# ---------------------------------------------------------------------------


def test_build_passage_and_fact_nodes():
    """_build_passage_and_fact_nodes creates correct structure."""
    extractor = LLMGraphExtractor.__new__(LLMGraphExtractor)

    episodes = [
        GraphEpisode(
            episode_id="ep1",
            scope_id="scope",
            content="Alice works at Apple.",
        )
    ]

    alice_id = _stable_node_id("scope", "Person", "Alice")
    apple_id = _stable_node_id("scope", "Organization", "Apple")

    entity_nodes = {
        ("Person", "alice"): GraphNode(
            node_id=alice_id, scope_id="scope", name="Alice",
            labels=["Person"], source_episode_ids=["ep1"],
        ),
        ("Organization", "apple"): GraphNode(
            node_id=apple_id, scope_id="scope", name="Apple",
            labels=["Organization"], source_episode_ids=["ep1"],
        ),
    }

    entity_edges = [
        GraphEdge(
            edge_id="e1", scope_id="scope",
            source_node_id=alice_id, target_node_id=apple_id,
            edge_type="WORKS_FOR", fact="Alice works at Apple",
            source_episode_ids=["ep1"],
        )
    ]

    passages, facts, extra_edges = extractor._build_passage_and_fact_nodes(
        scope_id="scope",
        episodes=episodes,
        entity_nodes=entity_nodes,
        entity_edges=entity_edges,
        ontology=GRAPH_ONTOLOGY,
    )

    # One passage per episode
    assert len(passages) == 1
    assert passages[0].labels == ["Passage"]
    assert passages[0].attributes["source_episode_id"] == "ep1"

    # One fact per edge with fact text
    assert len(facts) == 1
    assert facts[0].labels == ["Fact"]
    assert facts[0].attributes["subject"] == "Alice"
    assert facts[0].attributes["object_value"] == "Apple"

    # Edges: MENTIONS (passage→alice, passage→apple) + SUBJECT_OF + OBJECT_OF + EVIDENCES
    edge_types = [e.edge_type for e in extra_edges]
    assert "MENTIONS" in edge_types
    assert "SUBJECT_OF" in edge_types
    assert "OBJECT_OF" in edge_types
    assert "EVIDENCES" in edge_types


def test_build_skipped_without_ontology_support():
    """Enrichment skipped if ontology lacks Passage/Fact types."""
    from ctxforge.graph.ontology import GraphOntology

    # Minimal ontology without Passage/Fact
    minimal = GraphOntology(
        entity_types={"Person": None},
        edge_types={},
        allowed_edges={},
    )

    extractor = LLMGraphExtractor.__new__(LLMGraphExtractor)
    episodes = [GraphEpisode(episode_id="ep1", scope_id="s", content="test")]
    passages, facts, edges = extractor._build_passage_and_fact_nodes(
        scope_id="s", episodes=episodes,
        entity_nodes={}, entity_edges=[], ontology=minimal,
    )
    assert passages == [] and facts == [] and edges == []


# ---------------------------------------------------------------------------
# PPR node type weights tests
# ---------------------------------------------------------------------------


def test_default_node_type_weights():
    """Default weights give Passage lower priority than entities."""
    assert DEFAULT_NODE_TYPE_WEIGHTS["Person"] == 1.0
    assert DEFAULT_NODE_TYPE_WEIGHTS["Fact"] == 0.8
    assert DEFAULT_NODE_TYPE_WEIGHTS["Passage"] == 0.3
    assert DEFAULT_NODE_TYPE_WEIGHTS["Passage"] < DEFAULT_NODE_TYPE_WEIGHTS["Fact"]


# ---------------------------------------------------------------------------
# Format context tests
# ---------------------------------------------------------------------------


def test_format_context_multi_type():
    """format_graph_context renders entity, fact, and passage nodes separately."""
    nodes = [
        GraphNode(node_id="n1", scope_id="s", name="Alice", labels=["Person"]),
        GraphNode(node_id="n2", scope_id="s", name="Alice works at Apple",
                  labels=["Fact"], attributes={"confidence": 0.95},
                  summary="Alice works at Apple"),
        GraphNode(node_id="n3", scope_id="s", name="passage_0",
                  labels=["Passage"], summary="Alice mentioned working at Apple"),
    ]

    output = format_graph_context(edges=[], nodes=nodes, episodes=[])

    assert "<ENTITIES>" in output
    assert "Alice" in output
    assert "<STRUCTURED_FACTS>" in output
    assert "FACT: Alice works at Apple" in output
    assert "confidence: 0.95" in output
    assert "<EVIDENCE>" in output
    assert "Alice mentioned working at Apple" in output
