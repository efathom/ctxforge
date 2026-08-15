"""Tests for KNN-based entity linking (SAME_AS edges)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.graph.maintenance.entity_linking import EntityLinker, _stable_edge_id
from ctxforge.protocols.graph import GraphNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, name: str, embedding: list[float] | None = None) -> GraphNode:
    return GraphNode(
        node_id=nid,
        scope_id="test",
        name=name,
        labels=["Entity"],
        name_embedding=embedding,
    )


def _mock_store(existing_edge_ids: set[str] | None = None) -> MagicMock:
    store = MagicMock()
    existing = existing_edge_ids or set()

    async def get_edges(scope_id, ids):
        return [MagicMock(edge_id=eid) for eid in ids if eid in existing]

    store.get_edges_by_ids = AsyncMock(side_effect=get_edges)
    store.upsert_edges = AsyncMock()
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_similar_entities():
    """Nodes with similar embeddings get linked."""
    nodes = [
        _node("a", "NYC", [1.0, 0.0, 0.0]),
        _node("b", "New York City", [0.99, 0.1, 0.0]),
    ]
    store = _mock_store()
    linker = EntityLinker(similarity_threshold=0.9)

    edges = await linker.link_entities(nodes, store, "test")

    assert len(edges) == 1
    assert edges[0].edge_type == "SAME_AS"
    assert "NYC" in edges[0].fact
    assert "New York City" in edges[0].fact
    store.upsert_edges.assert_awaited_once()


@pytest.mark.asyncio
async def test_threshold_filtering():
    """Below-threshold pairs are not linked."""
    nodes = [
        _node("a", "Alice", [1.0, 0.0, 0.0]),
        _node("b", "Bob", [0.0, 1.0, 0.0]),  # orthogonal
    ]
    store = _mock_store()
    linker = EntityLinker(similarity_threshold=0.9)

    edges = await linker.link_entities(nodes, store, "test")
    assert edges == []


@pytest.mark.asyncio
async def test_no_self_links():
    """A node is never linked to itself."""
    nodes = [_node("a", "Alice", [1.0, 0.0])]
    store = _mock_store()
    linker = EntityLinker(similarity_threshold=0.5)

    edges = await linker.link_entities(nodes, store, "test")
    assert edges == []


@pytest.mark.asyncio
async def test_deduplication():
    """Running twice with same data doesn't create duplicate edges."""
    nodes = [
        _node("a", "NYC", [1.0, 0.0]),
        _node("b", "New York", [0.99, 0.01]),
    ]
    eid = _stable_edge_id("test", "a", "b")
    store = _mock_store(existing_edge_ids={eid})
    linker = EntityLinker(similarity_threshold=0.9)

    edges = await linker.link_entities(nodes, store, "test")
    assert edges == []


@pytest.mark.asyncio
async def test_empty_embeddings():
    """Nodes without embeddings are skipped gracefully."""
    nodes = [
        _node("a", "Alice", None),
        _node("b", "Bob", None),
    ]
    store = _mock_store()
    linker = EntityLinker()

    edges = await linker.link_entities(nodes, store, "test")
    assert edges == []


@pytest.mark.asyncio
async def test_max_neighbors():
    """Each node links to at most max_neighbors peers."""
    # Create 5 nearly-identical nodes
    nodes = [
        _node(f"n{i}", f"Entity{i}", [1.0, float(i) * 0.01])
        for i in range(5)
    ]
    store = _mock_store()
    linker = EntityLinker(similarity_threshold=0.9, max_neighbors=2)

    edges = await linker.link_entities(nodes, store, "test")
    # Each node has at most 2 links
    from collections import Counter
    counts = Counter()
    for e in edges:
        counts[e.source_node_id] += 1
        counts[e.target_node_id] += 1
    for count in counts.values():
        assert count <= 2


@pytest.mark.asyncio
async def test_similarity_in_attributes():
    """Created edges store similarity score in attributes."""
    nodes = [
        _node("a", "NYC", [1.0, 0.0]),
        _node("b", "New York", [0.99, 0.01]),
    ]
    store = _mock_store()
    linker = EntityLinker(similarity_threshold=0.9)

    edges = await linker.link_entities(nodes, store, "test")
    assert len(edges) == 1
    assert "similarity" in edges[0].attributes
    assert edges[0].attributes["method"] == "knn_embedding"


def test_stable_edge_id_order_independent():
    """Edge ID is the same regardless of argument order."""
    id1 = _stable_edge_id("scope", "a", "b")
    id2 = _stable_edge_id("scope", "b", "a")
    assert id1 == id2
