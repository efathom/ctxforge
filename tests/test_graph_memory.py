from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.context_engine import CtxForge
from ctxforge.graph.ontology import GraphOntology
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphNode, GraphSearchFilters
from ctxforge.storage.memory.session import InMemorySessionStore


class _EdgeAttrs(BaseModel):
    strength: float = 1.0


class _EntityAttrs(BaseModel):
    pass


@pytest.mark.asyncio
async def test_in_memory_graph_store_filters_expired_edges():
    store = InMemoryGraphStore()
    scope = "u"

    n1 = GraphNode(node_id="n1", scope_id=scope, name="Alice", labels=["Person"])
    n2 = GraphNode(node_id="n2", scope_id=scope, name="Coffee", labels=["Thing"])
    await store.upsert_nodes(scope, [n1, n2])

    expired = GraphEdge(
        edge_id="e1",
        scope_id=scope,
        source_node_id="n1",
        target_node_id="n2",
        edge_type="LIKES",
        fact="Alice likes coffee",
        invalid_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    current = GraphEdge(
        edge_id="e2",
        scope_id=scope,
        source_node_id="n1",
        target_node_id="n2",
        edge_type="LIKES",
        fact="Alice likes coffee",
        invalid_at=None,
    )
    await store.upsert_edges(scope, [expired, current])

    out = await store.search(scope, "coffee", scope="edges", limit=10)
    assert [e.edge_id for e in out.edges] == ["e2"]


@pytest.mark.asyncio
async def test_in_memory_graph_store_as_of_controls_valid_only():
    store = InMemoryGraphStore()
    scope = "u"

    n1 = GraphNode(node_id="n1", scope_id=scope, name="Alice", labels=["Person"])
    n2 = GraphNode(node_id="n2", scope_id=scope, name="Acme", labels=["Org"])
    await store.upsert_nodes(scope, [n1, n2])

    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    edge = GraphEdge(
        edge_id="e1",
        scope_id=scope,
        source_node_id="n1",
        target_node_id="n2",
        edge_type="WORKS_FOR",
        fact="Alice works for Acme",
        invalid_at=cutoff,
    )
    await store.upsert_edges(scope, [edge])

    out_now = await store.search(scope, "Acme", scope="edges", limit=10)
    assert out_now.edges == []

    out_past = await store.search(
        scope,
        "Acme",
        scope="edges",
        limit=10,
        filters=GraphSearchFilters(valid_only=True, as_of=cutoff - timedelta(days=1)),
    )
    assert [e.edge_id for e in out_past.edges] == ["e1"]


@pytest.mark.asyncio
async def test_in_memory_graph_store_semantic_node_search():
    store = InMemoryGraphStore()
    scope = "u"

    n1 = GraphNode(node_id="n1", scope_id=scope, name="Acme", labels=["Organization"], name_embedding=[1.0, 0.0])
    n2 = GraphNode(node_id="n2", scope_id=scope, name="Berlin", labels=["Location"], name_embedding=[0.0, 1.0])
    await store.upsert_nodes(scope, [n1, n2])

    out = await store.search_nodes_semantic(
        scope,
        query_vector=[0.9, 0.1],
        limit=5,
        filters=GraphSearchFilters(node_labels=["Organization"]),
    )
    assert [n.node_id for n in out] == ["n1"]


@pytest.mark.asyncio
async def test_in_memory_graph_store_invalidate_edges_hides_from_valid_only():
    store = InMemoryGraphStore()
    scope = "u"

    n1 = GraphNode(node_id="n1", scope_id=scope, name="Alice", labels=["Person"])
    n2 = GraphNode(node_id="n2", scope_id=scope, name="Acme", labels=["Organization"])
    await store.upsert_nodes(scope, [n1, n2])

    e1 = GraphEdge(
        edge_id="e1",
        scope_id=scope,
        source_node_id="n1",
        target_node_id="n2",
        edge_type="WORKS_FOR",
        fact="Alice works for Acme",
    )
    await store.upsert_edges(scope, [e1])

    await store.invalidate_edges(scope, ["e1"], invalid_at=datetime.now(timezone.utc))
    out = await store.search(scope, "Acme", scope="edges", limit=10, filters=GraphSearchFilters(valid_only=True))
    assert out.edges == []


def test_ontology_validates_allowed_edge_pairs():
    ontology = GraphOntology(
        entity_types={"Person": _EntityAttrs, "Org": _EntityAttrs},
        edge_types={"WORKS_FOR": _EdgeAttrs},
        allowed_edges={"WORKS_FOR": [("Person", "Org")]},
    )

    assert ontology.is_edge_allowed("WORKS_FOR", "Person", "Org") is True
    assert ontology.is_edge_allowed("WORKS_FOR", "Org", "Person") is False


@pytest.mark.asyncio
async def test_prepare_context_includes_graph_section_when_enabled():
    store = InMemoryGraphStore()
    session_store = InMemorySessionStore()

    cfg = DEFAULT_CONFIG.merge_with({"graph": {"enabled": True}})
    engine = CtxForge(
        config=cfg,
        session_store=session_store,
        memory_store=None,  # type: ignore
        graph_store=store,
        graph_extractor=None,
        graph_ontology=None,
    )

    # Create a session and add graph facts.
    await session_store.save(await session_store.load("s1", "u"))
    await store.upsert_nodes("u", [GraphNode(node_id="n1", scope_id="u", name="Alice", labels=["Person"])])
    await store.upsert_edges(
        "u",
        [
            GraphEdge(
                edge_id="e1",
                scope_id="u",
                source_node_id="n1",
                target_node_id="n1",
                edge_type="KNOWS",
                fact="Alice knows Alice",
            )
        ],
    )

    ctx = await engine.prepare_context(session_id="s1", user_id="u", user_input="Who does Alice know?", include_memories=False)
    section = ctx.get_section(cfg.graph.section_name)
    assert section is not None
    assert "FACTS" in section.content
    assert "Alice knows Alice" in section.content


