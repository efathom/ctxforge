from __future__ import annotations

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.communities.builder import (
    CommunityBuildConfig,
    CommunityBuilder,
    label_propagation_clusters,
)
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphNode
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class _NoopExtractor:
    async def extract(self, *, scope_id: str, episodes, ontology, model=None):
        return [], []


@pytest.mark.asyncio
async def test_label_propagation_clusters_two_components():
    """
    Deterministic clustering should separate two disconnected components.
    """
    node_ids = ["a", "b", "c", "x", "y"]
    edges = [
        GraphEdge(edge_id="e1", scope_id="s", source_node_id="a", target_node_id="b", edge_type="R"),
        GraphEdge(edge_id="e2", scope_id="s", source_node_id="b", target_node_id="c", edge_type="R"),
        GraphEdge(edge_id="e3", scope_id="s", source_node_id="x", target_node_id="y", edge_type="R"),
    ]
    clusters = label_propagation_clusters(node_ids=node_ids, edges=edges, max_iters=10)
    assert any(set(c) == {"a", "b", "c"} for c in clusters)
    assert any(set(c) == {"x", "y"} for c in clusters)


@pytest.mark.asyncio
async def test_inmemory_store_community_lookup_by_overlap():
    """
    Communities should be returned ranked by overlap with the requested node ids.
    """
    scope = "u"
    store = InMemoryGraphStore()

    # Seed nodes/edges (not used directly by lookup).
    await store.upsert_nodes(scope, [GraphNode(node_id="a", scope_id=scope, name="A"), GraphNode(node_id="b", scope_id=scope, name="B")])

    builder = CommunityBuilder(llm_provider=None, embedding_provider=None)
    comms, mems = await builder.build(
        scope_id=scope,
        nodes=[
            GraphNode(node_id="a", scope_id=scope, name="A"),
            GraphNode(node_id="b", scope_id=scope, name="B"),
            GraphNode(node_id="c", scope_id=scope, name="C"),
        ],
        edges=[GraphEdge(edge_id="e", scope_id=scope, source_node_id="a", target_node_id="b", edge_type="R")],
        config=CommunityBuildConfig(min_cluster_size=2, max_communities=5, max_concurrency=2),
    )
    await store.upsert_communities(scope, comms)
    await store.upsert_memberships(scope, mems)

    out = await store.get_communities_for_nodes(scope, ["a", "b"], limit=5)
    assert out
    assert out[0].overlap is not None and out[0].overlap >= 2


@pytest.mark.asyncio
async def test_engine_graph_section_includes_communities_block():
    """
    After rebuilding communities, the graph context should include a <COMMUNITIES> block.
    """
    scope = "u"
    store = InMemoryGraphStore()

    await store.upsert_nodes(
        scope,
        [
            GraphNode(node_id="p", scope_id=scope, name="Alice", labels=["Person"]),
            GraphNode(node_id="o", scope_id=scope, name="Acme", labels=["Organization"]),
        ],
    )
    await store.upsert_edges(
        scope,
        [
            GraphEdge(
                edge_id="e1",
                scope_id=scope,
                source_node_id="p",
                target_node_id="o",
                edge_type="WORKS_FOR",
                fact="Alice works for Acme",
            )
        ],
    )

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "graph": {
                "enabled": True,
                "communities": {"enabled": True, "rebuild_every_n_episodes": 0, "max_communities": 5},
                "retrieval": {"enabled": True, "methods": ["keyword"], "seed_k": 3, "bfs_max_depth": 1},
            }
        }
    )

    graph_service = GraphService(
        config=cfg,
        graph_store=store,
        graph_extractor=_NoopExtractor(),  # type: ignore
        graph_ontology=object(),  # type: ignore
        community_builder=CommunityBuilder(llm_provider=None, embedding_provider=None),
        background_tasks=set(),
    )
    _engine = CtxForge(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        graph_service=graph_service,
    )

    # Rebuild and then fetch a graph section that should reference the clustered nodes.
    await graph_service.rebuild_communities(scope_id=scope)
    section = await graph_service.build_section(user_id=scope, query="Acme")
    assert section is not None
    assert "<COMMUNITIES>" in section


