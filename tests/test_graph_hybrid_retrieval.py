from __future__ import annotations

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphNode
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class _NoopExtractor:
    async def extract(self, *, scope_id: str, episodes, ontology, model=None):
        return [], []


@pytest.mark.asyncio
async def test_graph_section_bfs_expands_to_multi_hop_edges():
    """
    The hybrid graph section builder should expand beyond direct keyword matches by traversing
    the neighborhood (BFS) starting from seeded nodes.
    """
    scope_id = "u"
    store = InMemoryGraphStore()

    await store.upsert_nodes(
        scope_id,
        [
            GraphNode(node_id="p", scope_id=scope_id, name="Alice", labels=["Person"]),
            GraphNode(node_id="o", scope_id=scope_id, name="Acme", labels=["Organization"]),
            GraphNode(node_id="c", scope_id=scope_id, name="Berlin", labels=["Location"]),
        ],
    )
    await store.upsert_edges(
        scope_id,
        [
            GraphEdge(
                edge_id="e1",
                scope_id=scope_id,
                source_node_id="p",
                target_node_id="o",
                edge_type="WORKS_FOR",
                fact="Alice WORKS_FOR Acme",
            ),
            GraphEdge(
                edge_id="e2",
                scope_id=scope_id,
                source_node_id="o",
                target_node_id="c",
                edge_type="LOCATED_IN",
                fact="Acme is LOCATED_IN Berlin",
            ),
        ],
    )

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "methods": ["keyword", "bfs"],
                    "seed_k": 2,
                    "bfs_max_depth": 2,
                    "bfs_edges_per_node": 10,
                    "rerank_enabled": True,
                    "reranker": "rrf",
                    "rerank_top_k": 20,
                    "max_facts": 20,
                    "max_entities": 20,
                },
            }
        }
    )

    graph_service = GraphService(
        config=cfg,
        graph_store=store,
        graph_extractor=_NoopExtractor(),  # type: ignore
        graph_ontology=object(),  # type: ignore
        background_tasks=set(),
    )
    _engine = CtxForge(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        graph_service=graph_service,
    )

    section = await graph_service.build_section(user_id=scope_id, query="Acme")
    assert section is not None
    # Direct match edge should appear.
    assert "WORKS_FOR" in section or "Alice WORKS_FOR Acme" in section
    # BFS should pull in the 2-hop neighbor edge even though "Berlin" isn't in the query.
    assert "LOCATED_IN" in section or "Acme is LOCATED_IN Berlin" in section


