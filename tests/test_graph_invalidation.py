from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.session import Session
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode, GraphSearchFilters
from ctxforge.protocols.graph_maintenance import EdgeInvalidationPlan, IGraphContradictionDetector
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class _DummyExtractor:
    async def extract(self, *, scope_id: str, episodes: list[GraphEpisode], ontology, model=None):
        # Return a single conflicting edge: user now works for Globex.
        nodes = [
            GraphNode(node_id="p1", scope_id=scope_id, name="Alice", labels=["Person"]),
            GraphNode(node_id="o2", scope_id=scope_id, name="Globex", labels=["Organization"]),
        ]
        edges = [
            GraphEdge(
                edge_id="e_new",
                scope_id=scope_id,
                source_node_id="p1",
                target_node_id="o2",
                edge_type="WORKS_FOR",
                fact="Alice works for Globex",
            )
        ]
        return nodes, edges


class _InvalidateAllCandidates(IGraphContradictionDetector):
    async def detect_contradictions(
        self,
        *,
        scope_id: str,
        new_edge: GraphEdge,
        candidate_edges: list[GraphEdge],
        nodes: list[GraphNode],
        episodes: list[GraphEpisode],
        model=None,
    ) -> EdgeInvalidationPlan:
        return EdgeInvalidationPlan(invalidate_edge_ids=[e.edge_id for e in candidate_edges])


@pytest.mark.asyncio
async def test_engine_graph_ingestion_invalidates_conflicting_edges():
    store = InMemoryGraphStore()
    scope = "u"

    # Pre-existing fact: Alice works for Acme.
    await store.upsert_nodes(
        scope,
        [
            GraphNode(node_id="p1", scope_id=scope, name="Alice", labels=["Person"]),
            GraphNode(node_id="o1", scope_id=scope, name="Acme", labels=["Organization"]),
        ],
    )
    await store.upsert_edges(
        scope,
        [
            GraphEdge(
                edge_id="e_old",
                scope_id=scope,
                source_node_id="p1",
                target_node_id="o1",
                edge_type="WORKS_FOR",
                fact="Alice works for Acme",
                invalid_at=None,
            )
        ],
    )

    cfg = DEFAULT_CONFIG.merge_with({"graph": {"enabled": True, "invalidation": {"enabled": True}}})
    graph_service = GraphService(
        config=cfg,
        graph_store=store,
        graph_extractor=_DummyExtractor(),  # type: ignore
        graph_ontology=object(),  # type: ignore
        contradiction_detector=_InvalidateAllCandidates(),
        background_tasks=set(),
    )
    _engine = CtxForge(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        graph_service=graph_service,
    )

    s = Session(user_id=scope)
    await graph_service.ingest_turn(session=s, user_input="I now work for Globex", assistant_response="ok")

    out_valid = await store.search(scope, "works", scope="edges", limit=10)
    assert any(e.edge_id == "e_new" for e in out_valid.edges)
    assert all(e.edge_id != "e_old" for e in out_valid.edges)

    out_past = await store.search(
        scope,
        "works",
        scope="edges",
        limit=10,
        filters=GraphSearchFilters(valid_only=True, as_of=datetime.now(timezone.utc) - timedelta(days=1)),
    )
    assert any(e.edge_id == "e_old" for e in out_past.edges)


