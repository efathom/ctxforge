from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.session import Session
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode, GraphSearchFilters
from ctxforge.protocols.graph_maintenance import EdgeTemporalInfo, IGraphEdgeTemporalExtractor
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class _DummyExtractor:
    async def extract(self, *, scope_id: str, episodes: list[GraphEpisode], ontology, model=None):
        nodes = [
            GraphNode(node_id="p1", scope_id=scope_id, name="Alice", labels=["Person"]),
            GraphNode(node_id="o1", scope_id=scope_id, name="Acme", labels=["Organization"]),
        ]
        edges = [
            GraphEdge(
                edge_id="e1",
                scope_id=scope_id,
                source_node_id="p1",
                target_node_id="o1",
                edge_type="WORKS_FOR",
                fact="Alice works for Acme",
                valid_at=None,
                invalid_at=None,
            )
        ]
        return nodes, edges


class _FixedTemporalExtractor(IGraphEdgeTemporalExtractor):
    async def extract_temporal_info(self, *, scope_id: str, edge: GraphEdge, episodes: list[GraphEpisode], model=None):
        return EdgeTemporalInfo(valid_at="2020-01-01T00:00:00Z", invalid_at=None)


@pytest.mark.asyncio
async def test_engine_applies_temporal_enrichment_before_persisting_edges():
    store = InMemoryGraphStore()
    scope = "u"

    cfg = DEFAULT_CONFIG.merge_with({"graph": {"enabled": True, "temporal": {"enabled": True}}})
    graph_service = GraphService(
        config=cfg,
        graph_store=store,
        graph_extractor=_DummyExtractor(),  # type: ignore
        graph_ontology=object(),  # type: ignore
        temporal_extractor=_FixedTemporalExtractor(),
        background_tasks=set(),
    )
    _engine = CtxForge(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        graph_service=graph_service,
    )

    s = Session(user_id=scope)
    await graph_service.ingest_turn(
        session=s,
        user_input="I started working for Acme on 2020-01-01",
        assistant_response="ok",
    )

    out = await store.search(scope, "Acme", scope="edges", limit=10, filters=GraphSearchFilters(valid_only=True))
    assert len(out.edges) == 1
    assert out.edges[0].valid_at is not None
    assert out.edges[0].valid_at.replace(tzinfo=timezone.utc) <= datetime(2020, 1, 2, tzinfo=timezone.utc)


