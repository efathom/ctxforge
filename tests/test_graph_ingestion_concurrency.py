from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.core.session import Session
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode
from ctxforge.protocols.graph_maintenance import (
    EdgeInvalidationPlan,
    EdgeTemporalInfo,
    IGraphContradictionDetector,
)
from ctxforge.protocols.llm import IEmbeddingProvider
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


class _DummyExtractor:
    async def extract(self, *, scope_id: str, episodes: list[GraphEpisode], ontology, model=None):
        nodes = [GraphNode(node_id=f"n{i}", scope_id=scope_id, name=f"Node {i}", labels=["Person"]) for i in range(30)]
        edges = [
            GraphEdge(
                edge_id=f"e{i}",
                scope_id=scope_id,
                source_node_id="n0",
                target_node_id=f"n{i+1}",
                edge_type="WORKS_FOR",
                fact=f"n0 WORKS_FOR n{i+1}",
            )
            for i in range(20)
        ]
        return nodes, edges


@dataclass
class _Counters:
    in_flight: int = 0
    max_in_flight: int = 0

    async def enter(self):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        # Yield to let concurrency build up deterministically.
        await asyncio.sleep(0.01)

    def exit(self):
        self.in_flight -= 1


class _FakeEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, counters: _Counters):
        self._c = counters

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake"

    @property
    def embedding_dimension(self) -> int:
        return 3

    async def embed(self, texts, model: Optional[str] = None, **kwargs):
        # Force the engine down the embed_single fallback by returning mismatch.
        class _R:
            embeddings = []

        return _R()

    async def embed_single(self, text: str, model: Optional[str] = None, **kwargs):
        await self._c.enter()
        try:
            return [0.1, 0.2, 0.3]
        finally:
            self._c.exit()


class _FakeTemporalExtractor:
    def __init__(self, counters: _Counters):
        self._c = counters

    async def extract_temporal_info(self, *, scope_id: str, edge: GraphEdge, episodes: list[GraphEpisode], model=None):
        await self._c.enter()
        try:
            return EdgeTemporalInfo(valid_at=None, invalid_at=None)
        finally:
            self._c.exit()


class _FakeContradictionDetector(IGraphContradictionDetector):
    def __init__(self, counters: _Counters):
        self._c = counters

    async def detect_contradictions(
        self,
        *,
        scope_id: str,
        new_edge: GraphEdge,
        candidate_edges: list[GraphEdge],
        nodes: list[GraphNode],
        episodes: list[GraphEpisode],
        model: str | None = None,
    ) -> EdgeInvalidationPlan:
        await self._c.enter()
        try:
            return EdgeInvalidationPlan(invalidate_edge_ids=[])
        finally:
            self._c.exit()


@pytest.mark.asyncio
async def test_graph_ingestion_concurrency_is_bounded():
    """
    Ensure Phase 6 concurrency limits cap concurrent embedding/temporal/invalidation calls.
    """
    scope_id = "u"
    store = InMemoryGraphStore()

    emb_c = _Counters()
    tmp_c = _Counters()
    inv_c = _Counters()

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "graph": {
                "enabled": True,
                "embeddings": {"enabled": True, "max_concurrency": 3},
                "temporal": {"enabled": True, "max_concurrency": 2},
                "invalidation": {"enabled": True, "max_concurrency": 4, "candidate_limit": 1},
            }
        }
    )

    graph_service = GraphService(
        config=cfg,
        graph_store=store,
        graph_extractor=_DummyExtractor(),  # type: ignore
        graph_ontology=object(),  # type: ignore
        embedding_provider=_FakeEmbeddingProvider(emb_c),
        temporal_extractor=_FakeTemporalExtractor(tmp_c),  # type: ignore
        contradiction_detector=_FakeContradictionDetector(inv_c),
        background_tasks=set(),
    )
    _engine = CtxForge(
        config=cfg,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        graph_service=graph_service,
    )

    s = Session(user_id=scope_id)
    await graph_service.ingest_turn(session=s, user_input="hi", assistant_response="ok")

    assert emb_c.max_in_flight <= 3
    assert tmp_c.max_in_flight <= 2
    assert inv_c.max_in_flight <= 4


