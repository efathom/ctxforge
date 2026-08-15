import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphNode
from ctxforge.protocols.llm import EmbeddingResponse, IEmbeddingProvider


class DeterministicEmbeddingProvider(IEmbeddingProvider):
    """Minimal deterministic embedding provider for tests."""

    @property
    def name(self) -> str:
        return "deterministic-embedder"

    @property
    def default_model(self) -> str:
        return "deterministic"

    @property
    def embedding_dimension(self) -> int:
        return 2

    async def embed(self, texts, model=None, **kwargs):
        return EmbeddingResponse(embeddings=[self._vec(t) for t in texts], model=model or self.default_model)

    async def embed_single(self, text, model=None, **kwargs):
        return self._vec(text)

    def _vec(self, text: str):
        t = (text or "").lower()
        # Simple keyword-based 2D embedding.
        if "acme" in t:
            return [1.0, 0.0]
        if "globex" in t:
            return [0.0, 1.0]
        # default
        return [0.5, 0.5]


@pytest.mark.asyncio
async def test_vector_fused_triplets_prefers_edges_connected_to_semantic_seed_nodes():
    cfg = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "embeddings": {"enabled": True},
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "local",
                    "methods": ["vector_fused_triplets"],
                    "include_entities": True,
                    "max_facts": 10,
                    "max_entities": 10,
                    "vector_fused_triplets": {
                        "enabled": True,
                        "seed_node_k": 5,
                        "seed_edge_k": 5,
                        "max_candidate_edges": 50,
                        "max_output_edges": 5,
                        "edge_score_mode": "relation_keyword",
                    },
                },
            },
        }
    )

    store = InMemoryGraphStore()
    embedder = DeterministicEmbeddingProvider()
    svc = GraphService(config=cfg, graph_store=store, embedding_provider=embedder)

    scope = "u"
    # Nodes: Acme is semantically close to "Acme", Globex is not.
    await store.upsert_nodes(
        scope,
        [
            GraphNode(node_id="n_acme", scope_id=scope, name="Acme", labels=["Organization"], name_embedding=[1.0, 0.0]),
            GraphNode(node_id="n_globex", scope_id=scope, name="Globex", labels=["Organization"], name_embedding=[0.0, 1.0]),
            GraphNode(node_id="n_user", scope_id=scope, name="User", labels=["Person"], name_embedding=[0.5, 0.5]),
        ],
    )

    await store.upsert_edges(
        scope,
        [
            GraphEdge(
                edge_id="e1",
                scope_id=scope,
                source_node_id="n_user",
                target_node_id="n_acme",
                edge_type="WORKS_FOR",
                fact="User WORKS_FOR Acme",
            ),
            GraphEdge(
                edge_id="e2",
                scope_id=scope,
                source_node_id="n_user",
                target_node_id="n_globex",
                edge_type="WORKS_FOR",
                fact="User WORKS_FOR Globex",
            ),
        ],
    )

    rr = await svc.build_retrieval_result(user_id=scope, query="Where do I work? Acme")
    assert rr is not None
    assert rr.edges

    # Expect top edge to be the Acme one (semantic endpoint score).
    assert rr.edges[0].attrs.get("fact") in {"User WORKS_FOR Acme", "User WORKS_FOR Globex"}
    assert "vf_score_total" in rr.edges[0].attrs
    top_fact = rr.edges[0].attrs.get("fact")
    assert top_fact == "User WORKS_FOR Acme"


@pytest.mark.asyncio
async def test_vector_fused_triplets_falls_back_to_keyword_edge_scoring_when_embeddings_missing():
    cfg = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "embeddings": {"enabled": False},
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "global",
                    "methods": ["vector_fused_triplets"],
                    "include_entities": False,
                    "max_facts": 10,
                    "vector_fused_triplets": {"enabled": True, "edge_score_mode": "relation_keyword"},
                },
            },
        }
    )

    store = InMemoryGraphStore()
    svc = GraphService(config=cfg, graph_store=store, embedding_provider=None)

    scope = "u"
    await store.upsert_nodes(
        scope,
        [
            GraphNode(node_id="n1", scope_id=scope, name="Acme", labels=["Organization"]),
            GraphNode(node_id="n2", scope_id=scope, name="User", labels=["Person"]),
        ],
    )
    await store.upsert_edges(
        scope,
        [
            GraphEdge(
                edge_id="e1",
                scope_id=scope,
                source_node_id="n2",
                target_node_id="n1",
                edge_type="WORKS_FOR",
                fact="User WORKS_FOR Acme",
            ),
            GraphEdge(
                edge_id="e2",
                scope_id=scope,
                source_node_id="n2",
                target_node_id="n1",
                edge_type="LIKES",
                fact="User LIKES Berlin",
            ),
        ],
    )

    # Without embeddings, the fused triplet recipe relies on keyword overlap; include "Acme" so edge search is possible.
    rr = await svc.build_retrieval_result(user_id=scope, query="Do I work for Acme?")
    assert rr is not None
    assert rr.edges
    assert rr.edges[0].attrs.get("fact") == "User WORKS_FOR Acme"


