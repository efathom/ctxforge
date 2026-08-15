import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphNode


@pytest.mark.asyncio
async def test_ppr_reranks_nodes_by_reset_bias_and_is_deterministic():
    cfg_uniform = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "local",
                    "methods": ["keyword", "bfs"],  # keep it simple
                    "bfs_max_depth": 3,  # ensure candidate subgraph includes the full chain
                    "include_entities": True,
                    "max_facts": 10,
                    "max_entities": 10,
                    "ppr": {"enabled": True, "reset_mode": "uniform", "damping": 0.85, "max_iters": 100, "tol": 1e-10},
                },
            },
        }
    )
    cfg_biased = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "local",
                    "methods": ["keyword", "bfs"],
                    "bfs_max_depth": 3,
                    "include_entities": True,
                    "max_facts": 10,
                    "max_entities": 10,
                    "ppr": {"enabled": True, "reset_mode": "node_scores", "damping": 0.85, "max_iters": 100, "tol": 1e-10},
                },
            },
        }
    )

    store = InMemoryGraphStore()
    svc_uniform = GraphService(config=cfg_uniform, graph_store=store, embedding_provider=None)
    svc_biased = GraphService(config=cfg_biased, graph_store=store, embedding_provider=None)

    scope = "u"
    # Build a chain A-B-C-D (B and C are central).
    nodes = [
        GraphNode(node_id="A", scope_id=scope, name="A", labels=["Entity"], attributes={"vf_score": 10.0}),
        GraphNode(node_id="B", scope_id=scope, name="B", labels=["Entity"], attributes={"vf_score": 1.0}),
        GraphNode(node_id="C", scope_id=scope, name="C", labels=["Entity"], attributes={"vf_score": 1.0}),
        GraphNode(node_id="D", scope_id=scope, name="D", labels=["Entity"], attributes={"vf_score": 1.0}),
    ]
    await store.upsert_nodes(scope, nodes)
    await store.upsert_edges(
        scope,
        [
            GraphEdge(edge_id="eAB", scope_id=scope, source_node_id="A", target_node_id="B", edge_type="REL", fact="A-B"),
            GraphEdge(edge_id="eBC", scope_id=scope, source_node_id="B", target_node_id="C", edge_type="REL", fact="B-C"),
            GraphEdge(edge_id="eCD", scope_id=scope, source_node_id="C", target_node_id="D", edge_type="REL", fact="C-D"),
        ],
    )

    rr_u1 = await svc_uniform.build_retrieval_result(user_id=scope, query="A")
    rr_u2 = await svc_uniform.build_retrieval_result(user_id=scope, query="A")
    assert rr_u1 is not None and rr_u2 is not None
    assert [n.node_id for n in rr_u1.nodes] == [n.node_id for n in rr_u2.nodes]

    rr_b = await svc_biased.build_retrieval_result(user_id=scope, query="A")
    assert rr_b is not None

    # Bias should improve A's rank relative to uniform reset.
    order_uniform = [n.node_id for n in rr_u1.nodes]
    order_biased = [n.node_id for n in rr_b.nodes]
    assert "A" in order_uniform and "A" in order_biased
    assert order_biased.index("A") <= order_uniform.index("A")

    # Sanity: endpoints should not dominate more than immediate neighbors in a chain.
    assert order_biased.index("B") < order_biased.index("D")


@pytest.mark.asyncio
async def test_ppr_interacts_with_token_budget_by_preserving_top_ranked_nodes():
    cfg = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "local",
                    "methods": ["keyword", "bfs"],
                    "include_entities": True,
                    "max_facts": 10,
                    "max_entities": 10,
                    "max_entity_tokens": 1,  # force truncation (one item best-effort)
                    "ppr": {"enabled": True},
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
            GraphNode(node_id="A", scope_id=scope, name="Alpha", labels=["Entity"], attributes={"vf_score": 10.0}),
            GraphNode(node_id="B", scope_id=scope, name="Beta", labels=["Entity"], attributes={"vf_score": 1.0}),
            GraphNode(node_id="C", scope_id=scope, name="Gamma", labels=["Entity"], attributes={"vf_score": 1.0}),
        ],
    )
    await store.upsert_edges(
        scope,
        [
            GraphEdge(edge_id="eAB", scope_id=scope, source_node_id="A", target_node_id="B", edge_type="REL", fact="A-B"),
            GraphEdge(edge_id="eBC", scope_id=scope, source_node_id="B", target_node_id="C", edge_type="REL", fact="B-C"),
        ],
    )

    rr = await svc.build_retrieval_result(user_id=scope, query="Alpha")
    assert rr is not None
    # Because budgeting runs after rerank, top node should survive truncation.
    assert rr.nodes
    assert rr.nodes[0].node_id == "A"


