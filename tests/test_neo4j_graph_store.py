import os
from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.config.base import Neo4jGraphStoreConfig
from ctxforge.graph.stores.neo4j import Neo4jGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphNode, GraphSearchFilters


def _neo4j_env_available() -> bool:
    return bool(os.getenv("NEO4J_URL") and os.getenv("NEO4J_USERNAME") and os.getenv("NEO4J_PASSWORD"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _neo4j_env_available(), reason="Neo4j env vars not set (NEO4J_URL/NEO4J_USERNAME/NEO4J_PASSWORD)")
async def test_neo4j_store_upsert_and_search_roundtrip():
    cfg = Neo4jGraphStoreConfig(
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE"),
        create_indexes=False,
    )
    store = Neo4jGraphStore(cfg)
    scope = "test_scope"
    try:
        n1 = GraphNode(node_id="n1", scope_id=scope, name="Alice", labels=["Person"])
        n2 = GraphNode(node_id="n2", scope_id=scope, name="Bob", labels=["Person"])
        await store.upsert_nodes(scope, [n1, n2])

        e = GraphEdge(
            edge_id="e1",
            scope_id=scope,
            source_node_id="n1",
            target_node_id="n2",
            edge_type="KNOWS",
            fact="Alice knows Bob",
        )
        await store.upsert_edges(scope, [e])

        out = await store.search(
            scope,
            "Alice knows",
            scope="edges",
            limit=5,
            filters=GraphSearchFilters(valid_only=True, edge_ids=["e1"]),
        )
        assert any(x.edge_id == "e1" for x in out.edges)

        expired_at = datetime.now(timezone.utc) - timedelta(days=1)
        expired = GraphEdge(
            edge_id="e2",
            scope_id=scope,
            source_node_id="n1",
            target_node_id="n2",
            edge_type="KNOWS",
            fact="Alice knows Bob (old)",
            invalid_at=expired_at,
        )
        await store.upsert_edges(scope, [expired])

        out_valid = await store.search(
            scope,
            "knows",
            scope="edges",
            limit=10,
            filters=GraphSearchFilters(valid_only=True, as_of=datetime.now(timezone.utc)),
        )
        assert any(x.edge_id == "e1" for x in out_valid.edges)
        assert all(x.edge_id != "e2" for x in out_valid.edges)
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not _neo4j_env_available(), reason="Neo4j env vars not set (NEO4J_URL/NEO4J_USERNAME/NEO4J_PASSWORD)")
async def test_neo4j_semantic_node_search_best_effort():
    cfg = Neo4jGraphStoreConfig(
        url=os.environ["NEO4J_URL"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE"),
        create_indexes=True,
        vector_dimensions=3,
        vector_index_name="ce_entity_name_embedding_test3",
    )
    store = Neo4jGraphStore(cfg)
    scope = "test_scope_vector"
    try:
        n1 = GraphNode(node_id="n1", scope_id=scope, name="Acme", labels=["Organization"], name_embedding=[1.0, 0.0, 0.0])
        n2 = GraphNode(node_id="n2", scope_id=scope, name="Berlin", labels=["Location"], name_embedding=[0.0, 1.0, 0.0])
        await store.upsert_nodes(scope, [n1, n2])

        try:
            out = await store.search_nodes_semantic(
                scope,
                query_vector=[0.9, 0.1, 0.0],
                limit=5,
                filters=GraphSearchFilters(node_labels=["Organization"]),
            )
        except Exception as e:
            # Neo4j installations without vector indexing support will raise here.
            pytest.skip(f"Neo4j vector search not available: {e}")

        assert any(n.node_id == "n1" for n in out)
    finally:
        await store.close()


