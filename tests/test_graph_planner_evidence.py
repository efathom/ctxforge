import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.protocols.graph import GraphEdge, GraphEpisode
from ctxforge.protocols.llm import ChatMessage


class DummyTokenizer:
    name = "dummy"

    def count_tokens(self, text: str, model=None) -> int:  # noqa: ANN001
        return len((text or "").split())

    def count_message_tokens(self, messages: list[ChatMessage], model=None) -> int:  # noqa: ANN001
        return sum(self.count_tokens(m.content, model=model) for m in messages)


@pytest.mark.asyncio
async def test_global_mode_returns_edges_when_local_does_not():
    store = InMemoryGraphStore()
    await store.upsert_edges(
        "u",
        [
            GraphEdge(
                edge_id="e1",
                scope_id="u",
                source_node_id="n1",
                target_node_id="n2",
                edge_type="reports",
                fact="Revenue increased by 10% in Q4",
            )
        ],
    )

    base = DEFAULT_CONFIG.merge_with(
        {
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "methods": ["keyword"],  # no semantic, no bfs
                    "seed_k": 0,
                    "bfs_max_depth": 0,
                    "bfs_edges_per_node": 0,
                    "max_facts": 10,
                    "include_entities": False,
                    "evidence_enabled": False,
                }
            }
        }
    )

    # Force local planner path: should not return edge keyword hits.
    cfg_local = base.merge_with({"graph": {"retrieval": {"planner_mode": "local"}}})
    svc_local = GraphService(config=cfg_local, graph_store=store, tokenizer_provider=DummyTokenizer())
    out_local = await svc_local.build_section(user_id="u", query="revenue")
    assert out_local is None

    # Force global planner path: should return edge keyword hits.
    cfg_global = base.merge_with({"graph": {"retrieval": {"planner_mode": "global"}}})
    svc_global = GraphService(config=cfg_global, graph_store=store, tokenizer_provider=DummyTokenizer())
    out_global = await svc_global.build_section(user_id="u", query="revenue")
    assert out_global is not None
    assert "<FACTS>" in out_global
    assert "Revenue increased" in out_global


@pytest.mark.asyncio
async def test_evidence_enabled_appends_evidence_block_and_budgets():
    store = InMemoryGraphStore()
    await store.add_episodes(
        "u",
        [
            GraphEpisode(episode_id="ep1", scope_id="u", content="Alpha beta gamma revenue details here."),
            GraphEpisode(episode_id="ep2", scope_id="u", content="More revenue info with extra words " * 40),
        ],
    )
    await store.upsert_edges(
        "u",
        [
            GraphEdge(
                edge_id="e1",
                scope_id="u",
                source_node_id="n1",
                target_node_id="n2",
                edge_type="mentions",
                fact="Revenue is discussed",
            )
        ],
    )

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "graph": {
                "enabled": True,
                "retrieval": {
                    "enabled": True,
                    "planner_mode": "global",
                    "methods": ["keyword"],
                    "seed_k": 0,
                    "bfs_max_depth": 0,
                    "bfs_edges_per_node": 0,
                    "max_facts": 10,
                    "include_entities": False,
                    "evidence_enabled": True,
                    "max_evidence_items": 10,
                    "max_evidence_tokens": 20,  # force truncation
                }
            }
        }
    )

    svc = GraphService(config=cfg, graph_store=store, tokenizer_provider=DummyTokenizer())
    out = await svc.build_section(user_id="u", query="revenue")
    assert out is not None
    assert "<RETRIEVAL_PLAN>" in out
    assert "<EVIDENCE>" in out
    assert "EP:" in out


