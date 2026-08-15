import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import ComponentRegistry


@pytest.mark.asyncio
async def test_engine_factory_wraps_memory_retriever_when_rerank_enabled():
    reg = ComponentRegistry()
    factory = EngineFactory(component_registry=reg)

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "llm": {"provider": "mock"},
            "retrieval": {
                "strategy": "keyword",
                "rerank_enabled": True,
                "reranker": "llm",
                "rerank_top_k": 5,
            },
        }
    )

    engine = await factory.build(cfg)
    try:
        assert engine._retriever is not None
        assert "rerank:llm" in engine._retriever.name
    finally:
        await engine.close()


