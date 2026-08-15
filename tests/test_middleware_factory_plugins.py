import pytest

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import ComponentRegistry


@pytest.mark.asyncio
async def test_middleware_factory_plugin_runs_in_prepare_pipeline():
    cfg = TESTING_CONFIG.merge_with(
        {
            "plugins": {"modules": ["ctxforge.tests.plugin_fixtures.middleware_factory_plugin_module"]},
            "pipelines": {
                "prepare": {
                    "chain": [
                        {
                            "type": "suffix_input",
                            "enabled": True,
                            "priority": 999,
                            "phases": ["prepare_input"],
                            "config": {"suffix": " [from-factory]"},
                        }
                    ]
                }
            },
        }
    )

    engine = await EngineFactory(component_registry=ComponentRegistry()).build(cfg)
    ctx = await engine.prepare_context(session_id="s1", user_id="u1", user_input="Hello")
    assert ctx.current_query.endswith(" [from-factory]")

