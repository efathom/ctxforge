import pytest

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import ComponentRegistry
from ctxforge.tests.plugin_fixtures import lifecycle_plugin_module


@pytest.mark.asyncio
async def test_factory_initializes_and_engine_closes_owned_resources():
    lifecycle_plugin_module.INITIALIZED = False
    lifecycle_plugin_module.CLOSED = False

    cfg = TESTING_CONFIG.merge_with(
        {
            "storage": {"session": {"backend": "redis"}},
            "plugins": {"modules": ["ctxforge.tests.plugin_fixtures.lifecycle_plugin_module"]},
        }
    )

    factory = EngineFactory(component_registry=ComponentRegistry())
    engine = await factory.build(cfg)

    assert lifecycle_plugin_module.INITIALIZED is True

    await engine.close()
    assert lifecycle_plugin_module.CLOSED is True

