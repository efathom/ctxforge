import importlib

import pytest

from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import ComponentRegistry


@pytest.mark.asyncio
async def test_plugins_modules_register_function_is_called_and_registers_component():
    # Fresh registry so we can assert exactly what was registered.
    reg = ComponentRegistry()

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "plugins": {
                "modules": ["ctxforge.tests.plugin_fixtures.plugin_module"],
                "registrations": [],
            }
        }
    )

    # Ensure module flag starts false.
    mod = importlib.import_module("ctxforge.tests.plugin_fixtures.plugin_module")
    mod.REGISTER_CALLED = False

    factory = EngineFactory(component_registry=reg)
    engine = await factory.build(cfg)
    await engine.close()

    assert mod.REGISTER_CALLED is True
    assert reg.get_reranker("plugin_test") is not None


@pytest.mark.asyncio
async def test_plugins_classpath_registration_registers_component():
    reg = ComponentRegistry()

    cfg = DEFAULT_CONFIG.merge_with(
        {
            "plugins": {
                "modules": [],
                "registrations": [
                    {
                        "component_type": "reranker",
                        "name": "classpath_test",
                        "class_path": "ctxforge.tests.plugin_fixtures.classpath_components:ClassPathReranker",
                    }
                ],
            }
        }
    )

    factory = EngineFactory(component_registry=reg)
    engine = await factory.build(cfg)
    await engine.close()

    cls = reg.get_reranker("classpath_test")
    assert cls is not None
    assert cls.__name__ == "ClassPathReranker"


