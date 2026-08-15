import pytest

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import ComponentRegistry
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore


@pytest.mark.asyncio
async def test_record_turn_pipeline_metadata_reaches_record_pipeline():
    reg = ComponentRegistry()

    seen = {"value": None}

    @reg.register_middleware("echo_meta")
    class EchoMetaMiddleware:
        @property
        def name(self) -> str:
            return "echo_meta"

        async def process(self, context, next):
            # record_input_output phase: metadata should be present
            if getattr(context, "phase", None) == "record_input_output":
                seen["value"] = context.get_metadata("foo")
                # also modify processed_input so we can assert via stored event
                context.processed_input = (context.processed_input or "") + f" [foo={seen['value']}]"
            return await next(context)

    cfg = TESTING_CONFIG.merge_with(
        {
            "pipelines": {
                "record": {
                    "chain": [
                        {
                            "type": "echo_meta",
                            "enabled": True,
                            "priority": 999,
                            "phases": ["record_input_output"],
                            "config": {},
                        }
                    ]
                }
            }
        }
    )

    factory = EngineFactory(component_registry=reg)
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    engine = await factory.create(config=cfg, session_store=session_store, memory_store=memory_store)

    await engine.record_turn(
        session_id="s1",
        user_id="u1",
        user_input="hello",
        assistant_response="ok",
        pipeline_metadata={"foo": "bar"},
    )

    sess = await engine.get_session("s1", "u1")
    assert seen["value"] == "bar"
    assert sess.events[-2].content.endswith("[foo=bar]")  # user event


