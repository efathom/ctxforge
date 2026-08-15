from __future__ import annotations

from ctxforge.engine.registry import ComponentRegistry
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction


class DemoMarkerMiddleware(BaseMiddleware):
    def __init__(self, key: str = "demo_plugin_loaded", value: str = "true", enabled: bool = True):
        super().__init__(enabled=enabled)
        self._key = key
        self._value = value

    @property
    def name(self) -> str:
        return "demo_marker"

    async def _do_process(self, context: MiddlewareContext, next: NextFunction) -> MiddlewareContext:
        context.set_metadata(self._key, self._value)
        return await next(context)


def register(registry: ComponentRegistry) -> None:
    registry.register_middleware("demo_marker")(DemoMarkerMiddleware)


