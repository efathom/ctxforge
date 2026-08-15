"""
Plugin fixture: middleware factory registration.
"""

from __future__ import annotations

from typing import Any, Optional

from ctxforge.engine.deps import EngineDeps
from ctxforge.engine.registry import ComponentRegistry
from ctxforge.middleware.protocol import IMiddleware, MiddlewareContext, NextFunction


class SuffixInputMiddleware(IMiddleware):
    def __init__(self, suffix: str):
        self._suffix = suffix

    @property
    def name(self) -> str:
        return "suffix_input"

    async def process(self, context: MiddlewareContext, next: NextFunction) -> MiddlewareContext:
        context.processed_input = f"{context.processed_input}{self._suffix}"
        return await next(context)


class SuffixInputMiddlewareFactory:
    def create(self, *, config: dict[str, Any], deps: EngineDeps) -> Optional[IMiddleware]:
        # Demonstrate deps access: if user disabled plugins or config missing, skip.
        suffix = str(config.get("suffix", ""))
        if not suffix:
            return None
        return SuffixInputMiddleware(suffix=suffix)


def register(registry: ComponentRegistry) -> None:
    registry.register_middleware_factory("suffix_input")(SuffixInputMiddlewareFactory())

