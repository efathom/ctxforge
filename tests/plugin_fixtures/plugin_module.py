"""
Plugin module fixture.

Exposes register(registry) so EngineFactory can register into a specific registry instance.
"""

from ctxforge.protocols.retriever import IReranker, RetrievalResult


REGISTER_CALLED = False


class PluginTestReranker(IReranker):
    @property
    def name(self) -> str:
        return "plugin_test"

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return results[:top_k] if top_k else results


def register(registry) -> None:
    global REGISTER_CALLED
    REGISTER_CALLED = True
    registry.register_component("reranker", "plugin_test", PluginTestReranker)


