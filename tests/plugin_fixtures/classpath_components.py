from ctxforge.protocols.retriever import IReranker, RetrievalResult


class ClassPathReranker(IReranker):
    @property
    def name(self) -> str:
        return "classpath_test"

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return list(reversed(results))[:top_k] if top_k else list(reversed(results))


