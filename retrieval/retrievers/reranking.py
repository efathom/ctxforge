"""
Reranking retriever wrapper.

Wraps a base IRetriever and applies an IReranker to its results.
"""

import logging
from typing import Any, List, Optional, Union

from ctxforge.core.memory import MemoryItem
from ctxforge.protocols.context import ContextRetrievalResult, IContextReranker
from ctxforge.protocols.retriever import IReranker, IRetriever, RetrievalConfig, RetrievalResult

logger = logging.getLogger(__name__)


class RerankingRetriever(IRetriever):
    def __init__(self, base: IRetriever, reranker: Union[IReranker, IContextReranker[MemoryItem]]):
        self._base = base
        self._reranker = reranker

    @property
    def name(self) -> str:
        return f"{self._base.name}+rerank:{self._reranker.name}"

    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        config = config or RetrievalConfig()
        results = await self._base.retrieve(query=query, user_id=user_id, config=config)
        if not results:
            return results
        try:
            # Prefer calling with `RetrievalResult` (memory) first for backward compatibility.
            # If the reranker is actually a context reranker, it may fail; we then retry with
            # `ContextRetrievalResult[MemoryItem]`.
            reranked_any: Any = await self._reranker.rerank(query=query, results=results, top_k=config.limit)
        except Exception:
            try:
                ctx_results = [
                    ContextRetrievalResult(
                        item=r.memory,
                        score=r.score,
                        retrieval_method=r.retrieval_method,
                        metadata=r.metadata or {},
                    )
                    for r in results
                ]
                reranked_any = await self._reranker.rerank(query=query, results=ctx_results, top_k=config.limit)
            except Exception as e:
                logger.warning(
                    f"Reranking failed (reranker={getattr(self._reranker, 'name', 'unknown')}): {type(e).__name__}: {e}"
                )
                return results[: config.limit]

        if not reranked_any:
            return []

        first = reranked_any[0]
        # Memory-style reranker output.
        if hasattr(first, "memory"):
            return reranked_any[: config.limit]

        # Context-style reranker output (ContextRetrievalResult[MemoryItem]).
        if hasattr(first, "item"):
            converted = [
                RetrievalResult(
                    memory=r.item,
                    score=r.score,
                    retrieval_method=r.retrieval_method,
                    metadata=getattr(r, "metadata", None) or {},
                )
                for r in reranked_any
            ]
            return converted[: config.limit]

        # Unknown output type: fail open.
        return results[: config.limit]

    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        return await self._base.retrieve_by_embedding(embedding, user_id, config)

    async def retrieve_related(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 5,
    ) -> List[RetrievalResult]:
        return await self._base.retrieve_related(memory_id, user_id, limit)


