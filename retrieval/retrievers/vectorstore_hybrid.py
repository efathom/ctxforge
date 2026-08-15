"""
Vectorstore-backed hybrid memory retriever.

Uses vector store similarity + keyword overlap, combining the two scores.
"""

import re
from typing import List, Optional

from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.indexers.memory import MemoryIndexer
from ctxforge.retrieval.retrievers.base import BaseRetriever


def _keyword_match_score(query: str, content: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower()))
    content_words = set(re.findall(r"\w+", content.lower()))
    if not query_words:
        return 0.0
    matches = query_words & content_words
    return len(matches) / len(query_words)


class VectorStoreHybridRetriever(BaseRetriever, IRetriever):
    def __init__(
        self,
        memory_store: IMemoryStore,
        indexer: MemoryIndexer,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        super().__init__(memory_store)
        self._indexer = indexer
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight

    @property
    def name(self) -> str:
        return "hybrid_vectorstore"

    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        config = config or RetrievalConfig()

        candidates = await self._indexer.search(
            query=query,
            scope_id=user_id,
            limit=max(config.limit * 10, config.limit),
            min_score=0.0,  # we apply min_score after combining
        )
        if not candidates:
            return []

        results: List[RetrievalResult] = []
        for c in candidates:
            mem = await self._memory_store.get(c.item_id)
            if mem is None:
                continue
            keyword_score = _keyword_match_score(query, mem.content)
            combined = (
                self._semantic_weight * float(c.score)
                + self._keyword_weight * float(keyword_score)
            )
            combined = min(1.0, max(0.0, combined))
            if combined < config.min_score:
                continue
            results.append(
                RetrievalResult(
                    memory=mem,
                    score=combined,
                    retrieval_method="hybrid",
                    metadata={
                        "semantic_score": float(c.score),
                        "keyword_score": float(keyword_score),
                        "semantic_weight": self._semantic_weight,
                        "keyword_weight": self._keyword_weight,
                        **(c.metadata or {}),
                    },
                )
            )

        filtered_memories = self._apply_filters([r.memory for r in results], config)
        filtered_ids = {m.memory_id for m in filtered_memories}
        results = [r for r in results if r.memory.memory_id in filtered_ids]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: config.limit]


