"""
Vectorstore-backed semantic memory retriever.

Uses a vector store index (via MemoryIndexer) for candidate selection,
while IMemoryStore remains the source-of-truth for MemoryItem payloads.
"""

from typing import List, Optional

from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.indexers.memory import MemoryIndexer
from ctxforge.retrieval.retrievers.base import BaseRetriever


class VectorStoreSemanticRetriever(BaseRetriever, IRetriever):
    def __init__(self, memory_store: IMemoryStore, indexer: MemoryIndexer):
        super().__init__(memory_store)
        self._indexer = indexer

    @property
    def name(self) -> str:
        return "semantic_vectorstore"

    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        config = config or RetrievalConfig()

        # Pull candidates from vector store
        candidates = await self._indexer.search(
            query=query,
            scope_id=user_id,
            limit=max(config.limit * 5, config.limit),
            min_score=config.min_score,
        )
        if not candidates:
            return []

        # Hydrate MemoryItems from IMemoryStore
        results: List[RetrievalResult] = []
        for c in candidates:
            mem = await self._memory_store.get(c.item_id)
            if mem is None:
                continue
            results.append(
                RetrievalResult(
                    memory=mem,
                    score=c.score,
                    retrieval_method="semantic",
                    metadata=dict(c.metadata or {}),
                )
            )

        # Apply standard filters and limit
        filtered_memories = self._apply_filters([r.memory for r in results], config)
        filtered_ids = {m.memory_id for m in filtered_memories}
        results = [r for r in results if r.memory.memory_id in filtered_ids]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: config.limit]


