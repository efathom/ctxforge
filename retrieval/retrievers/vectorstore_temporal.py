"""
Vectorstore-backed temporal memory retriever.

Uses vector store similarity as the semantic score, then applies recency weighting.
"""

import math
from datetime import datetime, timezone
from typing import List, Optional

from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.indexers.memory import MemoryIndexer
from ctxforge.retrieval.retrievers.base import BaseRetriever


def _compute_recency_weight(created_at: datetime, half_life_days: float = 7.0) -> float:
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400
    decay_rate = math.log(2) / half_life_days
    weight = math.exp(-decay_rate * age_days)
    return max(0.0, min(1.0, weight))


class VectorStoreTemporalRetriever(BaseRetriever, IRetriever):
    def __init__(
        self,
        memory_store: IMemoryStore,
        indexer: MemoryIndexer,
        recency_weight: float = 0.3,
        semantic_weight: float = 0.7,
        half_life_days: float = 7.0,
    ):
        super().__init__(memory_store)
        self._indexer = indexer
        self._recency_weight = recency_weight
        self._semantic_weight = semantic_weight
        self._half_life_days = half_life_days

    @property
    def name(self) -> str:
        return "temporal_vectorstore"

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
            min_score=0.0,
        )
        if not candidates:
            return []

        results: List[RetrievalResult] = []
        for c in candidates:
            mem = await self._memory_store.get(c.item_id)
            if mem is None:
                continue

            semantic = float(c.score)
            recency = _compute_recency_weight(mem.created_at, self._half_life_days)
            combined = self._semantic_weight * semantic + self._recency_weight * recency
            combined = min(1.0, max(0.0, combined))
            if combined < config.min_score:
                continue

            results.append(
                RetrievalResult(
                    memory=mem,
                    score=combined,
                    retrieval_method="temporal",
                    metadata={
                        "semantic_score": semantic,
                        "recency_score": recency,
                        "semantic_weight": self._semantic_weight,
                        "recency_weight": self._recency_weight,
                        "half_life_days": self._half_life_days,
                        **(c.metadata or {}),
                    },
                )
            )

        filtered_memories = self._apply_filters([r.memory for r in results], config)
        filtered_ids = {m.memory_id for m in filtered_memories}
        results = [r for r in results if r.memory.memory_id in filtered_ids]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: config.limit]


