"""
Salience-aware retriever implementation.

Blends similarity, reinforcement (access count), and recency into a single
salience score for memory ranking.
"""

import math
from datetime import datetime, timezone
from typing import List, Optional

from ctxforge.engine.registry import registry
from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.retrievers.base import BaseRetriever
from ctxforge.retrieval.utils import EmbeddingFunc
from ctxforge.utils.math import cosine_similarity


def compute_salience_score(
    similarity: float,
    access_count: int,
    accessed_at: Optional[datetime],
    half_life_days: float = 30.0,
) -> float:
    """Compute a salience score blending similarity, reinforcement, and recency.

    Formula:
        salience = similarity * log(access_count + 1) * recency_factor

    When *access_count* is 0 the reinforcement factor is ``log(1) = 0`` and
    the overall salience is 0.  This is intentional cold-start behaviour:
    a memory must be accessed at least once to gain salience.

    Args:
        similarity: Cosine similarity between query and memory (0-1).
        access_count: How many times the memory has been accessed.
        accessed_at: Last access time (naive assumed UTC).  ``None`` → 1.0.
        half_life_days: Days for recency to halve.

    Returns:
        Non-negative salience score.
    """
    reinforcement_factor = math.log(access_count + 1)

    if accessed_at is None:
        recency_factor = 1.0
    else:
        now = datetime.now(timezone.utc)
        if accessed_at.tzinfo is None:
            accessed_at = accessed_at.replace(tzinfo=timezone.utc)
        days_ago = max(0.0, (now - accessed_at).total_seconds() / 86400)
        recency_factor = math.exp(-0.693 * days_ago / half_life_days)

    return similarity * reinforcement_factor * recency_factor


@registry.register_retriever("salience")
class SalienceRetriever(BaseRetriever, IRetriever):
    """Retriever that ranks memories by a combined salience score.

    Example:
        >>> retriever = SalienceRetriever(
        ...     memory_store, embedding_func, half_life_days=30.0
        ... )
        >>> results = await retriever.retrieve("recent projects", "user_123")
    """

    def __init__(
        self,
        memory_store: IMemoryStore,
        embedding_func: EmbeddingFunc,
        half_life_days: float = 30.0,
    ):
        super().__init__(memory_store)
        self._embed = embedding_func
        self._half_life_days = half_life_days

    @property
    def name(self) -> str:
        return "salience"

    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        config = config or RetrievalConfig()
        query_embedding = await self._embed(query)
        return await self.retrieve_by_embedding(query_embedding, user_id, config)

    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        config = config or RetrievalConfig()

        memories = await self._get_user_memories(user_id, limit=100)
        memories = self._apply_filters(memories, config)

        results: List[RetrievalResult] = []
        for memory in memories:
            similarity = 0.0
            if memory.embedding:
                similarity = cosine_similarity(embedding, memory.embedding)

            score = compute_salience_score(
                similarity=similarity,
                access_count=memory.access_count,
                accessed_at=memory.accessed_at,
                half_life_days=self._half_life_days,
            )

            if score >= config.min_score:
                results.append(RetrievalResult(
                    memory=memory,
                    score=score,
                    retrieval_method=self.name,
                    metadata={
                        "similarity": similarity,
                        "access_count": memory.access_count,
                        "half_life_days": self._half_life_days,
                    },
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:config.limit]

    async def retrieve_related(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 5,
    ) -> List[RetrievalResult]:
        reference = await self._memory_store.get(memory_id)
        if reference is None:
            return []

        if reference.embedding:
            query_embedding = reference.embedding
        else:
            query_embedding = await self._embed(reference.content)

        config = RetrievalConfig(limit=limit + 1)
        results = await self.retrieve_by_embedding(query_embedding, user_id, config)

        return [r for r in results if r.memory.memory_id != memory_id][:limit]
