"""
Temporal retriever implementation.

Applies recency weighting to retrieval results.
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


def compute_recency_weight(
    created_at: datetime,
    half_life_days: float = 7.0,
) -> float:
    """
    Compute a recency weight using exponential decay.
    
    Memories decay in importance over time with a configurable half-life.
    
    Args:
        created_at: When the memory was created
        half_life_days: Days until the weight is halved
        
    Returns:
        Weight between 0 and 1
    """
    now = datetime.now(timezone.utc)
    
    # Handle naive datetime
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    
    age_days = (now - created_at).total_seconds() / 86400
    
    # Exponential decay: weight = 0.5 ^ (age / half_life)
    decay_rate = math.log(2) / half_life_days
    weight = math.exp(-decay_rate * age_days)
    
    return max(0.0, min(1.0, weight))


@registry.register_retriever("temporal")
class TemporalRetriever(BaseRetriever, IRetriever):
    """
    Temporal retriever with recency weighting.
    
    Combines semantic similarity with recency scoring so that
    more recent memories are ranked higher.
    
    Example:
        >>> retriever = TemporalRetriever(
        ...     memory_store,
        ...     embedding_func,
        ...     recency_weight=0.3,
        ...     half_life_days=7.0,
        ... )
        >>> results = await retriever.retrieve("recent projects", "user_123")
    """
    
    def __init__(
        self,
        memory_store: IMemoryStore,
        embedding_func: EmbeddingFunc,
        recency_weight: float = 0.3,
        semantic_weight: float = 0.7,
        half_life_days: float = 7.0,
    ):
        """
        Initialize with memory store and weights.
        
        Args:
            memory_store: The memory store to search
            embedding_func: Async function to compute embeddings
            recency_weight: Weight for recency (0-1)
            semantic_weight: Weight for semantic similarity (0-1)
            half_life_days: Days until recency weight halves
        """
        super().__init__(memory_store)
        self._embed = embedding_func
        self._recency_weight = recency_weight
        self._semantic_weight = semantic_weight
        self._half_life_days = half_life_days
    
    @property
    def name(self) -> str:
        return "temporal"
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve with recency-weighted scoring.
        """
        config = config or RetrievalConfig()
        query_embedding = await self._embed(query)
        return await self.retrieve_by_embedding(query_embedding, user_id, config)
    
    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve by embedding with recency weighting.
        """
        config = config or RetrievalConfig()
        
        memories = await self._get_user_memories(user_id, limit=100)
        memories = self._apply_filters(memories, config)
        
        results = []
        for memory in memories:
            # Compute semantic score
            semantic_score = 0.0
            if memory.embedding:
                semantic_score = cosine_similarity(embedding, memory.embedding)
            
            # Compute recency score
            recency_score = compute_recency_weight(
                memory.created_at,
                self._half_life_days,
            )
            
            # Combine scores
            combined_score = (
                self._semantic_weight * semantic_score +
                self._recency_weight * recency_score
            )
            combined_score = min(1.0, max(0.0, combined_score))
            
            if combined_score >= config.min_score:
                results.append(RetrievalResult(
                    memory=memory,
                    score=combined_score,
                    retrieval_method=self.name,
                    metadata={
                        "semantic_score": semantic_score,
                        "recency_score": recency_score,
                        "semantic_weight": self._semantic_weight,
                        "recency_weight": self._recency_weight,
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
        """Retrieve memories related to a given memory."""
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

