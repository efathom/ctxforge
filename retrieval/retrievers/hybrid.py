"""
Hybrid retriever implementation.

Combines semantic and keyword-based retrieval for better results.
"""

import re
from typing import List, Optional

from ctxforge.engine.registry import registry
from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.retrievers.base import BaseRetriever
from ctxforge.retrieval.utils import EmbeddingFunc
from ctxforge.utils.math import cosine_similarity


def keyword_match_score(query: str, content: str) -> float:
    """
    Compute a simple keyword match score.
    
    Uses case-insensitive word overlap.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    content_words = set(re.findall(r'\w+', content.lower()))
    
    if not query_words:
        return 0.0
    
    # Score is percentage of query words found
    matches = query_words & content_words
    return len(matches) / len(query_words)


@registry.register_retriever("hybrid")
class HybridRetriever(BaseRetriever, IRetriever):
    """
    Hybrid retriever combining semantic and keyword search.
    
    Uses a weighted combination of:
    - Semantic similarity (embedding-based)
    - Keyword matching (word overlap)
    
    This often provides better results than either approach alone.
    
    Example:
        >>> retriever = HybridRetriever(
        ...     memory_store,
        ...     embedding_func,
        ...     semantic_weight=0.7,
        ...     keyword_weight=0.3,
        ... )
        >>> results = await retriever.retrieve("vegetarian recipes", "user_123")
    """
    
    def __init__(
        self,
        memory_store: IMemoryStore,
        embedding_func: EmbeddingFunc,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Initialize with memory store and weights.
        
        Args:
            memory_store: The memory store to search
            embedding_func: Async function to compute embeddings
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
        """
        super().__init__(memory_store)
        self._embed = embedding_func
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid semantic + keyword matching.
        """
        config = config or RetrievalConfig()
        
        # Compute query embedding
        query_embedding = await self._embed(query)
        
        # Get and filter memories
        memories = await self._get_user_memories(user_id, limit=100)
        memories = self._apply_filters(memories, config)
        
        # Score each memory with combined score
        results = []
        for memory in memories:
            keyword_score = keyword_match_score(query, memory.content)
            
            semantic_score = 0.0
            if memory.embedding:
                semantic_score = cosine_similarity(query_embedding, memory.embedding)
            
            # Combine scores
            combined_score = (
                self._semantic_weight * semantic_score +
                self._keyword_weight * keyword_score
            )
            combined_score = min(1.0, max(0.0, combined_score))
            
            if combined_score >= config.min_score:
                results.append(RetrievalResult(
                    memory=memory,
                    score=combined_score,
                    retrieval_method=self.name,
                    metadata={
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score,
                        "semantic_weight": self._semantic_weight,
                        "keyword_weight": self._keyword_weight,
                    },
                ))
        
        # Sort by combined score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:config.limit]
    
    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve by embedding (semantic only, no keyword matching).
        """
        config = config or RetrievalConfig()
        
        memories = await self._get_user_memories(user_id, limit=100)
        memories = self._apply_filters(memories, config)
        
        results = []
        for memory in memories:
            if memory.embedding:
                score = cosine_similarity(embedding, memory.embedding)
                
                if score >= config.min_score:
                    results.append(RetrievalResult(
                        memory=memory,
                        score=score,
                        retrieval_method="semantic",
                        metadata={},
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
        
        # Use content for hybrid search
        config = RetrievalConfig(limit=limit + 1)
        results = await self.retrieve(reference.content, user_id, config)
        
        return [r for r in results if r.memory.memory_id != memory_id][:limit]

