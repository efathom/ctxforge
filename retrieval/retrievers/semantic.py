"""
Semantic retriever implementation.

Uses embeddings for similarity-based retrieval.
"""

from typing import List, Optional

from ctxforge.engine.registry import registry
from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.retrievers.base import BaseRetriever
from ctxforge.retrieval.utils import EmbeddingFunc
from ctxforge.utils.math import cosine_similarity


@registry.register_retriever("semantic")
class SemanticRetriever(BaseRetriever, IRetriever):
    """
    Semantic retriever using embedding similarity.
    
    Computes embeddings for queries and compares them against
    stored memory embeddings using cosine similarity.
    
    Example:
        >>> async def embed(text: str) -> List[float]:
        ...     return await embedding_model.embed(text)
        >>> 
        >>> retriever = SemanticRetriever(memory_store, embed)
        >>> results = await retriever.retrieve("vegetarian food", "user_123")
    """
    
    def __init__(
        self,
        memory_store: IMemoryStore,
        embedding_func: EmbeddingFunc,
    ):
        """
        Initialize with memory store and embedding function.
        
        Args:
            memory_store: The memory store to search
            embedding_func: Async function to compute embeddings
        """
        super().__init__(memory_store)
        self._embed = embedding_func
    
    @property
    def name(self) -> str:
        return "semantic"
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories using semantic similarity.
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
        Retrieve memories using a pre-computed embedding.
        """
        config = config or RetrievalConfig()
        
        # Get all memories for the user
        memories = await self._get_user_memories(user_id, limit=100)
        memories = self._apply_filters(memories, config)
        
        # Score each memory by cosine similarity
        results = self._score_by_embedding(memories, embedding, config)
        
        return results[:config.limit]
    
    async def retrieve_related(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories related to a given memory using embeddings.
        """
        reference = await self._memory_store.get(memory_id)
        if reference is None:
            return []
        
        # Use embedding if available, otherwise embed content
        if reference.embedding:
            query_embedding = reference.embedding
        else:
            query_embedding = await self._embed(reference.content)
        
        config = RetrievalConfig(limit=limit + 1)
        results = await self.retrieve_by_embedding(query_embedding, user_id, config)
        
        # Exclude the reference memory
        return [r for r in results if r.memory.memory_id != memory_id][:limit]
    
    def _score_by_embedding(
        self,
        memories: list,
        embedding: List[float],
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Score memories by cosine similarity to the embedding."""
        results = []
        
        for memory in memories:
            if memory.embedding:
                score = cosine_similarity(embedding, memory.embedding)
                
                if score >= config.min_score:
                    results.append(RetrievalResult(
                        memory=memory,
                        score=score,
                        retrieval_method=self.name,
                        metadata={"embedding_dim": len(embedding)},
                    ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

