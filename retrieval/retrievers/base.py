"""
Base retriever implementation.

Provides a base class with common functionality and a simple retriever.

Implements both IRetriever (memory-specific) and IContextRetriever[MemoryItem]
(generic context) interfaces for maximum flexibility.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.engine.registry import registry
from ctxforge.protocols.context import ContextRetrievalResult, IContextRetriever
from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.utils import apply_memory_filters


class BaseRetriever(ABC, IContextRetriever[MemoryItem]):
    """
    Base class for all retrievers.
    
    Provides common functionality:
    - Memory store access
    - Filtering utilities
    - Default retrieve_related implementation
    """
    
    def __init__(self, memory_store: IMemoryStore):
        """
        Initialize with a memory store.
        
        Args:
            memory_store: The memory store to search
        """
        self._memory_store = memory_store
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this retriever strategy."""
        ...
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories matching the query."""
        ...
    
    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve by embedding.
        
        Default implementation falls back to text search.
        Override in embedding-aware retrievers.
        """
        return await self.retrieve("", user_id, config)
    
    async def retrieve_related(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories related to a given memory.
        
        Default implementation uses the memory's content as query.
        Override for embedding-based similarity.
        """
        reference = await self._memory_store.get(memory_id)
        if reference is None:
            return []
        
        config = RetrievalConfig(limit=limit + 1)  # +1 to exclude self
        results = await self.retrieve(reference.content, user_id, config)
        
        # Exclude the reference memory itself
        return [r for r in results if r.memory.memory_id != memory_id][:limit]
    
    def _apply_filters(
        self,
        memories: List[MemoryItem],
        config: RetrievalConfig,
    ) -> List[MemoryItem]:
        """Apply configuration filters to memories."""
        return apply_memory_filters(memories, config)
    
    async def _get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """Get memories for a user from the store."""
        return await self._memory_store.get_by_user(
            user_id,
            limit=limit,
            include_inactive=include_inactive,
        )
    
    # =========================================================================
    # IContextRetriever[MemoryItem] Protocol Implementation
    # =========================================================================
    # These methods provide a generic interface compatible with the
    # IContextRetriever protocol. They delegate to the IRetriever methods.
    # 
    # Protocol Mapping:
    #   - scope_id → user_id
    #   - limit → config.limit
    #   - Returns List[MemoryItem] or List[ContextRetrievalResult[MemoryItem]]
    # =========================================================================
    
    async def retrieve_items(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[MemoryItem]:
        """
        Retrieve memory items using the generic IContextRetriever interface.
        
        This is the IContextRetriever.retrieve implementation, renamed to
        avoid signature conflict with IRetriever.retrieve.
        
        Args:
            query: The search query
            scope_id: The user_id (maps from scope_id in IContextRetriever)
            limit: Maximum number of items to return
            **kwargs: Additional keyword arguments (passed to RetrievalConfig)
            
        Returns:
            List of MemoryItem objects, ordered by relevance
        """
        config = RetrievalConfig(limit=limit, **{
            k: v for k, v in kwargs.items() 
            if k in RetrievalConfig.__annotations__
        })
        results = await self.retrieve(query, scope_id, config)
        return [r.memory for r in results]
    
    async def retrieve_with_scores(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[ContextRetrievalResult[MemoryItem]]:
        """
        Retrieve memory items with their relevance scores.
        
        Implements IContextRetriever.retrieve_with_scores for memories.
        
        Args:
            query: The search query
            scope_id: The user_id
            limit: Maximum number of results to return
            **kwargs: Additional keyword arguments
            
        Returns:
            List of ContextRetrievalResult containing items and scores
        """
        config = RetrievalConfig(limit=limit, **{
            k: v for k, v in kwargs.items() 
            if k in RetrievalConfig.__annotations__
        })
        results = await self.retrieve(query, scope_id, config)
        
        return [
            ContextRetrievalResult(
                item=r.memory,
                score=r.score,
                retrieval_method=r.retrieval_method,
                metadata=r.metadata or {},
            )
            for r in results
        ]


@registry.register_retriever("simple")
class SimpleRetriever(BaseRetriever, IRetriever):
    """
    Simple retriever that delegates to the memory store.
    
    This is the default retriever - it uses whatever search
    capability the memory store provides.
    
    Example:
        >>> retriever = SimpleRetriever(memory_store)
        >>> results = await retriever.retrieve("user preferences", "user_123")
    """
    
    @property
    def name(self) -> str:
        return "simple"
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories using the memory store's search.
        """
        config = config or RetrievalConfig()
        
        # Build memory query
        memory_query = MemoryQuery(
            user_id=user_id,
            query_text=query or None,
            types=list(config.memory_types) if config.memory_types else None,
            tags=list(config.tags) if config.tags else None,
            min_confidence=config.min_confidence,
            include_inactive=config.include_inactive,
            limit=config.limit,
        )
        
        # Get memories from store
        memories = await self._memory_store.search(memory_query)
        
        # Convert to RetrievalResult with position-based scoring
        results = []
        for i, memory in enumerate(memories):
            score = max(0.1, 1.0 - (i * 0.1))
            results.append(RetrievalResult(
                memory=memory,
                score=score,
                retrieval_method=self.name,
                metadata={"position": i},
            ))
        
        # Apply minimum score filter
        if config.min_score > 0:
            results = [r for r in results if r.score >= config.min_score]
        
        return results

