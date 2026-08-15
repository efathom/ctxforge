"""
Retriever Protocol Interface.

Defines the contract for memory retrieval strategies.
Supports various RAG approaches including semantic search,
hybrid search, and reranking.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ctxforge.core.memory import MemoryItem, MemoryType


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    
    memory: MemoryItem
    score: float  # Relevance score (0.0 to 1.0)
    retrieval_method: str  # e.g., "semantic", "keyword", "hybrid"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    
    limit: int = 5
    min_score: float = 0.0
    memory_types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    min_confidence: float = 0.0
    include_inactive: bool = False
    rerank: bool = False
    rerank_model: Optional[str] = None
    metadata_filters: Optional[Dict[str, Any]] = None


@runtime_checkable
class IRetriever(Protocol):
    """
    Protocol for memory retrieval.
    
    Implementations provide different retrieval strategies:
    - Semantic search (embedding similarity)
    - Keyword/BM25 search
    - Hybrid search (semantic + keyword)
    - Temporal weighting
    - Cross-encoder reranking
    
    Example implementations:
    - SemanticRetriever: Pure embedding-based search
    - HybridRetriever: Combined semantic + BM25
    - TemporalRetriever: Recency-weighted search
    - RerankingRetriever: Two-stage retrieval with reranking
    """
    
    @property
    def name(self) -> str:
        """The name of this retriever strategy."""
        ...
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: The search query
            user_id: The user to search memories for
            config: Optional retrieval configuration
            
        Returns:
            List of retrieval results, ordered by relevance
        """
        ...
    
    async def retrieve_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories using a pre-computed embedding.
        
        Useful when the query embedding is already available.
        
        Args:
            embedding: The query embedding vector
            user_id: The user to search memories for
            config: Optional retrieval configuration
            
        Returns:
            List of retrieval results, ordered by relevance
        """
        ...
    
    async def retrieve_related(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories related to a given memory.
        
        Useful for exploring memory connections.
        
        Args:
            memory_id: The reference memory ID
            user_id: The user to search memories for
            limit: Maximum number of results
            
        Returns:
            List of related memories
        """
        ...


@runtime_checkable
class IReranker(Protocol):
    """
    Protocol for result reranking.
    
    Rerankers take an initial set of retrieval results and
    reorder them using a more expensive but accurate scoring method.
    
    Example implementations:
    - CrossEncoderReranker: Uses cross-encoder models
    - LLMReranker: Uses LLM for relevance scoring
    - RecencyReranker: Boosts recent memories
    """
    
    @property
    def name(self) -> str:
        """The name of this reranker."""
        ...
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results.
        
        Args:
            query: The original query
            results: The initial retrieval results
            top_k: Optional limit on results to return
            
        Returns:
            Reranked results
        """
        ...

