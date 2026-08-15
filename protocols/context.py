"""
Context Item Protocol Interfaces.

Defines the contracts for context items (memories, expertise items, and future types)
that can be stored, retrieved, and used to build context for LLM interactions.

This provides a unified abstraction layer over different item types while
respecting their different scoping and lifecycle semantics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)


@runtime_checkable
class IContextItem(Protocol):
    """
    Protocol for any item that can be part of context.
    
    This defines the minimal interface that all context items must support,
    enabling generic retrieval, indexing, and context assembly.
    
    Implementations:
        - MemoryItem: User-scoped long-term memories
        - ExpertiseItem: Collection-scoped knowledge items
        - (Future) DocumentItem, ToolItem, etc.
    
    Note: Implementations may have additional fields specific to their domain.
    The protocol ensures a common interface for generic operations.
    """
    
    @property
    def item_id(self) -> str:
        """
        Unique identifier for this item.
        
        For MemoryItem, this maps to memory_id.
        For ExpertiseItem, this is item_id.
        """
        ...
    
    @property
    def content(self) -> str:
        """The textual content of this item."""
        ...
    
    @property
    def is_active(self) -> bool:
        """Whether the item is currently active (not soft-deleted)."""
        ...
    
    @property
    def embedding(self) -> Optional[List[float]]:
        """Optional vector embedding for semantic search."""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Additional metadata associated with the item."""
        ...
    
    @property
    def created_at(self) -> datetime:
        """When the item was created."""
        ...
    
    @property
    def updated_at(self) -> datetime:
        """When the item was last updated."""
        ...
    
    def to_prompt_format(self) -> str:
        """
        Convert the item to a format suitable for LLM prompt inclusion.
        
        Returns:
            A string representation optimized for prompt context.
        """
        ...


# Type variable for context items
T = TypeVar('T', bound=IContextItem)


@dataclass
class ContextRetrievalResult(Generic[T]):
    """
    Generic result from a context retrieval operation.
    
    Works with any item type that implements IContextItem.
    This is the generic equivalent of RetrievalResult (for memories)
    and ExpertiseRetrievalResult (for expertise items).
    """
    
    item: T
    score: float  # Relevance score (0.0 to 1.0)
    retrieval_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate score is in valid range."""
        if not 0.0 <= self.score <= 1.0:
            # Clamp to valid range
            self.score = max(0.0, min(1.0, self.score))


@runtime_checkable
class IContextRetriever(Protocol[T]):
    """
    Protocol for generic context item retrieval.
    
    This provides a unified interface for retrieval across different
    item types, while allowing type-specific implementations.
    
    The scope_id parameter represents different things for different item types:
        - For memories: user_id
        - For expertise: expertise_id
        - For documents: collection_id
    
    Method Naming:
        - retrieve_items: Returns just the items (no scores)
        - retrieve_with_scores: Returns items with relevance scores
        
    The method names are chosen to avoid conflicts with type-specific
    retriever interfaces (e.g., IRetriever.retrieve for memories).
    
    Implementations:
        - BaseRetriever (for MemoryItem via retrieve_items/retrieve_with_scores)
        - ExpertiseRetriever (for ExpertiseItem)
        - (Future) DocumentRetriever, etc.
    """
    
    @property
    def name(self) -> str:
        """The name of this retriever."""
        ...
    
    async def retrieve_items(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
    ) -> List[T]:
        """
        Retrieve relevant items for a query within a scope.
        
        Args:
            query: The search query
            scope_id: The scope to search within (user_id, expertise_id, etc.)
            limit: Maximum number of items to return
            
        Returns:
            List of items, ordered by relevance
        """
        ...
    
    async def retrieve_with_scores(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
    ) -> List[ContextRetrievalResult[T]]:
        """
        Retrieve items with their relevance scores.
        
        Args:
            query: The search query
            scope_id: The scope to search within
            limit: Maximum number of results to return
            
        Returns:
            List of retrieval results with scores
        """
        ...


@runtime_checkable
class IContextReranker(Protocol[T]):
    """
    Protocol for reranking generic context retrieval results.

    This is the unified reranker interface for any `IContextItem` type
    (e.g., MemoryItem, ExpertiseItem). It operates on `ContextRetrievalResult[T]`
    to avoid duplicated result wrappers per domain.
    """

    @property
    def name(self) -> str:
        """The name of this reranker."""
        ...

    async def rerank(
        self,
        query: str,
        results: List[ContextRetrievalResult[T]],
        top_k: Optional[int] = None,
    ) -> List[ContextRetrievalResult[T]]:
        """Rerank retrieval results."""
        ...


@dataclass
class IndexSearchResult:
    """
    Result from an index search operation.
    
    Contains the item ID and score, but not the full item.
    The caller can use the ID to load the full item from a store.
    """
    
    item_id: str
    score: float  # Similarity score (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IContextIndexer(Protocol[T]):
    """
    Protocol for indexing context items for semantic search.
    
    Provides a unified interface for indexing any type of context item
    into vector stores.
    
    Note: The indexer deals with embeddings and IDs. To get full items,
    use IContextRetriever which combines indexer + store.
    
    Implementations:
        - ExpertiseIndexer
        - (could add) MemoryIndexer
    """
    
    async def index_item(self, item: T, scope_id: str) -> None:
        """
        Index a single item.
        
        Args:
            item: The item to index
            scope_id: The scope this item belongs to
        """
        ...
    
    async def remove_item(self, item_id: str, scope_id: str) -> bool:
        """
        Remove an item from the index.
        
        Args:
            item_id: ID of the item to remove
            scope_id: The scope this item belongs to
            
        Returns:
            True if the item was removed
        """
        ...
    
    async def search(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
    ) -> List[IndexSearchResult]:
        """
        Search for items by semantic similarity.
        
        Args:
            query: The search query
            scope_id: The scope to search within
            limit: Maximum number of results
            
        Returns:
            List of search results with item IDs and scores
        """
        ...


@runtime_checkable
class IContextStore(Protocol[T]):
    """
    Protocol for storing context items.
    
    Provides a unified interface for CRUD operations on context items.
    Different implementations may have different scoping semantics.
    
    Implementations:
        - IMemoryStore (user-scoped)
        - IExpertiseStore (collection-scoped)
    """
    
    async def get(self, item_id: str, scope_id: str) -> Optional[T]:
        """
        Get an item by ID within a scope.
        
        Args:
            item_id: The item ID
            scope_id: The scope to search within
            
        Returns:
            The item if found, None otherwise
        """
        ...
    
    async def save(self, item: T, scope_id: str) -> None:
        """
        Save or update an item.
        
        Args:
            item: The item to save
            scope_id: The scope this item belongs to
        """
        ...
    
    async def delete(self, item_id: str, scope_id: str) -> bool:
        """
        Delete an item.
        
        Args:
            item_id: The item ID
            scope_id: The scope this item belongs to
            
        Returns:
            True if the item was deleted
        """
        ...
    
    async def search(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
    ) -> List[T]:
        """
        Search for items.
        
        Args:
            query: Search query
            scope_id: The scope to search within
            limit: Maximum results
            
        Returns:
            List of matching items
        """
        ...

