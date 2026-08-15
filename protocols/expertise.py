"""
Protocol interfaces for the Expertise system.

These protocols define the contracts that expertise components must implement,
enabling pluggable, duck-typed extensibility.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from ctxforge.core.expertise import (
    CompletedTurn,
    CurationPlan,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseUsageLog,
    ReflectionResult,
    TurnOutcome,
)


@runtime_checkable
class IExpertiseStore(Protocol):
    """
    Protocol for expertise storage.
    
    Follows the same patterns as ISessionStore and IMemoryStore,
    providing persistence for expertise knowledge bases.
    
    Implementations:
        - InMemoryExpertiseStore: For testing and development
        - PostgresExpertiseStore: Production persistence with full-text search
        - RedisExpertiseStore: Caching and fast access
    """
    
    async def save(self, expertise: Expertise) -> None:
        """
        Save or update an expertise knowledge base.
        
        Args:
            expertise: The expertise to save
        """
        ...
    
    async def load(self, expertise_id: str) -> Optional[Expertise]:
        """
        Load an expertise by ID.
        
        Args:
            expertise_id: Unique identifier of the expertise
            
        Returns:
            The expertise if found, None otherwise
        """
        ...
    
    async def delete(self, expertise_id: str) -> bool:
        """
        Delete an expertise knowledge base.
        
        Args:
            expertise_id: ID of the expertise to delete
            
        Returns:
            True if the expertise was deleted, False if not found
        """
        ...
    
    async def list_expertise(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Expertise]:
        """
        List expertise knowledge bases.
        
        Args:
            domain: Optional domain filter
            limit: Maximum number to return
            offset: Offset for pagination
            
        Returns:
            List of expertise matching criteria
        """
        ...
    
    async def add_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """
        Add an item to an expertise.
        
        Args:
            expertise_id: ID of the expertise
            item: Item to add
        """
        ...
    
    async def update_item(self, expertise_id: str, item: ExpertiseItem) -> None:
        """
        Update an existing item.
        
        Args:
            expertise_id: ID of the expertise
            item: Updated item (matched by item_id)
        """
        ...
    
    async def remove_item(self, expertise_id: str, item_id: str) -> bool:
        """
        Remove an item from an expertise.
        
        Args:
            expertise_id: ID of the expertise
            item_id: ID of the item to remove
            
        Returns:
            True if the item was removed, False if not found
        """
        ...
    
    async def get_item(
        self,
        expertise_id: str,
        item_id: str,
    ) -> Optional[ExpertiseItem]:
        """
        Get a single item by ID.
        
        Args:
            expertise_id: ID of the expertise
            item_id: ID of the item
            
        Returns:
            The item if found, None otherwise
        """
        ...
    
    async def get_items_by_section(
        self,
        expertise_id: str,
        section: ExpertiseSection,
    ) -> List[ExpertiseItem]:
        """
        Get all items in a section.
        
        Args:
            expertise_id: ID of the expertise
            section: Section to filter by
            
        Returns:
            List of items in the section
        """
        ...
    
    async def update_item_counts(
        self,
        expertise_id: str,
        item_id: str,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
    ) -> None:
        """
        Update helpful/harmful counts for an item.
        
        Args:
            expertise_id: ID of the expertise
            item_id: ID of the item
            helpful_delta: Amount to add to helpful count
            harmful_delta: Amount to add to harmful count
        """
        ...
    
    async def search_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """
        Search items by text content.
        
        Args:
            expertise_id: ID of the expertise
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching items
        """
        ...
    
    async def log_usage(self, log: ExpertiseUsageLog) -> None:
        """
        Log expertise usage in a turn.
        
        Args:
            log: Usage log entry
        """
        ...
    
    async def get_usage_stats(self, expertise_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for an expertise.
        
        Args:
            expertise_id: ID of the expertise
            
        Returns:
            Dictionary with usage statistics
        """
        ...


@runtime_checkable
class IExpertiseRetriever(Protocol):
    """
    Protocol for expertise retrieval.
    
    Follows the same patterns as IRetriever, providing semantic
    search capabilities for expertise items.
    
    Implementations:
        - ExpertiseRetriever: Semantic search using vector stores
        - HybridExpertiseRetriever: Combines semantic and keyword search
    """
    
    async def retrieve(
        self,
        query: str,
        expertise_id: str,
        limit: int = 10,
        sections: Optional[List[ExpertiseSection]] = None,
        min_effectiveness: float = 0.0,
    ) -> List[ExpertiseItem]:
        """
        Retrieve relevant expertise items for a query.
        
        Args:
            query: The query text
            expertise_id: ID of the expertise to search
            limit: Maximum items to return
            sections: Optional list of sections to filter by
            min_effectiveness: Minimum effectiveness score (0.0-1.0)
            
        Returns:
            List of relevant expertise items, ordered by relevance
        """
        ...

    async def retrieve_items(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[ExpertiseItem]:
        """
        Retrieve relevant expertise items using the generic IContextRetriever interface.

        Args:
            query: The query text
            scope_id: ID of the expertise to search (maps to expertise_id)
            limit: Maximum items to return
            **kwargs: Additional provider-specific arguments

        Returns:
            List of relevant expertise items, ordered by relevance
        """
        ...


# Note: ExpertiseRetrievalResult is defined in expertise/retriever.py
# to avoid circular imports. We use 'Any' here for the protocol definition.
@runtime_checkable
class IExpertiseReranker(Protocol):
    """
    Protocol for expertise result reranking.
    
    Follows the same pattern as IReranker from protocols/retriever.py,
    but operates on ExpertiseRetrievalResult instead of RetrievalResult.
    
    Implementations:
        - EffectivenessReranker: Reranks by helpful/harmful counts
        - CrossEncoderExpertiseReranker: Uses cross-encoder models
    """
    
    @property
    def name(self) -> str:
        """The name of this reranker."""
        ...
    
    async def rerank(
        self,
        query: str,
        results: List[Any],  # List[ExpertiseRetrievalResult]
        top_k: Optional[int] = None,
    ) -> List[Any]:  # List[ExpertiseRetrievalResult]
        """
        Rerank expertise retrieval results.
        
        Args:
            query: The original query
            results: The initial retrieval results (ExpertiseRetrievalResult)
            top_k: Optional limit on results to return
            
        Returns:
            Reranked results
        """
        ...


@runtime_checkable
class IReflector(Protocol):
    """
    Protocol for turn reflection.
    
    The reflector analyzes completed turns and provides feedback
    on which expertise items were helpful or harmful.
    
    Implementations:
        - ExpertiseReflector: LLM-based reflection
        - RuleBasedReflector: Rule-based reflection for testing
    """
    
    async def reflect(
        self,
        turn: CompletedTurn,
        items_used: List[ExpertiseItem],
        outcome: TurnOutcome,
    ) -> ReflectionResult:
        """
        Analyze a turn and provide feedback on expertise items.
        
        Args:
            turn: The completed conversation turn
            items_used: List of expertise items that were used
            outcome: The outcome of the turn
            
        Returns:
            ReflectionResult with item feedback and insights
        """
        ...


@runtime_checkable
class ICurator(Protocol):
    """
    Protocol for expertise curation.
    
    The curator evolves expertise based on reflection feedback,
    performing ADD, UPDATE, MERGE, and DELETE operations.
    
    Implementations:
        - ExpertiseCurator: LLM-based curation
        - RuleBasedCurator: Rule-based curation for testing
    """
    
    async def curate(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> Tuple[Expertise, CurationPlan]:
        """
        Generate and apply curation plan to evolve expertise.
        
        Args:
            expertise: The current expertise
            reflection: Recent reflection result
            usage_stats: Usage statistics
            
        Returns:
            Tuple of (updated expertise, curation plan applied)
        """
        ...

