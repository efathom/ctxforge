"""
Unified Cross-Store Retriever.

A single retrieval interface that searches across multiple knowledge stores
(expertise, memories, semantic models) and provides merged, ranked results.

This simplifies agent interaction by providing a single search_knowledge
tool instead of requiring the agent to know about different stores.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResultSource(str, Enum):
    """Source of a retrieved result."""
    EXPERTISE = "expertise"
    MEMORY = "memory"
    SEMANTIC_MODEL = "semantic_model"
    GRAPH = "graph"
    EVENTS_INTENT = "events_intent"
    EXTERNAL = "external"


class RetrievalResult(BaseModel):
    """
    A single result from unified retrieval.
    
    Provides a consistent structure regardless of the source store.
    """
    content: str
    score: float = Field(ge=0.0, le=1.0)
    source: ResultSource
    source_id: str = ""  # ID in the source store
    
    # Optional metadata
    knowledge_type: Optional[str] = None  # rule, pattern, gotcha, etc.
    section: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """Convert to string for LLM context."""
        prefix = ""
        if self.knowledge_type:
            type_icons = {
                "rule": "📋",
                "pattern": "📝",
                "gotcha": "⚠️",
                "procedure": "📌",
                "definition": "📖",
                "preference": "⭐",
            }
            icon = type_icons.get(self.knowledge_type, "💡")
            prefix = f"{icon} [{self.knowledge_type.upper()}] "
        
        return f"{prefix}{self.content}"


class RetrievalQuery(BaseModel):
    """
    A unified query for cross-store retrieval.
    """
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Filters
    sources: Optional[List[ResultSource]] = None  # None = search all
    knowledge_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    
    # Settings
    max_results: int = 10
    min_score: float = 0.0
    include_metadata: bool = True


class StoreAdapter(Protocol):
    """Protocol for adapting different stores to unified retrieval."""
    
    async def search(
        self, query: str, limit: int, **kwargs
    ) -> List[RetrievalResult]:
        """Search the store and return unified results."""
        ...


@dataclass
class RegisteredStore:
    """A registered store with its adapter and configuration."""
    name: str
    source: ResultSource
    adapter: StoreAdapter
    priority: int = 5  # 1-10, higher = searched first
    enabled: bool = True


class UnifiedRetriever:
    """
    Unified retrieval across multiple knowledge stores.
    
    Provides a single search interface that:
    1. Searches all registered stores in parallel
    2. Merges results with score normalization
    3. Applies cross-store ranking and filtering
    4. Returns a unified result set
    
    Example usage:
    ```python
    retriever = UnifiedRetriever()
    
    # Register stores
    retriever.register_store(
        name="expertise",
        source=ResultSource.EXPERTISE,
        adapter=ExpertiseStoreAdapter(expertise_store),
    )
    retriever.register_store(
        name="memories",
        source=ResultSource.MEMORY,
        adapter=MemoryStoreAdapter(memory_store),
    )
    
    # Search across all stores
    results = await retriever.search("How do I handle date formatting?")
    
    # Provide to agent as tool
    @tool
    async def search_knowledge(query: str) -> str:
        results = await retriever.search(query)
        return retriever.format_results(results)
    ```
    """
    
    def __init__(
        self,
        score_weights: Optional[Dict[ResultSource, float]] = None,
        merge_strategy: str = "interleave",  # interleave, source_first, score_only
    ):
        """
        Initialize the unified retriever.
        
        Args:
            score_weights: Weight multipliers for each source (default all 1.0)
            merge_strategy: How to merge results from different sources
        """
        self._stores: Dict[str, RegisteredStore] = {}
        self._weights = score_weights or {}
        self._merge_strategy = merge_strategy
    
    def register_store(
        self,
        name: str,
        source: ResultSource,
        adapter: StoreAdapter,
        priority: int = 5,
        enabled: bool = True,
    ) -> None:
        """
        Register a store for unified search.
        
        Args:
            name: Unique name for this store
            source: The source type
            adapter: Adapter implementing the StoreAdapter protocol
            priority: Search priority (higher = first)
            enabled: Whether this store is active
        """
        self._stores[name] = RegisteredStore(
            name=name,
            source=source,
            adapter=adapter,
            priority=priority,
            enabled=enabled,
        )
        logger.info(f"Registered store '{name}' with source {source.value}")
    
    def unregister_store(self, name: str) -> bool:
        """
        Unregister a store.
        
        Args:
            name: Name of the store to remove
            
        Returns:
            True if store was removed
        """
        if name in self._stores:
            del self._stores[name]
            return True
        return False
    
    def enable_store(self, name: str, enabled: bool = True) -> bool:
        """
        Enable or disable a store.
        
        Args:
            name: Name of the store
            enabled: Whether to enable or disable
            
        Returns:
            True if store was found and updated
        """
        if name in self._stores:
            self._stores[name].enabled = enabled
            return True
        return False
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[ResultSource]] = None,
        knowledge_types: Optional[List[str]] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Search across all registered stores.
        
        Args:
            query: The search query
            max_results: Maximum total results
            sources: Limit to specific sources (None = all)
            knowledge_types: Limit to specific knowledge types
            min_score: Minimum score threshold
            user_id: Optional user ID for personalized results
            **kwargs: Additional arguments passed to adapters
            
        Returns:
            List of merged and ranked RetrievalResults
        """
        # Filter to enabled stores and requested sources
        active_stores = [
            store for store in self._stores.values()
            if store.enabled and (sources is None or store.source in sources)
        ]
        
        if not active_stores:
            logger.warning("No active stores for search")
            return []
        
        # Search all stores in parallel
        search_tasks = []
        for store in active_stores:
            task = self._search_store(store, query, max_results, user_id, kwargs)
            search_tasks.append(task)
        
        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Collect all results
        all_results = []
        for store, results in zip(active_stores, results_lists, strict=False):
            if isinstance(results, Exception):
                logger.warning(f"Search failed for store {store.name}: {results}")
                continue
            all_results.extend(results)
        
        # Apply score weights
        for result in all_results:
            weight = self._weights.get(result.source, 1.0)
            result.score *= weight
        
        # Filter by knowledge type
        if knowledge_types:
            all_results = [
                r for r in all_results
                if r.knowledge_type is None or r.knowledge_type in knowledge_types
            ]
        
        # Filter by min score
        all_results = [r for r in all_results if r.score >= min_score]
        
        # Merge and rank
        merged = self._merge_results(all_results, max_results)
        
        return merged
    
    async def _search_store(
        self,
        store: RegisteredStore,
        query: str,
        limit: int,
        user_id: Optional[str],
        kwargs: Dict,
    ) -> List[RetrievalResult]:
        """Search a single store."""
        try:
            if user_id:
                results = await store.adapter.search(
                    query, limit=limit, user_id=user_id, **kwargs
                )
            else:
                results = await store.adapter.search(query, limit=limit, **kwargs)
            
            # Ensure source is set
            for result in results:
                if not result.source:
                    result.source = store.source
            
            return results
        except Exception as e:
            logger.error(f"Error searching store {store.name}: {e}")
            raise
    
    def _merge_results(
        self, results: List[RetrievalResult], max_results: int
    ) -> List[RetrievalResult]:
        """Merge results according to the merge strategy."""
        if self._merge_strategy == "score_only":
            # Pure score-based ranking
            sorted_results = sorted(results, key=lambda r: -r.score)
            return sorted_results[:max_results]
        
        elif self._merge_strategy == "source_first":
            # Group by source, then by score within each group
            by_source: Dict[ResultSource, List[RetrievalResult]] = {}
            for r in results:
                if r.source not in by_source:
                    by_source[r.source] = []
                by_source[r.source].append(r)
            
            # Sort each group by score
            for source_results in by_source.values():
                source_results.sort(key=lambda r: -r.score)
            
            # Flatten (source order determined by first appearance)
            merged = []
            for source_results in by_source.values():
                merged.extend(source_results)
            
            return merged[:max_results]
        
        else:  # interleave (default)
            # Interleave results from different sources
            by_source: Dict[ResultSource, List[RetrievalResult]] = {}
            for r in results:
                if r.source not in by_source:
                    by_source[r.source] = []
                by_source[r.source].append(r)
            
            # Sort each group by score
            for source_results in by_source.values():
                source_results.sort(key=lambda r: -r.score)
            
            # Interleave using round-robin
            merged = []
            source_queues = list(by_source.values())
            while len(merged) < max_results and source_queues:
                new_queues = []
                for queue in source_queues:
                    if queue:
                        merged.append(queue.pop(0))
                        if queue:
                            new_queues.append(queue)
                source_queues = new_queues
            
            return merged[:max_results]
    
    def format_results(
        self,
        results: List[RetrievalResult],
        include_source: bool = True,
        include_score: bool = False,
    ) -> str:
        """
        Format results as a string for LLM context.
        
        Args:
            results: Results to format
            include_source: Include source labels
            include_score: Include relevance scores
            
        Returns:
            Formatted string
        """
        if not results:
            return "No relevant knowledge found."
        
        lines = ["**Retrieved Knowledge:**\n"]
        
        for i, result in enumerate(results, 1):
            line_parts = [f"{i}."]
            
            if include_source:
                line_parts.append(f"[{result.source.value}]")
            
            line_parts.append(result.to_context_string())
            
            if include_score:
                line_parts.append(f"(score: {result.score:.2f})")
            
            lines.append(" ".join(line_parts))
        
        return "\n".join(lines)


# Convenience adapters for common store types


class SimpleSearchAdapter:
    """
    Simple adapter that wraps a search function.
    
    Example:
    ```python
    adapter = SimpleSearchAdapter(
        search_fn=my_search_function,
        result_mapper=lambda item: RetrievalResult(
            content=item.content,
            score=item.score,
            source=ResultSource.EXPERTISE,
        ),
    )
    ```
    """
    
    def __init__(
        self,
        search_fn: Callable,
        result_mapper: Callable[[Any], RetrievalResult],
        source: ResultSource = ResultSource.EXTERNAL,
    ):
        """
        Initialize the adapter.
        
        Args:
            search_fn: Async function that performs search
            result_mapper: Function to convert results to RetrievalResult
            source: Default source for results
        """
        self._search_fn = search_fn
        self._mapper = result_mapper
        self._source = source
    
    async def search(
        self, query: str, limit: int = 10, **kwargs
    ) -> List[RetrievalResult]:
        """Perform search and map results."""
        raw_results = await self._search_fn(query, limit=limit, **kwargs)
        
        mapped = []
        for item in raw_results:
            result = self._mapper(item)
            if not result.source:
                result.source = self._source
            mapped.append(result)
        
        return mapped
