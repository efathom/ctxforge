"""
Expertise Retriever.

Retrieves relevant expertise items using semantic search and reranking,
following patterns from the existing retrieval infrastructure.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ctxforge.core.expertise import (
    ExpertiseItem,
    ExpertiseSection,
)
from ctxforge.protocols.context import (
    ContextRetrievalResult,
    IContextReranker,
    IContextRetriever,
)
from ctxforge.protocols.expertise import IExpertiseRetriever, IExpertiseStore
from ctxforge.retrieval.indexers.expertise import ExpertiseIndexer
from ctxforge.vectorstores.protocol import QueryFilter

logger = logging.getLogger(__name__)


class ExpertiseRetrievalResult(ContextRetrievalResult[ExpertiseItem]):
    """
    Concrete retrieval result type for expertise.

    This is a thin wrapper over `ContextRetrievalResult[ExpertiseItem]` so callers
    can use `isinstance(..., ExpertiseRetrievalResult)` without hitting
    "subscripted generics cannot be used with class and instance checks".
    """

    pass


@dataclass
class ExpertiseRetrievalConfig:
    """Configuration for expertise retrieval."""
    
    limit: int = 10
    min_score: float = 0.0
    sections: Optional[List[ExpertiseSection]] = None
    min_effectiveness: float = 0.0
    include_inactive: bool = False
    rerank: bool = False
    boost_high_performing: bool = True


class ExpertiseRetriever(IExpertiseRetriever, IContextRetriever[ExpertiseItem]):
    """
    Retrieves relevant expertise items.
    
    Follows BaseRetriever patterns and reuses existing rerankers.
    Combines semantic search with effectiveness-based scoring.
    
    This class implements both:
    - IExpertiseRetriever: Domain-specific expertise retrieval protocol
    - IContextRetriever[ExpertiseItem]: Generic context retrieval protocol
      (where expertise_id serves as scope_id)
    
    Example:
        >>> retriever = ExpertiseRetriever(expertise_store, indexer)
        >>> items = await retriever.retrieve("error handling", expertise_id)
    """
    
    def __init__(
        self,
        expertise_store: IExpertiseStore,
        indexer: ExpertiseIndexer,
        reranker: Optional[IContextReranker[ExpertiseItem]] = None,
        effectiveness_weight: float = 0.2,
    ):
        """
        Initialize the retriever.
        
        Args:
            expertise_store: Store for loading expertise data
            indexer: Indexer for semantic search
            reranker: Optional reranker for result reordering
            effectiveness_weight: Weight for effectiveness score in final ranking (0-1)
        """
        self._store = expertise_store
        self._indexer = indexer
        self._reranker = reranker
        self._effectiveness_weight = effectiveness_weight
    
    @property
    def name(self) -> str:
        """The name of this retriever."""
        return "expertise_retriever"
    
    async def retrieve_items(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[ExpertiseItem]:
        """
        Retrieve expertise items (IContextRetriever protocol method).
        
        This is the generic interface. For more options, use retrieve() directly.
        
        Args:
            query: The query text
            scope_id: ID of the expertise to search (maps to expertise_id)
            limit: Maximum items to return
            **kwargs: Additional arguments (sections, min_effectiveness)
            
        Returns:
            List of relevant expertise items, ordered by relevance
        """
        return await self.retrieve(
            query=query,
            expertise_id=scope_id,
            limit=limit,
            sections=kwargs.get('sections'),
            min_effectiveness=kwargs.get('min_effectiveness', 0.0),
        )
    
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
        
        This is the full-featured method. For the generic interface,
        use retrieve_items() which conforms to IContextRetriever.
        
        Args:
            query: The query text
            expertise_id: ID of the expertise to search
            limit: Maximum items to return
            sections: Optional list of sections to filter by
            min_effectiveness: Minimum effectiveness score (0.0-1.0)
            
        Returns:
            List of relevant expertise items, ordered by relevance
        """
        config = ExpertiseRetrievalConfig(
            limit=limit,
            sections=sections,
            min_effectiveness=min_effectiveness,
        )
        
        results = await self.retrieve_detailed(query, expertise_id, config)
        return [r.item for r in results]
    
    async def retrieve_with_scores(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
    ) -> List[ContextRetrievalResult[ExpertiseItem]]:
        """
        Retrieve items with scores (IContextRetriever protocol method).
        
        This is the protocol-conforming signature. For more options,
        use retrieve_detailed() instead.
        
        Args:
            query: The query text
            scope_id: ID of the expertise to search (expertise_id)
            limit: Maximum items to return
            
        Returns:
            List of ContextRetrievalResult with scores
        """
        config = ExpertiseRetrievalConfig(limit=limit)
        results = await self.retrieve_detailed(query, scope_id, config)
        
        # Convert to generic ContextRetrievalResult
        return [
            ContextRetrievalResult(
                item=r.item,
                score=r.score,
                retrieval_method=r.retrieval_method,
                metadata=r.metadata,
            )
            for r in results
        ]
    
    async def retrieve_detailed(
        self,
        query: str,
        expertise_id: str,
        config: Optional[ExpertiseRetrievalConfig] = None,
    ) -> List[ExpertiseRetrievalResult]:
        """
        Retrieve expertise items with detailed configuration.
        
        Args:
            query: The query text
            expertise_id: ID of the expertise to search
            config: Retrieval configuration
            
        Returns:
            List of retrieval results with scores
        """
        config = config or ExpertiseRetrievalConfig()
        
        # Build filters
        filters: List[QueryFilter] = []
        if config.sections:
            section_values = [s.value for s in config.sections]
            filters.append(QueryFilter(field="section", operator="in", value=section_values))
        
        if config.min_effectiveness > 0:
            filters.append(QueryFilter(
                field="effectiveness_score",
                operator="gte",
                value=config.min_effectiveness,
            ))
        
        if not config.include_inactive:
            filters.append(QueryFilter(field="is_active", operator="eq", value=True))
        
        # Search using indexer
        vector_results = await self._indexer.search_raw(
            query=query,
            expertise_id=expertise_id,
            limit=config.limit * 2 if config.rerank else config.limit,
            filters=filters if filters else None,
            min_score=config.min_score,
        )
        
        if not vector_results:
            return []
        
        # Load full item data from store
        expertise = await self._store.load(expertise_id)
        if not expertise:
            logger.warning(f"Expertise {expertise_id} not found")
            return []
        
        # Convert to retrieval results
        results: List[ExpertiseRetrievalResult] = []
        for vr in vector_results:
            item = expertise.get_item(vr.id)
            if item:
                # Calculate combined score
                semantic_score = vr.score
                
                if config.boost_high_performing:
                    # Boost score based on effectiveness
                    effectiveness_boost = item.effectiveness_score * self._effectiveness_weight
                    combined_score = (
                        semantic_score * (1 - self._effectiveness_weight) +
                        effectiveness_boost
                    )
                else:
                    combined_score = semantic_score
                
                results.append(
                    ExpertiseRetrievalResult(
                        item=item,
                        score=combined_score,
                        retrieval_method="semantic",
                        metadata={
                            "semantic_score": semantic_score,
                            "effectiveness_score": item.effectiveness_score,
                            "section": item.section.value,
                        },
                    )
                )
        
        # Apply reranking if configured
        if config.rerank and self._reranker:
            results = await self._apply_reranking(query, results)
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:config.limit]
    
    async def retrieve_by_section(
        self,
        query: str,
        expertise_id: str,
        section: ExpertiseSection,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """
        Retrieve items from a specific section.
        
        Args:
            query: The query text
            expertise_id: ID of the expertise
            section: Section to search
            limit: Maximum items to return
            
        Returns:
            List of relevant items from the section
        """
        return await self.retrieve(
            query=query,
            expertise_id=expertise_id,
            limit=limit,
            sections=[section],
        )
    
    async def retrieve_high_performing(
        self,
        query: str,
        expertise_id: str,
        limit: int = 10,
    ) -> List[ExpertiseItem]:
        """
        Retrieve only high-performing items.
        
        Args:
            query: The query text
            expertise_id: ID of the expertise
            limit: Maximum items to return
            
        Returns:
            List of high-performing items relevant to the query
        """
        config = ExpertiseRetrievalConfig(
            limit=limit * 2,  # Get more to filter
            min_effectiveness=0.7,  # High effectiveness threshold
            boost_high_performing=True,
        )
        
        results = await self.retrieve_detailed(query, expertise_id, config)
        
        # Further filter for high-performing items
        high_performing = [
            r for r in results
            if r.item.is_high_performing
        ]
        
        return [r.item for r in high_performing[:limit]]
    
    async def retrieve_related(
        self,
        item_id: str,
        expertise_id: str,
        limit: int = 5,
    ) -> List[ExpertiseItem]:
        """
        Retrieve items related to a given item.
        
        Args:
            item_id: ID of the reference item
            expertise_id: ID of the expertise
            limit: Maximum items to return
            
        Returns:
            List of related items
        """
        # Get similar items from indexer
        similar = await self._indexer.get_similar_items(
            item_id=item_id,
            expertise_id=expertise_id,
            limit=limit,
        )
        
        if not similar:
            return []
        
        # Load full item data
        expertise = await self._store.load(expertise_id)
        if not expertise:
            return []
        
        items = []
        for vr in similar:
            item = expertise.get_item(vr.id)
            if item:
                items.append(item)
        
        return items
    
    async def _apply_reranking(
        self,
        query: str,
        results: List[ExpertiseRetrievalResult],
    ) -> List[ExpertiseRetrievalResult]:
        """
        Apply reranking to results using the unified IContextReranker interface.
        """
        try:
            return await self._reranker.rerank(query=query, results=results)
        except Exception as e:
            logger.warning(
                f"Expertise reranking failed (reranker={getattr(self._reranker, 'name', 'unknown')}): "
                f"{type(e).__name__}: {e}"
            )
            return results


class HybridExpertiseRetriever(IExpertiseRetriever):
    """
    Hybrid retriever combining semantic search with keyword matching.
    
    Uses both vector search and text search from the expertise store,
    merging results with configurable weights.
    
    Example:
        >>> retriever = HybridExpertiseRetriever(
        ...     expertise_store, indexer, semantic_weight=0.7
        ... )
        >>> items = await retriever.retrieve("calculate discount", expertise_id)
    """
    
    def __init__(
        self,
        expertise_store: IExpertiseStore,
        indexer: ExpertiseIndexer,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            expertise_store: Store for expertise data and keyword search
            indexer: Indexer for semantic search
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
        """
        self._store = expertise_store
        self._indexer = indexer
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
    
    @property
    def name(self) -> str:
        """The name of this retriever."""
        return "hybrid_expertise_retriever"
    
    async def retrieve(
        self,
        query: str,
        expertise_id: str,
        limit: int = 10,
        sections: Optional[List[ExpertiseSection]] = None,
        min_effectiveness: float = 0.0,
    ) -> List[ExpertiseItem]:
        """
        Retrieve using hybrid search.
        
        Args:
            query: The query text
            expertise_id: ID of the expertise to search
            limit: Maximum items to return
            sections: Optional list of sections to filter by
            min_effectiveness: Minimum effectiveness score
            
        Returns:
            List of relevant expertise items
        """
        # Get semantic results
        filters: List[QueryFilter] = []
        if sections:
            section_values = [s.value for s in sections]
            filters.append(QueryFilter(field="section", operator="in", value=section_values))
        
        semantic_results = await self._indexer.search_raw(
            query=query,
            expertise_id=expertise_id,
            limit=limit * 2,
            filters=filters if filters else None,
        )
        
        # Get keyword results from store
        keyword_results = await self._store.search_items(
            expertise_id=expertise_id,
            query=query,
            limit=limit * 2,
        )
        
        # Merge results
        item_scores: Dict[str, float] = {}
        item_map: Dict[str, ExpertiseItem] = {}
        
        # Process semantic results
        for vr in semantic_results:
            score = vr.score * self._semantic_weight
            item_scores[vr.id] = item_scores.get(vr.id, 0.0) + score
        
        # Process keyword results (assign position-based scores)
        for i, item in enumerate(keyword_results):
            position_score = max(0.1, 1.0 - (i * 0.05))
            score = position_score * self._keyword_weight
            item_scores[item.item_id] = item_scores.get(item.item_id, 0.0) + score
            item_map[item.item_id] = item
        
        # Load expertise for items from semantic search
        expertise = await self._store.load(expertise_id)
        if expertise:
            for item in expertise.items:
                if item.item_id in item_scores and item.item_id not in item_map:
                    item_map[item.item_id] = item
        
        # Sort and filter
        sorted_ids = sorted(
            item_scores.keys(),
            key=lambda x: item_scores[x],
            reverse=True,
        )
        
        results: List[ExpertiseItem] = []
        for item_id in sorted_ids:
            item = item_map.get(item_id)
            if item and item.is_active:
                if min_effectiveness <= 0 or item.effectiveness_score >= min_effectiveness:
                    results.append(item)
                    if len(results) >= limit:
                        break
        
        return results

