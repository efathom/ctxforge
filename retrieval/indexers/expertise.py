"""
Expertise Vector Indexer.

Indexes expertise items using existing vector store infrastructure,
enabling semantic search capabilities for knowledge retrieval.

Implements IContextIndexer protocol for generic context operations.
"""

import logging
from typing import List, Optional

from ctxforge.core.expertise import Expertise, ExpertiseItem, ExpertiseSection
from ctxforge.protocols.context import IContextIndexer, IndexSearchResult
from ctxforge.protocols.llm import IEmbeddingProvider
from ctxforge.vectorstores.protocol import (
    IVectorStore,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
)

logger = logging.getLogger(__name__)


class ExpertiseIndexer(IContextIndexer[ExpertiseItem]):
    """
    Indexes expertise items using existing vector store infrastructure.
    
    Implements IContextIndexer[ExpertiseItem] protocol for generic context operations.
    Reuses IVectorStore implementations (ChromaDB, Pinecone, Weaviate)
    and IEmbeddingProvider for generating embeddings.
    
    Protocol Mapping:
        - scope_id → expertise_id
        - item → ExpertiseItem
    
    Example:
        >>> indexer = ExpertiseIndexer(vector_store, embedding_provider)
        >>> await indexer.index_all(expertise)
        >>> results = await indexer.search("how to handle errors", expertise_id, limit=5)
    """
    
    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_provider: IEmbeddingProvider,
        namespace_prefix: str = "expertise",
    ):
        """
        Initialize the indexer.
        
        Args:
            vector_store: Vector store for indexing and searching
            embedding_provider: Provider for generating embeddings
            namespace_prefix: Prefix for namespacing expertise in the vector store
        """
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._namespace_prefix = namespace_prefix
    
    def _get_namespace(self, expertise_id: str) -> str:
        """Generate namespace for an expertise."""
        # Use underscore separator for compatibility with ChromaDB collection naming
        return f"{self._namespace_prefix}_{expertise_id}"
    
    async def index_item(
        self,
        item: ExpertiseItem,
        expertise_id: str,
    ) -> None:
        """
        Index a single expertise item.
        
        Args:
            item: The expertise item to index
            expertise_id: ID of the expertise this item belongs to
        """
        # Generate embedding if not already present
        if item.embedding:
            embedding = item.embedding
        else:
            embedding = await self._embedding_provider.embed_single(item.content)
            item.embedding = embedding
        
        # Create vector record with metadata
        record = VectorRecord(
            id=item.item_id,
            embedding=embedding,
            content=item.content,
            metadata={
                "expertise_id": expertise_id,
                "section": item.section.value,
                "helpful_count": item.helpful_count,
                "harmful_count": item.harmful_count,
                "effectiveness_score": item.effectiveness_score,
                "is_active": item.is_active,
                "source": item.source or "",
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            },
        )
        
        namespace = self._get_namespace(expertise_id)
        await self._vector_store.upsert([record], namespace=namespace)
        
        logger.debug(f"Indexed expertise item {item.item_id} in namespace {namespace}")
    
    async def index_all(
        self,
        expertise: Expertise,
        only_active: bool = True,
        batch_size: int = 100,
    ) -> int:
        """
        Index all items in an expertise.
        
        Args:
            expertise: The expertise to index
            only_active: Whether to only index active items
            batch_size: Number of items to process per batch
            
        Returns:
            Number of items indexed
        """
        items = expertise.active_items if only_active else expertise.items
        
        if not items:
            logger.info(f"No items to index for expertise {expertise.expertise_id}")
            return 0
        
        # Generate embeddings in batches
        records: List[VectorRecord] = []
        namespace = self._get_namespace(expertise.expertise_id)
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Get items that need embeddings
            items_needing_embeddings = [item for item in batch if not item.embedding]
            if items_needing_embeddings:
                contents = [item.content for item in items_needing_embeddings]
                response = await self._embedding_provider.embed(contents)
                embeddings = response.embeddings
                
                for item, embedding in zip(items_needing_embeddings, embeddings, strict=False):
                    item.embedding = embedding
            
            # Create records for all items in batch
            for item in batch:
                record = VectorRecord(
                    id=item.item_id,
                    embedding=item.embedding,
                    content=item.content,
                    metadata={
                        "expertise_id": expertise.expertise_id,
                        "section": item.section.value,
                        "helpful_count": item.helpful_count,
                        "harmful_count": item.harmful_count,
                        "effectiveness_score": item.effectiveness_score,
                        "is_active": item.is_active,
                        "source": item.source or "",
                        "created_at": item.created_at.isoformat(),
                        "updated_at": item.updated_at.isoformat(),
                    },
                )
                records.append(record)
            
            # Upsert batch
            if records:
                await self._vector_store.upsert(records, namespace=namespace)
                records = []
        
        logger.info(
            f"Indexed {len(items)} items for expertise {expertise.expertise_id}"
        )
        return len(items)
    
    async def remove_item(
        self,
        item_id: str,
        expertise_id: str,
    ) -> bool:
        """
        Remove an item from the index.
        
        Args:
            item_id: ID of the item to remove
            expertise_id: ID of the expertise
            
        Returns:
            True if the item was removed
        """
        namespace = self._get_namespace(expertise_id)
        count = await self._vector_store.delete([item_id], namespace=namespace)
        
        if count > 0:
            logger.debug(f"Removed item {item_id} from index")
            return True
        return False
    
    async def remove_all(self, expertise_id: str) -> bool:
        """
        Remove all items for an expertise from the index.
        
        Args:
            expertise_id: ID of the expertise
            
        Returns:
            True if the namespace was deleted
        """
        namespace = self._get_namespace(expertise_id)
        result = await self._vector_store.delete_namespace(namespace)
        
        if result:
            logger.info(f"Removed all items for expertise {expertise_id}")
        return result
    
    async def search(
        self,
        query: str,
        expertise_id: str,
        limit: int = 10,
        filters: Optional[List[QueryFilter]] = None,
        min_score: float = 0.0,
    ) -> List[IndexSearchResult]:
        """
        Search for expertise items by semantic similarity.
        
        Conforms to IContextIndexer protocol (expertise_id serves as scope_id).
        
        Args:
            query: The search query
            expertise_id: ID of the expertise to search (scope_id in protocol)
            limit: Maximum number of results
            filters: Optional metadata filters
            min_score: Minimum similarity score
            
        Returns:
            List of IndexSearchResult with item IDs and scores
        """
        raw_results = await self.search_raw(
            query=query,
            expertise_id=expertise_id,
            limit=limit,
            filters=filters,
            min_score=min_score,
        )
        
        # Convert VectorQueryResult to IndexSearchResult
        return [
            IndexSearchResult(
                item_id=r.id,
                score=r.score,
                metadata=r.metadata or {},
            )
            for r in raw_results
        ]
    
    async def search_raw(
        self,
        query: str,
        expertise_id: str,
        limit: int = 10,
        filters: Optional[List[QueryFilter]] = None,
        min_score: float = 0.0,
    ) -> List[VectorQueryResult]:
        """
        Search with raw VectorQueryResult output.
        
        Lower-level method that returns full vector store results
        including embeddings and content if available.
        
        Args:
            query: The search query
            expertise_id: ID of the expertise to search
            limit: Maximum number of results
            filters: Optional metadata filters
            min_score: Minimum similarity score
            
        Returns:
            List of VectorQueryResult ordered by similarity
        """
        # Generate query embedding
        query_embedding = await self._embedding_provider.embed_single(query)
        
        namespace = self._get_namespace(expertise_id)
        
        # Add filter for active items by default
        all_filters = list(filters or [])
        all_filters.append(QueryFilter(field="is_active", operator="eq", value=True))
        
        results = await self._vector_store.query(
            embedding=query_embedding,
            top_k=limit,
            namespace=namespace,
            filters=all_filters,
            include_metadata=True,
        )
        
        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        
        return results
    
    async def search_by_embedding(
        self,
        embedding: List[float],
        expertise_id: str,
        limit: int = 10,
        filters: Optional[List[QueryFilter]] = None,
        min_score: float = 0.0,
    ) -> List[VectorQueryResult]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: The query embedding vector
            expertise_id: ID of the expertise to search
            limit: Maximum number of results
            filters: Optional metadata filters
            min_score: Minimum similarity score
            
        Returns:
            List of vector query results ordered by similarity
        """
        namespace = self._get_namespace(expertise_id)
        
        # Add filter for active items by default
        all_filters = list(filters or [])
        all_filters.append(QueryFilter(field="is_active", operator="eq", value=True))
        
        results = await self._vector_store.query(
            embedding=embedding,
            top_k=limit,
            namespace=namespace,
            filters=all_filters,
            include_metadata=True,
        )
        
        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        
        return results
    
    async def search_by_section(
        self,
        query: str,
        expertise_id: str,
        sections: List[ExpertiseSection],
        limit: int = 10,
    ) -> List[VectorQueryResult]:
        """
        Search within specific sections.
        
        Args:
            query: The search query
            expertise_id: ID of the expertise to search
            sections: Sections to filter by
            limit: Maximum number of results
            
        Returns:
            List of vector query results
        """
        section_values = [s.value for s in sections]
        filters = [QueryFilter(field="section", operator="in", value=section_values)]
        
        return await self.search_raw(query, expertise_id, limit, filters)
    
    async def get_similar_items(
        self,
        item_id: str,
        expertise_id: str,
        limit: int = 5,
    ) -> List[VectorQueryResult]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: ID of the reference item
            expertise_id: ID of the expertise
            limit: Maximum number of results
            
        Returns:
            List of similar items (excluding the reference)
        """
        namespace = self._get_namespace(expertise_id)
        
        # Query by ID
        results = await self._vector_store.query_by_id(
            vector_id=item_id,
            top_k=limit + 1,  # +1 to account for self
            namespace=namespace,
            filters=[QueryFilter(field="is_active", operator="eq", value=True)],
        )
        
        # Remove self from results
        return [r for r in results if r.id != item_id][:limit]
    
    async def count(self, expertise_id: str) -> int:
        """
        Count indexed items for an expertise.
        
        Args:
            expertise_id: ID of the expertise
            
        Returns:
            Number of indexed items
        """
        namespace = self._get_namespace(expertise_id)
        return await self._vector_store.count(namespace=namespace)
    
    async def update_item_metadata(
        self,
        item: ExpertiseItem,
        expertise_id: str,
    ) -> None:
        """
        Update metadata for an indexed item (e.g., after count updates).
        
        This re-indexes the item with updated metadata.
        
        Args:
            item: The updated expertise item
            expertise_id: ID of the expertise
        """
        await self.index_item(item, expertise_id)

