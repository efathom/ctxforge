"""
Memory Vector Indexer.

Indexes memory items using existing vector store infrastructure,
enabling semantic search capabilities for memory retrieval.

Implements IContextIndexer protocol for generic context operations.
"""

import logging
from typing import Any, Dict, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.protocols.context import IContextIndexer, IndexSearchResult
from ctxforge.protocols.llm import IEmbeddingProvider
from ctxforge.vectorstores.protocol import (
    IVectorStore,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
)

logger = logging.getLogger(__name__)


class MemoryIndexer(IContextIndexer[MemoryItem]):
    """
    Indexes memory items using existing vector store infrastructure.
    
    Implements IContextIndexer[MemoryItem] protocol for generic context operations.
    Reuses IVectorStore implementations (ChromaDB, Pinecone, Weaviate)
    and IEmbeddingProvider for generating embeddings.
    
    Protocol Mapping:
        - scope_id → user_id
        - item → MemoryItem
    
    Example:
        >>> indexer = MemoryIndexer(vector_store, embedding_provider)
        >>> await indexer.index_item(memory, user_id="user_123")
        >>> results = await indexer.search("vegetarian preferences", "user_123", limit=5)
    """
    
    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_provider: IEmbeddingProvider,
        namespace_prefix: str = "memory",
    ):
        """
        Initialize the indexer.
        
        Args:
            vector_store: Vector store for indexing and searching
            embedding_provider: Provider for generating embeddings
            namespace_prefix: Prefix for namespacing memories in the vector store
        """
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._namespace_prefix = namespace_prefix
    
    def _get_namespace(self, user_id: str) -> str:
        """Generate namespace for a user's memories."""
        # Decision: always map memory namespace to user_id
        return user_id
    
    def _get_vector_id(self, memory_id: str, user_id: str) -> str:
        """Generate unique vector ID for a memory."""
        # IDs are unique within a namespace, so memory_id alone is sufficient.
        return memory_id
    
    def _build_metadata(self, item: MemoryItem) -> Dict[str, Any]:
        """Build metadata dict for vector store."""
        meta: Dict[str, Any] = {
            "memory_id": item.memory_id,
            "user_id": item.user_id,
            "type": item.type.value,
            "source": item.source.value,
            "confidence_score": item.confidence_score,
            "is_active": item.is_active,
            # Many vector stores only accept scalar metadata values
            "tags": ", ".join(item.tags) if item.tags else "",
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            # Multi-view indexing metadata
            "keywords": ", ".join(item.keywords) if item.keywords else "",
            "persons": ", ".join(item.persons) if item.persons else "",
            "locations": ", ".join(item.locations) if item.locations else "",
            "topics": ", ".join(item.topics) if item.topics else "",
        }
        if item.event_timestamp:
            meta["event_timestamp"] = item.event_timestamp.isoformat()
        # Include custom metadata fields
        for k, v in item.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        return meta
    
    def _build_indexable_content(self, item: MemoryItem) -> str:
        """Build content string for embedding.

        Prefers the disambiguated ``restatement`` over raw ``content``
        so that vector embeddings capture self-contained facts.
        """
        text = item.restatement if item.restatement else item.content
        parts = [text]
        
        # Include type for context
        parts.append(f"Type: {item.type.value}")
        
        # Include tags
        if item.tags:
            parts.append(f"Tags: {', '.join(item.tags)}")
        
        return " | ".join(parts)
    
    async def index_item(
        self,
        item: MemoryItem,
        scope_id: str,  # user_id
    ) -> None:
        """
        Index a single memory item.
        
        If the item already has an embedding, it will be used.
        Otherwise, a new embedding will be generated.
        
        Args:
            item: The memory item to index
            scope_id: The user_id (maps to scope_id in IContextIndexer)
        """
        # Use existing embedding or generate new one
        if item.embedding:
            embedding = item.embedding
        else:
            content = self._build_indexable_content(item)
            embedding = await self._embedding_provider.embed_single(content)
        
        vector_id = self._get_vector_id(item.memory_id, scope_id)
        namespace = self._get_namespace(scope_id)
        metadata = self._build_metadata(item)
        
        record = VectorRecord(
            id=vector_id,
            embedding=embedding,
            metadata=metadata,
            content=item.content,
        )
        
        await self._vector_store.upsert([record], namespace=namespace)
        
        logger.debug(f"Indexed memory {item.memory_id} in namespace {namespace}")
    
    async def index_all(
        self,
        items: List[MemoryItem],
        scope_id: str,
    ) -> int:
        """
        Index multiple memory items in batch.
        
        Args:
            items: List of memory items to index
            scope_id: The user_id
            
        Returns:
            Number of items successfully indexed
        """
        if not items:
            return 0
        
        namespace = self._get_namespace(scope_id)
        records = []
        
        for item in items:
            # Generate embedding if needed
            if item.embedding:
                embedding = item.embedding
            else:
                content = self._build_indexable_content(item)
                embedding = await self._embedding_provider.embed_single(content)
            
            vector_id = self._get_vector_id(item.memory_id, scope_id)
            metadata = self._build_metadata(item)
            
            records.append(VectorRecord(
                id=vector_id,
                embedding=embedding,
                metadata=metadata,
                content=item.content,
            ))
        
        await self._vector_store.upsert(records, namespace=namespace)
        
        logger.info(f"Indexed {len(records)} memories in namespace {namespace}")
        return len(records)
    
    async def remove_item(
        self,
        item_id: str,  # memory_id
        scope_id: str,  # user_id
    ) -> bool:
        """
        Remove a memory from the index.
        
        Args:
            item_id: The memory_id
            scope_id: The user_id
            
        Returns:
            True if removal was attempted (vector stores may not confirm)
        """
        vector_id = self._get_vector_id(item_id, scope_id)
        namespace = self._get_namespace(scope_id)
        
        await self._vector_store.delete(
            ids=[vector_id],
            namespace=namespace,
        )
        
        logger.debug(f"Removed memory {item_id} from namespace {namespace}")
        return True
    
    async def remove_all(self, scope_id: str) -> bool:
        """
        Remove all memories for a user from the index.
        
        Args:
            scope_id: The user_id
            
        Returns:
            True if removal was attempted
        """
        namespace = self._get_namespace(scope_id)
        result = await self._vector_store.delete_namespace(namespace)
        if result:
            logger.info(f"Removed all memories from namespace {namespace}")
        return result
    
    async def search(
        self,
        query: str,
        scope_id: str,  # user_id
        limit: int = 10,
        filters: Optional[List[QueryFilter]] = None,
        min_score: float = 0.0,
    ) -> List[IndexSearchResult]:
        """
        Search for memories by semantic similarity.
        
        Args:
            query: The search query
            scope_id: The user_id
            limit: Maximum number of results
            filters: Optional filters (e.g., by type, tags)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with memory IDs and scores
        """
        # Generate query embedding
        query_embedding = await self._embedding_provider.embed_single(query)
        
        return await self.search_by_embedding(
            embedding=query_embedding,
            scope_id=scope_id,
            limit=limit,
            filters=filters,
            min_score=min_score,
        )
    
    async def search_by_embedding(
        self,
        embedding: List[float],
        scope_id: str,
        limit: int = 10,
        filters: Optional[List[QueryFilter]] = None,
        min_score: float = 0.0,
    ) -> List[IndexSearchResult]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: The query embedding vector
            scope_id: The user_id
            limit: Maximum number of results
            filters: Optional filters
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with memory IDs and scores
        """
        namespace = self._get_namespace(scope_id)
        
        # Query vector store
        results = await self._vector_store.query(
            embedding=embedding,
            top_k=limit,
            namespace=namespace,
            filters=filters,
        )
        
        # Filter by minimum score and convert to IndexSearchResult
        search_results = []
        for r in results:
            if r.score >= min_score:
                # Extract memory_id from metadata or vector_id
                memory_id = r.metadata.get("memory_id", r.id.split(":")[-1])
                search_results.append(IndexSearchResult(
                    item_id=memory_id,
                    score=r.score,
                    metadata=r.metadata,
                ))
        
        logger.debug(
            f"Search in {namespace} returned {len(search_results)} results "
            f"(limit={limit}, min_score={min_score})"
        )
        
        return search_results
    
    async def search_raw(
        self,
        query: str,
        scope_id: str,
        limit: int = 10,
        filters: Optional[List[QueryFilter]] = None,
        min_score: float = 0.0,
    ) -> List[VectorQueryResult]:
        """
        Search and return raw vector store results.
        
        This provides access to the full VectorQueryResult for advanced use cases.
        
        Args:
            query: The search query
            scope_id: The user_id
            limit: Maximum number of results
            filters: Optional filters
            min_score: Minimum similarity score threshold
            
        Returns:
            List of raw VectorQueryResult objects
        """
        query_embedding = await self._embedding_provider.embed_single(query)
        namespace = self._get_namespace(scope_id)
        
        results = await self._vector_store.query(
            embedding=query_embedding,
            top_k=limit,
            namespace=namespace,
            filters=filters,
        )
        
        return [r for r in results if r.score >= min_score]

