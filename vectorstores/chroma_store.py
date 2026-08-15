# mypy: disable-error-code="attr-defined"
"""
ChromaDB Vector Store Implementation.

Provides integration with ChromaDB, an open-source embedding database
that can run locally or connect to a cloud service.

Features:
- Local persistence or in-memory operation
- Collection-based organization (similar to namespaces)
- Metadata filtering with rich query language
- Optional embedding function integration
"""

import asyncio
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import Field

from ctxforge.core.exceptions import (
    ConfigurationError,
    ContextEngineError,
)
from ctxforge.vectorstores.protocol import (
    DistanceMetric,
    IVectorStore,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
    VectorStoreConfig,
)


class VectorStoreError(ContextEngineError):
    """Error during vector store operations."""
    pass


class ChromaConfig(VectorStoreConfig):
    """Configuration for ChromaDB vector store."""
    
    collection_name: str = Field(
        default="memories",
        description="Name of the ChromaDB collection"
    )
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory for persistent storage (None for in-memory)"
    )
    host: Optional[str] = Field(
        default=None,
        description="ChromaDB server host (for client mode)"
    )
    port: Optional[int] = Field(
        default=8000,
        description="ChromaDB server port"
    )
    ssl: bool = Field(
        default=False,
        description="Use SSL for connection"
    )
    tenant: str = Field(
        default="default_tenant",
        description="Tenant name for multi-tenancy"
    )
    database: str = Field(
        default="default_database",
        description="Database name"
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token for cloud"
    )
    create_collection_if_missing: bool = Field(
        default=True,
        description="Create collection if it doesn't exist"
    )


class ChromaDBStore(IVectorStore):
    """
    ChromaDB vector store implementation.
    
    Supports local persistence, in-memory operation, and client-server mode.
    Uses collections for organization (can simulate namespaces).
    
    Example:
        # Local persistent mode
        config = ChromaConfig(
            collection_name="memories",
            persist_directory="./chroma_data",
            dimension=1536,
        )
        store = ChromaDBStore(config)
        await store.initialize()
        
        # Client mode (connecting to server)
        config = ChromaConfig(
            collection_name="memories",
            host="localhost",
            port=8000,
        )
    """
    
    def __init__(
        self,
        config: ChromaConfig,
        embedding_function: Optional[Callable[[List[str]], Awaitable[List[List[float]]]]] = None,
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            config: ChromaDB configuration
            embedding_function: Optional async function to generate embeddings
                                (ChromaDB can use this for document operations)
        """
        self._config = config
        self._embedding_function = embedding_function
        self._client = None
        self._collection = None
        self._collections_cache: Dict[str, Any] = {}
        self._initialized = False
    
    @property
    def name(self) -> str:
        """The name of this vector store implementation."""
        return f"chromadb:{self._config.collection_name}"
    
    @property
    def dimension(self) -> int:
        """The dimension of vectors this store handles."""
        return self._config.dimension
    
    @property
    def metric(self) -> DistanceMetric:
        """The distance metric used for similarity."""
        return self._config.metric
    
    def _metric_to_chroma(self) -> str:
        """Convert our metric enum to ChromaDB format."""
        mapping = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip",  # inner product
        }
        return mapping.get(self._config.metric, "cosine")
    
    async def initialize(self) -> None:
        """
        Initialize connection to ChromaDB.
        
        Creates collection if configured and it doesn't exist.
        """
        if self._initialized:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ConfigurationError(
                "chromadb package not installed. "
                "Install with: pip install chromadb"
            ) from None
        
        # Determine client mode
        if self._config.host:
            # Client mode (connecting to server)
            self._client = chromadb.HttpClient(
                host=self._config.host,
                port=self._config.port,
                ssl=self._config.ssl,
                tenant=self._config.tenant,
                database=self._config.database,
                headers={"Authorization": f"Bearer {self._config.auth_token}"} 
                        if self._config.auth_token else None,
            )
        elif self._config.persist_directory:
            # Persistent local mode
            self._client = chromadb.PersistentClient(
                path=self._config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                ),
            )
        else:
            # In-memory mode
            self._client = chromadb.Client(
                settings=Settings(
                    anonymized_telemetry=False,
                ),
            )
        
        # Get or create collection
        await self._get_or_create_collection(self._config.collection_name)
        
        self._initialized = True
    
    async def _get_or_create_collection(self, name: str) -> Any:
        """Get or create a collection."""
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        loop = asyncio.get_event_loop()
        
        if self._config.create_collection_if_missing:
            collection = await loop.run_in_executor(
                None,
                lambda: self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": self._metric_to_chroma()},
                )
            )
        else:
            try:
                collection = await loop.run_in_executor(
                    None,
                    lambda: self._client.get_collection(name=name)
                )
            except Exception as e:
                raise ConfigurationError(
                    f"Collection '{name}' does not exist: {e}"
                ) from e
        
        self._collections_cache[name] = collection
        
        if name == self._config.collection_name:
            self._collection = collection
        
        return collection
    
    async def close(self) -> None:
        """Close the connection."""
        self._collection = None
        self._collections_cache.clear()
        self._client = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            raise VectorStoreError(
                "Store not initialized. Call initialize() first."
            )
    
    def _get_collection_name(self, namespace: Optional[str]) -> str:
        """
        Get the collection name for a namespace.
        
        ChromaDB uses collections instead of namespaces, so we create
        separate collections for each namespace.
        """
        if namespace:
            return f"{self._config.collection_name}_{namespace}"
        return self._config.namespace or self._config.collection_name
    
    async def _get_collection(self, namespace: Optional[str] = None) -> Any:
        """Get the appropriate collection."""
        name = self._get_collection_name(namespace)
        
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        return await self._get_or_create_collection(name)
    
    def _build_where_filter(
        self, 
        filters: Optional[List[QueryFilter]]
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where filter from QueryFilter list."""
        if not filters:
            return None
        
        if len(filters) == 1:
            return filters[0].to_chroma()
        
        # Multiple filters: combine with $and
        return {"$and": [f.to_chroma() for f in filters]}
    
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Insert or update vectors in ChromaDB.
        
        Args:
            vectors: List of vectors to upsert
            namespace: Optional namespace (becomes collection suffix)
            
        Returns:
            Number of vectors upserted
        """
        self._ensure_initialized()
        
        if not vectors:
            return 0
        
        collection = await self._get_collection(namespace)
        
        # Prepare data for ChromaDB
        ids = [v.id for v in vectors]
        embeddings = [v.embedding for v in vectors]
        metadatas = [v.metadata or {} for v in vectors]
        documents = [v.content for v in vectors if v.content]
        
        # Batch upsert
        loop = asyncio.get_event_loop()
        batch_size = self._config.batch_size
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = documents[i:i + batch_size] if documents else None
            
            await loop.run_in_executor(
                None,
                partial(
                    collection.upsert,
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents if batch_documents else None,
                ),
            )
        
        return len(vectors)
    
    async def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
        include_embedding: bool = False,
        include_metadata: bool = True,
    ) -> List[VectorQueryResult]:
        """
        Query for similar vectors.
        
        Args:
            embedding: The query vector
            top_k: Number of results to return
            namespace: Optional namespace
            filters: Optional metadata filters
            include_embedding: Whether to include vectors in results
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of query results, ordered by similarity
        """
        self._ensure_initialized()
        
        collection = await self._get_collection(namespace)
        where_filter = self._build_where_filter(filters)
        
        # Build include list
        include = ["distances"]
        if include_embedding:
            include.append("embeddings")
        if include_metadata:
            include.append("metadatas")
        include.append("documents")
        
        # Run query
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=include,
            )
        )
        
        # Convert results (ChromaDB returns nested lists)
        query_results = []
        if result["ids"] and result["ids"][0]:
            ids = result["ids"][0]
            distances = result["distances"][0] if result.get("distances") else [0] * len(ids)
            embeddings_list = result.get("embeddings", [[None] * len(ids)])[0] if include_embedding else [None] * len(ids)
            metadatas = result.get("metadatas", [[{}] * len(ids)])[0] if include_metadata else [{}] * len(ids)
            documents = result.get("documents", [[None] * len(ids)])[0]
            
            for i, vec_id in enumerate(ids):
                # Convert distance to similarity score
                # For cosine: distance is between 0-2, convert to 0-1 similarity
                distance = distances[i]
                if self._config.metric == DistanceMetric.COSINE:
                    score = 1 - (distance / 2)
                elif self._config.metric == DistanceMetric.DOT_PRODUCT:
                    score = distance  # Higher is better for dot product
                else:
                    # Euclidean: convert distance to similarity
                    score = 1 / (1 + distance)
                
                query_results.append(VectorQueryResult(
                    id=vec_id,
                    score=score,
                    embedding=embeddings_list[i] if embeddings_list else None,
                    metadata=metadatas[i] if metadatas else {},
                    content=documents[i] if documents else None,
                ))
        
        return query_results
    
    async def query_by_id(
        self,
        vector_id: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
        include_embedding: bool = False,
    ) -> List[VectorQueryResult]:
        """
        Query for vectors similar to an existing vector.
        
        Args:
            vector_id: ID of the reference vector
            top_k: Number of results to return
            namespace: Optional namespace
            filters: Optional metadata filters
            include_embedding: Whether to include vectors in results
            
        Returns:
            List of similar vectors, excluding the reference
        """
        self._ensure_initialized()
        
        # First, fetch the reference vector
        records = await self.fetch([vector_id], namespace=namespace)
        
        if vector_id not in records:
            raise VectorStoreError(f"Vector '{vector_id}' not found")
        
        reference = records[vector_id]
        
        # Query using its embedding
        results = await self.query(
            embedding=reference.embedding,
            top_k=top_k + 1,  # +1 to exclude self
            namespace=namespace,
            filters=filters,
            include_embedding=include_embedding,
        )
        
        # Filter out self
        return [r for r in results if r.id != vector_id][:top_k]
    
    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> Dict[str, VectorRecord]:
        """
        Fetch vectors by their IDs.
        
        Args:
            ids: List of vector IDs to fetch
            namespace: Optional namespace
            
        Returns:
            Dictionary mapping IDs to VectorRecords
        """
        self._ensure_initialized()
        
        if not ids:
            return {}
        
        collection = await self._get_collection(namespace)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: collection.get(
                ids=ids,
                include=["embeddings", "metadatas", "documents"],
            )
        )
        
        # Convert to VectorRecords
        records = {}
        if result["ids"]:
            for i, vec_id in enumerate(result["ids"]):
                embedding = result["embeddings"][i] if result.get("embeddings") else []
                metadata = result["metadatas"][i] if result.get("metadatas") else {}
                content = result["documents"][i] if result.get("documents") else None
                
                records[vec_id] = VectorRecord(
                    id=vec_id,
                    embedding=embedding,
                    metadata=metadata,
                    content=content,
                )
        
        return records
    
    async def delete(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace
            
        Returns:
            Number of vectors deleted
        """
        self._ensure_initialized()
        
        if not ids:
            return 0
        
        collection = await self._get_collection(namespace)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: collection.delete(ids=ids)
        )
        
        return len(ids)
    
    async def delete_by_filter(
        self,
        filters: List[QueryFilter],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Delete vectors matching filter conditions.
        
        Args:
            filters: Filter conditions
            namespace: Optional namespace
            
        Returns:
            Number of vectors deleted
        """
        self._ensure_initialized()
        
        collection = await self._get_collection(namespace)
        where_filter = self._build_where_filter(filters)
        
        if not where_filter:
            return 0
        
        # Get count before
        count_before = await self.count(namespace=namespace)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: collection.delete(where=where_filter)
        )
        
        # Get count after
        count_after = await self.count(namespace=namespace)
        
        return max(0, count_before - count_after)
    
    async def delete_namespace(
        self,
        namespace: str,
    ) -> bool:
        """
        Delete all vectors in a namespace (deletes the collection).
        
        Args:
            namespace: The namespace to delete
            
        Returns:
            True if deleted
        """
        self._ensure_initialized()
        
        collection_name = self._get_collection_name(namespace)
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.delete_collection(name=collection_name)
            )
            # Remove from cache
            self._collections_cache.pop(collection_name, None)
            return True
        except Exception:
            return False
    
    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces (collections) in the store.
        
        Returns:
            List of namespace names
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        collections = await loop.run_in_executor(
            None,
            lambda: self._client.list_collections()
        )
        
        # Extract namespace suffixes from collection names
        prefix = f"{self._config.collection_name}_"
        namespaces = []
        for coll in collections:
            name = coll.name
            if name == self._config.collection_name:
                namespaces.append("")  # Default namespace
            elif name.startswith(prefix):
                namespaces.append(name[len(prefix):])
        
        return namespaces
    
    async def count(
        self,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
    ) -> int:
        """
        Count vectors in the collection.
        
        Args:
            namespace: Optional namespace
            filters: Optional filters (not supported, ignored)
            
        Returns:
            Number of vectors
        """
        self._ensure_initialized()
        
        collection = await self._get_collection(namespace)
        
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(
            None,
            lambda: collection.count()
        )
        
        return count
    
    async def describe(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection information
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        collections = await loop.run_in_executor(
            None,
            lambda: self._client.list_collections()
        )
        
        collection_info = {}
        for coll in collections:
            count = await loop.run_in_executor(
                None,
                lambda c=coll: c.count()
            )
            collection_info[coll.name] = {
                "count": count,
                "metadata": coll.metadata,
            }
        
        return {
            "name": self._config.collection_name,
            "dimension": self._config.dimension,
            "metric": self._metric_to_chroma(),
            "collections": collection_info,
            "persist_directory": self._config.persist_directory,
            "host": self._config.host,
        }

