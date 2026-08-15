# mypy: disable-error-code="attr-defined"
"""
Pinecone Vector Store Implementation.

Provides integration with Pinecone, a cloud-native vector database
optimized for large-scale similarity search with low latency.

Features:
- Namespace-based multi-tenancy
- Metadata filtering
- Serverless and pod-based deployments
- High-throughput batch operations
"""

import asyncio
from typing import Any, Dict, List, Optional

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


class PineconeConfig(VectorStoreConfig):
    """Configuration for Pinecone vector store."""
    
    api_key: str = Field(
        ...,
        description="Pinecone API key"
    )
    environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment (for pod-based, e.g., 'us-west1-gcp')"
    )
    index_name: str = Field(
        ...,
        description="Name of the Pinecone index"
    )
    host: Optional[str] = Field(
        default=None,
        description="Index host URL (for serverless)"
    )
    create_index_if_missing: bool = Field(
        default=False,
        description="Create index if it doesn't exist"
    )
    serverless_cloud: Optional[str] = Field(
        default="aws",
        description="Cloud provider for serverless (aws, gcp, azure)"
    )
    serverless_region: Optional[str] = Field(
        default="us-east-1",
        description="Region for serverless deployment"
    )
    pod_type: Optional[str] = Field(
        default=None,
        description="Pod type for pod-based deployment (e.g., 'p1.x1')"
    )
    replicas: int = Field(
        default=1,
        description="Number of replicas for pod-based",
        ge=1
    )
    pool_threads: int = Field(
        default=4,
        description="Number of threads for connection pool",
        ge=1
    )


class PineconeStore(IVectorStore):
    """
    Pinecone vector store implementation.
    
    Supports both serverless and pod-based Pinecone deployments.
    Uses namespaces for multi-tenant isolation.
    
    Example:
        config = PineconeConfig(
            api_key="your-api-key",
            index_name="memories",
            dimension=1536,
        )
        store = PineconeStore(config)
        await store.initialize()
        
        # Upsert vectors
        await store.upsert([
            VectorRecord(id="mem_1", embedding=[...], metadata={"user": "123"})
        ])
        
        # Query similar vectors
        results = await store.query(embedding=[...], top_k=5)
    """
    
    def __init__(self, config: PineconeConfig):
        """
        Initialize Pinecone store.
        
        Args:
            config: Pinecone configuration
        """
        self._config = config
        self._client = None
        self._index = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        """The name of this vector store implementation."""
        return f"pinecone:{self._config.index_name}"
    
    @property
    def dimension(self) -> int:
        """The dimension of vectors this store handles."""
        return self._config.dimension
    
    @property
    def metric(self) -> DistanceMetric:
        """The distance metric used for similarity."""
        return self._config.metric
    
    def _metric_to_pinecone(self) -> str:
        """Convert our metric enum to Pinecone format."""
        mapping = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "euclidean",
            DistanceMetric.DOT_PRODUCT: "dotproduct",
        }
        return mapping.get(self._config.metric, "cosine")
    
    async def initialize(self) -> None:
        """
        Initialize connection to Pinecone.
        
        Creates index if configured and it doesn't exist.
        """
        if self._initialized:
            return
        
        try:
            from pinecone import Pinecone  # ServerlessSpec, PodSpec available if needed
        except ImportError:
            raise ConfigurationError(
                "pinecone-client package not installed. "
                "Install with: pip install pinecone-client"
            ) from None
        
        # Initialize client
        self._client = Pinecone(
            api_key=self._config.api_key,
            pool_threads=self._config.pool_threads,
        )
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self._client.list_indexes()]
        
        if self._config.index_name not in existing_indexes:
            if self._config.create_index_if_missing:
                await self._create_index()
            else:
                raise ConfigurationError(
                    f"Index '{self._config.index_name}' does not exist. "
                    "Set create_index_if_missing=True to create it."
                )
        
        # Get index reference
        if self._config.host:
            self._index = self._client.Index(
                name=self._config.index_name,
                host=self._config.host,
            )
        else:
            self._index = self._client.Index(self._config.index_name)
        
        self._initialized = True
    
    async def _create_index(self) -> None:
        """Create a new Pinecone index."""
        from pinecone import PodSpec, ServerlessSpec
        
        if self._config.pod_type:
            # Pod-based deployment
            spec = PodSpec(
                environment=self._config.environment,
                pod_type=self._config.pod_type,
                replicas=self._config.replicas,
            )
        else:
            # Serverless deployment
            spec = ServerlessSpec(
                cloud=self._config.serverless_cloud,
                region=self._config.serverless_region,
            )
        
        self._client.create_index(
            name=self._config.index_name,
            dimension=self._config.dimension,
            metric=self._metric_to_pinecone(),
            spec=spec,
        )
        
        # Wait for index to be ready
        while not self._client.describe_index(self._config.index_name).status.ready:
            await asyncio.sleep(1)
    
    async def close(self) -> None:
        """Close the connection."""
        self._index = None
        self._client = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            raise VectorStoreError(
                "Store not initialized. Call initialize() first."
            )
    
    def _get_namespace(self, namespace: Optional[str]) -> str:
        """Get the namespace to use."""
        return namespace or self._config.namespace or ""
    
    def _build_filter(self, filters: Optional[List[QueryFilter]]) -> Optional[Dict[str, Any]]:
        """Build Pinecone filter from QueryFilter list."""
        if not filters:
            return None
        
        if len(filters) == 1:
            return filters[0].to_pinecone()
        
        # Multiple filters: combine with $and
        return {"$and": [f.to_pinecone() for f in filters]}
    
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Insert or update vectors in Pinecone.
        
        Args:
            vectors: List of vectors to upsert
            namespace: Optional namespace override
            
        Returns:
            Number of vectors upserted
        """
        self._ensure_initialized()
        
        if not vectors:
            return 0
        
        ns = self._get_namespace(namespace)
        
        # Convert to Pinecone format
        pinecone_vectors = []
        for vec in vectors:
            record = {
                "id": vec.id,
                "values": vec.embedding,
            }
            if vec.metadata:
                record["metadata"] = vec.metadata
            pinecone_vectors.append(record)
        
        # Batch upsert
        upserted_count = 0
        batch_size = self._config.batch_size
        
        for i in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[i:i + batch_size]
            # Run in executor for async compatibility
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda b=batch: self._index.upsert(vectors=b, namespace=ns)
            )
            upserted_count += result.upserted_count
        
        return upserted_count
    
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
            namespace: Optional namespace to search in
            filters: Optional metadata filters
            include_embedding: Whether to include vectors in results
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of query results, ordered by similarity
        """
        self._ensure_initialized()
        
        ns = self._get_namespace(namespace)
        filter_dict = self._build_filter(filters)
        
        # Run query in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._index.query(
                vector=embedding,
                top_k=top_k,
                namespace=ns,
                filter=filter_dict,
                include_values=include_embedding,
                include_metadata=include_metadata,
            )
        )
        
        # Convert results
        query_results = []
        for match in result.matches:
            query_results.append(VectorQueryResult(
                id=match.id,
                score=match.score,
                embedding=match.values if include_embedding else None,
                metadata=match.metadata or {},
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
        
        ns = self._get_namespace(namespace)
        filter_dict = self._build_filter(filters)
        
        # Run query in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._index.query(
                id=vector_id,
                top_k=top_k + 1,  # +1 to exclude self
                namespace=ns,
                filter=filter_dict,
                include_values=include_embedding,
                include_metadata=True,
            )
        )
        
        # Convert and filter out self
        query_results = []
        for match in result.matches:
            if match.id != vector_id:
                query_results.append(VectorQueryResult(
                    id=match.id,
                    score=match.score,
                    embedding=match.values if include_embedding else None,
                    metadata=match.metadata or {},
                ))
        
        return query_results[:top_k]
    
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
        
        ns = self._get_namespace(namespace)
        
        # Run fetch in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._index.fetch(ids=ids, namespace=ns)
        )
        
        # Convert to VectorRecords
        records = {}
        for vec_id, vec_data in result.vectors.items():
            records[vec_id] = VectorRecord(
                id=vec_id,
                embedding=vec_data.values,
                metadata=vec_data.metadata or {},
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
            Number of vectors deleted (Pinecone doesn't return exact count)
        """
        self._ensure_initialized()
        
        if not ids:
            return 0
        
        ns = self._get_namespace(namespace)
        
        # Batch delete
        batch_size = self._config.batch_size
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            await loop.run_in_executor(
                None,
                lambda b=batch: self._index.delete(ids=b, namespace=ns)
            )
        
        return len(ids)  # Pinecone delete doesn't return count
    
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
            Estimated number deleted (Pinecone doesn't return exact count)
        """
        self._ensure_initialized()
        
        ns = self._get_namespace(namespace)
        filter_dict = self._build_filter(filters)
        
        if not filter_dict:
            return 0
        
        # Get count before delete
        stats_before = await self.count(namespace=namespace)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._index.delete(filter=filter_dict, namespace=ns)
        )
        
        # Get count after delete
        stats_after = await self.count(namespace=namespace)
        
        return max(0, stats_before - stats_after)
    
    async def delete_namespace(
        self,
        namespace: str,
    ) -> bool:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: The namespace to delete
            
        Returns:
            True if deletion was initiated
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._index.delete(delete_all=True, namespace=namespace)
        )
        
        return True
    
    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the index.
        
        Returns:
            List of namespace names
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            lambda: self._index.describe_index_stats()
        )
        
        # Extract namespaces from stats
        namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        return namespaces
    
    async def count(
        self,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
    ) -> int:
        """
        Count vectors in the store.
        
        Note: Pinecone doesn't support filtered counts natively.
        For filtered counts, this performs a query with large top_k.
        
        Args:
            namespace: Optional namespace
            filters: Optional filters (approximate count only)
            
        Returns:
            Number of vectors
        """
        self._ensure_initialized()
        
        ns = self._get_namespace(namespace)
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            lambda: self._index.describe_index_stats()
        )
        
        if ns and stats.namespaces:
            ns_stats = stats.namespaces.get(ns)
            return ns_stats.vector_count if ns_stats else 0
        
        return stats.total_vector_count
    
    async def describe(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index information
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            lambda: self._index.describe_index_stats()
        )
        
        return {
            "name": self._config.index_name,
            "dimension": stats.dimension,
            "total_vector_count": stats.total_vector_count,
            "namespaces": {
                ns: {"vector_count": ns_stats.vector_count}
                for ns, ns_stats in (stats.namespaces or {}).items()
            },
            "metric": self._metric_to_pinecone(),
        }

