# mypy: disable-error-code="attr-defined,func-returns-value"
"""
Weaviate Vector Store Implementation.

Provides integration with Weaviate, an open-source vector database
with a GraphQL API and support for hybrid search.

Features:
- GraphQL-based API
- Hybrid search (vector + keyword)
- Multi-tenancy support
- Schema-based class definitions
- Batch import capabilities
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


class WeaviateConfig(VectorStoreConfig):
    """Configuration for Weaviate vector store."""
    
    url: str = Field(
        default="http://localhost:8080",
        description="Weaviate server URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Weaviate API key (for cloud instances)"
    )
    class_name: str = Field(
        default="Memory",
        description="Weaviate class name for storing vectors"
    )
    text_key: str = Field(
        default="content",
        description="Property name for text content"
    )
    additional_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers for requests"
    )
    grpc_port: Optional[int] = Field(
        default=50051,
        description="gRPC port for faster operations"
    )
    startup_period: int = Field(
        default=5,
        description="Seconds to wait for startup health check"
    )
    timeout_config: Dict[str, int] = Field(
        default_factory=lambda: {"query": 30, "insert": 60},
        description="Timeout configuration in seconds"
    )
    create_class_if_missing: bool = Field(
        default=True,
        description="Create class schema if it doesn't exist"
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid (vector + keyword) search"
    )


class WeaviateStore(IVectorStore):
    """
    Weaviate vector store implementation.
    
    Supports Weaviate Cloud Services (WCS) and self-hosted instances.
    Uses classes for organization and tenants for multi-tenancy.
    
    Example:
        # Local instance
        config = WeaviateConfig(
            url="http://localhost:8080",
            class_name="Memory",
            dimension=1536,
        )
        store = WeaviateStore(config)
        await store.initialize()
        
        # Cloud instance
        config = WeaviateConfig(
            url="https://your-instance.weaviate.network",
            api_key="your-api-key",
            class_name="Memory",
        )
    """
    
    def __init__(self, config: WeaviateConfig):
        """
        Initialize Weaviate store.
        
        Args:
            config: Weaviate configuration
        """
        self._config = config
        self._client = None
        self._collection = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        """The name of this vector store implementation."""
        return f"weaviate:{self._config.class_name}"
    
    @property
    def dimension(self) -> int:
        """The dimension of vectors this store handles."""
        return self._config.dimension
    
    @property
    def metric(self) -> DistanceMetric:
        """The distance metric used for similarity."""
        return self._config.metric
    
    def _metric_to_weaviate(self) -> str:
        """Convert our metric enum to Weaviate format."""
        mapping = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2-squared",
            DistanceMetric.DOT_PRODUCT: "dot",
        }
        return mapping.get(self._config.metric, "cosine")
    
    async def initialize(self) -> None:
        """
        Initialize connection to Weaviate.
        
        Creates class schema if configured and it doesn't exist.
        """
        if self._initialized:
            return
        
        try:
            import weaviate
            from weaviate.classes.config import Configure  # noqa: F401
            from weaviate.classes.init import Auth
        except ImportError:
            raise ConfigurationError(
                "weaviate-client package not installed. "
                "Install with: pip install weaviate-client"
            ) from None
        
        # Build connection parameters
        loop = asyncio.get_event_loop()
        
        if self._config.api_key:
            # Cloud connection
            self._client = await loop.run_in_executor(
                None,
                lambda: weaviate.connect_to_weaviate_cloud(
                    cluster_url=self._config.url,
                    auth_credentials=Auth.api_key(self._config.api_key),
                    additional_headers=self._config.additional_headers,
                )
            )
        else:
            # Local connection
            # Parse URL for host and port
            from urllib.parse import urlparse
            parsed = urlparse(self._config.url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080
            
            self._client = await loop.run_in_executor(
                None,
                lambda: weaviate.connect_to_local(
                    host=host,
                    port=port,
                    grpc_port=self._config.grpc_port,
                    additional_headers=self._config.additional_headers,
                )
            )
        
        # Check if class exists, create if needed
        await self._ensure_class_exists()
        
        self._initialized = True
    
    async def _ensure_class_exists(self) -> None:
        """Ensure the Weaviate class schema exists."""
        from weaviate.classes.config import Configure, DataType, Property
        
        loop = asyncio.get_event_loop()
        
        # Check if collection exists
        exists = await loop.run_in_executor(
            None,
            lambda: self._client.collections.exists(self._config.class_name)
        )
        
        if not exists:
            if not self._config.create_class_if_missing:
                raise ConfigurationError(
                    f"Class '{self._config.class_name}' does not exist. "
                    "Set create_class_if_missing=True to create it."
                )
            
            # Create the collection with schema
            await loop.run_in_executor(
                None,
                lambda: self._client.collections.create(
                    name=self._config.class_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=getattr(
                            Configure.VectorIndex.Distance,
                            self._metric_to_weaviate().upper().replace("-", "_"),
                            Configure.VectorIndex.Distance.COSINE
                        )
                    ),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="memory_id", data_type=DataType.TEXT),
                        Property(name="user_id", data_type=DataType.TEXT),
                        Property(name="memory_type", data_type=DataType.TEXT),
                        Property(name="created_at", data_type=DataType.DATE),
                        Property(name="metadata_json", data_type=DataType.TEXT),
                    ],
                )
            )
        
        # Get collection reference
        self._collection = self._client.collections.get(self._config.class_name)
    
    async def close(self) -> None:
        """Close the connection."""
        if self._client:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.close)
        self._collection = None
        self._client = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            raise VectorStoreError(
                "Store not initialized. Call initialize() first."
            )
    
    def _get_tenant(self, namespace: Optional[str]) -> Optional[str]:
        """Get tenant name from namespace."""
        return namespace or self._config.namespace
    
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Insert or update vectors in Weaviate.
        
        Args:
            vectors: List of vectors to upsert
            namespace: Optional namespace (used as tenant)
            
        Returns:
            Number of vectors upserted
        """
        self._ensure_initialized()
        
        if not vectors:
            return 0
        
        import json

        from weaviate.classes.data import DataObject
        
        # Prepare data objects
        data_objects = []
        for vec in vectors:
            properties = {
                "content": vec.content or "",
                "memory_id": vec.id,
                "metadata_json": json.dumps(vec.metadata) if vec.metadata else "{}",
            }
            # Add any additional metadata as properties
            for key, value in (vec.metadata or {}).items():
                if key in ["user_id", "memory_type"]:
                    properties[key] = str(value)
            
            data_objects.append(DataObject(
                uuid=vec.id if self._is_valid_uuid(vec.id) else None,
                properties=properties,
                vector=vec.embedding,
            ))
        
        # Batch insert
        batch_size = self._config.batch_size
        upserted = 0
        
        for i in range(0, len(data_objects), batch_size):
            batch = data_objects[i:i + batch_size]
            
            with self._collection.batch.dynamic() as batch_inserter:
                for obj in batch:
                    batch_inserter.add_object(
                        properties=obj.properties,
                        uuid=obj.uuid,
                        vector=obj.vector,
                    )
            
            upserted += len(batch)
        
        return upserted
    
    def _is_valid_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        import uuid
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False
    
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
            namespace: Optional namespace (tenant)
            filters: Optional metadata filters
            include_embedding: Whether to include vectors in results
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of query results, ordered by similarity
        """
        self._ensure_initialized()
        
        import json

        from weaviate.classes.query import MetadataQuery
        
        loop = asyncio.get_event_loop()
        
        # Build filter if provided
        weaviate_filter = self._build_filter(filters) if filters else None
        
        # Configure what to return
        return_metadata = MetadataQuery(
            distance=True,
            certainty=True,
        )
        
        # Perform query
        result = await loop.run_in_executor(
            None,
            lambda: self._collection.query.near_vector(
                near_vector=embedding,
                limit=top_k,
                filters=weaviate_filter,
                include_vector=include_embedding,
                return_metadata=return_metadata,
            )
        )
        
        # Convert results
        query_results = []
        for obj in result.objects:
            # Parse metadata from JSON
            metadata = {}
            if include_metadata and obj.properties.get("metadata_json"):
                try:
                    metadata = json.loads(obj.properties["metadata_json"])
                except json.JSONDecodeError:
                    pass
            
            # Calculate score from distance
            # Weaviate returns distance; convert to similarity
            distance = obj.metadata.distance if obj.metadata else 0
            if self._config.metric == DistanceMetric.COSINE:
                score = 1 - distance
            elif self._config.metric == DistanceMetric.DOT_PRODUCT:
                score = distance  # For dot product, higher is better
            else:
                score = 1 / (1 + distance)
            
            query_results.append(VectorQueryResult(
                id=str(obj.uuid) if obj.uuid else obj.properties.get("memory_id", ""),
                score=score,
                embedding=obj.vector.get("default") if include_embedding and obj.vector else None,
                metadata=metadata,
                content=obj.properties.get("content"),
            ))
        
        return query_results
    
    def _build_filter(self, filters: List[QueryFilter]) -> Any:
        """Build Weaviate filter from QueryFilter list."""
        from weaviate.classes.query import Filter
        
        if not filters:
            return None
        
        # Build individual filters
        weaviate_filters = []
        for f in filters:
            prop_filter = Filter.by_property(f.field)
            
            if f.operator == "eq":
                weaviate_filters.append(prop_filter.equal(f.value))
            elif f.operator == "ne":
                weaviate_filters.append(prop_filter.not_equal(f.value))
            elif f.operator == "gt":
                weaviate_filters.append(prop_filter.greater_than(f.value))
            elif f.operator == "gte":
                weaviate_filters.append(prop_filter.greater_or_equal(f.value))
            elif f.operator == "lt":
                weaviate_filters.append(prop_filter.less_than(f.value))
            elif f.operator == "lte":
                weaviate_filters.append(prop_filter.less_or_equal(f.value))
            elif f.operator == "contains":
                weaviate_filters.append(prop_filter.like(f"*{f.value}*"))
            elif f.operator == "in":
                weaviate_filters.append(prop_filter.contains_any(f.value))
        
        if len(weaviate_filters) == 1:
            return weaviate_filters[0]
        
        # Combine with AND
        combined = weaviate_filters[0]
        for wf in weaviate_filters[1:]:
            combined = combined & wf
        
        return combined
    
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
        
        # Fetch the reference vector
        records = await self.fetch([vector_id], namespace=namespace)
        
        if vector_id not in records:
            raise VectorStoreError(f"Vector '{vector_id}' not found")
        
        reference = records[vector_id]
        
        # Query using its embedding
        results = await self.query(
            embedding=reference.embedding,
            top_k=top_k + 1,
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
        
        import json
        
        loop = asyncio.get_event_loop()
        records = {}
        
        for vec_id in ids:
            try:
                if self._is_valid_uuid(vec_id):
                    obj = await loop.run_in_executor(
                        None,
                        lambda vid=vec_id: self._collection.query.fetch_object_by_id(
                            uuid=vid,
                            include_vector=True,
                        )
                    )
                    
                    if obj:
                        metadata = {}
                        if obj.properties.get("metadata_json"):
                            try:
                                metadata = json.loads(obj.properties["metadata_json"])
                            except json.JSONDecodeError:
                                pass
                        
                        records[vec_id] = VectorRecord(
                            id=vec_id,
                            embedding=obj.vector.get("default", []) if obj.vector else [],
                            metadata=metadata,
                            content=obj.properties.get("content"),
                        )
            except Exception:
                continue  # Skip failed fetches
        
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
        
        loop = asyncio.get_event_loop()
        deleted = 0
        
        for vec_id in ids:
            try:
                if self._is_valid_uuid(vec_id):
                    success = await loop.run_in_executor(
                        None,
                        lambda vid=vec_id: self._collection.data.delete_by_id(uuid=vid)
                    )
                    if success:
                        deleted += 1
            except Exception:
                continue
        
        return deleted
    
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
        
        if not filters:
            return 0
        
        weaviate_filter = self._build_filter(filters)
        
        loop = asyncio.get_event_loop()
        
        # Get count before
        count_before = await self.count(namespace=namespace)
        
        # Delete by filter
        await loop.run_in_executor(
            None,
            lambda: self._collection.data.delete_many(where=weaviate_filter)
        )
        
        # Get count after
        count_after = await self.count(namespace=namespace)
        
        return max(0, count_before - count_after)
    
    async def delete_namespace(
        self,
        namespace: str,
    ) -> bool:
        """
        Delete all vectors for a tenant.
        
        Note: In Weaviate, this requires multi-tenancy to be enabled.
        For non-tenant setups, this will fail.
        
        Args:
            namespace: The namespace (tenant) to delete
            
        Returns:
            True if deleted
        """
        self._ensure_initialized()
        
        # For multi-tenant setups, we would delete the tenant
        # For now, we can delete all objects (not recommended for production)
        loop = asyncio.get_event_loop()
        
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.collections.delete(f"{self._config.class_name}_{namespace}")
            )
            return True
        except Exception:
            return False
    
    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces (tenants or classes).
        
        Returns:
            List of namespace names
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        
        # List all collections
        collections = await loop.run_in_executor(
            None,
            lambda: list(self._client.collections.list_all().keys())
        )
        
        # Extract namespaces from collection names
        prefix = f"{self._config.class_name}_"
        namespaces = []
        for name in collections:
            if name == self._config.class_name:
                namespaces.append("")
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
            filters: Optional filters
            
        Returns:
            Number of vectors
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        
        weaviate_filter = self._build_filter(filters) if filters else None
        
        result = await loop.run_in_executor(
            None,
            lambda: self._collection.aggregate.over_all(
                filters=weaviate_filter,
                total_count=True,
            )
        )
        
        return result.total_count if result else 0
    
    async def describe(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection information
        """
        self._ensure_initialized()
        
        loop = asyncio.get_event_loop()
        
        # Get collection config
        config = await loop.run_in_executor(
            None,
            lambda: self._collection.config.get()
        )
        
        # Get count
        count = await self.count()
        
        return {
            "name": self._config.class_name,
            "dimension": self._config.dimension,
            "metric": self._metric_to_weaviate(),
            "total_count": count,
            "properties": [p.name for p in config.properties] if config.properties else [],
            "url": self._config.url,
        }

