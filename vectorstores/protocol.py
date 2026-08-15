"""
Vector Store Protocol Interface.

Defines the contract for vector database integrations.
Supports operations like upsert, query, delete, and metadata filtering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class DistanceMetric(str, Enum):
    """Supported distance/similarity metrics."""
    
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class VectorRecord:
    """
    A record to store in a vector database.
    
    Attributes:
        id: Unique identifier for the vector
        embedding: The vector embedding
        metadata: Additional metadata to store with the vector
        content: Optional text content (for stores that support it)
    """
    
    id: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Vector record ID cannot be empty")
        if not self.embedding:
            raise ValueError("Vector embedding cannot be empty")


@dataclass
class VectorQueryResult:
    """
    Result from a vector similarity query.
    
    Attributes:
        id: The ID of the matched vector
        score: Similarity score (interpretation depends on metric)
        embedding: Optional - the stored embedding vector
        metadata: Metadata stored with the vector
        content: Optional text content if stored
    """
    
    id: str
    score: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None


class VectorStoreConfig(BaseModel):
    """Base configuration for vector stores."""
    
    dimension: int = Field(
        default=1536,
        description="Dimension of embedding vectors",
        ge=1
    )
    metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE,
        description="Distance metric for similarity"
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional namespace/partition for isolation"
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for bulk operations",
        ge=1
    )


@dataclass
class QueryFilter:
    """
    Filter conditions for vector queries.
    
    Supports basic comparison operators for metadata filtering.
    """
    
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, nin, contains
    value: Any
    
    def to_pinecone(self) -> Dict[str, Any]:
        """Convert to Pinecone filter format."""
        op_map = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "gte": "$gte",
            "lt": "$lt",
            "lte": "$lte",
            "in": "$in",
            "nin": "$nin",
        }
        if self.operator == "eq":
            return {self.field: self.value}
        return {self.field: {op_map.get(self.operator, "$eq"): self.value}}
    
    def to_chroma(self) -> Dict[str, Any]:
        """Convert to ChromaDB filter format."""
        op_map = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "gte": "$gte",
            "lt": "$lt",
            "lte": "$lte",
            "in": "$in",
            "nin": "$nin",
            "contains": "$contains",
        }
        return {self.field: {op_map.get(self.operator, "$eq"): self.value}}
    
    def to_weaviate(self) -> Dict[str, Any]:
        """Convert to Weaviate filter format."""
        op_map = {
            "eq": "Equal",
            "ne": "NotEqual",
            "gt": "GreaterThan",
            "gte": "GreaterThanEqual",
            "lt": "LessThan",
            "lte": "LessThanEqual",
            "in": "ContainsAny",
            "contains": "Like",
        }
        return {
            "path": [self.field],
            "operator": op_map.get(self.operator, "Equal"),
            "valueText" if isinstance(self.value, str) else "valueNumber": self.value,
        }


@runtime_checkable
class IVectorStore(Protocol):
    """
    Protocol for vector database operations.
    
    Implementations provide storage and similarity search for embedding vectors.
    Supports namespaces for multi-tenant isolation and metadata filtering.
    
    Example implementations:
    - PineconeStore: Cloud-native with managed infrastructure
    - ChromaDBStore: Local-first with cloud sync option
    - WeaviateStore: Enterprise-grade with GraphQL API
    - QdrantStore: High-performance Rust-based store
    
    All methods are async to support high-throughput operations.
    """
    
    @property
    def name(self) -> str:
        """The name of this vector store implementation."""
        ...
    
    @property
    def dimension(self) -> int:
        """The dimension of vectors this store handles."""
        ...
    
    @property
    def metric(self) -> DistanceMetric:
        """The distance metric used for similarity."""
        ...
    
    async def initialize(self) -> None:
        """
        Initialize the vector store connection.
        
        Should be called before any operations. Creates indexes/collections
        if they don't exist.
        
        Raises:
            ConnectionError: If connection fails
            ConfigurationError: If configuration is invalid
        """
        ...
    
    async def close(self) -> None:
        """
        Close the vector store connection.
        
        Should be called when done using the store.
        """
        ...
    
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Insert or update vectors in the store.
        
        Args:
            vectors: List of vectors to upsert
            namespace: Optional namespace override
            
        Returns:
            Number of vectors successfully upserted
            
        Raises:
            VectorStoreError: If upsert fails
        """
        ...
    
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
            List of query results, ordered by similarity (best first)
            
        Raises:
            VectorStoreError: If query fails
        """
        ...
    
    async def query_by_id(
        self,
        vector_id: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
        include_embedding: bool = False,
    ) -> List[VectorQueryResult]:
        """
        Query for vectors similar to an existing vector by ID.
        
        Args:
            vector_id: ID of the reference vector
            top_k: Number of results to return
            namespace: Optional namespace to search in
            filters: Optional metadata filters
            include_embedding: Whether to include vectors in results
            
        Returns:
            List of query results, excluding the reference vector
            
        Raises:
            NotFoundError: If vector_id doesn't exist
            VectorStoreError: If query fails
        """
        ...
    
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
            Dictionary mapping IDs to VectorRecords (missing IDs omitted)
            
        Raises:
            VectorStoreError: If fetch fails
        """
        ...
    
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
            
        Raises:
            VectorStoreError: If delete fails
        """
        ...
    
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
            
        Raises:
            VectorStoreError: If delete fails
        """
        ...
    
    async def delete_namespace(
        self,
        namespace: str,
    ) -> bool:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: The namespace to delete
            
        Returns:
            True if namespace was deleted
            
        Raises:
            VectorStoreError: If delete fails
        """
        ...
    
    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the store.
        
        Returns:
            List of namespace names
        """
        ...
    
    async def count(
        self,
        namespace: Optional[str] = None,
        filters: Optional[List[QueryFilter]] = None,
    ) -> int:
        """
        Count vectors in the store.
        
        Args:
            namespace: Optional namespace to count in
            filters: Optional filters to apply
            
        Returns:
            Number of matching vectors
        """
        ...
    
    async def describe(self) -> Dict[str, Any]:
        """
        Get statistics and metadata about the store.
        
        Returns:
            Dictionary with store information (dimensions, count, etc.)
        """
        ...

