"""
Vector Store implementations for the ctxforge framework.

Provides integrations with various vector databases for semantic memory
retrieval. Each implementation follows the IVectorStore protocol.

Available stores:
- PineconeStore: Cloud-native vector database with namespaces
- ChromaDBStore: Local/cloud vector database
- WeaviateStore: Enterprise-grade vector search

Usage:
    from ctxforge.vectorstores import PineconeStore, ChromaDBStore
    
    # Create a Pinecone store
    store = PineconeStore(
        api_key="your-api-key",
        index_name="memories",
        namespace="user_123"
    )
    
    # Upsert vectors
    await store.upsert(vectors=[...])
    
    # Query similar vectors
    results = await store.query(embedding=[...], top_k=5)
"""

from ctxforge.vectorstores.chroma_store import (
    ChromaConfig,
    ChromaDBStore,
)
from ctxforge.vectorstores.pinecone_store import (
    PineconeConfig,
    PineconeStore,
)
from ctxforge.vectorstores.protocol import (
    IVectorStore,
    VectorQueryResult,
    VectorRecord,
    VectorStoreConfig,
)
from ctxforge.vectorstores.weaviate_store import (
    WeaviateConfig,
    WeaviateStore,
)

__all__ = [
    # Protocol
    "IVectorStore",
    "VectorRecord",
    "VectorQueryResult",
    "VectorStoreConfig",
    # Pinecone
    "PineconeStore",
    "PineconeConfig",
    # ChromaDB
    "ChromaDBStore",
    "ChromaConfig",
    # Weaviate
    "WeaviateStore",
    "WeaviateConfig",
]

