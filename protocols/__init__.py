"""
Protocol interfaces for the ctxforge framework.

Protocols define the contracts that components must implement,
enabling pluggable, duck-typed extensibility. Any class that
implements the required methods can be used, regardless of inheritance.
"""

from ctxforge.protocols.context import (
    ContextRetrievalResult,
    IContextIndexer,
    IContextItem,
    IContextReranker,
    IContextRetriever,
    IContextStore,
    IndexSearchResult,
)
from ctxforge.protocols.expertise import (
    ICurator,
    IExpertiseReranker,
    IExpertiseRetriever,
    IExpertiseStore,
    IReflector,
)
from ctxforge.protocols.extractor import IMemoryExtractor
from ctxforge.protocols.llm import (
    IEmbeddingProvider,
    ILLMProvider,
)
from ctxforge.protocols.retriever import IRetriever
from ctxforge.protocols.storage import (
    IMemoryStore,
    ISessionStore,
)
from ctxforge.protocols.vectorstore import IVectorStore

__all__ = [
    # Storage
    "ISessionStore",
    "IMemoryStore",
    # Vector Stores
    "IVectorStore",
    # LLM
    "ILLMProvider",
    "IEmbeddingProvider",
    # Retrieval
    "IRetriever",
    # Extraction
    "IMemoryExtractor",
    # Expertise
    "IExpertiseStore",
    "IExpertiseRetriever",
    "IExpertiseReranker",
    "IReflector",
    "ICurator",
    # Context (generic protocols)
    "IContextItem",
    "IContextRetriever",
    "IContextReranker",
    "IContextIndexer",
    "IContextStore",
    "ContextRetrievalResult",
    "IndexSearchResult",
]

