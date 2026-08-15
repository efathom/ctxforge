"""
Retrieval implementations.

Provides various strategies for retrieving and indexing context items:

Indexers (for vector search):
- MemoryIndexer: Indexes memory items
- ExpertiseIndexer: Indexes expertise items

Retrievers:
- BaseRetriever: Abstract base for memory retrievers
- SimpleRetriever: Basic memory store search
- SemanticRetriever: Embedding-based similarity search
- HybridRetriever: Combined semantic + keyword search
- TemporalRetriever: Recency-weighted retrieval
- ExpertiseRetriever: Expertise item retrieval
- HybridExpertiseRetriever: Combined search for expertise

Rerankers:
- RecencyReranker: Boosts recent memories
- ScoreThresholdReranker: Filters by score threshold
- DiversityReranker: Promotes result diversity
- EffectivenessReranker: Prioritizes expertise by effectiveness

Utilities:
- EmbeddingFunc: Type alias for embedding functions
- apply_memory_filters: Filter memories by config
"""

# Indexers
from ctxforge.retrieval.aggregation_builder import AggregationBuilder

# Enhanced structures and fast-path retrieval
from ctxforge.retrieval.enhanced_structures import (
    EnhancedMemoryIndex,
    EntityAggregation,
    QueryCache,
    RelationTriple,
)
from ctxforge.retrieval.fast_path_retriever import (
    FastPathResult,
    FastPathRetriever,
)
from ctxforge.retrieval.indexers import (
    ExpertiseIndexer,
    MemoryIndexer,
)

# Rerankers
from ctxforge.retrieval.rerankers import (
    DiversityReranker,
    EffectivenessReranker,
    RecencyReranker,
    ScoreThresholdReranker,
)

# Retrievers
from ctxforge.retrieval.retrievers import (
    BaseRetriever,
    ExpertiseRetrievalConfig,
    ExpertiseRetrievalResult,
    ExpertiseRetriever,
    HybridExpertiseRetriever,
    HybridRetriever,
    SemanticRetriever,
    SimpleRetriever,
    TemporalRetriever,
)

# Utilities
from ctxforge.retrieval.utils import (
    EmbeddingFunc,
    apply_memory_filters,
)

__all__ = [
    # Indexers
    "MemoryIndexer",
    "ExpertiseIndexer",
    # Base
    "BaseRetriever",
    # Memory Retrievers
    "SimpleRetriever",
    "SemanticRetriever",
    "HybridRetriever",
    "TemporalRetriever",
    # Expertise Retrievers
    "ExpertiseRetriever",
    "HybridExpertiseRetriever",
    "ExpertiseRetrievalResult",
    "ExpertiseRetrievalConfig",
    # Memory Rerankers
    "RecencyReranker",
    "ScoreThresholdReranker",
    "DiversityReranker",
    # Expertise Rerankers
    "EffectivenessReranker",
    # Utilities
    "EmbeddingFunc",
    "apply_memory_filters",
    # Enhanced structures
    "EntityAggregation",
    "RelationTriple",
    "QueryCache",
    "EnhancedMemoryIndex",
    # Aggregation builder
    "AggregationBuilder",
    # Fast-path retrieval
    "FastPathRetriever",
    "FastPathResult",
]
