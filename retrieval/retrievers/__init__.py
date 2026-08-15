"""
Retriever implementations.

Provides various strategies for retrieving context items:
- BaseRetriever: Abstract base for memory retrievers
- SimpleRetriever: Basic memory store search
- SemanticRetriever: Embedding-based similarity search
- HybridRetriever: Combined semantic + keyword search (memory)
- TemporalRetriever: Recency-weighted retrieval
- ExpertiseRetriever: Expertise item retrieval
- HybridExpertiseRetriever: Combined search for expertise
"""

from ctxforge.retrieval.retrievers.base import BaseRetriever, SimpleRetriever
from ctxforge.retrieval.retrievers.expertise import (
    ExpertiseRetrievalConfig,
    ExpertiseRetrievalResult,
    ExpertiseRetriever,
    HybridExpertiseRetriever,
)
from ctxforge.retrieval.retrievers.hybrid import HybridRetriever
from ctxforge.retrieval.retrievers.salience import SalienceRetriever
from ctxforge.retrieval.retrievers.semantic import SemanticRetriever
from ctxforge.retrieval.retrievers.temporal import TemporalRetriever

__all__ = [
    # Base
    "BaseRetriever",
    "SimpleRetriever",
    # Memory retrievers
    "SemanticRetriever",
    "HybridRetriever",
    "TemporalRetriever",
    "SalienceRetriever",
    # Expertise retrievers
    "ExpertiseRetriever",
    "HybridExpertiseRetriever",
    "ExpertiseRetrievalResult",
    "ExpertiseRetrievalConfig",
]

