"""
Shared utilities for the ctxforge framework.

This module provides common utilities used across the framework:
- Math utilities (cosine similarity, vector operations)
- Similarity calculators (text, embedding-based)
- Text processing utilities
"""

from ctxforge.utils.hashing import compute_content_hash
from ctxforge.utils.math import cosine_similarity
from ctxforge.utils.references import (
    build_reference_map,
    extract_references,
    format_as_citations,
    strip_references,
)
from ctxforge.utils.similarity import (
    AsyncToSyncAdapter,
    EmbeddingSimilarityCalculator,
    IAsyncSimilarityCalculator,
    ISimilarityCalculator,
    LevenshteinSimilarityCalculator,
    TextSimilarityCalculator,
    calculate_text_similarity,
    normalize_for_comparison,
)

__all__ = [
    # Math utilities
    "cosine_similarity",
    # Hashing
    "compute_content_hash",
    # References
    "extract_references",
    "strip_references",
    "format_as_citations",
    "build_reference_map",
    # Similarity calculators
    "ISimilarityCalculator",
    "IAsyncSimilarityCalculator",
    "TextSimilarityCalculator",
    "LevenshteinSimilarityCalculator",
    "EmbeddingSimilarityCalculator",
    "AsyncToSyncAdapter",
    "calculate_text_similarity",
    "normalize_for_comparison",
]

