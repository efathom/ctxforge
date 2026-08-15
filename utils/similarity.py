"""
Similarity Calculator Strategy Pattern.

Provides a protocol and implementations for calculating text similarity.
This allows for flexible similarity algorithms (text-based, embedding-based, etc.)
to be injected into extractors, consolidators, and retrievers.
"""

import re
from difflib import SequenceMatcher
from typing import Awaitable, Callable, List, Protocol, runtime_checkable

from ctxforge.utils.math import cosine_similarity


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for similarity comparison.
    
    More aggressive normalization for deduplication.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized lowercase text
    """
    if not text:
        return ""
    
    # Strip and collapse whitespace
    text = re.sub(r'\s+', ' ', text.strip()).lower()
    
    # Remove common filler words for better matching
    fillers = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'that', 'this']
    words = text.split()
    words = [w for w in words if w not in fillers]
    
    return ' '.join(words)


@runtime_checkable
class ISimilarityCalculator(Protocol):
    """
    Protocol for text similarity calculation.
    
    Implementations can use various algorithms:
    - Text-based (SequenceMatcher, Levenshtein, etc.)
    - Embedding-based (cosine similarity of vectors)
    - Hybrid approaches
    
    All implementations should return a score between 0.0 and 1.0,
    where 1.0 means identical and 0.0 means completely different.
    """
    
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        ...


@runtime_checkable
class IAsyncSimilarityCalculator(Protocol):
    """
    Async protocol for text similarity calculation.
    
    Use this for similarity calculations that require async operations,
    such as fetching embeddings from an API.
    """
    
    async def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts asynchronously.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        ...


class TextSimilarityCalculator(ISimilarityCalculator):
    """
    Default text-based similarity calculator.
    
    Uses Python's SequenceMatcher for a quick similarity score.
    Good for detecting near-duplicates in text.
    
    Example:
        calculator = TextSimilarityCalculator()
        score = calculator.calculate("I love coffee", "I really love coffee")
        # Returns ~0.85
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize the calculator.
        
        Args:
            normalize: Whether to normalize text before comparison
        """
        self._normalize = normalize
    
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using SequenceMatcher.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        if self._normalize:
            t1 = normalize_for_comparison(text1)
            t2 = normalize_for_comparison(text2)
        else:
            t1 = text1
            t2 = text2
        
        if not t1 or not t2:
            return 0.0
        
        return SequenceMatcher(None, t1, t2).ratio()


class LevenshteinSimilarityCalculator(ISimilarityCalculator):
    """
    Levenshtein distance-based similarity calculator.
    
    Uses edit distance for similarity. Requires the `rapidfuzz` or
    `python-Levenshtein` package for optimal performance.
    
    Falls back to SequenceMatcher if not available.
    
    Example:
        calculator = LevenshteinSimilarityCalculator()
        score = calculator.calculate("hello", "hallo")
        # Returns 0.8 (1 edit out of 5 characters)
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize the calculator.
        
        Args:
            normalize: Whether to normalize text before comparison
        """
        self._normalize = normalize
        self._use_rapidfuzz = False
        
        try:
            from rapidfuzz import fuzz
            self._fuzz = fuzz
            self._use_rapidfuzz = True
        except ImportError:
            pass
    
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using Levenshtein distance.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        if self._normalize:
            t1 = normalize_for_comparison(text1)
            t2 = normalize_for_comparison(text2)
        else:
            t1 = text1
            t2 = text2
        
        if not t1 or not t2:
            return 0.0
        
        if self._use_rapidfuzz:
            # rapidfuzz returns 0-100, normalize to 0-1
            return self._fuzz.ratio(t1, t2) / 100.0
        else:
            # Fallback to SequenceMatcher
            return SequenceMatcher(None, t1, t2).ratio()


class EmbeddingSimilarityCalculator(IAsyncSimilarityCalculator):
    """
    Embedding-based similarity calculator.
    
    Uses vector embeddings for semantic similarity.
    More accurate for meaning-based comparison but requires
    an embedding provider.
    
    Example:
        from ctxforge.llm import OpenAIEmbeddingProvider
        
        embedder = OpenAIEmbeddingProvider(api_key="...")
        calculator = EmbeddingSimilarityCalculator(embedder)
        
        score = await calculator.calculate(
            "I love coffee",
            "I enjoy drinking espresso"
        )
        # Returns high score due to semantic similarity
    """
    
    def __init__(
        self,
        embedding_func: Callable[[str], Awaitable[List[float]]],
        cache_embeddings: bool = True,
    ):
        """
        Initialize the calculator.
        
        Args:
            embedding_func: Async function that returns embeddings for text
            cache_embeddings: Whether to cache embeddings for repeated texts
        """
        self._embedding_func = embedding_func
        self._cache_embeddings = cache_embeddings
        self._cache: dict = {}
    
    async def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using embedding cosine similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Get embeddings (with optional caching)
        emb1 = await self._get_embedding(text1)
        emb2 = await self._get_embedding(text2)
        
        if not emb1 or not emb2:
            return 0.0
        
        similarity = cosine_similarity(emb1, emb2)
        # Clamp to [0, 1] range (cosine similarity can be [-1, 1])
        return max(0.0, min(1.0, similarity))
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        if self._cache_embeddings and text in self._cache:
            return self._cache[text]
        
        embedding = await self._embedding_func(text)
        
        if self._cache_embeddings:
            self._cache[text] = embedding
        
        return embedding
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


class AsyncToSyncAdapter(ISimilarityCalculator):
    """
    Adapter to use async calculators in sync contexts.
    
    Warning: This runs the event loop synchronously and should
    only be used when async is not available.
    
    Example:
        async_calc = EmbeddingSimilarityCalculator(embedder)
        sync_calc = AsyncToSyncAdapter(async_calc)
        
        score = sync_calc.calculate("text1", "text2")
    """
    
    def __init__(self, async_calculator: IAsyncSimilarityCalculator):
        """
        Initialize the adapter.
        
        Args:
            async_calculator: The async calculator to wrap
        """
        self._async_calculator = async_calculator
    
    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity by running async calculator synchronously.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't run sync in running loop, return 0
                return 0.0
            return loop.run_until_complete(
                self._async_calculator.calculate(text1, text2)
            )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self._async_calculator.calculate(text1, text2)
            )


# Default calculator instance for convenience
default_similarity_calculator = TextSimilarityCalculator()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using the default calculator.
    
    This is a convenience function that uses TextSimilarityCalculator.
    For more control, instantiate a calculator directly.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    return default_similarity_calculator.calculate(text1, text2)

