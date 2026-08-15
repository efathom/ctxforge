"""
Base reranker implementation.

Provides abstract base class and shared utilities for rerankers.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

# Type variable for generic reranker results
T = TypeVar('T')


@dataclass
class RerankedResult(Generic[T]):
    """
    Generic reranked result wrapper.
    
    Wraps any retrieval result type with reranking metadata.
    """
    result: T
    original_score: float
    new_score: float
    boost: float
    method_suffix: str


class BaseReranker(ABC, Generic[T]):
    """
    Abstract base class for all rerankers.
    
    Provides common functionality:
    - Score normalization
    - Result limiting
    - Common utilities
    
    Subclasses must implement:
    - name property
    - _calculate_boost method
    - _get_score method
    - _create_result method
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this reranker."""
        ...
    
    @abstractmethod
    def _get_score(self, result: T) -> float:
        """Get the current score from a result."""
        ...
    
    @abstractmethod
    def _calculate_boost(self, result: T, query: str) -> float:
        """Calculate the boost value for a result."""
        ...
    
    @abstractmethod
    def _create_result(
        self,
        original: T,
        new_score: float,
        method_suffix: str,
        extra_metadata: dict,
    ) -> T:
        """Create a new result with updated score and metadata."""
        ...
    
    async def rerank(
        self,
        query: str,
        results: List[T],
        top_k: Optional[int] = None,
    ) -> List[T]:
        """
        Rerank results using the boost calculation.
        
        Args:
            query: The original query
            results: The initial retrieval results
            top_k: Optional limit on results to return
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        reranked: List[T] = []
        
        for result in results:
            original_score = self._get_score(result)
            boost = self._calculate_boost(result, query)
            new_score = self._normalize_score(original_score + boost)
            
            reranked.append(self._create_result(
                original=result,
                new_score=new_score,
                method_suffix=self.name,
                extra_metadata={
                    "original_score": original_score,
                    f"{self.name}_boost": boost,
                },
            ))
        
        # Sort by new score
        reranked.sort(key=lambda r: self._get_score(r), reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0.0, 1.0] range."""
        return max(0.0, min(1.0, score))


class ContentSimilarityMixin:
    """
    Mixin providing content similarity calculation.
    
    Can be used by diversity-based rerankers for any item type.
    """
    
    def _compute_content_similarity(self, content_a: str, content_b: str) -> float:
        """
        Compute simple content similarity based on word overlap (Jaccard).
        
        Args:
            content_a: First content string
            content_b: Second content string
            
        Returns:
            Similarity score between 0 and 1
        """
        words_a = set(re.findall(r'\w+', content_a.lower()))
        words_b = set(re.findall(r'\w+', content_b.lower()))
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0


class ThresholdFilterMixin:
    """
    Mixin providing threshold filtering functionality.
    
    Can be used by any reranker that needs to filter by score.
    """
    
    def _filter_by_threshold(
        self,
        results: List[T],
        min_score: float,
        score_getter,
    ) -> List[T]:
        """
        Filter results by minimum score threshold.
        
        Args:
            results: Results to filter
            min_score: Minimum score to keep
            score_getter: Function to get score from result
            
        Returns:
            Filtered results
        """
        return [r for r in results if score_getter(r) >= min_score]

