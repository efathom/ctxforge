"""
Expertise reranker implementations.

Rerankers for expertise retrieval results that use
effectiveness metrics and other expertise-specific criteria.
"""

from typing import List, Optional

from ctxforge.core.expertise import ExpertiseItem
from ctxforge.protocols.context import IContextReranker
from ctxforge.retrieval.rerankers.base import (
    BaseReranker,
    ContentSimilarityMixin,
)
from ctxforge.retrieval.retrievers.expertise import ExpertiseRetrievalResult


class ExpertiseRerankerBase(BaseReranker[ExpertiseRetrievalResult]):
    """
    Base class for expertise-specific rerankers.
    
    Implements the abstract methods from BaseReranker for
    ExpertiseRetrievalResult type (expertise retrieval).
    """
    
    def _get_score(self, result: ExpertiseRetrievalResult) -> float:
        """Get the current score from an expertise result."""
        return result.score
    
    def _create_result(
        self,
        original: ExpertiseRetrievalResult,
        new_score: float,
        method_suffix: str,
        extra_metadata: dict,
    ) -> ExpertiseRetrievalResult:
        """Create a new ExpertiseRetrievalResult with updated score."""
        return ExpertiseRetrievalResult(
            item=original.item,
            score=new_score,
            retrieval_method=f"{original.retrieval_method}+{method_suffix}",
            metadata={
                **original.metadata,
                **extra_metadata,
            },
        )


class EffectivenessReranker(ExpertiseRerankerBase, IContextReranker[ExpertiseItem]):
    """
    Reranker that prioritizes expertise items by effectiveness.
    
    Implements IExpertiseReranker protocol.
    Combines the semantic similarity score with effectiveness metrics
    to promote items that have proven helpful in practice.
    
    Example:
        >>> reranker = EffectivenessReranker(helpful_weight=1.5, harmful_weight=-3.0)
        >>> reranked = await reranker.rerank(query, results)
    """
    
    def __init__(
        self,
        helpful_weight: float = 1.0,
        harmful_weight: float = -2.0,
        min_usage_for_boost: int = 3,
        recency_factor: float = 0.1,
    ):
        """
        Initialize the reranker.
        
        Args:
            helpful_weight: Weight for helpful count contribution
            harmful_weight: Weight for harmful count contribution (typically negative)
            min_usage_for_boost: Minimum usage count before effectiveness affects ranking
            recency_factor: Boost for recently updated items
        """
        self._helpful_weight = helpful_weight
        self._harmful_weight = harmful_weight
        self._min_usage = min_usage_for_boost
        self._recency_factor = recency_factor
    
    @property
    def name(self) -> str:
        """The name of this reranker."""
        return "effectiveness"
    
    def _calculate_boost(self, result: ExpertiseRetrievalResult, query: str) -> float:
        """Calculate effectiveness boost based on usage statistics."""
        item = result.item
        
        if item.total_usage >= self._min_usage:
            return self._calculate_effectiveness_boost(item)
        return 0.0
    
    def _calculate_effectiveness_boost(self, item: ExpertiseItem) -> float:
        """
        Calculate effectiveness boost for an item.
        
        Uses a sigmoid-like normalization to prevent extreme values.
        """
        # Raw effectiveness score
        helpful_contrib = item.helpful_count * self._helpful_weight
        harmful_contrib = item.harmful_count * self._harmful_weight
        
        raw_boost = (helpful_contrib + harmful_contrib) / (item.total_usage + 1)
        
        # Normalize to [-0.3, 0.3] range
        normalized_boost = max(-0.3, min(0.3, raw_boost * 0.1))
        
        return normalized_boost


class UsageRecencyReranker(ExpertiseRerankerBase):
    """
    Reranker that boosts recently used expertise items.
    
    Items that have been used more recently get a boost,
    which can help surface "hot" knowledge items.
    """
    
    def __init__(
        self,
        recency_boost: float = 0.15,
        max_age_days: float = 14.0,
    ):
        """
        Initialize with boost parameters.
        
        Args:
            recency_boost: Maximum boost for recently used items (0-1)
            max_age_days: Age at which boost becomes zero
        """
        self._recency_boost = recency_boost
        self._max_age_seconds = max_age_days * 86400
    
    @property
    def name(self) -> str:
        return "usage_recency"
    
    def _calculate_boost(self, result: ExpertiseRetrievalResult, query: str) -> float:
        """Calculate recency boost based on last update time."""
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        updated_at = result.item.updated_at
        
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        
        age_seconds = (now - updated_at).total_seconds()
        
        if age_seconds <= self._max_age_seconds:
            return self._recency_boost * (1 - age_seconds / self._max_age_seconds)
        return 0.0


class ExpertiseDiversityReranker(ContentSimilarityMixin, ExpertiseRerankerBase):
    """
    Reranker that promotes diversity in expertise results.
    
    Uses Maximal Marginal Relevance (MMR) to balance relevance
    with diversity, avoiding redundant expertise items.
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize with diversity weight.
        
        Args:
            diversity_weight: Balance between relevance and diversity (0-1)
                0 = pure relevance, 1 = pure diversity
        """
        self._diversity_weight = diversity_weight
    
    @property
    def name(self) -> str:
        return "expertise_diversity"
    
    def _calculate_boost(self, result: ExpertiseRetrievalResult, query: str) -> float:
        """Not used for MMR-based reranking."""
        return 0.0
    
    async def rerank(
        self,
        query: str,
        results: List[ExpertiseRetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[ExpertiseRetrievalResult]:
        """Rerank using MMR for diversity."""
        if len(results) <= 1:
            return results
        
        selected: List[ExpertiseRetrievalResult] = []
        remaining = list(results)
        limit = top_k or len(results)
        
        while remaining and len(selected) < limit:
            best_result = None
            best_mmr = float('-inf')
            
            for candidate in remaining:
                # Relevance component
                relevance = candidate.score
                
                # Diversity component (max similarity to already selected)
                if selected:
                    max_similarity = max(
                        self._compute_content_similarity(
                            candidate.item.content,
                            s.item.content,
                        )
                        for s in selected
                    )
                else:
                    max_similarity = 0.0
                
                # MMR score
                mmr = (1 - self._diversity_weight) * relevance - \
                      self._diversity_weight * max_similarity
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_result = candidate
            
            if best_result:
                selected.append(self._create_result(
                    original=best_result,
                    new_score=best_result.score,
                    method_suffix=self.name,
                    extra_metadata={"mmr_score": best_mmr},
                ))
                remaining.remove(best_result)
        
        return selected
