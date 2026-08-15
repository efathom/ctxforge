"""
Memory reranker implementations.

Rerankers take initial retrieval results and reorder them
using additional scoring criteria for memory items.
"""

from datetime import datetime, timezone
from typing import List, Optional

from ctxforge.engine.registry import registry
from ctxforge.protocols.retriever import IReranker, RetrievalResult
from ctxforge.retrieval.rerankers.base import (
    BaseReranker,
    ContentSimilarityMixin,
    ThresholdFilterMixin,
)


class MemoryRerankerBase(BaseReranker[RetrievalResult]):
    """
    Base class for memory-specific rerankers.
    
    Implements the abstract methods from BaseReranker for
    RetrievalResult type (memory retrieval).
    """
    
    def _get_score(self, result: RetrievalResult) -> float:
        """Get the current score from a memory result."""
        return result.score
    
    def _create_result(
        self,
        original: RetrievalResult,
        new_score: float,
        method_suffix: str,
        extra_metadata: dict,
    ) -> RetrievalResult:
        """Create a new RetrievalResult with updated score."""
        return RetrievalResult(
            memory=original.memory,
            score=new_score,
            retrieval_method=f"{original.retrieval_method}+{method_suffix}",
            metadata={
                **(original.metadata or {}),
                **extra_metadata,
            },
        )


@registry.register_reranker("recency")
class RecencyReranker(MemoryRerankerBase, IReranker):
    """
    Reranker that boosts recent memories.
    
    Takes initial retrieval results and adjusts scores based on
    how recently the memories were created.
    
    Example:
        >>> reranker = RecencyReranker(recency_boost=0.2)
        >>> reranked = await reranker.rerank("query", initial_results)
    """
    
    def __init__(
        self,
        recency_boost: float = 0.2,
        max_age_days: float = 30.0,
    ):
        """
        Initialize with boost parameters.
        
        Args:
            recency_boost: Maximum boost for recent items (0-1)
            max_age_days: Age at which boost becomes zero
        """
        self._recency_boost = recency_boost
        self._max_age_days = max_age_days
    
    @property
    def name(self) -> str:
        return "recency"
    
    def _calculate_boost(self, result: RetrievalResult, query: str) -> float:
        """Calculate recency boost based on memory creation time."""
        now = datetime.now(timezone.utc)
        created_at = result.memory.created_at
        
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        age_days = (now - created_at).total_seconds() / 86400
        
        # Linear decay of boost
        if age_days <= self._max_age_days:
            return self._recency_boost * (1 - age_days / self._max_age_days)
        return 0.0


@registry.register_reranker("threshold")
class ScoreThresholdReranker(ThresholdFilterMixin, IReranker):
    """
    Reranker that filters results by score threshold.
    
    Simple reranker that removes results below a threshold
    and optionally caps the number of results.
    """
    
    def __init__(
        self,
        min_score: float = 0.5,
        max_results: Optional[int] = None,
    ):
        """
        Initialize with thresholds.
        
        Args:
            min_score: Minimum score to keep (0-1)
            max_results: Maximum results to return
        """
        self._min_score = min_score
        self._max_results = max_results
    
    @property
    def name(self) -> str:
        return "threshold"
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Filter results by score threshold."""
        # Filter by threshold
        filtered = self._filter_by_threshold(
            results,
            self._min_score,
            lambda r: r.score,
        )
        
        # Apply limits
        limit = top_k or self._max_results
        if limit:
            filtered = filtered[:limit]
        
        return filtered


@registry.register_reranker("diversity")
class DiversityReranker(ContentSimilarityMixin, IReranker):
    """
    Reranker that promotes diversity in results.
    
    Uses Maximal Marginal Relevance (MMR) to balance
    relevance with diversity, avoiding redundant results.
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
        return "diversity"
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank using MMR for diversity."""
        if len(results) <= 1:
            return results
        
        selected: List[RetrievalResult] = []
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
                            candidate.memory.content,
                            s.memory.content,
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
                selected.append(RetrievalResult(
                    memory=best_result.memory,
                    score=best_result.score,
                    retrieval_method=f"{best_result.retrieval_method}+diversity",
                    metadata={
                        **(best_result.metadata or {}),
                        "mmr_score": best_mmr,
                    },
                ))
                remaining.remove(best_result)
        
        return selected
