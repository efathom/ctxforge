"""
Rank fusion utilities for combining results from multiple retrieval sources.
"""

from typing import Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: List[List[T]],
    key_fn: Callable[[T], str],
    k: int = 60,
    limit: Optional[int] = None,
) -> List[T]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each item's score is ``sum(1 / (k + rank))`` across all lists it appears in,
    where ``rank`` is 1-based.

    Args:
        ranked_lists: Ordered result lists from different retrieval sources.
        key_fn: Function to extract a unique identifier from each item.
        k: Smoothing constant (default 60, per the original RRF paper).
        limit: Maximum number of items to return.  ``None`` returns all.

    Returns:
        Merged list ordered by descending RRF score.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, T] = {}

    for ranked in ranked_lists:
        for rank_idx, item in enumerate(ranked):
            item_key = key_fn(item)
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + rank_idx + 1)
            if item_key not in items:
                items[item_key] = item

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    result = [items[k_] for k_ in sorted_keys]
    if limit is not None:
        result = result[:limit]
    return result
