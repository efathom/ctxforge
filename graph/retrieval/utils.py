from __future__ import annotations

"""
Utilities for graph retrieval.

This module contains small, deterministic helpers used by the engine when building
hybrid graph retrieval results (e.g., combining multiple rankings).
"""

from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def rrf_scores(rankings: Sequence[Sequence[str]], *, k: int = 60) -> Dict[str, float]:
    """
    Compute Reciprocal Rank Fusion (RRF) scores for ids across multiple rankings.

    For each ranking list, an id at rank r (0-based) contributes:
        score += 1 / (k + r + 1)

    Args:
        rankings: Ordered sequences of ids (best first).
        k: RRF constant controlling how quickly rank contributions decay.

    Returns:
        Mapping from id -> fused score.
    """
    k = max(1, int(k))
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for r, _id in enumerate(ranking):
            if not _id:
                continue
            scores[_id] = scores.get(_id, 0.0) + 1.0 / float(k + r + 1)
    return scores


def stable_unique(items: Iterable[T], *, key_fn) -> List[T]:
    """Return items deduped by `key_fn` while preserving first-seen order."""
    seen = set()
    out: List[T] = []
    for it in items:
        k = key_fn(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def order_ids_by_score(scores: Dict[str, float], *, tie_breaker: Sequence[str] | None = None) -> List[str]:
    """
    Order ids by descending score, with a stable tie-breaker ordering when provided.
    """
    tb_rank: Dict[str, int] = {}
    if tie_breaker is not None:
        tb_rank = {x: i for i, x in enumerate(tie_breaker) if x}

    def sort_key(item: Tuple[str, float]) -> Tuple[float, int]:
        _id, s = item
        return (s, -tb_rank.get(_id, -10**9))

    ordered = sorted(scores.items(), key=sort_key, reverse=True)
    return [i for i, _ in ordered]


