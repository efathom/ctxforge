"""
Generic greedy budget packing utility.

Given a list of items, a token budget, and functions to extract text
and count tokens, greedily packs items in order until the budget is
exhausted.
"""

from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


def _default_token_count(text: str) -> int:
    """Approximate token count using word-split heuristic."""
    return len(text.split())


def budget_pack(
    items: List[T],
    budget: int,
    text_fn: Callable[[T], str],
    token_fn: Optional[Callable[[str], int]] = None,
) -> List[T]:
    """Greedily pack items into a token budget.

    Items are considered in the order given.  Each item is included if its
    token cost fits within the remaining budget.  Items that exceed the
    remaining budget are skipped (not truncated).

    Args:
        items: Candidate items in priority order (highest priority first).
        budget: Maximum token budget.
        text_fn: Extracts the text representation from an item.
        token_fn: Counts tokens for a text string.
            Defaults to a word-split heuristic.

    Returns:
        Subset of *items* that fit within *budget*, preserving order.
    """
    if budget <= 0 or not items:
        return []

    count_fn = token_fn or _default_token_count
    packed: List[T] = []
    remaining = budget

    for item in items:
        text = text_fn(item)
        cost = count_fn(text)
        if cost <= remaining:
            packed.append(item)
            remaining -= cost

    return packed
