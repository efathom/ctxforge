"""
Graph query planner.

Adds a light planning step that decides how to seed / expand
graph retrieval:
- local: entity-first (low-level keywords) + BFS expansion
- global: fact/relation-first (high-level keywords) + endpoint expansion
- hybrid: combine both
- auto: choose based on extracted keywords + fallbacks

This module is intentionally dependency-light; it uses a heuristic keyword extractor by
default (no LLM required).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Sequence

GraphRetrievalMode = Literal["local", "global", "hybrid"]


@dataclass(frozen=True)
class GraphKeywords:
    low_level: List[str]
    high_level: List[str]


@dataclass(frozen=True)
class GraphQueryPlan:
    mode: GraphRetrievalMode
    keywords: GraphKeywords
    reason: str


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']{1,}")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "to",
    "of",
    "in",
    "on",
    "with",
    "at",
    "by",
    "from",
    "about",
    "into",
    "over",
    "after",
    "before",
    "between",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "as",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "them",
    "us",
    "my",
    "your",
    "their",
    "our",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "please",
}


def _unique_preserve(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def extract_keywords_heuristic(
    query: str,
    *,
    max_low_level: int,
    max_high_level: int,
) -> GraphKeywords:
    """
    Best-effort keyword extraction without an LLM.

    - low_level: "entity-ish" tokens (capitalized words, alphanumerics, longer terms)
    - high_level: remaining salient tokens (excluding stopwords + low_level)
    """
    text = (query or "").strip()
    if not text:
        return GraphKeywords(low_level=[], high_level=[])

    tokens = _WORD_RE.findall(text)
    if not tokens:
        return GraphKeywords(low_level=[], high_level=[])

    low_candidates: List[str] = []
    high_candidates: List[str] = []

    for t in tokens:
        tl = t.lower()
        if tl in _STOPWORDS:
            continue

        # Prefer entity-ish signals for low-level keywords.
        is_cap = t[0].isupper()
        is_long = len(t) >= 6
        has_digit = any(ch.isdigit() for ch in t)
        if is_cap or has_digit or is_long:
            low_candidates.append(t)
        else:
            high_candidates.append(t)

    low = _unique_preserve(low_candidates)[: max(0, int(max_low_level))]

    low_set = set([x.lower() for x in low])
    high = [t for t in high_candidates if t.lower() not in low_set]
    high = _unique_preserve(high)[: max(0, int(max_high_level))]

    return GraphKeywords(low_level=low, high_level=high)


def plan_mode(
    *,
    planner_mode: str,
    keywords: GraphKeywords,
    fallback_to_global_if_no_entities: bool,
    fallback_to_local_if_no_themes: bool,
) -> GraphQueryPlan:
    pm = (planner_mode or "auto").strip().lower()
    if pm in ("local", "global", "hybrid"):
        return GraphQueryPlan(
            mode=pm,  # type: ignore[return-value]
            keywords=keywords,
            reason=f"forced planner_mode={pm}",
        )

    # auto
    if not keywords.low_level and fallback_to_global_if_no_entities:
        return GraphQueryPlan(mode="global", keywords=keywords, reason="auto: no low-level keywords -> global")
    if not keywords.high_level and fallback_to_local_if_no_themes:
        return GraphQueryPlan(mode="local", keywords=keywords, reason="auto: no high-level keywords -> local")
    return GraphQueryPlan(mode="hybrid", keywords=keywords, reason="auto: both keyword sets present -> hybrid")


