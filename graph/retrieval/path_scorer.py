"""
Node scoring and budget enforcement for graph path mining.

Scores nodes based on relevance signals (entity match, date match, keyword overlap,
recency) and enforces a node budget (min_nodes to max_nodes) using a threshold
percentage of the maximum score.

Scoring formula (per node):
- Target entity in node name or summary: +100
- Target entity in labels: +80
- Query date in content: +80
- Word overlap with query: +10 per overlapping word
- Keyword overlap (high-precision): +15 per overlapping keyword
- Node has timestamp: +5
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Set, Tuple

from ctxforge.config.base import GraphPathMiningConfig
from ctxforge.protocols.graph import GraphNode


def _extract_date_from_query(query: str) -> Optional[str]:
    """Extract a date string from a query if present."""
    query_lower = query.lower()
    date_match = re.search(
        r"(\d{1,2})\s*(st|nd|rd|th)?\s*"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s*,?\s*(\d{4})?",
        query_lower,
    )
    if date_match:
        return date_match.group(0)
    return None


def _node_text(node: GraphNode) -> str:
    """Combine all text fields of a node into a single lowercase string."""
    parts = [node.name or ""]
    if node.summary:
        parts.append(node.summary)
    for label in (node.labels or []):
        parts.append(label)
    for _key, val in (node.attributes or {}).items():
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, set)):
            parts.extend(str(v) for v in val)
    return " ".join(parts).lower()


def _node_keywords(node: GraphNode) -> Set[str]:
    """Extract keyword-like tokens from a node."""
    keywords: Set[str] = set()
    for key in ("keywords", "tags"):
        vals = (node.attributes or {}).get(key)
        if isinstance(vals, (list, set)):
            keywords.update(str(v).lower() for v in vals)
    for label in (node.labels or []):
        keywords.add(label.lower())
    keywords.add(node.name.lower())
    return keywords


def score_node(
    node: GraphNode,
    *,
    query_words: Set[str],
    target_entity: Optional[str] = None,
    query_date: Optional[str] = None,
) -> float:
    """
    Score a single node based on relevance to the query.

    Returns a non-negative float score.
    """
    score = 0.0
    text = _node_text(node)

    # 1. Target entity match (highest priority)
    if target_entity:
        target_lower = target_entity.lower()
        name_lower = (node.name or "").lower()
        labels_lower = [label.lower() for label in (node.labels or [])]

        if target_lower in name_lower or target_lower in text:
            score += 100
        elif target_lower in labels_lower:
            score += 80

    # 2. Date match
    if query_date and query_date in text:
        score += 80

    # 3. Word overlap with query
    text_words = set(text.split())
    overlap = len(query_words & text_words)
    score += overlap * 10

    # 4. Keyword overlap (high-precision signals)
    keywords = _node_keywords(node)
    keyword_overlap = len(query_words & keywords)
    score += keyword_overlap * 15

    # 5. Recency bonus
    for key in ("created_at", "timestamp"):
        if isinstance((node.attributes or {}).get(key), datetime):
            score += 5
            break

    return score


def rank_and_limit_nodes(
    nodes: List[GraphNode],
    *,
    query: str,
    target_entity: Optional[str] = None,
    config: GraphPathMiningConfig,
) -> List[GraphNode]:
    """
    Score all nodes, apply threshold filtering, and enforce the node budget.

    Returns a list of nodes within [min_nodes, max_nodes] bounds.
    """
    if not nodes:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())
    query_date = _extract_date_from_query(query)

    scored: List[Tuple[GraphNode, float]] = [
        (node, score_node(node, query_words=query_words, target_entity=target_entity, query_date=query_date))
        for node in nodes
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    max_score = scored[0][1] if scored else 0
    threshold = max_score * config.node_score_threshold_pct

    # Filter by threshold
    relevant = [(n, s) for n, s in scored if s >= threshold]

    # Apply strict bounds
    if len(relevant) < config.min_nodes:
        relevant = scored[: min(config.min_nodes, len(scored))]
    elif len(relevant) > config.max_nodes:
        relevant = relevant[: config.max_nodes]

    return [n for n, _ in relevant]
