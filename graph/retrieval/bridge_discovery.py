"""
Bridge candidate discovery for disconnected graph regions.

When two nodes in a query-specific subgraph are chronologically adjacent but have no
direct entity or temporal link, bridge discovery searches for an intermediate node
that connects them. This is a Steiner-tree approximation: find b* such that
b* = argmax cos(E(q_ij), v_m) subject to t_m in [t_i, t_j].

Three search strategies are tried in order:
1. Entity-based: query using entities unique to each endpoint
2. Keyword combination: union of top keywords/entities from both endpoints
3. Content-based: extract salient nouns from node content

The bridge candidate must be temporally between the two endpoints (or within a
configurable proximity window of either).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Set, Tuple

from ctxforge.config.base import GraphPathMiningConfig
from ctxforge.graph.retrieval.types import BridgeConnection
from ctxforge.protocols.graph import (
    GraphEdge,
    GraphNode,
    IGraphStore,
)
from ctxforge.protocols.llm import IEmbeddingProvider

logger = logging.getLogger(__name__)


def _parse_datetime(ts: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to timezone-naive UTC for comparison."""
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _hours_between(t1: Optional[datetime], t2: Optional[datetime]) -> Optional[float]:
    """Return the absolute hours between two datetimes, or None if either is missing."""
    dt1 = _parse_datetime(t1)
    dt2 = _parse_datetime(t2)
    if dt1 is None or dt2 is None:
        return None
    return abs((dt1 - dt2).total_seconds()) / 3600.0


def _is_between(ts: Optional[datetime], ts1: Optional[datetime], ts2: Optional[datetime]) -> bool:
    """Check if ts is temporally between ts1 and ts2."""
    t = _parse_datetime(ts)
    t1 = _parse_datetime(ts1)
    t2 = _parse_datetime(ts2)
    if t is None or t1 is None or t2 is None:
        return False
    lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
    return lo <= t <= hi


def _node_entities(node: GraphNode) -> Set[str]:
    """Extract entity-like identifiers from a node."""
    entities: Set[str] = set()
    entities.add(node.name)
    for label in (node.labels or []):
        entities.add(label)
    for key in ("persons", "entities", "keywords"):
        vals = (node.attributes or {}).get(key)
        if isinstance(vals, (list, set)):
            entities.update(str(v) for v in vals)
    return entities


def _node_keywords(node: GraphNode) -> Set[str]:
    """Extract keyword-like tokens from a node."""
    keywords: Set[str] = set()
    keywords.add(node.name)
    if node.summary:
        for word in node.summary.split():
            if len(word) > 3 and word[0].isupper():
                keywords.add(word)
    for key in ("keywords",):
        vals = (node.attributes or {}).get(key)
        if isinstance(vals, (list, set)):
            keywords.update(str(v) for v in vals)
    return keywords


def check_connection(
    n1: GraphNode,
    n2: GraphNode,
    edges: List[GraphEdge],
    *,
    temporal_flow_hours: float = 6.0,
) -> Optional[str]:
    """
    Check if two nodes are directly connected.

    Returns the connection type string if connected, or None if disconnected.
    Checks:
    1. Explicit edge between the two nodes
    2. Entity overlap (shared names/labels/attributes)
    3. Temporal proximity (within ``temporal_flow_hours``)
    """
    # Check explicit edges
    for edge in edges:
        if (edge.source_node_id == n1.node_id and edge.target_node_id == n2.node_id) or \
           (edge.source_node_id == n2.node_id and edge.target_node_id == n1.node_id):
            return "edge_link"

    # Check entity overlap
    e1 = _node_entities(n1)
    e2 = _node_entities(n2)
    if e1.intersection(e2):
        return "entity_link"

    # Check temporal proximity
    t1 = (n1.attributes or {}).get("created_at") or (n1.attributes or {}).get("timestamp")
    t2 = (n2.attributes or {}).get("created_at") or (n2.attributes or {}).get("timestamp")
    if isinstance(t1, datetime) and isinstance(t2, datetime):
        hours = _hours_between(t1, t2)
        if hours is not None and hours < temporal_flow_hours:
            return "temporal_flow"

    return None


async def find_bridge_candidates(
    *,
    scope_id: str,
    disconnected_pairs: List[Tuple[GraphNode, GraphNode]],
    graph_store: IGraphStore,
    embedding_provider: Optional[IEmbeddingProvider],
    config: GraphPathMiningConfig,
    existing_node_ids: Set[str],
) -> Tuple[List[GraphNode], List[BridgeConnection]]:
    """
    Search for bridge nodes that connect disconnected node pairs.

    For each disconnected pair (n1, n2), tries three search strategies:
    1. Entity-based: query using entities unique to each endpoint
    2. Keyword combination: union of keywords/entities from both endpoints
    3. Content-based: salient nouns from node summaries

    Returns:
        Tuple of (bridge_nodes, bridge_connections) discovered.
    """
    if not disconnected_pairs or not config.bridge_discovery_enabled:
        return [], []

    bridge_nodes: List[GraphNode] = []
    bridge_connections: List[BridgeConnection] = []
    seen_bridge_ids: Set[str] = set(existing_node_ids)

    for n1, n2 in disconnected_pairs:
        # Check temporal window
        t1 = (n1.attributes or {}).get("created_at") or (n1.attributes or {}).get("timestamp")
        t2 = (n2.attributes or {}).get("created_at") or (n2.attributes or {}).get("timestamp")
        if isinstance(t1, datetime) and isinstance(t2, datetime):
            hours = _hours_between(t1, t2)
            if hours is not None and (hours <= 1.0 or hours >= config.temporal_window_hours):
                continue
        # If no timestamps, still try bridge discovery

        bridge = await _find_single_bridge(
            scope_id=scope_id,
            n1=n1,
            n2=n2,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            config=config,
            exclude_ids=seen_bridge_ids,
        )
        if bridge is not None:
            seen_bridge_ids.add(bridge.node_id)
            bridge_nodes.append(bridge)
            bridge_connections.append(
                BridgeConnection(
                    source_node_id=n1.node_id,
                    bridge_node_id=bridge.node_id,
                    target_node_id=n2.node_id,
                    bridge_type="inferred",
                )
            )

    logger.debug(
        "Bridge discovery: %d pairs checked, %d bridges found",
        len(disconnected_pairs),
        len(bridge_nodes),
    )
    return bridge_nodes, bridge_connections


async def _find_single_bridge(
    *,
    scope_id: str,
    n1: GraphNode,
    n2: GraphNode,
    graph_store: IGraphStore,
    embedding_provider: Optional[IEmbeddingProvider],
    config: GraphPathMiningConfig,
    exclude_ids: Set[str],
) -> Optional[GraphNode]:
    """Try three strategies to find a bridge node between n1 and n2."""

    e1 = _node_entities(n1)
    e2 = _node_entities(n2)
    k1 = _node_keywords(n1)
    k2 = _node_keywords(n2)

    # Strategy 1: Entity-based bridge
    shared = e1 & e2
    bridge_entities = (e1 | e2) - shared
    if bridge_entities:
        query = " ".join(list(bridge_entities)[:4])
        candidate = await _search_bridge_candidate(
            scope_id=scope_id,
            query=query,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            config=config,
            exclude_ids=exclude_ids,
            n1=n1,
            n2=n2,
        )
        if candidate is not None:
            return candidate

    # Strategy 2: Keyword combination
    all_entities = list(e1 | e2)[:3]
    all_keywords = list(k1 | k2)[:5]
    if all_entities or all_keywords:
        query = " ".join(all_entities + all_keywords)
        candidate = await _search_bridge_candidate(
            scope_id=scope_id,
            query=query,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            config=config,
            exclude_ids=exclude_ids,
            n1=n1,
            n2=n2,
        )
        if candidate is not None:
            return candidate

    # Strategy 3: Content-based (salient nouns from summaries)
    words1 = [w for w in (n1.summary or "").split() if len(w) > 4 and w[0].isupper()][:2]
    words2 = [w for w in (n2.summary or "").split() if len(w) > 4 and w[0].isupper()][:2]
    if words1 or words2:
        query = " ".join(words1 + words2)
        candidate = await _search_bridge_candidate(
            scope_id=scope_id,
            query=query,
            graph_store=graph_store,
            embedding_provider=embedding_provider,
            config=config,
            exclude_ids=exclude_ids,
            n1=n1,
            n2=n2,
        )
        if candidate is not None:
            return candidate

    return None


async def _search_bridge_candidate(
    *,
    scope_id: str,
    query: str,
    graph_store: IGraphStore,
    embedding_provider: Optional[IEmbeddingProvider],
    config: GraphPathMiningConfig,
    exclude_ids: Set[str],
    n1: GraphNode,
    n2: GraphNode,
) -> Optional[GraphNode]:
    """
    Search for a bridge candidate using semantic or keyword search.

    Time-aware filtering: prefer candidates temporally between the two endpoints,
    or within proximity_hours of either endpoint.
    """
    candidates: List[GraphNode] = []

    # Try semantic search first (if embedding provider is available)
    if embedding_provider is not None:
        try:
            qv = await embedding_provider.embed_single(query)
            if qv:
                semantic_results = await graph_store.search_nodes_semantic(
                    scope_id,
                    qv,
                    limit=config.bridge_search_top_k,
                )
                candidates.extend(semantic_results)
        except Exception:
            pass

    # Fall back to keyword search
    if not candidates:
        try:
            result = await graph_store.search(
                scope_id,
                query,
                scope="nodes",
                limit=config.bridge_search_top_k,
            )
            candidates.extend(result.nodes)
        except Exception:
            pass

    # Filter and select the best candidate
    t1 = (n1.attributes or {}).get("created_at") or (n1.attributes or {}).get("timestamp")
    t2 = (n2.attributes or {}).get("created_at") or (n2.attributes or {}).get("timestamp")

    for candidate in candidates:
        if candidate.node_id in exclude_ids:
            continue

        t_bridge = (candidate.attributes or {}).get("created_at") or (candidate.attributes or {}).get("timestamp")

        # If timestamps are available, apply time-aware filtering
        if isinstance(t1, datetime) and isinstance(t2, datetime) and isinstance(t_bridge, datetime):
            if _is_between(t_bridge, t1, t2):
                return candidate
            dt1 = _hours_between(t_bridge, t1)
            dt2 = _hours_between(t_bridge, t2)
            if dt1 is not None and dt2 is not None:
                if dt1 < config.bridge_proximity_hours or dt2 < config.bridge_proximity_hours:
                    return candidate
        else:
            # No timestamp info - accept first valid candidate
            return candidate

    return None
