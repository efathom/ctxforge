"""
Multi-hop path mining via DFS.

Discovers reasoning paths through a query-specific subgraph. Given a set of nodes
and edges, enumerates all paths of length [min_path_length, max_path_depth] using
depth-first search from every node.

Paths are:
- Deduplicated by their ordered node ID tuple
- Ranked by length (longer first), then by earliest timestamp (earlier first)
- Capped at max_paths

This implements the path mining step 
P_q = {p in G_q | min_len <= |p| <= L, temporally consistent}
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from ctxforge.config.base import GraphPathMiningConfig
from ctxforge.graph.retrieval.types import ReasoningPath
from ctxforge.protocols.graph import GraphEdge, GraphNode

logger = logging.getLogger(__name__)


def _get_node_timestamp(node: GraphNode) -> Optional[datetime]:
    """Extract a timestamp from a node's attributes."""
    for key in ("created_at", "timestamp", "valid_at"):
        val = (node.attributes or {}).get(key)
        if isinstance(val, datetime):
            return val
    return None


def _build_adjacency(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, GraphNode]]:
    """
    Build an adjacency list from nodes and edges.

    Returns:
        (adj, node_map) where adj maps node_id -> [(neighbor_id, edge_type), ...]
        and node_map maps node_id -> GraphNode.
    """
    node_map: Dict[str, GraphNode] = {n.node_id: n for n in nodes}
    node_ids = set(node_map.keys())

    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        src = edge.source_node_id
        tgt = edge.target_node_id
        if src in node_ids and tgt in node_ids:
            adj[src].append((tgt, edge.edge_type))
            # Also add reverse direction for undirected traversal
            adj[tgt].append((src, edge.edge_type))

    return adj, node_map


def discover_reasoning_paths(
    *,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    config: GraphPathMiningConfig,
) -> List[ReasoningPath]:
    """
    Discover reasoning paths via DFS from every node.

    Args:
        nodes: The subgraph nodes to traverse.
        edges: The subgraph edges defining connectivity.
        config: Path mining configuration (depth, limits, etc.).

    Returns:
        A list of ReasoningPath objects, ranked by length (longer first)
        then by earliest timestamp (earlier first), deduplicated and capped.
    """
    if not nodes or not edges:
        return []

    adj, node_map = _build_adjacency(nodes, edges)
    max_depth = config.max_path_depth
    min_length = config.min_path_length

    all_paths: List[Tuple[List[str], List[str]]] = []  # (node_ids, edge_types)

    def dfs(
        current_id: str,
        path_ids: List[str],
        path_edge_types: List[str],
        visited: Set[str],
    ) -> None:
        """DFS to discover reasoning paths."""
        # Stop condition: reached max depth
        if len(path_ids) >= max_depth:
            if len(path_ids) >= min_length:
                all_paths.append((path_ids[:], path_edge_types[:]))
            return

        # Check if current node has any unvisited neighbors
        neighbors = adj.get(current_id, [])
        has_unvisited = any(nid not in visited for nid, _ in neighbors)

        # If no more neighbors to explore, save current path (leaf node)
        if not has_unvisited and len(path_ids) >= min_length:
            all_paths.append((path_ids[:], path_edge_types[:]))
            return

        # Explore neighbors
        for next_id, edge_type in neighbors:
            if next_id not in visited:
                visited.add(next_id)
                path_ids.append(next_id)
                path_edge_types.append(edge_type)
                dfs(next_id, path_ids, path_edge_types, visited)
                path_edge_types.pop()
                path_ids.pop()
                visited.remove(next_id)

    # Start DFS from each node
    for node in nodes:
        visited: Set[str] = {node.node_id}
        dfs(node.node_id, [node.node_id], [], visited)

    # Deduplicate paths by their ordered node ID tuple
    unique_paths: List[Tuple[List[str], List[str]]] = []
    seen: Set[Tuple[str, ...]] = set()

    def _path_sort_key(item: Tuple[List[str], List[str]]) -> Tuple[int, str]:
        """Sort by length (longer first), then earliest timestamp (earlier first)."""
        nids, _ = item
        length = len(nids)
        earliest = "9999"
        for nid in nids:
            n = node_map.get(nid)
            if n is not None:
                ts = _get_node_timestamp(n)
                if ts is not None:
                    ts_str = ts.isoformat()
                    if ts_str < earliest:
                        earliest = ts_str
        # Negate length for descending sort; use earliest for ascending
        return (-length, earliest)

    sorted_paths = sorted(all_paths, key=_path_sort_key)

    for nids, etypes in sorted_paths:
        path_key = tuple(nids)
        if path_key not in seen:
            unique_paths.append((nids, etypes))
            seen.add(path_key)

    # Cap at max_paths (scale with graph size)
    max_paths = min(len(nodes), config.max_paths)
    capped = unique_paths[:max_paths]

    result = [
        ReasoningPath(node_ids=nids, edge_types=etypes)
        for nids, etypes in capped
    ]

    logger.debug(
        "Path mining: %d total paths found, %d unique, %d returned",
        len(all_paths),
        len(unique_paths),
        len(result),
    )
    return result
