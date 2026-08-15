"""
In-memory graph store.

This backend is intended for tests and lightweight local usage. It stores:
- episodes keyed by (scope_id, episode_id)
- nodes keyed by (scope_id, node_id)
- edges keyed by (scope_id, edge_id)

It supports:
- keyword search over node/edge text fields,
- temporal filtering via `GraphSearchFilters.as_of` + `valid_only`,
- semantic node search when `GraphNode.name_embedding` is populated.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ctxforge.retrieval.enhanced_structures import EnhancedMemoryIndex

from ctxforge.protocols.graph import (
    GraphCommunity,
    GraphEdge,
    GraphEpisode,
    GraphNode,
    GraphSearchFilters,
    GraphSearchResult,
    GraphSearchScope,
    IGraphStore,
)
from ctxforge.utils.math import cosine_similarity


def _stable_id(*parts: str) -> str:
    """Generate a deterministic UUID5 from normalized string parts."""
    raw = "|".join([p.strip().lower() for p in parts if p is not None])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


class InMemoryGraphStore(IGraphStore):
    """In-process implementation of `IGraphStore` for tests and demos."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._episodes: Dict[Tuple[str, str], GraphEpisode] = {}
        self._nodes: Dict[Tuple[str, str], GraphNode] = {}
        self._nodes_by_name: Dict[Tuple[str, str], str] = {}
        self._edges: Dict[Tuple[str, str], GraphEdge] = {}
        self._communities: Dict[Tuple[str, str], GraphCommunity] = {}
        self._memberships: Dict[Tuple[str, str], set[str]] = {}  # (scope_id, community_id) -> {node_id}
        self._enhanced_indexes: Dict[str, object] = {}  # scope_id -> EnhancedMemoryIndex

    async def add_episodes(self, scope_id: str, episodes: List[GraphEpisode]) -> int:
        """Store/overwrite episodes by `(scope_id, episode_id)`."""
        async with self._lock:
            n = 0
            for ep in episodes:
                ep.scope_id = scope_id
                key = (scope_id, ep.episode_id)
                self._episodes[key] = ep
                n += 1
            return n

    async def upsert_nodes(self, scope_id: str, nodes: List[GraphNode]) -> int:
        """Store/overwrite nodes by `(scope_id, node_id)`; assign deterministic ids if missing."""
        async with self._lock:
            n = 0
            for node in nodes:
                node.scope_id = scope_id
                if not node.node_id:
                    node.node_id = _stable_id(scope_id, "node", node.name, ",".join(sorted(node.labels or [])))
                key = (scope_id, node.node_id)
                self._nodes[key] = node
                self._nodes_by_name[(scope_id, node.name.strip().lower())] = node.node_id
                n += 1
            return n

    async def upsert_edges(self, scope_id: str, edges: List[GraphEdge]) -> int:
        """Store/overwrite edges by `(scope_id, edge_id)`; assign deterministic ids if missing."""
        async with self._lock:
            n = 0
            for edge in edges:
                edge.scope_id = scope_id
                if not edge.edge_id:
                    edge.edge_id = _stable_id(
                        scope_id,
                        "edge",
                        edge.source_node_id,
                        edge.edge_type,
                        edge.target_node_id,
                        edge.fact or "",
                    )
                key = (scope_id, edge.edge_id)
                self._edges[key] = edge
                n += 1
            return n

    async def get_edges_by_ids(self, scope_id: str, edge_ids: List[str]) -> List[GraphEdge]:
        """Fetch edges by id (no ordering guarantee)."""
        want = set(edge_ids or [])
        if not want:
            return []
        async with self._lock:
            out: List[GraphEdge] = []
            for (sid, _), edge in self._edges.items():
                if sid != scope_id:
                    continue
                if edge.edge_id in want:
                    out.append(edge)
            return out

    async def invalidate_edges(
        self,
        scope_id: str,
        edge_ids: List[str],
        *,
        invalid_at: datetime,
    ) -> int:
        """Mark a list of edges invalid by setting their `invalid_at` timestamp."""

        want = set(edge_ids or [])
        if not want:
            return 0
        n = 0
        async with self._lock:
            for (sid, _), edge in self._edges.items():
                if sid != scope_id:
                    continue
                if edge.edge_id in want:
                    edge.invalid_at = invalid_at
                    n += 1
        return n

    async def delete_scope(self, scope_id: str) -> int:
        """Delete all graph objects for a scope_id (episodes, nodes, edges)."""
        async with self._lock:
            ep_keys = [k for k in self._episodes.keys() if k[0] == scope_id]
            node_keys = [k for k in self._nodes.keys() if k[0] == scope_id]
            edge_keys = [k for k in self._edges.keys() if k[0] == scope_id]
            for k in ep_keys:
                del self._episodes[k]
            for k in node_keys:
                del self._nodes[k]
            for k in edge_keys:
                del self._edges[k]

            # Rebuild name map without deleted scope
            self._nodes_by_name = {k: v for k, v in self._nodes_by_name.items() if k[0] != scope_id}

            # Delete derived community artifacts for the scope as well.
            comm_keys = [k for k in self._communities.keys() if k[0] == scope_id]
            for k in comm_keys:
                del self._communities[k]
            mem_keys = [k for k in self._memberships.keys() if k[0] == scope_id]
            for k in mem_keys:
                del self._memberships[k]

            # Delete enhanced index for the scope.
            self._enhanced_indexes.pop(scope_id, None)

            return len(ep_keys) + len(node_keys) + len(edge_keys) + len(comm_keys)

    async def upsert_communities(self, scope_id: str, communities: List[GraphCommunity]) -> int:
        """Insert/update community nodes for a scope."""
        if not communities:
            return 0
        async with self._lock:
            n = 0
            for c in communities:
                c.scope_id = scope_id
                self._communities[(scope_id, c.community_id)] = c
                n += 1
            return n

    async def upsert_memberships(self, scope_id: str, memberships: List[tuple[str, str]]) -> int:
        """Insert membership edges (community_id -> node_id) for a scope."""
        if not memberships:
            return 0
        async with self._lock:
            n = 0
            for community_id, node_id in memberships:
                if not community_id or not node_id:
                    continue
                key = (scope_id, community_id)
                if key not in self._memberships:
                    self._memberships[key] = set()
                if node_id not in self._memberships[key]:
                    self._memberships[key].add(node_id)
                    n += 1
            return n

    async def get_communities_for_nodes(
        self,
        scope_id: str,
        node_ids: List[str],
        *,
        limit: int = 10,
    ) -> List[GraphCommunity]:
        """Return communities ranked by overlap with `node_ids` (then member_count)."""
        want = set([x for x in (node_ids or []) if x])
        if not want or limit <= 0:
            return []
        async with self._lock:
            scored: List[tuple[int, int, GraphCommunity]] = []
            for (sid, cid), c in self._communities.items():
                if sid != scope_id:
                    continue
                members = self._memberships.get((sid, cid), set())
                overlap = len(want.intersection(members))
                if overlap <= 0:
                    continue
                scored.append(
                    (
                        overlap,
                        int(c.member_count),
                        GraphCommunity(
                            community_id=c.community_id,
                            scope_id=scope_id,
                            name=c.name,
                            summary=c.summary,
                            member_count=int(c.member_count),
                            updated_at=c.updated_at,
                            name_embedding=c.name_embedding,
                            summary_embedding=c.summary_embedding,
                            overlap=overlap,
                        ),
                    )
                )
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [c for _, _, c in scored[:limit]]

    async def delete_communities(self, scope_id: str) -> int:
        """Delete all community nodes + memberships for a scope."""
        async with self._lock:
            comm_keys = [k for k in self._communities.keys() if k[0] == scope_id]
            for k in comm_keys:
                del self._communities[k]
            mem_keys = [k for k in self._memberships.keys() if k[0] == scope_id]
            for k in mem_keys:
                del self._memberships[k]
            return len(comm_keys)

    async def save_enhanced_index(self, scope_id: str, index: EnhancedMemoryIndex) -> None:
        """Persist the enhanced memory index for a scope."""
        async with self._lock:
            self._enhanced_indexes[scope_id] = index

    async def load_enhanced_index(self, scope_id: str) -> Optional[EnhancedMemoryIndex]:
        """Load the enhanced memory index for a scope."""
        async with self._lock:
            return self._enhanced_indexes.get(scope_id)

    async def search_nodes_semantic(
        self,
        scope_id: str,
        query_vector: List[float],
        *,
        limit: int = 20,
        filters: Optional[GraphSearchFilters] = None,
    ) -> List[GraphNode]:
        """Rank nodes by cosine similarity between `query_vector` and `GraphNode.name_embedding`."""
        filters = filters or GraphSearchFilters()
        want_labels = set([label.lower() for label in (filters.node_labels or [])])
        want_node_ids = set([x for x in (filters.node_ids or []) if x])
        limit = max(0, int(limit))

        async with self._lock:
            scored: List[Tuple[float, GraphNode]] = []
            for (sid, _), node in self._nodes.items():
                if sid != scope_id:
                    continue
                if want_node_ids and node.node_id not in want_node_ids:
                    continue
                if want_labels:
                    node_labels = set([label.lower() for label in (node.labels or [])])
                    if node_labels.isdisjoint(want_labels):
                        continue
                if not node.name_embedding:
                    continue
                score = cosine_similarity(query_vector, node.name_embedding)
                if score <= 0.0:
                    continue
                scored.append((score, node))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [n for _, n in scored[:limit]]

    async def search(
        self,
        scope_id: str,
        query: str,
        *,
        scope: GraphSearchScope,
        limit: int = 20,
        filters: Optional[GraphSearchFilters] = None,
        center_node_id: Optional[str] = None,
    ) -> GraphSearchResult:
        """Keyword search over stored nodes/edges/episodes with optional temporal filtering."""
        filters = filters or GraphSearchFilters()
        q = (query or "").strip().lower()
        as_of = filters.as_of or datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        want_node_ids = set([x for x in (filters.node_ids or []) if x])
        want_edge_ids = set([x for x in (filters.edge_ids or []) if x])

        def match_text(text: str) -> int:
            if not q:
                return 1
            if not text:
                return 0
            t = text.lower()
            if q in t:
                return 3
            q_words = set(q.split())
            t_words = set(t.split())
            return len(q_words & t_words)

        res = GraphSearchResult()

        async with self._lock:
            if scope == "episodes":
                scored = []
                for (sid, _), ep in self._episodes.items():
                    if sid != scope_id:
                        continue
                    score = match_text(ep.content)
                    if score <= 0:
                        continue
                    scored.append((score, ep.created_at, ep))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                res.episodes = [x[2] for x in scored[: max(0, limit)]]
                return res

            if scope == "nodes":
                want_labels = set([label.lower() for label in (filters.node_labels or [])])
                scored = []
                for (sid, _), node in self._nodes.items():
                    if sid != scope_id:
                        continue
                    if want_node_ids and node.node_id not in want_node_ids:
                        continue
                    if want_labels:
                        node_labels = set([label.lower() for label in (node.labels or [])])
                        if node_labels.isdisjoint(want_labels):
                            continue
                    score = max(
                        match_text(node.name),
                        match_text(node.summary or ""),
                        match_text(" ".join([f"{k}:{v}" for k, v in (node.attributes or {}).items()])),
                    )
                    if score <= 0:
                        continue
                    scored.append((score, node))
                scored.sort(key=lambda x: x[0], reverse=True)
                res.nodes = [x[1] for x in scored[: max(0, limit)]]
                return res

            # edges
            want_edge_types = set([t.lower() for t in (filters.edge_types or [])])
            scored = []
            for (sid, _), edge in self._edges.items():
                if sid != scope_id:
                    continue
                if want_edge_ids and edge.edge_id not in want_edge_ids:
                    continue
                if center_node_id and edge.source_node_id != center_node_id and edge.target_node_id != center_node_id:
                    continue
                if want_edge_types and edge.edge_type.lower() not in want_edge_types:
                    continue
                invalid_at = edge.invalid_at
                if invalid_at is not None and invalid_at.tzinfo is None:
                    invalid_at = invalid_at.replace(tzinfo=timezone.utc)
                if filters.valid_only and invalid_at is not None and invalid_at <= as_of:
                    continue
                score = max(
                    match_text(edge.fact or ""),
                    match_text(edge.edge_type),
                    match_text(" ".join([f"{k}:{v}" for k, v in (edge.attributes or {}).items()])),
                )
                if score <= 0:
                    continue
                scored.append((score, edge))
            scored.sort(key=lambda x: x[0], reverse=True)
            res.edges = [x[1] for x in scored[: max(0, limit)]]
            return res


