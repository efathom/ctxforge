"""
KNN-based entity linking for knowledge graphs.

Discovers semantically equivalent entity nodes across the graph using
cosine similarity on node embeddings and creates SAME_AS edges between
them.  This bridges different mentions of the same concept (e.g.,
"NYC" ↔ "New York City") so that graph traversal and PPR can
propagate relevance across them.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

from ctxforge.protocols.graph import GraphEdge, GraphNode, IGraphStore

logger = logging.getLogger(__name__)

# Default edge type for entity-linking edges.
SAME_AS_EDGE_TYPE = "SAME_AS"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _stable_edge_id(scope_id: str, id_a: str, id_b: str) -> str:
    """Deterministic edge ID for an undirected pair."""
    ordered = tuple(sorted([id_a, id_b]))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{scope_id}|{SAME_AS_EDGE_TYPE}|{ordered[0]}|{ordered[1]}"))


class EntityLinker:
    """Discovers and links semantically equivalent entity nodes via KNN.

    After linking, SAME_AS edges allow graph traversal algorithms (BFS,
    PPR) to treat different mentions of the same entity as connected.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_neighbors: int = 5,
        batch_size: int = 100,
    ):
        self._threshold = similarity_threshold
        self._max_neighbors = max_neighbors
        self._batch_size = batch_size

    async def link_entities(
        self,
        nodes: List[GraphNode],
        graph_store: IGraphStore,
        scope_id: str,
    ) -> List[GraphEdge]:
        """Find and persist SAME_AS edges between similar entity nodes.

        Args:
            nodes: Nodes to consider for linking (must have ``name_embedding``).
            graph_store: Store to persist new edges to.
            scope_id: Scope partition for the edges.

        Returns:
            List of newly created SAME_AS edges.
        """
        # Collect nodes that have embeddings.
        embedded: List[Tuple[str, List[float]]] = []
        for n in nodes:
            if n.name_embedding is not None and n.node_id:
                embedded.append((n.node_id, n.name_embedding))

        if len(embedded) < 2:
            return []

        # Build name lookup for fact descriptions.
        name_map: Dict[str, str] = {n.node_id: n.name for n in nodes}

        # Compute KNN pairs above threshold.
        pairs = self._compute_knn(embedded)

        if not pairs:
            return []

        # Deduplicate against existing SAME_AS edges.
        existing_ids = await self._get_existing_link_ids(graph_store, scope_id, pairs)

        new_edges: List[GraphEdge] = []
        for id_a, id_b, similarity in pairs:
            eid = _stable_edge_id(scope_id, id_a, id_b)
            if eid in existing_ids:
                continue

            name_a = name_map.get(id_a, id_a)
            name_b = name_map.get(id_b, id_b)
            edge = GraphEdge(
                edge_id=eid,
                scope_id=scope_id,
                source_node_id=id_a,
                target_node_id=id_b,
                edge_type=SAME_AS_EDGE_TYPE,
                fact=f"{name_a} is the same as {name_b}",
                attributes={"similarity": round(similarity, 4), "method": "knn_embedding"},
                valid_at=datetime.now(timezone.utc),
            )
            new_edges.append(edge)

        if new_edges:
            try:
                await graph_store.upsert_edges(scope_id, new_edges)
                logger.info("Entity linking created %d SAME_AS edges in scope %s", len(new_edges), scope_id)
            except Exception:
                logger.warning("Failed to persist SAME_AS edges", exc_info=True)
                return []

        return new_edges

    # ------------------------------------------------------------------
    # KNN computation
    # ------------------------------------------------------------------

    def _compute_knn(
        self,
        embedded: List[Tuple[str, List[float]]],
    ) -> List[Tuple[str, str, float]]:
        """Brute-force KNN via batched cosine similarity.

        Returns ``(node_id_a, node_id_b, similarity)`` tuples above
        threshold.  Skips self-pairs and returns at most
        ``max_neighbors`` links per node.
        """
        results: List[Tuple[str, str, float]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        neighbor_counts: Dict[str, int] = {}

        for i, (id_a, emb_a) in enumerate(embedded):
            scored: List[Tuple[str, float]] = []
            for j, (id_b, emb_b) in enumerate(embedded):
                if i == j:
                    continue
                sim = _cosine_similarity(emb_a, emb_b)
                if sim >= self._threshold:
                    scored.append((id_b, sim))

            scored.sort(key=lambda x: x[1], reverse=True)

            for id_b, sim in scored[:self._max_neighbors]:
                pair = tuple(sorted([id_a, id_b]))
                if pair in seen_pairs:
                    continue
                if neighbor_counts.get(id_a, 0) >= self._max_neighbors:
                    break
                if neighbor_counts.get(id_b, 0) >= self._max_neighbors:
                    continue

                seen_pairs.add(pair)
                neighbor_counts[id_a] = neighbor_counts.get(id_a, 0) + 1
                neighbor_counts[id_b] = neighbor_counts.get(id_b, 0) + 1
                results.append((id_a, id_b, sim))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_existing_link_ids(
        store: IGraphStore,
        scope_id: str,
        pairs: List[Tuple[str, str, float]],
    ) -> Set[str]:
        """Return edge IDs that already exist in the store."""
        ids: Set[str] = set()
        for id_a, id_b, _ in pairs:
            eid = _stable_edge_id(scope_id, id_a, id_b)
            try:
                existing = await store.get_edges_by_ids(scope_id, [eid])
                if existing:
                    ids.add(eid)
            except Exception:
                pass
        return ids
