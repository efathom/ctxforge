from __future__ import annotations

"""
Graph subsystem service.

Owns graph dependencies and implements:
- ingestion: episode -> extract -> maintenance -> upsert
- retrieval: hybrid seeding + BFS expansion + community lookup
- community rebuild pipeline trigger + explicit rebuild
"""

import asyncio
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from dateutil.parser import isoparse

from ctxforge.compaction.topology_view import TopologyAwareRenderer
from ctxforge.config.base import EngineConfig, GraphRetrievalConfig, GraphVectorFusedTripletsConfig
from ctxforge.core.session import Session
from ctxforge.graph.communities.builder import CommunityBuildConfig, CommunityBuilder
from ctxforge.graph.retrieval.bridge_discovery import check_connection, find_bridge_candidates
from ctxforge.graph.retrieval.path_miner import discover_reasoning_paths
from ctxforge.graph.retrieval.path_scorer import rank_and_limit_nodes
from ctxforge.graph.retrieval.planner import (
    GraphKeywords,
    GraphQueryPlan,
    extract_keywords_heuristic,
    plan_mode,
)
from ctxforge.graph.retrieval.types import (
    BridgeConnection,
    EvidenceItem,
    GraphEdgeHit,
    GraphNodeHit,
    GraphRetrievalResult,
    ReasoningPath,
)
from ctxforge.graph.retrieval.utils import order_ids_by_score, rrf_scores, stable_unique
from ctxforge.graph.utils import format_graph_context
from ctxforge.protocols.graph import (
    GraphCommunity,
    GraphEdge,
    GraphEpisode,
    GraphNode,
    GraphSearchFilters,
    IGraphExtractor,
    IGraphStore,
)
from ctxforge.protocols.graph_maintenance import (
    IGraphContradictionDetector,
    IGraphEdgeTemporalExtractor,
)
from ctxforge.protocols.llm import IEmbeddingProvider
from ctxforge.protocols.tokenizer import ITokenizerProvider
from ctxforge.utils.math import cosine_similarity

logger = logging.getLogger(__name__)


class GraphService:
    """
    Graph subsystem owned by the engine.

    This service owns graph dependencies entirely; the engine delegates all graph-related work
    (ingestion, retrieval, community rebuild) to this object.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        graph_store: IGraphStore,
        graph_extractor: Optional[IGraphExtractor] = None,
        graph_ontology: Optional[Any] = None,
        embedding_provider: Optional[IEmbeddingProvider] = None,
        tokenizer_provider: Optional[ITokenizerProvider] = None,
        contradiction_detector: Optional[IGraphContradictionDetector] = None,
        temporal_extractor: Optional[IGraphEdgeTemporalExtractor] = None,
        community_builder: Optional[CommunityBuilder] = None,
        entity_linker: Optional[Any] = None,
        background_tasks: Optional[Set[asyncio.Task]] = None,
    ):
        self._cfg = config
        self._store = graph_store
        self._extractor = graph_extractor
        self._ontology = graph_ontology
        self._embed = embedding_provider
        self._tokenizer = tokenizer_provider
        self._contradiction = contradiction_detector
        self._temporal = temporal_extractor
        self._communities = community_builder
        self._entity_linker = entity_linker
        self._background_tasks = background_tasks

        # Track ingestion cadence per scope to trigger periodic community rebuilds.
        self._episode_counts: Dict[str, int] = {}

    @property
    def store(self) -> IGraphStore:
        return self._store

    def _normalize_query_for_keyword_search(self, query: str) -> str:
        """
        Normalize a user query for keyword-based graph searches.

        This strips punctuation so token intersection matching works better across stores
        that split on whitespace.
        """
        q = (query or "").strip().lower()
        if not q:
            return ""
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", q)
        return " ".join(tokens)

    async def close(self) -> None:
        """Close graph resources if the backend supports it."""
        close_fn = getattr(self._store, "close", None)
        if close_fn is None:
            return
        out = close_fn()
        if asyncio.iscoroutine(out):
            await out

    async def ingest_turn(self, *, session: Session, user_input: str, assistant_response: str) -> None:
        """
        Ingest a single conversation turn into graph memory for a given `session.user_id` scope.

        This is the *write path* for graph memory. It turns (user_input, assistant_response) into:
        - a persisted `GraphEpisode` (the raw text evidence we can later retrieve), and
        - a set of extracted `GraphNode`/`GraphEdge` facts derived from the episode,
          optionally augmented with embeddings / temporal validity / contradiction invalidation.

        Design principles:
        - **Opt-in**: If graph is disabled, or if this engine is configured for retrieval-only
          (no extractor/ontology), this is a no-op.
        - **Best-effort**: Embeddings, temporal extraction, and contradiction invalidation are
          optional and should not break ingestion if they fail; we proceed with whatever data we have.
        - **Bounded work**: External calls (embedding batch, temporal, invalidation) are concurrency-limited.
        - **Side effects**: This function writes to the graph store (episodes/nodes/edges) and may trigger
          a community rebuild (background task) depending on cadence config.

        Flow:
        - Create and persist an episode for the turn (the evidence anchor for this write).
        - Extract nodes/edges using the configured graph extractor + ontology.
        - Optionally embed node names (when graph embeddings are enabled and an embedding provider exists).
        - Upsert nodes.
        - Optionally enrich edges with temporal bounds (`valid_at`/`invalid_at`) via temporal extractor.
        - Optionally invalidate contradicted existing edges (invalidation pipeline).
        - Upsert edges.
        - Optionally trigger a community rebuild on a cadence (sync or background task).
        """
        if not getattr(self._cfg, "graph", None) or not getattr(self._cfg.graph, "enabled", False):
            return
        # Ingestion requires an extractor + ontology. Retrieval-only configurations are allowed.
        if self._extractor is None or self._ontology is None:
            return

        # We store the full turn as a single episode, so later retrieval can cite the original
        # conversational evidence behind extracted facts.
        content = f"{user_input}\n{assistant_response}"
        ep = GraphEpisode(
            episode_id=str(uuid.uuid4()),
            scope_id=session.user_id,
            content=content,
            content_type="message",
            metadata={"session_id": session.session_id},
        )
        # Persist the episode first. Even if extraction fails, we retain the raw turn evidence.
        await self._store.add_episodes(session.user_id, [ep])
        self._episode_counts[session.user_id] = self._episode_counts.get(session.user_id, 0) + 1

        model = self._cfg.graph.extraction.model or self._cfg.llm.model
        # Extract structured nodes/edges from the new episode.
        nodes, edges = await self._extractor.extract(
            scope_id=session.user_id,
            episodes=[ep],
            ontology=self._ontology,
            model=model,
        )

        async def _gather_limited(coros, *, limit: int):
            # Helper for bounded-concurrency IO (embedding / temporal / invalidation).
            limit = max(1, int(limit))
            sem = asyncio.Semaphore(limit)

            async def run_one(c):
                async with sem:
                    return await c

            return await asyncio.gather(*[run_one(c) for c in coros], return_exceptions=True)

        # Node embeddings (optional)
        if (
            nodes
            and self._embed is not None
            and getattr(getattr(self._cfg.graph, "embeddings", None), "enabled", False)
        ):
            embed_model = getattr(self._cfg.graph.embeddings.embedding, "model", None)
            names = [(n.name or "").replace("\n", " ") for n in nodes]
            try:
                # Prefer batch embedding for performance; fall back to per-item calls on failure.
                resp = await self._embed.embed(names, model=embed_model)
                vectors = resp.embeddings or []
                if len(vectors) == len(nodes):
                    for n, v in zip(nodes, vectors, strict=True):
                        n.name_embedding = v
                else:
                    raise ValueError("embedding batch length mismatch")
            except Exception:
                max_c = int(getattr(self._cfg.graph.embeddings, "max_concurrency", 8))
                results = await _gather_limited(
                    [self._embed.embed_single(t, model=embed_model) for t in names],
                    limit=max_c,
                )
                for n, r in zip(nodes, results, strict=False):
                    if isinstance(r, Exception):
                        continue
                    if isinstance(r, list):
                        n.name_embedding = r

        if nodes:
            # Nodes can be upserted independently of edges; edges often reference node ids.
            await self._store.upsert_nodes(session.user_id, nodes)

        if edges:
            # Temporal enrichment (optional).
            if self._temporal is not None and getattr(getattr(self._cfg.graph, "temporal", None), "enabled", False):
                temporal_model = getattr(self._cfg.graph.temporal, "model", None) or self._cfg.llm.model
                max_c = int(getattr(self._cfg.graph.temporal, "max_concurrency", 4))
                temporal_results = await _gather_limited(
                    [
                        self._temporal.extract_temporal_info(
                            scope_id=session.user_id, edge=e, episodes=[ep], model=temporal_model
                        )
                        for e in edges
                    ],
                    limit=max_c,
                )
                for e, info in zip(edges, temporal_results, strict=False):
                    if isinstance(info, Exception) or info is None:
                        continue
                    if getattr(info, "valid_at", None):
                        try:
                            e.valid_at = isoparse(info.valid_at)  # type: ignore[arg-type]
                        except Exception:
                            pass
                    if getattr(info, "invalid_at", None):
                        try:
                            e.invalid_at = isoparse(info.invalid_at)  # type: ignore[arg-type]
                        except Exception:
                            pass

            # Edge invalidation (optional).
            if self._contradiction is not None and getattr(getattr(self._cfg.graph, "invalidation", None), "enabled", False):
                try:
                    candidate_limit = int(getattr(self._cfg.graph.invalidation, "candidate_limit", 25))
                except Exception:
                    candidate_limit = 25

                plan_model = getattr(self._cfg.graph.invalidation, "model", None) or self._cfg.llm.model
                max_c = int(getattr(self._cfg.graph.invalidation, "max_concurrency", 4))

                async def detect_for_edge(new_edge: GraphEdge) -> List[str]:
                    # Collect "nearby" candidate edges of the same type to compare against the new edge.
                    # This is intentionally bounded by `candidate_limit` to keep invalidation cheap.
                    candidates: List[Any] = []
                    try:
                        candidates.extend(
                            (
                                await self._store.search(
                                    session.user_id,
                                    "",
                                    scope="edges",
                                    limit=candidate_limit,
                                    filters=GraphSearchFilters(edge_types=[new_edge.edge_type], valid_only=True),
                                )
                            ).edges
                        )
                        candidates.extend(
                            (
                                await self._store.search(
                                    session.user_id,
                                    "",
                                    scope="edges",
                                    limit=candidate_limit,
                                    filters=GraphSearchFilters(edge_types=[new_edge.edge_type], valid_only=True),
                                    center_node_id=new_edge.source_node_id,
                                )
                            ).edges
                        )
                        candidates.extend(
                            (
                                await self._store.search(
                                    session.user_id,
                                    "",
                                    scope="edges",
                                    limit=candidate_limit,
                                    filters=GraphSearchFilters(edge_types=[new_edge.edge_type], valid_only=True),
                                    center_node_id=new_edge.target_node_id,
                                )
                            ).edges
                        )
                    except Exception:
                        candidates = []

                    dedup: Dict[str, Any] = {}
                    for c in candidates:
                        if not getattr(c, "edge_id", None):
                            continue
                        if c.edge_id == new_edge.edge_id:
                            continue
                        dedup[c.edge_id] = c

                    plan = await self._contradiction.detect_contradictions(
                        scope_id=session.user_id,
                        new_edge=new_edge,
                        candidate_edges=list(dedup.values()),
                        nodes=nodes,
                        episodes=[ep],
                        model=plan_model,
                    )
                    return list(plan.invalidate_edge_ids or [])

                invalidation_lists = await _gather_limited([detect_for_edge(e) for e in edges], limit=max_c)
                edge_ids_to_invalidate: List[str] = []
                for r in invalidation_lists:
                    if isinstance(r, Exception) or r is None:
                        continue
                    if isinstance(r, list):
                        edge_ids_to_invalidate.extend([x for x in r if isinstance(x, str)])

                edge_ids_to_invalidate = list(dict.fromkeys([x for x in edge_ids_to_invalidate if x]))
                if edge_ids_to_invalidate:
                    # We mark invalid edges with a timestamp; they can still be stored for history,
                    # but retrieval can filter to valid-only edges.
                    await self._store.invalidate_edges(
                        session.user_id, edge_ids_to_invalidate, invalid_at=datetime.now(timezone.utc)
                    )

            # Finally write (possibly temporal-enriched) edges for this episode.
            await self._store.upsert_edges(session.user_id, edges)

        # Entity linking (optional): discover SAME_AS edges among extracted nodes.
        el_cfg = getattr(self._cfg.graph, "entity_linking", None)
        if (
            self._entity_linker is not None
            and el_cfg is not None
            and getattr(el_cfg, "enabled", False)
            and getattr(el_cfg, "run_on_ingest", True)
        ):
            try:
                all_nodes = await self._store.search_nodes_semantic(
                    session.user_id, [], limit=200
                ) if self._embed else []
            except Exception:
                all_nodes = []
            if len(all_nodes) >= 2:
                try:
                    await self._entity_linker.link_entities(
                        nodes=all_nodes,
                        graph_store=self._store,
                        scope_id=session.user_id,
                    )
                except Exception:
                    logger.warning("Entity linking failed during ingest", exc_info=True)

        # Community rebuild trigger (optional, cadence-based).
        if getattr(getattr(self._cfg.graph, "communities", None), "enabled", False) and self._communities is not None:
            every = int(getattr(self._cfg.graph.communities, "rebuild_every_n_episodes", 0) or 0)
            if every > 0 and (self._episode_counts.get(session.user_id, 0) % every == 0):
                if self._background_tasks is not None:
                    # Prefer background rebuild to keep ingestion latency low.
                    task = asyncio.create_task(self.rebuild_communities(scope_id=session.user_id))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                else:
                    # No background task set provided; rebuild inline.
                    await self.rebuild_communities(scope_id=session.user_id)

    async def rebuild_communities(self, *, scope_id: str) -> int:
        """Recompute and persist derived communities for a scope."""
        if not getattr(getattr(self._cfg.graph, "communities", None), "enabled", False):
            return 0
        if self._communities is None:
            return 0

        max_nodes = int(getattr(self._cfg.graph.communities, "max_nodes", 0) or 0) or 2000
        max_edges = int(getattr(self._cfg.graph.communities, "max_edges", 0) or 0) or 4000
        filters = GraphSearchFilters(valid_only=True)

        nodes = (await self._store.search(scope_id, "", scope="nodes", limit=max_nodes, filters=filters)).nodes
        edges = (await self._store.search(scope_id, "", scope="edges", limit=max_edges, filters=filters)).edges

        cfg = CommunityBuildConfig(
            min_cluster_size=int(getattr(self._cfg.graph.communities, "min_cluster_size", 2)),
            max_communities=int(getattr(self._cfg.graph.communities, "max_communities", 5)),
            model=getattr(self._cfg.graph.communities, "model", None) or self._cfg.llm.model,
            max_concurrency=int(getattr(self._cfg.graph.communities, "max_concurrency", 4)),
        )
        communities, memberships = await self._communities.build(scope_id=scope_id, nodes=nodes, edges=edges, config=cfg)

        await self._store.delete_communities(scope_id)
        await self._store.upsert_communities(scope_id, communities)
        await self._store.upsert_memberships(scope_id, memberships)
        return len(communities)

    def _truncate_by_token_budget(self, items: List[Any], *, render: Any, max_tokens: int) -> List[Any]:
        """
        Truncate a list of items by cumulative token count.

        - If no tokenizer provider is configured, returns items unchanged.
        - `render(item)` must return a string representation for token counting.
        """
        if self._tokenizer is None:
            return items
        max_tokens = int(max_tokens or 0)
        if max_tokens <= 0:
            return items
        out: List[Any] = []
        used = 0
        for it in items:
            try:
                s = str(render(it) or "")
            except Exception:
                s = ""
            if not s:
                continue
            cost = int(self._tokenizer.count_tokens(s))
            if out and used + cost > max_tokens:
                break
            if not out and cost > max_tokens:
                # Always allow at least one item (best-effort).
                out.append(it)
                break
            out.append(it)
            used += cost
        return out

    async def _embed_query_vector(
        self, *, query: str, methods_set: Set[str], vft_enabled: bool
    ) -> Optional[List[float]]:
        """
        Compute query embedding once for semantic seeding / vector-fused recipe.

        Returns None on any embedding failure.
        """
        if not query.strip():
            return None
        if self._embed is None:
            return None
        if not getattr(getattr(self._cfg.graph, "embeddings", None), "enabled", False):
            return None
        if not (("semantic" in methods_set) or vft_enabled):
            return None
        try:
            embed_model = getattr(self._cfg.graph.embeddings.embedding, "model", None)
            return await self._embed.embed_single(query, model=embed_model)
        except Exception:
            return None

    async def _seed_nodes_for_scope(
        self,
        *,
        scope_id: str,
        keyword_query: str,
        methods_set: Set[str],
        seed_k: int,
        filters: GraphSearchFilters,
        qv: Optional[List[float]],
    ) -> Tuple[List[GraphNode], List[GraphNode]]:
        """
        Retrieve candidate "seed" nodes for a single scope, using the enabled seeding methods.

        This is a small helper used by multiple retrieval recipes. It does **not** decide which
        nodes ultimately become seeds (that’s handled by the recipe/planner). Instead, it returns
        two candidate lists that downstream logic can fuse (e.g., RRF) and/or score.

        Inputs:
        - `scope_id`: graph scope to search (typically `user_id` or `global_scope_id`)
        - `keyword_query`: normalized query for keyword/fulltext search (punctuation stripped)
        - `methods_set`: retrieval method flags (e.g. {"keyword", "semantic"})
        - `seed_k`: configured seed size; if 0, we skip seeding
        - `filters`: graph-level filters (valid_only, labels/types, as_of, etc.)
        - `qv`: optional query embedding vector for semantic seeding

        Behavior:
        - **Semantic seeding** (`search_nodes_semantic`) runs only when:
          - `qv` is present,
          - `"semantic"` is enabled in `methods_set`,
          - `seed_k > 0`.
        - **Keyword seeding** (`search(..., scope="nodes")`) runs only when:
          - `"keyword"` is enabled in `methods_set`,
          - `seed_k > 0`.
        - Both calls are **best-effort**; failures return an empty list.

        Returns:
        - `(semantic_nodes, keyword_nodes)`: two lists of `GraphNode` candidates.
        """
        semantic_nodes: List[GraphNode] = []
        keyword_nodes: List[GraphNode] = []

        # Semantic seeding: vector search over node embeddings (when available).
        if qv is not None and "semantic" in methods_set and seed_k > 0:
            try:
                semantic_nodes = await self._store.search_nodes_semantic(
                    scope_id, qv, limit=max(seed_k, 1), filters=filters
                )
            except Exception:
                semantic_nodes = []

        # Keyword/fulltext seeding: best-effort lexical match over node names/attrs.
        # We retrieve up to 2*seed_k to give downstream fusion some room.
        if "keyword" in methods_set and seed_k > 0:
            try:
                keyword_nodes = (
                    await self._store.search(
                        scope_id,
                        keyword_query,
                        scope="nodes",
                        limit=max(seed_k * 2, 1),
                        filters=filters,
                    )
                ).nodes
            except Exception:
                keyword_nodes = []

        return semantic_nodes, keyword_nodes

    async def _retrieve_recipe_vector_fused_triplets(
        self,
        *,
        scope_id: str,
        plan: GraphQueryPlan,
        kws: GraphKeywords,
        keyword_query: str,
        cfg_r: GraphRetrievalConfig,
        vft_cfg: GraphVectorFusedTripletsConfig,
        filters: GraphSearchFilters,
        qv: Optional[List[float]],
        semantic_nodes: List[GraphNode],
        keyword_nodes: List[GraphNode],
        seed_k: int,
        bfs_edges_per_node: int,
    ) -> Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode]]:
        """
        Vector+graph fused triplet scoring recipe.

        This is an opt-in retrieval recipe inspired by Cognee’s “graph+vector fused triplet scoring” idea:
        instead of ranking edges purely by keyword match or purely by traversal order, we score each
        candidate edge (a subject–predicate–object “triplet”) by combining:

        - **Endpoint relevance**: how relevant the source and target nodes are to the query
          (semantic similarity if embeddings exist, otherwise keyword overlap),
        - **Edge relevance**: how relevant the edge’s fact/relation text is to the query (MVP: keyword overlap),
        - **Weights**: `w_node` and `w_edge` from config.

        The output is a *bounded* set of high-scoring edges, plus the endpoint nodes required to render them.
        This keeps it cheap and makes it work well with our downstream token budgeting.

        Notes:
        - The planner’s `plan.mode` controls *how we source candidates*:
          - `local` / `hybrid`: pick top query-relevant seed nodes, then pull adjacency edges around them.
          - `global` / `hybrid`: pull keyword-matching edges directly (fact-first).
        - We attach `vf_score_*` attributes to edges/nodes so later stages can inspect/debug scoring.
        - Behavior is intentionally kept aligned with the prior inline implementation in `build_retrieval_result`.
        """
        cfg = vft_cfg

        def _tokens(s: str) -> Set[str]:
            # Tokenize with the same normalization used by keyword search, plus split common compounds
            # (e.g. WORKS_FOR -> ["works", "for"]) to improve lexical overlap robustness.
            raw = [t for t in self._normalize_query_for_keyword_search(s).split() if t]
            out: Set[str] = set()
            for tok in raw:
                out.add(tok)
                # Also split common compound forms (e.g. WORKS_FOR) to improve overlap robustness.
                for part in re.split(r"[_\-]+", tok):
                    part = part.strip()
                    if part:
                        out.add(part)
            return out

        q_tokens = _tokens(keyword_query)

        def _kw_score(text: str) -> float:
            # Simple overlap score in [0,1] against query tokens (MVP lexical relevance signal).
            if not q_tokens:
                return 0.0
            t = _tokens(text)
            if not t:
                return 0.0
            overlap = len(q_tokens & t)
            return float(overlap) / float(max(1, len(q_tokens)))

        def _node_score(n: GraphNode) -> float:
            # Prefer semantic similarity when embeddings are present; otherwise fall back to keyword overlap.
            s_sem = 0.0
            if qv is not None and n.name_embedding:
                try:
                    s_sem = float(cosine_similarity(qv, n.name_embedding))
                except Exception:
                    s_sem = 0.0
            s_kw = _kw_score(n.name or "")
            return max(s_sem, s_kw)

        # 1) Build a candidate node set from both semantic and keyword seeds, then score nodes.
        node_candidates = stable_unique(list(semantic_nodes) + list(keyword_nodes), key_fn=lambda n: n.node_id)
        node_scores: Dict[str, float] = {}
        for n in node_candidates:
            if n.node_id:
                node_scores[n.node_id] = _node_score(n)

        seed_node_k_v = int(getattr(cfg, "seed_node_k", 0) or 0) or int(seed_k or 0) or int(getattr(cfg_r, "max_entities", 20))
        seed_edge_k_v = int(getattr(cfg, "seed_edge_k", 0) or 0) or int(getattr(cfg_r, "max_facts", 20))
        max_candidate_edges = int(getattr(cfg, "max_candidate_edges", 500) or 0)
        max_output_edges = int(getattr(cfg, "max_output_edges", 20) or 0)
        w_node = float(getattr(cfg, "w_node", 1.0) or 0.0)
        w_edge = float(getattr(cfg, "w_edge", 1.0) or 0.0)
        edge_score_mode = str(getattr(cfg, "edge_score_mode", "relation_keyword") or "relation_keyword").strip().lower()

        # 2) Choose seed nodes (local/hybrid) by node relevance. Global mode does not rely on node seeds.
        seed_nodes: List[GraphNode] = []
        if plan.mode in ("local", "hybrid") and node_candidates:
            ordered = sorted(
                [n for n in node_candidates if n.node_id],
                key=lambda n: (node_scores.get(n.node_id, 0.0), n.node_id),
                reverse=True,
            )
            seed_nodes = ordered[:seed_node_k_v]
        seed_ids = [n.node_id for n in seed_nodes if n.node_id]

        # 3) Collect edge candidates:
        # - fact-first (global/hybrid): keyword match against edges/facts
        # - neighborhood-first (local/hybrid): pull adjacency edges around seed nodes (bounded)
        edges_keyword: List[GraphEdge] = []
        if plan.mode in ("global", "hybrid") or edge_score_mode != "none":
            try:
                edges_keyword = (
                    await self._store.search(
                        scope_id,
                        keyword_query,
                        scope="edges",
                        limit=max(seed_edge_k_v, 1),
                        filters=filters,
                    )
                ).edges
            except Exception:
                edges_keyword = []

        # Local mode: add adjacency edges around top seed nodes (bounded; reuse bfs_edges_per_node).
        adj_edges: List[GraphEdge] = []
        if plan.mode in ("local", "hybrid") and seed_ids:
            per_node = bfs_edges_per_node if bfs_edges_per_node > 0 else 12
            if max_candidate_edges > 0:
                per_node = max(1, min(per_node, max_candidate_edges // max(1, len(seed_ids))))
            for nid in seed_ids:
                try:
                    res = await self._store.search(
                        scope_id,
                        "",
                        scope="edges",
                        limit=per_node,
                        filters=filters,
                        center_node_id=nid,
                    )
                    adj_edges.extend(res.edges)
                except Exception:
                    continue

        # Merge and bound candidates to control cost.
        edge_candidates = stable_unique(edges_keyword + adj_edges, key_fn=lambda e: e.edge_id)
        if max_candidate_edges > 0:
            edge_candidates = edge_candidates[:max_candidate_edges]

        def _edge_score(e: GraphEdge) -> float:
            if edge_score_mode == "none":
                return 0.0
            # MVP: keyword overlap against fact + edge_type (future: fact_embedding).
            text = f"{e.fact or ''} {e.edge_type or ''}".strip()
            return _kw_score(text)

        # 4) Triplet scoring: combine endpoint node scores + edge score with configured weights.
        scored_edges: List[Tuple[float, GraphEdge]] = []
        for e in edge_candidates:
            src = e.source_node_id
            tgt = e.target_node_id
            s_src = float(node_scores.get(src, 0.0)) if src else 0.0
            s_tgt = float(node_scores.get(tgt, 0.0)) if tgt else 0.0
            s_edge = _edge_score(e)
            total = (w_node * (s_src + s_tgt)) + (w_edge * s_edge)
            # Record score components on the edge for later hit construction.
            e.attributes = dict(e.attributes or {})
            e.attributes["vf_score_total"] = total
            e.attributes["vf_score_node_src"] = s_src
            e.attributes["vf_score_node_tgt"] = s_tgt
            e.attributes["vf_score_edge"] = s_edge
            scored_edges.append((total, e))

        # 5) Select top edges by fused score (bounded by config).
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        limit_edges = min(int(getattr(cfg_r, "max_facts", 20)), max_output_edges or int(getattr(cfg_r, "max_facts", 20)))
        edges_final = [e for _, e in scored_edges[: max(0, limit_edges)]]

        # 6) Determine required nodes to render: endpoints of selected edges + seed nodes.
        node_ids_needed: Set[str] = set(seed_ids)
        for e in edges_final:
            if e.source_node_id:
                node_ids_needed.add(e.source_node_id)
            if e.target_node_id:
                node_ids_needed.add(e.target_node_id)

        nodes_final: List[GraphNode] = []
        if node_ids_needed and bool(getattr(cfg_r, "include_entities", True)):
            try:
                nodes_final = (
                    await self._store.search(
                        scope_id,
                        "",
                        scope="nodes",
                        limit=min(len(node_ids_needed), int(cfg_r.max_entities)),
                        filters=GraphSearchFilters(
                            node_labels=filters.node_labels,
                            edge_types=filters.edge_types,
                            valid_only=filters.valid_only,
                            as_of=filters.as_of,
                            node_ids=list(node_ids_needed),
                        ),
                    )
                ).nodes
            except Exception:
                nodes_final = seed_nodes[: int(cfg_r.max_entities)]

        # 7) Attach node scores for downstream hit construction/debugging.
        for n in nodes_final:
            if n.node_id and n.node_id in node_scores:
                n.attributes = dict(n.attributes or {})
                n.attributes["vf_score"] = float(node_scores[n.node_id])

        # 8) Evidence episodes (optional): attach supporting episodes/chunks for the chosen query terms.
        # This is orthogonal to triplet scoring: evidence is extra context, not part of the edge score.
        evidence_eps: List[GraphEpisode] = []
        if bool(getattr(cfg_r, "evidence_enabled", False)):
            try:
                q_terms = ", ".join(kws.low_level + kws.high_level) if (kws.low_level or kws.high_level) else keyword_query
                max_items = int(getattr(cfg_r, "max_evidence_items", 10) or 0)
                if max_items > 0 and q_terms.strip():
                    evidence_eps = (
                        await self._store.search(
                            scope_id,
                            q_terms,
                            scope="episodes",
                            limit=max_items,
                            filters=filters,
                        )
                    ).episodes
            except Exception:
                evidence_eps = []

        return edges_final, nodes_final, evidence_eps

    async def _retrieve_recipe_seed_bfs(
        self,
        *,
        scope_id: str,
        plan: GraphQueryPlan,
        kws: GraphKeywords,
        keyword_query: str,
        cfg_r: GraphRetrievalConfig,
        filters: GraphSearchFilters,
        semantic_nodes: List[GraphNode],
        keyword_nodes: List[GraphNode],
        methods_set: Set[str],
        seed_k: int,
        bfs_max_depth: int,
        bfs_edges_per_node: int,
        rerank_enabled: bool,
        rerank_top_k: int,
    ) -> Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode]]:
        """
        Seed (keyword/semantic) + optional BFS expansion + (optional) RRF rerank.

        This is the “classic” hybrid graph retrieval recipe used by the engine:

        - **Seed selection** (local/hybrid): pick a small set of relevant entity nodes using
          keyword and/or semantic seeding (already done upstream), then optionally fuse those
          candidate lists deterministically (RRF).
        - **Expansion** (optional): expand outward from the seed nodes via a bounded BFS over edges
          (`bfs_max_depth`, `bfs_edges_per_node`), producing a local neighborhood subgraph.
        - **Global signal** (global/hybrid): optionally pull keyword-matching edges directly.
        - **Edge fusion/rerank**: combine keyword edges and BFS edges (RRF when enabled; otherwise stable concat).
        - **Node closure**: fetch the endpoint nodes needed to render the chosen edges.
        - **Evidence** (optional): attach supporting episodes based on extracted keywords.

        The recipe is intentionally bounded (seed_k, max_facts, max_entities) so it works well with
        downstream token budgeting, and remains deterministic for tests and reproducibility.

        Behavior is intentionally kept aligned with the prior inline implementation in `build_retrieval_result`.
        """
        # 1) Seed nodes for local/hybrid plans.
        # For local/hybrid, we start from entities; for global-only the seed_ids are derived from edge hits instead.
        seed_nodes: List[GraphNode] = []
        if plan.mode in ("local", "hybrid"):
            if not methods_set:
                # No explicit method list: treat whatever we have as the seed list.
                seed_nodes = (keyword_nodes or semantic_nodes)[: max(seed_k, int(cfg_r.max_entities))]
            else:
                # Deterministic fusion: Reciprocal Rank Fusion (RRF) over semantic and keyword candidates.
                sem_ids = [n.node_id for n in semantic_nodes if n.node_id]
                key_ids = [n.node_id for n in keyword_nodes if n.node_id]
                fused = rrf_scores([sem_ids, key_ids], k=60)
                ordered_ids = order_ids_by_score(fused, tie_breaker=sem_ids + key_ids)
                seed_ids = ordered_ids[:seed_k] if seed_k > 0 else []
                by_id = {n.node_id: n for n in (semantic_nodes + keyword_nodes) if n.node_id}
                seed_nodes = [by_id[i] for i in seed_ids if i in by_id]

        seed_ids = [n.node_id for n in seed_nodes if n.node_id]

        # 2) Global/hybrid: gather keyword-matching edges directly (fact-first).
        edges_keyword: List[GraphEdge] = []
        if plan.mode in ("global", "hybrid"):
            try:
                edges_keyword = (
                    await self._store.search(
                        scope_id,
                        keyword_query,
                        scope="edges",
                        limit=max(int(cfg_r.max_facts), 1),
                        filters=filters,
                    )
                ).edges
            except Exception:
                edges_keyword = []

        if plan.mode == "global" and edges_keyword:
            # In global-only mode, we don't have entity seeds; instead, derive seed_ids from the
            # endpoints of the top keyword-matching edges, then expand from those endpoints.
            endpoint_ids: List[str] = []
            for e in edges_keyword:
                if e.source_node_id:
                    endpoint_ids.append(e.source_node_id)
                if e.target_node_id:
                    endpoint_ids.append(e.target_node_id)
            endpoint_ids = stable_unique(endpoint_ids, key_fn=lambda x: x)
            seed_ids = endpoint_ids[: max(seed_k, 1)] if seed_k > 0 else endpoint_ids[: max(int(cfg_r.seed_k), 1)]

        # 3) BFS expansion around the seed set (optional, bounded).
        bfs_edges: List[GraphEdge] = []
        bfs_rank: Dict[str, int] = {}
        visited_nodes = set(seed_ids)
        frontier = list(seed_ids)

        if "bfs" in methods_set and bfs_max_depth > 0 and bfs_edges_per_node > 0 and frontier:
            for depth in range(1, bfs_max_depth + 1):
                if not frontier:
                    break
                next_frontier: List[str] = []
                for center in frontier:
                    try:
                        # Fetch a bounded number of edges adjacent to the current frontier node.
                        res = await self._store.search(
                            scope_id,
                            "",
                            scope="edges",
                            limit=bfs_edges_per_node,
                            filters=filters,
                            center_node_id=center,
                        )
                    except Exception:
                        continue
                    for e in res.edges:
                        if not e.edge_id:
                            continue
                        bfs_edges.append(e)
                        bfs_rank.setdefault(e.edge_id, depth)
                        for nid in [e.source_node_id, e.target_node_id]:
                            if not nid or nid in visited_nodes:
                                continue
                            visited_nodes.add(nid)
                            next_frontier.append(nid)
                # Stabilize frontier ordering and avoid duplicates.
                seen = set()
                frontier = []
                for nid in next_frontier:
                    if nid in seen:
                        continue
                    seen.add(nid)
                    frontier.append(nid)

        edges_keyword = stable_unique(edges_keyword, key_fn=lambda e: e.edge_id)
        bfs_edges = stable_unique(bfs_edges, key_fn=lambda e: e.edge_id)

        # 4) Edge fusion / rerank:
        # - When rerank_enabled, RRF provides a deterministic way to mix "global" keyword edges
        #   with "local" BFS edges.
        # - Otherwise we keep a stable concat with de-duping.
        kw_ids = [e.edge_id for e in edges_keyword if e.edge_id]
        bfs_ids = [e.edge_id for e in sorted(bfs_edges, key=lambda e: bfs_rank.get(e.edge_id, 10**9)) if e.edge_id]

        if rerank_enabled:
            fused_edges = rrf_scores([kw_ids, bfs_ids], k=60)
            ordered_edge_ids = order_ids_by_score(fused_edges, tie_breaker=kw_ids + bfs_ids)
            ordered_edge_ids = ordered_edge_ids[: max(rerank_top_k, int(cfg_r.max_facts))]
            by_edge_id = {e.edge_id: e for e in (edges_keyword + bfs_edges) if e.edge_id}
            edges_final = [by_edge_id[eid] for eid in ordered_edge_ids if eid in by_edge_id]
        else:
            # local mode: edges_keyword is empty by construction, so this is BFS-only.
            edges_final = stable_unique(edges_keyword + bfs_edges, key_fn=lambda e: e.edge_id)

        edges_final = edges_final[: int(cfg_r.max_facts)]

        # 5) Node closure: collect the endpoint nodes needed to render the selected edges.
        node_ids_needed = set(seed_ids)
        for e in edges_final:
            if e.source_node_id:
                node_ids_needed.add(e.source_node_id)
            if e.target_node_id:
                node_ids_needed.add(e.target_node_id)

        nodes_final: List[GraphNode] = []
        if node_ids_needed and bool(getattr(cfg_r, "include_entities", True)):
            try:
                nodes_final = (
                    await self._store.search(
                        scope_id,
                        "",
                        scope="nodes",
                        limit=min(len(node_ids_needed), int(cfg_r.max_entities)),
                        filters=GraphSearchFilters(
                            node_labels=filters.node_labels,
                            edge_types=filters.edge_types,
                            valid_only=filters.valid_only,
                            as_of=filters.as_of,
                            node_ids=list(node_ids_needed),
                        ),
                    )
                ).nodes
            except Exception:
                nodes_final = seed_nodes[: int(cfg_r.max_entities)]

        # 6) Evidence episodes (optional): attach supporting episodes/chunks for the chosen query terms.
        evidence_eps: List[GraphEpisode] = []
        if bool(getattr(cfg_r, "evidence_enabled", False)):
            try:
                q_terms = ", ".join(kws.low_level + kws.high_level) if (kws.low_level or kws.high_level) else keyword_query
                max_items = int(getattr(cfg_r, "max_evidence_items", 10) or 0)
                evidence_eps = (
                    await self._store.search(
                        scope_id,
                        q_terms,
                        scope="episodes",
                        limit=max_items if max_items > 0 else int(cfg_r.max_episodes),
                        filters=filters,
                    )
                ).episodes
            except Exception:
                evidence_eps = []

        return edges_final, nodes_final, evidence_eps

    async def _retrieve_across_scopes(
        self,
        *,
        user_id: str,
        enable_global: bool,
        global_scope_id: str,
        retrieve_scope: Callable[[str], Awaitable[Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode]]]],
    ) -> Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode]]:
        edges, nodes, eps = await retrieve_scope(user_id)
        if enable_global:
            edges_g, nodes_g, eps_g = await retrieve_scope(global_scope_id)
            edges.extend(edges_g)
            nodes.extend(nodes_g)
            eps.extend(eps_g)

        # Dedupe by ids while preserving ordering (user scope first by construction).
        dedup_edges = stable_unique(edges, key_fn=lambda e: e.edge_id)
        dedup_nodes = stable_unique(nodes, key_fn=lambda n: n.node_id)
        dedup_eps = stable_unique(eps, key_fn=lambda ep: ep.episode_id)
        return dedup_edges, dedup_nodes, dedup_eps

    def _to_hits(
        self,
        *,
        edges: List[GraphEdge],
        nodes: List[GraphNode],
        episodes: List[GraphEpisode],
    ) -> Tuple[List[GraphNodeHit], List[GraphEdgeHit], List[EvidenceItem]]:
        """
        Convert store-layer graph objects into the structured retrieval output types.

        This is a shaping step:
        - `GraphNode`  -> `GraphNodeHit`
        - `GraphEdge`  -> `GraphEdgeHit`
        - `GraphEpisode` -> `EvidenceItem`

        Why this exists:
        - Keep `build_retrieval_result` orchestration code clean: retrieval recipes operate on store objects,
          while render/budget code operates on stable "hit" types.
        - Centralize the score policy used by downstream consumers:
          - If PPR reranking ran, we attach `ppr_score` onto node/edge `attributes` and prefer it here.
          - Otherwise, we fall back to vector-fused scores (`vf_score`, `vf_score_total`) when present.
          - Otherwise, we use a constant `1.0` as a best-effort placeholder.

        Note:
        - `attrs` intentionally includes both canonical fields (e.g. `labels`, `fact`) and the original
          backend attributes so callers can inspect debug values like `vf_score_*` / `ppr_score`.
        """
        # Nodes: score is the primary ranking signal (after optional PPR), used later for budgeting/ordering.
        node_hits = [
            GraphNodeHit(
                node_id=n.node_id,
                label=n.name,
                score=float((n.attributes or {}).get("ppr_score", (n.attributes or {}).get("vf_score", 1.0))),
                attrs={"labels": list(n.labels or []), **(n.attributes or {})},
            )
            for n in nodes
        ]
        # Edges: score follows the same precedence rule as nodes (PPR first, then vector-fused score).
        edge_hits = [
            GraphEdgeHit(
                edge_id=e.edge_id,
                source_id=e.source_node_id,
                target_id=e.target_node_id,
                relation=e.edge_type,
                score=float((e.attributes or {}).get("ppr_score", (e.attributes or {}).get("vf_score_total", 1.0))),
                attrs={"fact": e.fact, "labels": list(e.labels or []), **(e.attributes or {})},
            )
            for e in edges
        ]
        # Evidence: episodes are the raw textual anchors behind extracted facts (kept as-is; scoring is MVP=1.0).
        ev_items = [
            EvidenceItem(
                source="episode",
                source_id=ep.episode_id,
                content=ep.content,
                score=1.0,
                metadata={"created_at": ep.created_at.isoformat(), **(ep.metadata or {})},
            )
            for ep in episodes
        ]
        return node_hits, edge_hits, ev_items

    def _compute_ppr_scores(
        self,
        *,
        node_ids: List[str],
        edges: List[GraphEdge],
        reset_map: Dict[str, float],
        damping: float,
        max_iters: int,
        tol: float,
        directed: bool,
        edge_weight_attr: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Minimal personalized PageRank (PPR) implementation over a bounded candidate subgraph.

        - Works over the provided `node_ids` universe.
        - Uses `reset_map` as the personalization distribution (will be normalized).
        - Uses power iteration with a dangling-mass fix redistributed via reset.
        """
        # Build index mapping.
        uniq_node_ids = list(dict.fromkeys([nid for nid in node_ids if nid]))  # stable unique
        n = len(uniq_node_ids)
        if n == 0:
            return {}, {"iters": 0, "converged": True, "delta": 0.0}

        idx_of: Dict[str, int] = {nid: i for i, nid in enumerate(uniq_node_ids)}

        # Reset vector r (normalized).
        r = [0.0] * n
        for nid, w in (reset_map or {}).items():
            if nid in idx_of:
                try:
                    r[idx_of[nid]] = max(0.0, float(w))
                except Exception:
                    r[idx_of[nid]] = 0.0
        s = float(sum(r))
        if s <= 0.0:
            # fallback to uniform
            r = [1.0 / float(n)] * n
        else:
            r = [x / s for x in r]

        d = float(damping)
        max_iters = int(max_iters)
        tol = float(tol)

        # Adjacency list: u -> [(v, w), ...]
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for e in edges:
            src = getattr(e, "source_node_id", None)
            tgt = getattr(e, "target_node_id", None)
            if not src or not tgt:
                continue
            if src not in idx_of or tgt not in idx_of:
                continue

            w = 1.0
            try:
                if edge_weight_attr:
                    w_raw = (e.attributes or {}).get(edge_weight_attr, None)
                    if w_raw is not None:
                        w = float(w_raw)
            except Exception:
                w = 1.0
            if not math.isfinite(w) or w <= 0.0:
                w = 1.0

            u = idx_of[src]
            v = idx_of[tgt]
            adj.setdefault(u, []).append((v, w))
            if not directed:
                adj.setdefault(v, []).append((u, w))

        # Power iteration.
        p = list(r)  # start at reset
        delta = 0.0
        for it in range(max_iters):
            p_new = [(1.0 - d) * r_i for r_i in r]
            dangling_mass = 0.0
            for u in range(n):
                nbrs = adj.get(u, [])
                if not nbrs:
                    dangling_mass += p[u]
                    continue
                out_w = sum(w for _, w in nbrs) or 1.0
                pu = p[u]
                if pu == 0.0:
                    continue
                scale = d * pu / out_w
                for v, w in nbrs:
                    p_new[v] += scale * w

            if dangling_mass:
                dm = d * dangling_mass
                for i in range(n):
                    p_new[i] += dm * r[i]

            delta = sum(abs(p_new[i] - p[i]) for i in range(n))
            p = p_new
            if delta <= tol:
                scores = {nid: float(p[idx_of[nid]]) for nid in uniq_node_ids}
                return scores, {"iters": it + 1, "converged": True, "delta": float(delta)}

        scores = {nid: float(p[idx_of[nid]]) for nid in uniq_node_ids}
        return scores, {"iters": max_iters, "converged": False, "delta": float(delta)}

    def _build_ppr_reset_map(
        self,
        *,
        cfg_ppr: Any,
        nodes: List[GraphNode],
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Build the personalization / reset distribution used by PPR.

        In Personalized PageRank, the reset vector controls where the random walk "teleports" to.
        Intuitively, it encodes *what the query cares about* (within the candidate subgraph).

        We support two MVP modes:
        - **`reset_mode="node_scores"`**: use each node's existing relevance signal as its reset weight.
          Today that relevance signal is best-effort `node.attributes["vf_score"]` (set by the vector-fused
          recipe) and otherwise defaults to `1.0`.
        - **`reset_mode="uniform"`**: give each candidate node equal reset weight (pure centrality over the
          candidate subgraph).

        Cost/quality knobs:
        - `reset_top_k_nodes`: if > 0, we keep only the top-K highest-weight nodes in the reset map and set
          all others implicitly to 0. This keeps PPR "focused" and avoids smearing mass across many nodes.
        - `reset_min_weight`: drop nodes whose computed weight is below this threshold.

        Output:
        - A sparse dict `{node_id: weight}` (not normalized here; `_compute_ppr_scores` normalizes it).
        - Ordering is made deterministic by sorting by (weight desc, node_id desc) before top-k truncation.
        """
        reset_mode = str(getattr(cfg_ppr, "reset_mode", "node_scores") or "node_scores").strip().lower()
        top_k = int(getattr(cfg_ppr, "reset_top_k_nodes", 0) or 0)
        min_w = float(getattr(cfg_ppr, "reset_min_weight", 0.0) or 0.0)

        # Query-embedding mode: seed from cosine similarity between query and node embeddings.
        if reset_mode == "query_embedding" and query_embedding is not None:
            from ctxforge.graph.retrieval.pagerank import compute_seed_scores
            return compute_seed_scores(query_embedding, nodes, top_k=top_k or len(nodes))

        weights: List[Tuple[str, float]] = []
        for n in nodes:
            if not n.node_id:
                continue
            if reset_mode == "uniform":
                # Uniform reset: every node gets the same mass before normalization.
                w = 1.0
            else:
                # Score-based reset: bias teleportation toward nodes already deemed query-relevant.
                try:
                    w = float((n.attributes or {}).get("vf_score", 1.0))
                except Exception:
                    w = 1.0
            if w < min_w:
                continue
            weights.append((n.node_id, max(0.0, w)))

        # Sort for determinism and optionally top-k gate.
        weights.sort(key=lambda x: (x[1], x[0]), reverse=True)
        if top_k > 0:
            weights = weights[:top_k]

        return {nid: w for nid, w in weights}

    def _apply_ppr_rerank(
        self,
        *,
        cfg_r: GraphRetrievalConfig,
        edges: List[GraphEdge],
        nodes: List[GraphNode],
        episodes: List[GraphEpisode],
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode], Dict[str, Any]]:
        """
        Optionally apply Personalized PageRank (PPR) reranking to a *bounded* candidate subgraph.

        This is a post-retrieval, pre-budgeting step:
        - Retrieval recipes (seed+BFS, vector-fused triplets, etc.) produce a candidate `edges/nodes/episodes`.
        - If enabled, we run PPR **only over that candidate subgraph** and use the resulting scores to:
          - reorder `nodes` (and attach `node.attributes["ppr_score"]`),
          - optionally derive an edge score from endpoint node scores and reorder `edges`
            (and attach `edge.attributes["ppr_score"]`).
        - Token budgeting runs later, so PPR mainly influences *which* items survive truncation.

        Safety/compatibility:
        - When disabled, returns inputs unchanged and `debug={"enabled": False}`.
        - When the candidate subgraph is too small (configurable), we skip to avoid noise and overhead.

        Returns:
        - `(edges, nodes, episodes, debug)` where `debug` is included in `GraphRetrievalResult.debug["ppr"]`.
        """
        cfg_ppr = getattr(cfg_r, "ppr", None)
        if not cfg_ppr or not bool(getattr(cfg_ppr, "enabled", False)):
            return edges, nodes, episodes, {"enabled": False}

        # Skip PPR for tiny candidate graphs where the stationary distribution tends to be unstable/unhelpful.
        min_nodes = int(getattr(cfg_ppr, "min_nodes", 3) or 0)
        min_edges = int(getattr(cfg_ppr, "min_edges", 2) or 0)
        if len(nodes) < min_nodes or len(edges) < min_edges:
            return edges, nodes, episodes, {"enabled": True, "skipped": True, "reason": "too_small"}

        # Build the node universe for the PPR run. We include:
        # - any node objects we already fetched, plus
        # - endpoints referenced by candidate edges (even if the node object is missing).
        node_ids_universe: List[str] = []
        node_ids_universe.extend([n.node_id for n in nodes if n.node_id])
        for e in edges:
            if getattr(e, "source_node_id", None):
                node_ids_universe.append(e.source_node_id)
            if getattr(e, "target_node_id", None):
                node_ids_universe.append(e.target_node_id)
        node_ids_universe = list(dict.fromkeys([nid for nid in node_ids_universe if nid]))

        # Personalization vector: determines where the random-walk teleports.
        # By default this uses existing node relevance scores (vf_score), optionally sparsified by top-k.
        reset_map = self._build_ppr_reset_map(cfg_ppr=cfg_ppr, nodes=nodes, query_embedding=query_embedding)
        scores, meta = self._compute_ppr_scores(
            node_ids=node_ids_universe,
            edges=edges,
            reset_map=reset_map,
            damping=float(getattr(cfg_ppr, "damping", 0.85) or 0.85),
            max_iters=int(getattr(cfg_ppr, "max_iters", 50) or 50),
            tol=float(getattr(cfg_ppr, "tol", 1e-6) or 1e-6),
            directed=bool(getattr(cfg_ppr, "directed", False)),
            edge_weight_attr=str(getattr(cfg_ppr, "use_edge_weight_attr", "weight") or "weight"),
        )

        rerank_nodes = bool(getattr(cfg_ppr, "rerank_nodes", True))
        rerank_edges = bool(getattr(cfg_ppr, "rerank_edges", True))
        edge_score_mode = str(getattr(cfg_ppr, "edge_score_mode", "endpoint_sum") or "endpoint_sum").strip().lower()

        if rerank_nodes:
            # Attach score and reorder deterministically.
            # Note: PPR may rank a high-centrality node above a high-reset node; that is expected.
            for n in nodes:
                if not n.node_id:
                    continue
                n.attributes = dict(n.attributes or {})
                if n.node_id in scores:
                    n.attributes["ppr_score"] = float(scores[n.node_id])
            nodes = sorted(
                nodes,
                key=lambda n: (float(scores.get(n.node_id, 0.0)), str(n.node_id)),
                reverse=True,
            )

        if rerank_edges:
            # Attach an edge-level score derived from node-level PPR.
            # MVP: endpoint_sum makes edges connecting high-PPR nodes rise to the top.
            for e in edges:
                src = getattr(e, "source_node_id", None)
                tgt = getattr(e, "target_node_id", None)
                s = 0.0
                if edge_score_mode == "endpoint_sum":
                    s = float(scores.get(src, 0.0)) + float(scores.get(tgt, 0.0))
                e.attributes = dict(e.attributes or {})
                e.attributes["ppr_score"] = float(s)
            if edge_score_mode == "endpoint_sum":
                edges = sorted(
                    edges,
                    key=lambda e: (
                        float((e.attributes or {}).get("ppr_score", 0.0)),
                        str(getattr(e, "edge_id", "")),
                    ),
                    reverse=True,
                )

        # Keep debug payload small but sufficient for tuning and troubleshooting.
        debug = {
            "enabled": True,
            "skipped": False,
            "node_count": len(node_ids_universe),
            "edge_count": len(edges),
            "reset_mode": str(getattr(cfg_ppr, "reset_mode", "node_scores")),
            "reset_top_k_nodes": int(getattr(cfg_ppr, "reset_top_k_nodes", 0) or 0),
            "damping": float(getattr(cfg_ppr, "damping", 0.85) or 0.85),
            **meta,
        }
        return edges, nodes, episodes, debug

    def _apply_budgets(
        self,
        *,
        plan: GraphQueryPlan,
        cfg_r: GraphRetrievalConfig,
        node_hits: List[GraphNodeHit],
        edge_hits: List[GraphEdgeHit],
        ev_items: List[EvidenceItem],
    ) -> Tuple[List[GraphNodeHit], List[GraphEdgeHit], List[EvidenceItem]]:
        """
        Apply token budgets to the structured retrieval outputs (nodes/edges/evidence).

        This is the final “packing” step for graph retrieval before returning a `GraphRetrievalResult`.
        It intentionally runs **after** any reranking steps (e.g. PPR) so that truncation preserves the
        most relevant items.

        Budgets:
        - `max_entity_tokens`: budget for rendering entity lines (`<ENTITIES>`)
        - `max_relation_tokens`: budget for rendering fact/relation lines (`<FACTS>`)
        - `max_evidence_tokens`: budget for rendering evidence snippets (`<EVIDENCE>`)
        - `max_total_tokens`: end-to-end budget across all three components

        Enforcement strategy (best-effort):
        - Per-component budgets are enforced first via `_truncate_by_token_budget`.
          - If no tokenizer provider is configured, per-component budgets effectively no-op.
          - `_truncate_by_token_budget` is designed to keep **at least one** item when possible.
        - Then, if `max_total_tokens` is set and a tokenizer is available, we trim in this order:
          **evidence → edges → nodes**.
          Rationale: evidence is usually the most verbose, and facts/entities are more compact.
        """
        max_entity_tokens = int(getattr(cfg_r, "max_entity_tokens", 0) or 0)
        max_relation_tokens = int(getattr(cfg_r, "max_relation_tokens", 0) or 0)
        max_evidence_tokens = int(getattr(cfg_r, "max_evidence_tokens", 0) or 0)
        max_total_tokens = int(getattr(cfg_r, "max_total_tokens", 0) or 0)

        # Budget per component. Each list is truncated independently based on its rendering.
        node_hits_b = self._truncate_by_token_budget(
            node_hits, render=lambda x: f"- {x.label} ({', '.join(map(str, x.attrs.get('labels', []) or []))})", max_tokens=max_entity_tokens
        )
        edge_hits_b = self._truncate_by_token_budget(
            edge_hits, render=lambda x: f"- {x.attrs.get('fact') or x.relation} ({x.source_id}->{x.target_id})", max_tokens=max_relation_tokens
        )
        ev_items_b = self._truncate_by_token_budget(
            ev_items, render=lambda x: f"- {x.content}", max_tokens=max_evidence_tokens
        )

        # Total budget: trim evidence first (best-effort) then edges then nodes.
        if self._tokenizer is not None and max_total_tokens > 0:
            def tokens_for(ns, es, evs) -> int:
                # A lightweight proxy string for token counting. We count tokens on a compact summary
                # to avoid having to fully render the final section here.
                s = []
                s.append(f"plan={plan.mode} {plan.reason}")
                s.extend([f"N:{n.label}" for n in ns])
                s.extend([f"E:{e.attrs.get('fact') or e.relation}" for e in es])
                s.extend([f"EV:{ev.content}" for ev in evs])
                return int(self._tokenizer.count_tokens("\n".join(s)))

            # Trim order matters: evidence tends to be large and redundant, then edges, then nodes.
            while ev_items_b and tokens_for(node_hits_b, edge_hits_b, ev_items_b) > max_total_tokens:
                ev_items_b = ev_items_b[:-1]
            while edge_hits_b and tokens_for(node_hits_b, edge_hits_b, ev_items_b) > max_total_tokens:
                edge_hits_b = edge_hits_b[:-1]
            while node_hits_b and tokens_for(node_hits_b, edge_hits_b, ev_items_b) > max_total_tokens:
                node_hits_b = node_hits_b[:-1]

        return node_hits_b, edge_hits_b, ev_items_b

    async def build_retrieval_result(self, *, user_id: str, query: str) -> Optional[GraphRetrievalResult]:
        """
        Retrieval API: produce a plan + a budgeted subgraph + evidence.

        This is the **structured** graph retrieval entrypoint. It returns a `GraphRetrievalResult`
        that includes:
        - the retrieval plan (mode + reason),
        - a bounded set of node/edge hits,
        - optional evidence (episodes),
        - and debug metadata.

        High-level pipeline:
        - **Plan**: extract keywords and decide a retrieval mode (`local` / `global` / `hybrid`).
        - **Seed**: compute query embedding (optional) and retrieve candidate seed nodes (keyword/semantic).
        - **Retrieve**: run one retrieval recipe for each scope (user scope, plus optional global scope),
          producing candidate edges/nodes/evidence.
        - **Merge**: merge across scopes and stable-dedupe results.
        - **Rerank** (optional): apply PPR reranking over the bounded candidate subgraph.
        - **Shape**: convert store objects to stable "hit" types.
        - **Budget**: apply component and total token budgets as the final packing step.

        This structure supports richer rendering (planner/evidence blocks), two-step fusion workflows,
        and keeps `build_section()` backward compatible by letting it choose whether to use this
        structured path or the legacy string renderer.
        """
        if not getattr(self._cfg.graph, "enabled", False):
            return None
        if not getattr(self._cfg.graph.retrieval, "enabled", True):
            return None

        cfg_r = self._cfg.graph.retrieval

        # ------------------------------------------------------------------
        # 1) Plan: keyword extraction + mode decision
        # ------------------------------------------------------------------
        planner_mode = str(getattr(cfg_r, "planner_mode", "auto") or "auto")
        max_ll = int(getattr(cfg_r, "max_low_level_keywords", 10) or 0)
        max_hl = int(getattr(cfg_r, "max_high_level_keywords", 8) or 0)
        kws = extract_keywords_heuristic(query, max_low_level=max_ll, max_high_level=max_hl)
        keyword_query = self._normalize_query_for_keyword_search(query) or query

        plan: GraphQueryPlan = plan_mode(
            planner_mode=planner_mode,
            keywords=kws,
            fallback_to_global_if_no_entities=True,
            fallback_to_local_if_no_themes=True,
        )

        # ------------------------------------------------------------------
        # 2) Scope setup + core retrieval knobs
        # ------------------------------------------------------------------
        # Retrieve across scopes (user scope first). This gives the engine a simple global-memory hook.
        enable_global = bool(getattr(self._cfg.scopes, "enable_global", False))
        global_scope_id = str(getattr(self._cfg.scopes, "global_scope_id", "global"))
        if user_id == global_scope_id:
            enable_global = False

        filters = GraphSearchFilters(valid_only=bool(getattr(cfg_r, "valid_only", True)))
        methods = [m.lower() for m in (getattr(cfg_r, "methods", None) or [])]
        methods_set = set([m for m in methods if m])
        seed_k = int(getattr(cfg_r, "seed_k", 0) or 0)
        bfs_max_depth = int(getattr(cfg_r, "bfs_max_depth", 0) or 0)
        bfs_edges_per_node = int(getattr(cfg_r, "bfs_edges_per_node", 0) or 0)
        rerank_enabled = bool(getattr(cfg_r, "rerank_enabled", False))
        rerank_top_k = int(getattr(cfg_r, "rerank_top_k", 0) or 0)
        vft_cfg = getattr(cfg_r, "vector_fused_triplets", None)
        vft_enabled = bool(getattr(vft_cfg, "enabled", False)) and ("vector_fused_triplets" in methods_set)

        # ------------------------------------------------------------------
        # 3) Optional query embedding (used by semantic seeding and vector-fused recipe)
        # ------------------------------------------------------------------
        qv = await self._embed_query_vector(query=query, methods_set=methods_set, vft_enabled=vft_enabled)

        async def retrieve_scope(scope_id: str) -> Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode]]:
            # Per-scope retrieval:
            # - get candidate seed nodes (keyword + semantic)
            # - dispatch to the selected retrieval recipe
            semantic_nodes, keyword_nodes = await self._seed_nodes_for_scope(
                scope_id=scope_id,
                keyword_query=keyword_query,
                methods_set=methods_set,
                seed_k=seed_k,
                filters=filters,
                qv=qv,
            )
            if vft_enabled:
                return await self._retrieve_recipe_vector_fused_triplets(
                    scope_id=scope_id,
                    plan=plan,
                    kws=kws,
                    keyword_query=keyword_query,
                    cfg_r=cfg_r,
                    vft_cfg=vft_cfg,
                    filters=filters,
                    qv=qv,
                    semantic_nodes=semantic_nodes,
                    keyword_nodes=keyword_nodes,
                    seed_k=seed_k,
                    bfs_edges_per_node=bfs_edges_per_node,
                )

            return await self._retrieve_recipe_seed_bfs(
                scope_id=scope_id,
                plan=plan,
                kws=kws,
                keyword_query=keyword_query,
                cfg_r=cfg_r,
                filters=filters,
                semantic_nodes=semantic_nodes,
                keyword_nodes=keyword_nodes,
                methods_set=methods_set,
                seed_k=seed_k,
                bfs_max_depth=bfs_max_depth,
                bfs_edges_per_node=bfs_edges_per_node,
                rerank_enabled=rerank_enabled,
                rerank_top_k=rerank_top_k,
            )

        # ------------------------------------------------------------------
        # 4) Retrieve across scopes and stable-dedupe (user scope first)
        # ------------------------------------------------------------------
        dedup_edges, dedup_nodes, dedup_eps = await self._retrieve_across_scopes(
            user_id=user_id,
            enable_global=enable_global,
            global_scope_id=global_scope_id,
            retrieve_scope=retrieve_scope,
        )

        # ------------------------------------------------------------------
        # 5) Optional graph-aware rerank (PPR) over the bounded candidate subgraph
        # ------------------------------------------------------------------
        dedup_edges, dedup_nodes, dedup_eps, ppr_debug = self._apply_ppr_rerank(
            cfg_r=cfg_r,
            edges=dedup_edges,
            nodes=dedup_nodes,
            episodes=dedup_eps,
            query_embedding=qv,
        )

        # ------------------------------------------------------------------
        # 5b) Bridge discovery + path mining (Phase 5)
        # ------------------------------------------------------------------
        reasoning_paths: List[ReasoningPath] = []
        bridge_connections: List[BridgeConnection] = []
        pm_config = getattr(getattr(self._cfg, "memory_quality", None), "graph_path_mining", None)
        if pm_config is not None and getattr(pm_config, "enabled", False) and dedup_nodes and dedup_edges:
            # Identify disconnected pairs among chronologically adjacent nodes
            sorted_nodes = sorted(
                dedup_nodes,
                key=lambda n: str(
                    (n.attributes or {}).get("created_at")
                    or (n.attributes or {}).get("timestamp")
                    or "9999"
                ),
            )
            disconnected_pairs = []
            existing_node_ids = set(n.node_id for n in dedup_nodes)
            for i in range(len(sorted_nodes) - 1):
                conn = check_connection(
                    sorted_nodes[i],
                    sorted_nodes[i + 1],
                    dedup_edges,
                    temporal_flow_hours=float(getattr(pm_config, "temporal_flow_hours", 6.0)),
                )
                if conn is None:
                    disconnected_pairs.append((sorted_nodes[i], sorted_nodes[i + 1]))

            # Bridge discovery
            if disconnected_pairs and getattr(pm_config, "bridge_discovery_enabled", False):
                bridge_nodes, bridge_connections = await find_bridge_candidates(
                    scope_id=user_id,
                    disconnected_pairs=disconnected_pairs,
                    graph_store=self._store,
                    embedding_provider=self._embed,
                    config=pm_config,
                    existing_node_ids=existing_node_ids,
                )
                if bridge_nodes:
                    dedup_nodes.extend(bridge_nodes)

            # Path mining (DFS)
            reasoning_paths = discover_reasoning_paths(
                nodes=dedup_nodes,
                edges=dedup_edges,
                config=pm_config,
            )

            # Node scoring and budget
            dedup_nodes = rank_and_limit_nodes(
                dedup_nodes,
                query=query,
                config=pm_config,
            )

        # ------------------------------------------------------------------
        # 6) Shape to stable hit types + apply token budgets
        # ------------------------------------------------------------------
        node_hits, edge_hits, ev_items = self._to_hits(edges=dedup_edges, nodes=dedup_nodes, episodes=dedup_eps)
        node_hits_b, edge_hits_b, ev_items_b = self._apply_budgets(
            plan=plan, cfg_r=cfg_r, node_hits=node_hits, edge_hits=edge_hits, ev_items=ev_items
        )

        # If everything is empty after budgeting, do not emit a graph section at all.
        if not node_hits_b and not edge_hits_b and not ev_items_b:
            return None

        return GraphRetrievalResult(
            plan_mode=plan.mode,
            plan_reason=plan.reason,
            nodes=node_hits_b,
            edges=edge_hits_b,
            evidence=ev_items_b,
            debug={
                "keywords_low_level": kws.low_level,
                "keywords_high_level": kws.high_level,
                "methods": sorted(list(methods_set)),
                "vector_fused_triplets": {
                    "enabled": bool(vft_enabled),
                    "edge_score_mode": str(getattr(vft_cfg, "edge_score_mode", "relation_keyword")) if vft_cfg else None,
                },
                "ppr": ppr_debug,
                "tokenizer": getattr(self._tokenizer, "name", None) if self._tokenizer else None,
                "path_mining": {
                    "enabled": pm_config is not None and getattr(pm_config, "enabled", False),
                    "reasoning_paths_count": len(reasoning_paths),
                    "bridge_connections_count": len(bridge_connections),
                },
            },
            reasoning_paths=reasoning_paths,
            bridge_connections=bridge_connections,
        )

    async def build_section(self, *, user_id: str, query: str) -> Optional[str]:
        """Build the rendered graph context section (facts/entities/communities/episodes)."""
        if not getattr(self._cfg.graph, "enabled", False):
            return None
        if not getattr(self._cfg.graph.retrieval, "enabled", True):
            return None

        cfg_r = self._cfg.graph.retrieval
        if self._planner_path_enabled(cfg_r):
            rr = await self.build_retrieval_result(user_id=user_id, query=query)
            if rr is None:
                # When explicitly enabled (forced mode / evidence / budgets), do not fall back.
                return None
            communities = await self._fetch_communities_for_node_ids(
                user_id=user_id, node_ids=[n.node_id for n in rr.nodes if n.node_id]
            )
            return self._render_retrieval_result(rr, communities=communities)

        return await self._build_section_legacy(user_id=user_id, query=query)

    def _planner_path_enabled(self, cfg_r: GraphRetrievalConfig) -> bool:
        """
        Backward compatible gating: default configs keep using legacy retrieval unless explicitly enabled.
        """
        planner_mode = str(getattr(cfg_r, "planner_mode", "auto") or "auto").strip().lower()
        evidence_enabled = bool(getattr(cfg_r, "evidence_enabled", False))
        budgets_enabled = any(
            int(getattr(cfg_r, k, 0) or 0) > 0
            for k in ("max_entity_tokens", "max_relation_tokens", "max_evidence_tokens", "max_total_tokens")
        )
        return bool(evidence_enabled or budgets_enabled or (planner_mode != "auto"))

    async def _fetch_communities_for_node_ids(self, *, user_id: str, node_ids: List[str]) -> List[GraphCommunity]:
        if not node_ids:
            return []
        if not getattr(getattr(self._cfg.graph, "communities", None), "enabled", False):
            return []

        enable_global = bool(getattr(self._cfg.scopes, "enable_global", False))
        global_scope_id = str(getattr(self._cfg.scopes, "global_scope_id", "global"))
        if user_id == global_scope_id:
            enable_global = False

        try:
            communities: List[GraphCommunity] = await self._store.get_communities_for_nodes(
                user_id,
                node_ids,
                limit=int(getattr(self._cfg.graph.communities, "max_communities", 5)),
            )
            if enable_global:
                communities_g = await self._store.get_communities_for_nodes(
                    global_scope_id,
                    node_ids,
                    limit=int(getattr(self._cfg.graph.communities, "max_communities", 5)),
                )
                seen_c = set([(c.scope_id, c.community_id) for c in communities])
                for c in communities_g:
                    key = (c.scope_id, c.community_id)
                    if key in seen_c:
                        continue
                    seen_c.add(key)
                    communities.append(c)
            return communities
        except Exception:
            return []

    def _render_retrieval_result(self, rr: GraphRetrievalResult, *, communities: Optional[List[GraphCommunity]] = None) -> str:
        cfg_r = self._cfg.graph.retrieval

        # ----- Topology-aware mode (Phase 6) -----
        topo_cfg = getattr(cfg_r, "topology_aware", None)
        if topo_cfg is not None and getattr(topo_cfg, "enabled", False):
            return self._render_topology_aware(rr, communities=communities)

        # ----- Legacy flat-list mode -----
        lines: List[str] = []
        lines.append("<RETRIEVAL_PLAN>")
        lines.append(f"- mode={rr.plan_mode}")
        lines.append(f"- reason={rr.plan_reason}")

        if rr.edges:
            lines.append("")
            lines.append("<FACTS>")
            for e in rr.edges:
                fact = e.attrs.get("fact") or f"{e.relation} ({e.source_id} -> {e.target_id})"
                lines.append(f"- {fact}".strip())

        if bool(getattr(cfg_r, "include_entities", True)) and rr.nodes:
            lines.append("")
            lines.append("<ENTITIES>")
            for n in rr.nodes:
                labels = n.attrs.get("labels", [])
                labels_s = ", ".join([str(x) for x in labels]) if labels else ""
                lines.append(f"- {n.label} ({labels_s})".strip())

        if communities:
            lines.append("")
            lines.append("<COMMUNITIES>")
            for c in communities[: int(getattr(self._cfg.graph.communities, "max_communities", 5))]:
                overlap = f", overlap: {c.overlap}" if c.overlap is not None else ""
                lines.append(f"- {c.name} (members: {c.member_count}{overlap}) :: {c.summary}".strip())

        if rr.evidence:
            lines.append("")
            lines.append("<EVIDENCE>")
            for ev in rr.evidence:
                created_at = ev.metadata.get("created_at") or "unknown"
                content = (ev.content or "").replace("\n", " ").strip()
                if len(content) > 220:
                    content = content[:220].rstrip() + "..."
                lines.append(f"- [EP:{ev.source_id}] {created_at} :: {content}")

        # Reasoning paths (Phase 5)
        if rr.reasoning_paths:
            # Build a node_id -> label map for rendering
            node_label_map: Dict[str, str] = {}
            for idx, n in enumerate(rr.nodes, 1):
                node_label_map[n.node_id] = f"E{idx}"

            valid_paths = []
            seen_label_seqs: Set[Tuple[str, ...]] = set()
            for path in rr.reasoning_paths:
                labels = []
                for nid in path.node_ids:
                    label = node_label_map.get(nid)
                    if label:
                        labels.append(label)
                if len(labels) >= 2:
                    label_seq = tuple(labels)
                    if label_seq not in seen_label_seqs:
                        seen_label_seqs.add(label_seq)
                        valid_paths.append(labels)
            if valid_paths:
                lines.append("")
                lines.append("<REASONING_PATHS>")
                for i, labels in enumerate(valid_paths, 1):
                    lines.append(f"  {i}. {' -> '.join(labels)}")

        # Bridge connections (Phase 5)
        if rr.bridge_connections:
            lines.append("")
            lines.append(f"<BRIDGE_CONNECTIONS: {len(rr.bridge_connections)} inferred links found>")

        return "\n".join(lines).strip()

    def _render_topology_aware(
        self,
        rr: GraphRetrievalResult,
        *,
        communities: Optional[List[GraphCommunity]] = None,
    ) -> str:
        """Render using topology-aware structured format (Phase 6)."""
        cfg_r = self._cfg.graph.retrieval
        topo_cfg = cfg_r.topology_aware
        renderer = TopologyAwareRenderer(topo_cfg)
        text = renderer.render(rr)

        # Append communities if present (not part of topology view but still useful)
        if communities:
            comm_lines: List[str] = ["", "[Communities]"]
            for c in communities[: int(getattr(self._cfg.graph.communities, "max_communities", 5))]:
                overlap = f", overlap: {c.overlap}" if c.overlap is not None else ""
                comm_lines.append(f"- {c.name} (members: {c.member_count}{overlap}) :: {c.summary}".strip())
            text = text + "\n" + "\n".join(comm_lines)

        return text.strip()

    async def _build_section_legacy(self, *, user_id: str, query: str) -> Optional[str]:
        """
        Legacy string-rendering graph retrieval.

        This method implements the pre-planner/pre-evidence behavior of `GraphService.build_section()`.
        It is kept for backward compatibility so existing deployments get the same output shape
        (via `format_graph_context(...)`) unless the newer planner/evidence/budget path is explicitly enabled.

        High-level behavior:
        - Compute an optional query embedding (semantic seeding) when embeddings are enabled.
        - Per-scope retrieval:
          - seed nodes via keyword and/or semantic search
          - pull keyword-matching edges
          - optionally expand neighborhood via bounded BFS from the seed nodes
          - optionally fuse keyword edges and BFS edges using deterministic RRF
          - fetch the endpoint nodes needed to render those edges
          - optionally fetch episode hits (legacy `include_episodes` mode)
        - Merge user scope + optional global scope, stable-dedupe, and optionally fetch communities.
        - Render to a single legacy string via `format_graph_context(...)`.

        Notes:
        - This path does **not** use the new planner plan/evidence blocks and does **not** apply token budgets.
          It relies on count-based limits (`max_facts`, `max_entities`, `max_episodes`).
        """
        enable_global = bool(getattr(self._cfg.scopes, "enable_global", False))
        global_scope_id = str(getattr(self._cfg.scopes, "global_scope_id", "global"))
        if user_id == global_scope_id:
            enable_global = False

        filters = GraphSearchFilters(valid_only=bool(getattr(self._cfg.graph.retrieval, "valid_only", True)))
        # Normalize once and reuse: punctuation-stripped query works better for whitespace tokenizers.
        keyword_query = self._normalize_query_for_keyword_search(query) or query

        methods = [m.lower() for m in (getattr(self._cfg.graph.retrieval, "methods", None) or [])]
        methods_set = set([m for m in methods if m])
        seed_k = int(getattr(self._cfg.graph.retrieval, "seed_k", 0) or 0)
        bfs_max_depth = int(getattr(self._cfg.graph.retrieval, "bfs_max_depth", 0) or 0)
        bfs_edges_per_node = int(getattr(self._cfg.graph.retrieval, "bfs_edges_per_node", 0) or 0)
        rerank_enabled = bool(getattr(self._cfg.graph.retrieval, "rerank_enabled", False))
        rerank_top_k = int(getattr(self._cfg.graph.retrieval, "rerank_top_k", 0) or 0)

        # Compute query embedding once for semantic seeding (best-effort).
        qv = None
        if (
            "semantic" in methods_set
            and self._embed is not None
            and getattr(getattr(self._cfg.graph, "embeddings", None), "enabled", False)
            and query.strip()
        ):
            try:
                embed_model = getattr(self._cfg.graph.embeddings.embedding, "model", None)
                qv = await self._embed.embed_single(query, model=embed_model)
            except Exception:
                qv = None

        async def retrieve_scope(scope_id: str) -> Tuple[List[GraphEdge], List[GraphNode], List[GraphEpisode]]:
            # Per-scope retrieval for legacy rendering.
            semantic_nodes: List[GraphNode] = []
            keyword_nodes: List[GraphNode] = []

            # Semantic seed nodes.
            if qv is not None and "semantic" in methods_set and seed_k > 0:
                try:
                    semantic_nodes = await self._store.search_nodes_semantic(
                        scope_id, qv, limit=max(seed_k, 1), filters=filters
                    )
                except Exception:
                    semantic_nodes = []

            # Keyword seed nodes.
            if "keyword" in methods_set and seed_k > 0:
                try:
                    keyword_nodes = (
                        await self._store.search(scope_id, query, scope="nodes", limit=max(seed_k * 2, 1), filters=filters)
                    ).nodes
                except Exception:
                    keyword_nodes = []

            if not methods_set:
                # No explicit method list: accept whichever seed list is available.
                seed_nodes = (keyword_nodes or semantic_nodes)[: max(seed_k, int(self._cfg.graph.retrieval.max_entities))]
            else:
                # Deterministic fusion of semantic/keyword seeds via RRF.
                sem_ids = [n.node_id for n in semantic_nodes if n.node_id]
                key_ids = [n.node_id for n in keyword_nodes if n.node_id]
                fused = rrf_scores([sem_ids, key_ids], k=60)
                ordered_ids = order_ids_by_score(fused, tie_breaker=sem_ids + key_ids)
                seed_ids = ordered_ids[:seed_k] if seed_k > 0 else []
                by_id = {n.node_id: n for n in (semantic_nodes + keyword_nodes) if n.node_id}
                seed_nodes = [by_id[i] for i in seed_ids if i in by_id]

            seed_ids = [n.node_id for n in seed_nodes if n.node_id]

            # Keyword edges (fact-first).
            edges_keyword: List[GraphEdge] = []
            try:
                edges_keyword = (
                    await self._store.search(
                        scope_id, keyword_query, scope="edges", limit=max(int(self._cfg.graph.retrieval.max_facts), 1), filters=filters
                    )
                ).edges
            except Exception:
                edges_keyword = []

            # BFS expansion (neighborhood-first), bounded by depth and fanout per node.
            bfs_edges: List[GraphEdge] = []
            bfs_rank: Dict[str, int] = {}
            visited_nodes = set(seed_ids)
            frontier = list(seed_ids)

            if "bfs" in methods_set and bfs_max_depth > 0 and bfs_edges_per_node > 0 and frontier:
                for depth in range(1, bfs_max_depth + 1):
                    if not frontier:
                        break
                    next_frontier: List[str] = []
                    for center in frontier:
                        try:
                            res = await self._store.search(
                                scope_id,
                                "",
                                scope="edges",
                                limit=bfs_edges_per_node,
                                filters=filters,
                                center_node_id=center,
                            )
                        except Exception:
                            continue
                        for e in res.edges:
                            if not e.edge_id:
                                continue
                            bfs_edges.append(e)
                            bfs_rank.setdefault(e.edge_id, depth)
                            for nid in [e.source_node_id, e.target_node_id]:
                                if not nid or nid in visited_nodes:
                                    continue
                                visited_nodes.add(nid)
                                next_frontier.append(nid)
                    seen = set()
                    frontier = []
                    for nid in next_frontier:
                        if nid in seen:
                            continue
                        seen.add(nid)
                        frontier.append(nid)

            edges_keyword = stable_unique(edges_keyword, key_fn=lambda e: e.edge_id)
            bfs_edges = stable_unique(bfs_edges, key_fn=lambda e: e.edge_id)

            kw_ids = [e.edge_id for e in edges_keyword if e.edge_id]
            bfs_ids = [e.edge_id for e in sorted(bfs_edges, key=lambda e: bfs_rank.get(e.edge_id, 10**9)) if e.edge_id]

            if rerank_enabled:
                # Mix keyword edges and BFS edges deterministically.
                fused_edges = rrf_scores([kw_ids, bfs_ids], k=60)
                ordered_edge_ids = order_ids_by_score(fused_edges, tie_breaker=kw_ids + bfs_ids)
                ordered_edge_ids = ordered_edge_ids[: max(rerank_top_k, int(self._cfg.graph.retrieval.max_facts))]
                by_edge_id = {e.edge_id: e for e in (edges_keyword + bfs_edges) if e.edge_id}
                edges_final = [by_edge_id[eid] for eid in ordered_edge_ids if eid in by_edge_id]
            else:
                edges_final = stable_unique(edges_keyword + bfs_edges, key_fn=lambda e: e.edge_id)

            edges_final = edges_final[: int(self._cfg.graph.retrieval.max_facts)]

            # Fetch the nodes needed to render the final edge list.
            node_ids_needed = set(seed_ids)
            for e in edges_final:
                if e.source_node_id:
                    node_ids_needed.add(e.source_node_id)
                if e.target_node_id:
                    node_ids_needed.add(e.target_node_id)

            nodes_final: List[GraphNode] = []
            if node_ids_needed and bool(getattr(self._cfg.graph.retrieval, "include_entities", True)):
                try:
                    nodes_final = (
                        await self._store.search(
                            scope_id,
                            "",
                            scope="nodes",
                            limit=min(len(node_ids_needed), int(self._cfg.graph.retrieval.max_entities)),
                            filters=GraphSearchFilters(
                                node_labels=filters.node_labels,
                                edge_types=filters.edge_types,
                                valid_only=filters.valid_only,
                                as_of=filters.as_of,
                                node_ids=list(node_ids_needed),
                            ),
                        )
                    ).nodes
                except Exception:
                    nodes_final = seed_nodes[: int(self._cfg.graph.retrieval.max_entities)]

            episodes_final: List[GraphEpisode] = []
            if getattr(self._cfg.graph.retrieval, "include_episodes", False):
                # Legacy optional: episode hits are a separate retrieval mode from the newer evidence linking.
                try:
                    episodes_final = (
                        await self._store.search(
                            scope_id,
                            keyword_query,
                            scope="episodes",
                            limit=int(self._cfg.graph.retrieval.max_episodes),
                            filters=filters,
                        )
                    ).episodes
                except Exception:
                    episodes_final = []

            return edges_final, nodes_final, episodes_final

        # Retrieve user scope first, then optionally global scope, then stable-dedupe.
        edges, nodes, episodes = await retrieve_scope(user_id)

        if enable_global:
            edges_g, nodes_g, episodes_g = await retrieve_scope(global_scope_id)
            edges.extend(edges_g)
            nodes.extend(nodes_g)
            episodes.extend(episodes_g)

        # Dedupe by ids while preserving ordering (user scope first by construction).
        seen_edges = set()
        dedup_edges = []
        for e in edges:
            if e.edge_id in seen_edges:
                continue
            seen_edges.add(e.edge_id)
            dedup_edges.append(e)

        seen_nodes = set()
        dedup_nodes = []
        for n in nodes:
            if n.node_id in seen_nodes:
                continue
            seen_nodes.add(n.node_id)
            dedup_nodes.append(n)

        communities = await self._fetch_communities_for_node_ids(
            user_id=user_id, node_ids=[n.node_id for n in dedup_nodes if n.node_id]
        )

        if not dedup_edges and not dedup_nodes and not episodes and not communities:
            return None

        # Render with the legacy formatting helper to preserve output shape expected by older callers/tests.
        return format_graph_context(
            edges=dedup_edges[: int(self._cfg.graph.retrieval.max_facts)],
            nodes=dedup_nodes[: int(self._cfg.graph.retrieval.max_entities)],
            episodes=episodes[: int(self._cfg.graph.retrieval.max_episodes)],
            communities=communities[: int(getattr(self._cfg.graph.communities, "max_communities", 5))] if communities else None,
            include_entities=bool(getattr(self._cfg.graph.retrieval, "include_entities", True)),
            include_episodes=bool(getattr(self._cfg.graph.retrieval, "include_episodes", False)),
            include_communities=bool(getattr(getattr(self._cfg.graph, "communities", None), "enabled", False)),
        )


