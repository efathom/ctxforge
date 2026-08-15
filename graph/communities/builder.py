from __future__ import annotations

"""
Community building (derived clusters).

This module builds communities from an existing entity graph:
- cluster nodes by structure (label propagation on the adjacency graph)
- generate a short name + summary per cluster (LLM when available, deterministic fallback otherwise)
- optionally embed name/summary for semantic lookup
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from ctxforge.protocols.graph import GraphCommunity, GraphEdge, GraphNode
from ctxforge.protocols.llm import ChatMessage, IEmbeddingProvider, ILLMProvider


def _stable_community_id(scope_id: str, member_node_ids: Sequence[str]) -> str:
    """Generate a stable community id from scope_id + sorted member ids."""
    raw = "|".join([scope_id] + sorted([x.strip() for x in member_node_ids if x]))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def label_propagation_clusters(
    *,
    node_ids: List[str],
    edges: List[GraphEdge],
    max_iters: int = 20,
) -> List[List[str]]:
    """
    Deterministic label propagation clustering over an undirected adjacency graph.

    We treat all edges as undirected for the purpose of community structure.
    Edge multiplicity increases neighbor weight.
    """
    node_ids = [x for x in node_ids if x]
    if not node_ids:
        return []

    # Build weighted adjacency: u -> {v: weight}
    adj: Dict[str, Dict[str, int]] = {n: {} for n in node_ids}
    for e in edges:
        a = e.source_node_id
        b = e.target_node_id
        if a not in adj or b not in adj or a == b:
            continue
        adj[a][b] = adj[a].get(b, 0) + 1
        adj[b][a] = adj[b].get(a, 0) + 1

    # Initialize each node label to itself.
    labels: Dict[str, str] = {n: n for n in node_ids}

    for _ in range(max(1, int(max_iters))):
        changed = 0
        # Deterministic update order: sort node_ids.
        for n in sorted(node_ids):
            nbrs = adj.get(n) or {}
            if not nbrs:
                continue
            # Aggregate neighbor label weights.
            weights: Dict[str, int] = {}
            for nbr, w in nbrs.items():
                lbl = labels.get(nbr, nbr)
                weights[lbl] = weights.get(lbl, 0) + int(w)
            if not weights:
                continue
            # Pick the label with max weight; deterministic tie-breaker is lexicographic.
            best = sorted(weights.items(), key=lambda x: (-x[1], x[0]))[0][0]
            if labels[n] != best:
                labels[n] = best
                changed += 1
        if changed == 0:
            break

    clusters: Dict[str, List[str]] = {}
    for n, lbl in labels.items():
        clusters.setdefault(lbl, []).append(n)
    # Deterministic cluster ordering: by size desc then id.
    out = sorted([sorted(members) for members in clusters.values()], key=lambda xs: (-len(xs), xs[0]))
    return out


@dataclass
class CommunityBuildConfig:
    min_cluster_size: int = 2
    max_communities: int = 5
    model: Optional[str] = None
    max_concurrency: int = 4


class CommunityBuilder:
    """Build communities for a scope from nodes/edges and optionally enrich with LLM + embeddings."""

    def __init__(
        self,
        *,
        llm_provider: Optional[ILLMProvider] = None,
        embedding_provider: Optional[IEmbeddingProvider] = None,
    ):
        self._llm = llm_provider
        self._embed = embedding_provider

    async def build(
        self,
        *,
        scope_id: str,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        config: CommunityBuildConfig,
    ) -> Tuple[List[GraphCommunity], List[tuple[str, str]]]:
        # 1) Cluster (structure-only)
        clusters = label_propagation_clusters(
            node_ids=[n.node_id for n in nodes if n.node_id],
            edges=edges,
        )
        clusters = [c for c in clusters if len(c) >= int(config.min_cluster_size)]
        clusters = clusters[: max(0, int(config.max_communities))]

        node_by_id = {n.node_id: n for n in nodes if n.node_id}

        # 2) Name + summarize each cluster (LLM best-effort; deterministic fallback).
        sem = asyncio.Semaphore(max(1, int(config.max_concurrency)))

        async def build_one(member_ids: List[str]) -> GraphCommunity:
            async with sem:
                member_texts = []
                for nid in member_ids[:40]:
                    n = node_by_id.get(nid)
                    if not n:
                        continue
                    member_texts.append((n.summary or n.name or "").strip())
                member_texts = [t for t in member_texts if t]
                if not member_texts:
                    name = "Community"
                    summary = "Derived cluster of related entities."
                else:
                    name, summary = await self._summarize(member_texts, model=config.model)

                cid = _stable_community_id(scope_id, member_ids)
                return GraphCommunity(
                    community_id=cid,
                    scope_id=scope_id,
                    name=name,
                    summary=summary,
                    member_count=len(member_ids),
                    updated_at=datetime.now(timezone.utc),
                )

        communities = await asyncio.gather(*[build_one(m) for m in clusters], return_exceptions=True)
        comms: List[GraphCommunity] = [c for c in communities if isinstance(c, GraphCommunity)]

        # 3) Embed name + summary (best-effort; uses batch embedding).
        if self._embed is not None and comms:
            try:
                names = [c.name for c in comms]
                summaries = [c.summary for c in comms]
                # One batch call for name embeddings, one for summary embeddings.
                name_resp = await self._embed.embed(names)
                sum_resp = await self._embed.embed(summaries)
                if name_resp.embeddings and len(name_resp.embeddings) == len(comms):
                    for c, v in zip(comms, name_resp.embeddings, strict=True):
                        c.name_embedding = v
                if sum_resp.embeddings and len(sum_resp.embeddings) == len(comms):
                    for c, v in zip(comms, sum_resp.embeddings, strict=True):
                        c.summary_embedding = v
            except Exception:
                pass

        memberships: List[tuple[str, str]] = []
        for c, member_ids in zip(comms, clusters, strict=False):
            for nid in member_ids:
                memberships.append((c.community_id, nid))
        return comms, memberships

    async def _summarize(self, member_texts: List[str], *, model: Optional[str]) -> Tuple[str, str]:
        """
        Produce (name, summary) for a cluster.

        If an LLM provider is not available, return a deterministic fallback.
        """
        # Deterministic fallback: name from the first 1–2 items, summary as a short join.
        if self._llm is None:
            name = " / ".join(member_texts[:2])[:60]
            summary = "; ".join(member_texts[:8])[:400]
            return name or "Community", summary or "Derived cluster of related entities."

        prompt = (
            "You name and summarize a cluster of related entities.\n\n"
            "Rules:\n"
            "- Keep the name short (2-6 words).\n"
            "- The summary should be 2-4 sentences.\n"
            "- Only use the provided bullets.\n"
            "- Output JSON only: {\"name\": \"...\", \"summary\": \"...\"}\n"
        )
        bullets = "\n".join([f"- {t}" for t in member_texts[:40]])
        resp = await self._llm.chat(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=bullets),
            ],
            model=model,
            temperature=0.2,
            max_tokens=400,
        )
        # Best-effort parse (simple).
        import json

        from ctxforge.extraction.utils import extract_json_from_text

        js = extract_json_from_text(resp.content or "")
        if not js:
            return "Community", "; ".join(member_texts[:8])[:400]
        try:
            data = json.loads(js)
        except Exception:
            return "Community", "; ".join(member_texts[:8])[:400]
        name = str(data.get("name") or "Community").strip()
        summary = str(data.get("summary") or "; ".join(member_texts[:8])[:400]).strip()
        return name, summary


