from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from ctxforge.protocols.graph import GraphCommunity, GraphEdge, GraphEpisode, GraphNode


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    return dt.isoformat()


def format_graph_context(
    *,
    edges: List[GraphEdge],
    nodes: List[GraphNode],
    episodes: List[GraphEpisode],
    communities: Optional[List[GraphCommunity]] = None,
    include_entities: bool = True,
    include_episodes: bool = False,
    include_communities: bool = True,
) -> str:
    parts: List[str] = []

    if edges:
        parts.append("<FACTS>")
        for e in edges:
            validity = f"{_fmt_dt(e.valid_at)} - {(_fmt_dt(e.invalid_at) if e.invalid_at else 'present')}"
            fact = e.fact or f"{e.edge_type} ({e.source_node_id} -> {e.target_node_id})"
            parts.append(f"- {fact} [{validity}]")

    if include_entities and nodes:
        # Separate entity, fact, and passage nodes for clearer rendering.
        entity_nodes = [n for n in nodes if not (n.labels and n.labels[0] in ("Fact", "Passage"))]
        fact_nodes = [n for n in nodes if n.labels and n.labels[0] == "Fact"]
        passage_nodes = [n for n in nodes if n.labels and n.labels[0] == "Passage"]

        if entity_nodes:
            parts.append("")
            parts.append("<ENTITIES>")
            for n in entity_nodes:
                labels = ", ".join(n.labels) if n.labels else ""
                summary = f" :: {n.summary}" if n.summary else ""
                parts.append(f"- {n.name} ({labels}){summary}".strip())

        if fact_nodes:
            parts.append("")
            parts.append("<STRUCTURED_FACTS>")
            for n in fact_nodes:
                attrs = n.attributes or {}
                conf = attrs.get("confidence", "")
                conf_str = f" (confidence: {conf})" if conf else ""
                parts.append(f"- FACT: {n.summary or n.name}{conf_str}")

        if passage_nodes:
            parts.append("")
            parts.append("<EVIDENCE>")
            for n in passage_nodes:
                parts.append(f"- {n.summary or n.name}")

    if include_communities and communities:
        parts.append("")
        parts.append("<COMMUNITIES>")
        for c in communities:
            overlap = f", overlap: {c.overlap}" if c.overlap is not None else ""
            parts.append(f"- {c.name} (members: {c.member_count}{overlap}) :: {c.summary}".strip())

    if include_episodes and episodes:
        parts.append("")
        parts.append("<EPISODES>")
        for ep in episodes:
            parts.append(f"- {_fmt_dt(ep.created_at)} :: {ep.content}")

    return "\n".join([p for p in parts if p is not None])


