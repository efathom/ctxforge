"""
Topology-Aware Context Serialization.

Converts a ``GraphRetrievalResult`` into structured, evidence-grounded text
for LLM consumption.  Instead of flat lists of facts/entities/evidence, the
output uses:

- **Labeled facts** ``[F1]``, ``[F2]`` with optional inline evidence
- **Annotated reasoning paths** that reference fact labels and edge types
- **Bridge connection summaries** explaining inferred links

This follows the topology-aware contextualization approach (Section 2.3):
    C_graph = labeled_facts + reasoning_paths + bridge_summaries

The renderer is stateless and config-driven.  Enable it via
``graph.retrieval.topology_aware.enabled = true`` in the engine config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ctxforge.config.base import TopologySerializationConfig
from ctxforge.graph.retrieval.types import (
    BridgeConnection,
    EvidenceItem,
    GraphEdgeHit,
    GraphNodeHit,
    GraphRetrievalResult,
    ReasoningPath,
)

# ---------------------------------------------------------------------------
# Structured models
# ---------------------------------------------------------------------------


@dataclass
class LabeledFact:
    """A single labeled fact with optional inline evidence."""

    label: str  # e.g. "F1"
    node_id: str
    entity_name: str
    content: str
    timestamp: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class AnnotatedPath:
    """A reasoning path rendered with fact labels and edge types."""

    index: int
    labels: List[str]  # e.g. ["F1", "F2", "F3"]
    edge_types: List[str]  # e.g. ["KNOWS", "WORKS_AT"]
    summary: str  # e.g. "F1 --[KNOWS]--> F2 --[WORKS_AT]--> F3"


@dataclass
class BridgeSummary:
    """A human-readable summary of a bridge connection."""

    source_label: str
    bridge_label: str
    target_label: str
    source_name: str
    bridge_name: str
    target_name: str
    bridge_type: str
    description: str


@dataclass
class TopologyView:
    """Complete topology-aware view of a graph retrieval result."""

    facts: List[LabeledFact] = field(default_factory=list)
    paths: List[AnnotatedPath] = field(default_factory=list)
    bridges: List[BridgeSummary] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class TopologyAwareRenderer:
    """
    Renders a ``GraphRetrievalResult`` as topology-aware structured text.

    Usage::

        renderer = TopologyAwareRenderer(config)
        text = renderer.render(retrieval_result)
    """

    def __init__(self, config: TopologySerializationConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_view(self, rr: GraphRetrievalResult) -> TopologyView:
        """Build a structured ``TopologyView`` from a retrieval result."""
        node_label_map, node_name_map = self._build_label_maps(rr.nodes)

        facts = self._build_facts(rr.nodes, rr.edges, rr.evidence, node_label_map)
        paths = self._build_paths(rr.reasoning_paths, node_label_map, rr.edges)
        bridges = self._build_bridges(rr.bridge_connections, node_label_map, node_name_map)

        return TopologyView(facts=facts, paths=paths, bridges=bridges)

    def render(self, rr: GraphRetrievalResult) -> str:
        """Render a ``GraphRetrievalResult`` as topology-aware text."""
        view = self.build_view(rr)
        return self.render_view(view)

    def render_view(self, view: TopologyView) -> str:
        """Render a pre-built ``TopologyView`` to text."""
        lines: List[str] = []

        # Part 1: Labeled facts
        if view.facts:
            lines.append("[Facts from Graph]")
            for fact in view.facts:
                lines.append(self._render_fact(fact))
                for ev_line in fact.evidence:
                    lines.append(f"  {ev_line}")

        # Part 2: Reasoning paths
        if view.paths:
            if lines:
                lines.append("")
            lines.append("[Reasoning Paths]")
            for path in view.paths:
                lines.append(f"  {path.index}. {path.summary}")

        # Part 3: Bridge summaries
        if view.bridges:
            if lines:
                lines.append("")
            lines.append(f"[Bridge Connections: {len(view.bridges)} inferred links]")
            for bridge in view.bridges:
                lines.append(f"  - {bridge.description}")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_label_maps(
        self, nodes: List[GraphNodeHit]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build node_id -> label and node_id -> name maps."""
        prefix = self._cfg.fact_label_prefix
        label_map: Dict[str, str] = {}
        name_map: Dict[str, str] = {}
        for idx, n in enumerate(nodes, 1):
            label_map[n.node_id] = f"{prefix}{idx}"
            name_map[n.node_id] = n.label
        return label_map, name_map

    def _build_facts(
        self,
        nodes: List[GraphNodeHit],
        edges: List[GraphEdgeHit],
        evidence: List[EvidenceItem],
        label_map: Dict[str, str],
    ) -> List[LabeledFact]:
        """Build labeled facts from nodes with inline evidence."""
        max_chars = self._cfg.max_fact_content_chars
        max_ev = self._cfg.max_evidence_per_fact
        max_ev_chars = self._cfg.max_evidence_chars

        # Pre-index: which edges mention each node (for fact content)
        node_facts: Dict[str, List[str]] = {}
        for e in edges:
            fact_text = e.attrs.get("fact") or f"{e.relation} ({e.source_id} -> {e.target_id})"
            for nid in (e.source_id, e.target_id):
                if nid not in node_facts:
                    node_facts[nid] = []
                node_facts[nid].append(fact_text)

        # Pre-index: evidence by source_id (episode node)
        ev_by_source: Dict[str, List[EvidenceItem]] = {}
        for ev in evidence:
            if ev.source_id not in ev_by_source:
                ev_by_source[ev.source_id] = []
            ev_by_source[ev.source_id].append(ev)

        facts: List[LabeledFact] = []
        for n in nodes:
            label = label_map.get(n.node_id, "")
            if not label:
                continue

            # Content: prefer edge facts mentioning this node, else node summary
            related_facts = node_facts.get(n.node_id, [])
            if related_facts:
                content = "; ".join(related_facts)
            else:
                summary = n.attrs.get("summary", "")
                content = summary if summary else n.label

            content = content.replace("\n", " ").strip()
            if max_chars and len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."

            # Timestamp
            timestamp: Optional[str] = None
            if self._cfg.include_timestamps:
                ts = n.attrs.get("created_at") or n.attrs.get("timestamp")
                if ts is not None:
                    timestamp = str(ts)

            # Inline evidence
            ev_lines: List[str] = []
            if max_ev > 0:
                node_evidence = ev_by_source.get(n.node_id, [])
                for ev in node_evidence[:max_ev]:
                    ev_content = (ev.content or "").replace("\n", " ").strip()
                    if max_ev_chars and len(ev_content) > max_ev_chars:
                        ev_content = ev_content[:max_ev_chars].rstrip() + "..."
                    ev_date = ev.metadata.get("created_at", "")
                    if ev_date:
                        ev_lines.append(f"[Evidence: {ev_date}] {ev_content}")
                    else:
                        ev_lines.append(f"[Evidence] {ev_content}")

            facts.append(
                LabeledFact(
                    label=label,
                    node_id=n.node_id,
                    entity_name=n.label,
                    content=content,
                    timestamp=timestamp,
                    evidence=ev_lines,
                )
            )

        return facts

    def _build_paths(
        self,
        reasoning_paths: List[ReasoningPath],
        label_map: Dict[str, str],
        edges: List[GraphEdgeHit],
    ) -> List[AnnotatedPath]:
        """Build annotated reasoning paths with fact labels and edge types."""
        max_paths = self._cfg.max_reasoning_paths
        include_edge_types = self._cfg.include_edge_types_in_paths

        # Build edge lookup: (source, target) -> edge_type
        edge_type_map: Dict[Tuple[str, str], str] = {}
        for e in edges:
            edge_type_map[(e.source_id, e.target_id)] = e.relation
            edge_type_map[(e.target_id, e.source_id)] = e.relation

        valid_paths: List[AnnotatedPath] = []
        seen_label_seqs: Set[Tuple[str, ...]] = set()

        for path in reasoning_paths:
            labels: List[str] = []
            for nid in path.node_ids:
                label = label_map.get(nid)
                if label:
                    labels.append(label)
            if len(labels) < 2:
                continue

            label_seq = tuple(labels)
            if label_seq in seen_label_seqs:
                continue
            seen_label_seqs.add(label_seq)

            # Resolve edge types for the path
            resolved_edge_types: List[str] = []
            if include_edge_types and len(path.node_ids) >= 2:
                for i in range(len(path.node_ids) - 1):
                    src = path.node_ids[i]
                    tgt = path.node_ids[i + 1]
                    etype = edge_type_map.get((src, tgt))
                    if etype is None and i < len(path.edge_types):
                        etype = path.edge_types[i]
                    resolved_edge_types.append(etype or "related")

            # Build summary string
            if include_edge_types and resolved_edge_types:
                parts: List[str] = [labels[0]]
                for j, et in enumerate(resolved_edge_types):
                    if j + 1 < len(labels):
                        parts.append(f"--[{et}]-->")
                        parts.append(labels[j + 1])
                summary = " ".join(parts)
            else:
                summary = " -> ".join(labels)

            valid_paths.append(
                AnnotatedPath(
                    index=len(valid_paths) + 1,
                    labels=labels,
                    edge_types=resolved_edge_types,
                    summary=summary,
                )
            )

            if len(valid_paths) >= max_paths:
                break

        return valid_paths

    def _build_bridges(
        self,
        bridge_connections: List[BridgeConnection],
        label_map: Dict[str, str],
        name_map: Dict[str, str],
    ) -> List[BridgeSummary]:
        """Build bridge connection summaries."""
        max_bridges = self._cfg.max_bridge_summaries
        summaries: List[BridgeSummary] = []

        for bc in bridge_connections[:max_bridges]:
            src_label = label_map.get(bc.source_node_id, bc.source_node_id)
            brg_label = label_map.get(bc.bridge_node_id, bc.bridge_node_id)
            tgt_label = label_map.get(bc.target_node_id, bc.target_node_id)

            src_name = name_map.get(bc.source_node_id, bc.source_node_id)
            brg_name = name_map.get(bc.bridge_node_id, bc.bridge_node_id)
            tgt_name = name_map.get(bc.target_node_id, bc.target_node_id)

            description = (
                f"{src_name} [{src_label}] and {tgt_name} [{tgt_label}] "
                f"connected via {brg_name} [{brg_label}] ({bc.bridge_type})"
            )

            summaries.append(
                BridgeSummary(
                    source_label=src_label,
                    bridge_label=brg_label,
                    target_label=tgt_label,
                    source_name=src_name,
                    bridge_name=brg_name,
                    target_name=tgt_name,
                    bridge_type=bc.bridge_type,
                    description=description,
                )
            )

        return summaries

    def _render_fact(self, fact: LabeledFact) -> str:
        """Render a single labeled fact line."""
        if fact.timestamp:
            return f"[{fact.label}] {fact.timestamp}: {fact.content}"
        return f"[{fact.label}] {fact.content}"
