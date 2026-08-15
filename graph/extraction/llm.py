from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil.parser import isoparse

from ctxforge.core.alignment_types import AlignmentStatus
from ctxforge.extraction.alignment import WordAligner
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.graph.ontology import GraphOntology
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode, IGraphExtractor
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

DEFAULT_GRAPH_EXTRACTION_PROMPT = """You extract entities and relationships from episodes using a provided ontology.

Rules:
- Only use entity types and edge types that exist in the ontology.
- Only create edges that match allowed (source_type, target_type) pairs for that edge_type.
- Prefer explicit facts. Do not invent.
- For entity names, use EXACT text from the source when possible for better source grounding.
- Return valid JSON only.

JSON schema:
{
  "entities": [
    {"name": "string", "entity_type": "string", "attributes": { ... }, "summary": "string|null"}
  ],
  "edges": [
    {"source_name": "string", "source_type": "string", "edge_type": "string", "target_name": "string", "target_type": "string",
     "attributes": { ... }, "fact": "string|null", "valid_at": "ISO8601|null", "invalid_at": "ISO8601|null"}
  ]
}"""


@dataclass
class GraphExtractionConfig:
    """Configuration for graph extraction."""
    
    extraction_passes: int = 1
    enable_alignment: bool = True
    fuzzy_alignment_threshold: float = 0.75
    temperature: float = 0.0
    max_tokens: int = 1400


def _stable_node_id(scope_id: str, entity_type: str, name: str) -> str:
    raw = f"{scope_id}|{entity_type}|{name}".strip().lower()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _stable_edge_id(scope_id: str, src_id: str, edge_type: str, dst_id: str, fact: str | None) -> str:
    raw = f"{scope_id}|{src_id}|{edge_type}|{dst_id}|{fact or ''}".strip().lower()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


class LLMGraphExtractor(IGraphExtractor):
    """
    Enhanced LLM Graph Extractor with:
    - Multi-pass extraction for improved recall
    - Source text alignment for provenance
    """
    
    def __init__(
        self,
        llm_provider: ILLMProvider,
        *,
        system_prompt: str = DEFAULT_GRAPH_EXTRACTION_PROMPT,
        default_model: Optional[str] = None,
        max_tokens: int = 1400,
        default_config: Optional[GraphExtractionConfig] = None,
        aligner: Optional[WordAligner] = None,
    ):
        self._llm = llm_provider
        self._system_prompt = system_prompt
        self._default_model = default_model
        self._max_tokens = max_tokens
        self._config = default_config or GraphExtractionConfig()
        self._aligner = aligner  # Lazy initialization

    def _get_aligner(self) -> WordAligner:
        """Get or create word aligner."""
        if self._aligner is not None:
            return self._aligner
        return WordAligner(fuzzy_threshold=self._config.fuzzy_alignment_threshold)

    async def extract(
        self,
        *,
        scope_id: str,
        episodes: List[GraphEpisode],
        ontology: GraphOntology,
        model: Optional[str] = None,
        extraction_passes: Optional[int] = None,
        enable_alignment: Optional[bool] = None,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Extract with multi-pass and source grounding."""
        if not episodes:
            return [], []
        
        # Merge config
        passes = extraction_passes if extraction_passes is not None else self._config.extraction_passes
        align = enable_alignment if enable_alignment is not None else self._config.enable_alignment
        
        # Build episode text map for alignment
        episode_texts = {ep.episode_id: ep.content for ep in episodes}
        
        # Multi-pass extraction
        all_nodes_by_pass: List[Dict[Tuple[str, str], GraphNode]] = []
        all_edges_by_pass: List[List[GraphEdge]] = []
        
        for pass_num in range(1, passes + 1):
            nodes, edges = await self._single_pass_extract(
                scope_id=scope_id,
                episodes=episodes,
                ontology=ontology,
                model=model,
                pass_num=pass_num,
                enable_alignment=align,
                episode_texts=episode_texts,
            )
            all_nodes_by_pass.append(nodes)
            all_edges_by_pass.append(edges)
        
        # Merge results
        merged_nodes = self._merge_nodes(all_nodes_by_pass)
        merged_edges = self._merge_edges(all_edges_by_pass)

        # Enrich with Passage and Fact nodes if the ontology supports them
        passage_nodes, fact_nodes, extra_edges = self._build_passage_and_fact_nodes(
            scope_id=scope_id,
            episodes=episodes,
            entity_nodes=merged_nodes,
            entity_edges=merged_edges,
            ontology=ontology,
        )
        for pn in passage_nodes:
            key = (pn.labels[0] if pn.labels else "Passage", pn.name.lower())
            merged_nodes.setdefault(key, pn)
        for fn in fact_nodes:
            key = (fn.labels[0] if fn.labels else "Fact", fn.name.lower())
            merged_nodes.setdefault(key, fn)
        merged_edges.extend(extra_edges)

        return list(merged_nodes.values()), merged_edges
    
    async def _single_pass_extract(
        self,
        scope_id: str,
        episodes: List[GraphEpisode],
        ontology: GraphOntology,
        model: Optional[str],
        pass_num: int,
        enable_alignment: bool,
        episode_texts: Dict[str, str],
    ) -> Tuple[Dict[Tuple[str, str], GraphNode], List[GraphEdge]]:
        """Single extraction pass."""
        user_lines = [
            "Ontology entity types:",
            ", ".join(sorted(list(ontology.entity_types.keys()))),
            "",
            "Ontology edge types:",
            ", ".join(sorted(list(ontology.edge_types.keys()))),
            "",
            "Allowed edges:",
        ]
        for edge_type, pairs in sorted(ontology.allowed_edges.items()):
            pairs_s = ", ".join([f"{a}->{b}" for a, b in pairs])
            user_lines.append(f"- {edge_type}: {pairs_s}")
        user_lines.append("")
        user_lines.append("Episodes:")
        for ep in episodes:
            user_lines.append(f"- [{ep.episode_id}] ({ep.content_type}) {ep.content}")
        
        if pass_num > 1:
            user_lines.append("")
            user_lines.append(f"(Pass {pass_num}: Look for entities and relationships you may have missed.)")

        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content="\n".join(user_lines)),
        ]

        resp = await self._llm.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=self._config.temperature,
            max_tokens=self._max_tokens,
        )

        parsed = self._parse(resp.content or "")
        if parsed is None:
            return {}, []

        entities_raw, edges_raw = parsed
        nodes_by_key: Dict[Tuple[str, str], GraphNode] = {}

        for ent in entities_raw:
            name = (ent.get("name") or "").strip()
            etype = (ent.get("entity_type") or "").strip()
            if not name or not etype:
                continue
            if not ontology.is_entity_type_known(etype):
                continue
            attrs = ent.get("attributes") if isinstance(ent.get("attributes"), dict) else {}
            attrs = ontology.validate_entity_attributes(etype, attrs)
            summary = ent.get("summary") if isinstance(ent.get("summary"), str) else None
            node_id = _stable_node_id(scope_id, etype, name)
            
            node = GraphNode(
                node_id=node_id,
                scope_id=scope_id,
                name=name,
                labels=[etype],
                attributes=attrs,
                summary=summary,
                source_episode_ids=[ep.episode_id for ep in episodes],
            )
            
            # Align to source
            if enable_alignment:
                aligner = self._get_aligner()
                for ep in episodes:
                    result = aligner.align(name, ep.content)
                    if result.status != AlignmentStatus.UNALIGNED and result.char_span:
                        node.source_spans[ep.episode_id] = result.char_span
                        node.alignment_status = result.status
                        node.extraction_confidence = result.confidence
                        break  # Use first match
            
            nodes_by_key[(etype, name.lower())] = node

        graph_edges: List[GraphEdge] = []
        for e in edges_raw:
            src_name = (e.get("source_name") or "").strip()
            src_type = (e.get("source_type") or "").strip()
            dst_name = (e.get("target_name") or "").strip()
            dst_type = (e.get("target_type") or "").strip()
            edge_type = (e.get("edge_type") or "").strip()
            if not (src_name and src_type and dst_name and dst_type and edge_type):
                continue
            if not ontology.is_edge_type_known(edge_type):
                continue
            if not ontology.is_entity_type_known(src_type) or not ontology.is_entity_type_known(dst_type):
                continue
            if not ontology.is_edge_allowed(edge_type, src_type, dst_type):
                continue

            src = nodes_by_key.get((src_type, src_name.lower()))
            dst = nodes_by_key.get((dst_type, dst_name.lower()))
            if src is None or dst is None:
                continue

            attrs = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
            attrs = ontology.validate_edge_attributes(edge_type, attrs)
            fact = e.get("fact") if isinstance(e.get("fact"), str) else None

            valid_at = self._parse_dt(e.get("valid_at"))
            invalid_at = self._parse_dt(e.get("invalid_at"))
            edge_id = _stable_edge_id(scope_id, src.node_id, edge_type, dst.node_id, fact)

            edge = GraphEdge(
                edge_id=edge_id,
                scope_id=scope_id,
                source_node_id=src.node_id,
                target_node_id=dst.node_id,
                edge_type=edge_type,
                fact=fact,
                labels=[edge_type],
                attributes=attrs,
                valid_at=valid_at,
                invalid_at=invalid_at,
                source_episode_ids=[ep.episode_id for ep in episodes],
            )
            
            # Align fact to source
            if enable_alignment and fact:
                aligner = self._get_aligner()
                for ep in episodes:
                    result = aligner.align(fact, ep.content)
                    if result.status != AlignmentStatus.UNALIGNED and result.char_span:
                        edge.source_spans[ep.episode_id] = result.char_span
                        edge.alignment_status = result.status
                        edge.extraction_confidence = result.confidence
                        break
            
            graph_edges.append(edge)

        return nodes_by_key, graph_edges
    
    def _merge_nodes(
        self,
        nodes_by_pass: List[Dict[Tuple[str, str], GraphNode]],
    ) -> Dict[Tuple[str, str], GraphNode]:
        """Merge nodes from multiple passes, first-pass wins."""
        if len(nodes_by_pass) == 1:
            return nodes_by_pass[0]
        
        merged = dict(nodes_by_pass[0])
        
        for pass_nodes in nodes_by_pass[1:]:
            for key, node in pass_nodes.items():
                if key not in merged:
                    merged[key] = node
        
        return merged
    
    def _merge_edges(
        self,
        edges_by_pass: List[List[GraphEdge]],
    ) -> List[GraphEdge]:
        """Merge edges from multiple passes, avoid duplicates."""
        if len(edges_by_pass) == 1:
            return edges_by_pass[0]
        
        seen_ids = set()
        merged = []
        
        for pass_edges in edges_by_pass:
            for edge in pass_edges:
                if edge.edge_id not in seen_ids:
                    seen_ids.add(edge.edge_id)
                    merged.append(edge)
        
        return merged

    def _build_passage_and_fact_nodes(
        self,
        *,
        scope_id: str,
        episodes: List[GraphEpisode],
        entity_nodes: Dict[Tuple[str, str], GraphNode],
        entity_edges: List[GraphEdge],
        ontology: GraphOntology,
    ) -> Tuple[List[GraphNode], List[GraphNode], List[GraphEdge]]:
        """Create Passage and Fact nodes from extracted entities and edges.

        - One Passage node per episode.
        - MENTIONS edges from Passage to each entity mentioned in that episode.
        - One Fact node per edge that has a ``fact`` text.
        - EVIDENCES edges from Passage to Fact, SUBJECT_OF/OBJECT_OF from entity to Fact.

        Skipped entirely when the ontology does not define Passage/Fact types.
        """
        if not ontology.is_entity_type_known("Passage") or not ontology.is_entity_type_known("Fact"):
            return [], [], []

        passage_nodes: List[GraphNode] = []
        fact_nodes: List[GraphNode] = []
        extra_edges: List[GraphEdge] = []

        # Build node-id lookup.
        nid_to_node: Dict[str, GraphNode] = {}
        for node in entity_nodes.values():
            nid_to_node[node.node_id] = node

        # One Passage per episode.
        episode_passage_ids: Dict[str, str] = {}
        for idx, ep in enumerate(episodes):
            passage_id = _stable_node_id(scope_id, "Passage", ep.episode_id)
            episode_passage_ids[ep.episode_id] = passage_id
            content_preview = ep.content[:120].replace("\n", " ") if ep.content else ""
            passage_nodes.append(GraphNode(
                node_id=passage_id,
                scope_id=scope_id,
                name=f"passage_{idx}",
                labels=["Passage"],
                attributes={
                    "source_episode_id": ep.episode_id,
                    "chunk_index": idx,
                    "token_count": len(ep.content.split()) if ep.content else 0,
                },
                summary=content_preview,
                source_episode_ids=[ep.episode_id],
            ))

            # MENTIONS edges: Passage → entity nodes grounded in this episode.
            for node in entity_nodes.values():
                if ep.episode_id in node.source_episode_ids:
                    for lbl in node.labels:
                        if ontology.is_edge_allowed("MENTIONS", "Passage", lbl):
                            eid = _stable_edge_id(scope_id, passage_id, "MENTIONS", node.node_id, None)
                            extra_edges.append(GraphEdge(
                                edge_id=eid,
                                scope_id=scope_id,
                                source_node_id=passage_id,
                                target_node_id=node.node_id,
                                edge_type="MENTIONS",
                                labels=["MENTIONS"],
                                source_episode_ids=[ep.episode_id],
                            ))
                            break

        # One Fact node per edge with a fact description.
        for edge in entity_edges:
            if not edge.fact:
                continue
            fact_id = _stable_node_id(scope_id, "Fact", edge.fact)
            src_node = nid_to_node.get(edge.source_node_id)
            tgt_node = nid_to_node.get(edge.target_node_id)

            fact_nodes.append(GraphNode(
                node_id=fact_id,
                scope_id=scope_id,
                name=edge.fact[:80],
                labels=["Fact"],
                attributes={
                    "subject": src_node.name if src_node else "",
                    "predicate": edge.edge_type,
                    "object_value": tgt_node.name if tgt_node else "",
                    "confidence": edge.extraction_confidence,
                },
                summary=edge.fact,
                source_episode_ids=list(edge.source_episode_ids),
            ))

            # SUBJECT_OF: entity → fact
            if src_node:
                for lbl in src_node.labels:
                    if ontology.is_edge_allowed("SUBJECT_OF", lbl, "Fact"):
                        eid = _stable_edge_id(scope_id, src_node.node_id, "SUBJECT_OF", fact_id, None)
                        extra_edges.append(GraphEdge(
                            edge_id=eid, scope_id=scope_id,
                            source_node_id=src_node.node_id, target_node_id=fact_id,
                            edge_type="SUBJECT_OF", labels=["SUBJECT_OF"],
                            source_episode_ids=list(edge.source_episode_ids),
                        ))
                        break

            # OBJECT_OF: entity → fact
            if tgt_node:
                for lbl in tgt_node.labels:
                    if ontology.is_edge_allowed("OBJECT_OF", lbl, "Fact"):
                        eid = _stable_edge_id(scope_id, tgt_node.node_id, "OBJECT_OF", fact_id, None)
                        extra_edges.append(GraphEdge(
                            edge_id=eid, scope_id=scope_id,
                            source_node_id=tgt_node.node_id, target_node_id=fact_id,
                            edge_type="OBJECT_OF", labels=["OBJECT_OF"],
                            source_episode_ids=list(edge.source_episode_ids),
                        ))
                        break

            # EVIDENCES: passage → fact
            for ep_id in edge.source_episode_ids:
                passage_id = episode_passage_ids.get(ep_id)
                if passage_id and ontology.is_edge_allowed("EVIDENCES", "Passage", "Fact"):
                    eid = _stable_edge_id(scope_id, passage_id, "EVIDENCES", fact_id, None)
                    extra_edges.append(GraphEdge(
                        edge_id=eid, scope_id=scope_id,
                        source_node_id=passage_id, target_node_id=fact_id,
                        edge_type="EVIDENCES", labels=["EVIDENCES"],
                        source_episode_ids=[ep_id],
                    ))

        return passage_nodes, fact_nodes, extra_edges

    def _parse(self, text: str) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        json_str = extract_json_from_text(text or "")
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        entities = data.get("entities", [])
        edges = data.get("edges", [])
        if not isinstance(entities, list) or not isinstance(edges, list):
            return None
        entities = [x for x in entities if isinstance(x, dict)]
        edges = [x for x in edges if isinstance(x, dict)]
        return entities, edges

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return isoparse(value)
            except Exception:
                return None
        return None


