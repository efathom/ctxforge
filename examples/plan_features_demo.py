#!/usr/bin/env python3
"""
Implementation Plan Features Demo.

Demonstrates all features implemented across Phases 0-6:

  Phase 1 - Entropy Gate (extraction deduplication)
  Phase 2 - Conflict-Aware Consolidation (keyword overlap, LLM contradiction)
  Phase 3 - Consolidation enhancements (asymmetric formula, configurable prompts)
  Phase 4 - Enhanced Memory Indexing & Fast-Path Retrieval
  Phase 5 - Graph Bridge Discovery & Multi-Hop Path Mining
  Phase 6 - Topology-Aware Context Serialization

Each section is self-contained and runs without external services (no LLM API
keys, no databases). All demos use in-memory stores and mock providers.

Run:
    cd /path/to/ctxforge
    source venv/bin/activate
    python -m examples.plan_features_demo
"""

import asyncio
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure ctxforge is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxforge.compaction.topology_view import TopologyAwareRenderer
from ctxforge.config.base import (
    GraphPathMiningConfig,
    RetrievalFastPathConfig,
    TopologySerializationConfig,
)
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.extraction.consolidation.conflict_aware import (
    ConflictAwareConsolidator,
    ConsolidationAction,
)
from ctxforge.extraction.entropy_gate import EntropyGate, EntropyGateConfig, GateResult
from ctxforge.graph.retrieval.bridge_discovery import check_connection
from ctxforge.graph.retrieval.path_miner import discover_reasoning_paths
from ctxforge.graph.retrieval.path_scorer import rank_and_limit_nodes, score_node
from ctxforge.graph.retrieval.types import (
    BridgeConnection,
    EvidenceItem,
    GraphEdgeHit,
    GraphNodeHit,
    GraphRetrievalResult,
    ReasoningPath,
)
from ctxforge.protocols.graph import GraphEdge, GraphNode
from ctxforge.protocols.llm import EmbeddingResponse, IEmbeddingProvider
from ctxforge.retrieval.aggregation_builder import AggregationBuilder
from ctxforge.retrieval.fast_path_retriever import FastPathRetriever

# =============================================================================
# Mock Embedding Provider (for demos that need embeddings)
# =============================================================================


class _DemoEmbeddingProvider(IEmbeddingProvider):
    """Simple hash-based mock embedding provider for demo purposes."""

    @property
    def name(self) -> str:
        return "demo-hash-embedder"

    @property
    def dimension(self) -> int:
        return 64

    async def embed(self, texts: List[str]) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[self._hash_embed(t) for t in texts],
            model="demo-hash",
            usage={"total_tokens": sum(len(t.split()) for t in texts)},
        )

    async def embed_single(self, text: str) -> List[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding from text hash.

        Uses per-word hashing so that texts with overlapping words produce
        similar embeddings (high cosine similarity), while texts with
        different vocabulary produce dissimilar ones.
        """
        words = set(text.lower().split())
        vec = [0.0] * 64
        for w in words:
            h = hashlib.sha256(w.encode()).digest()
            for i in range(64):
                vec[i] += (h[i % len(h)] - 128) / 128.0
        # Normalize
        norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]


# =============================================================================
# Helpers
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def _make_memory(
    content: str,
    user_id: str = "demo-user",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    tags: Optional[List[str]] = None,
    created_at: Optional[datetime] = None,
) -> MemoryItem:
    m = MemoryItem(
        user_id=user_id,
        content=content,
        type=memory_type,
        source=MemorySource.USER_EXPLICIT,
        tags=tags or [],
    )
    if created_at:
        m.created_at = created_at
    return m


def _make_graph_node(
    node_id: str,
    name: str,
    labels: Optional[List[str]] = None,
    summary: str = "",
    attributes: Optional[Dict] = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        scope_id="demo",
        name=name,
        labels=labels or [],
        summary=summary,
        attributes=attributes or {},
    )


def _make_graph_edge(
    edge_id: str,
    source: str,
    target: str,
    edge_type: str = "RELATED",
    fact: str = "",
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id,
        scope_id="demo",
        source_node_id=source,
        target_node_id=target,
        edge_type=edge_type,
        fact=fact,
    )


# =============================================================================
# Phase 1: Entropy Gate
# =============================================================================


async def demo_entropy_gate() -> None:
    print_header("PHASE 1 - Entropy Gate (Extraction Deduplication)")

    print("""
The Entropy Gate prevents redundant LLM extraction calls by detecting when
new dialogue content is too similar to what has already been processed.
It uses embedding cosine similarity to measure novelty against a sliding
window of recent turns.
""")

    config = EntropyGateConfig(
        enabled=True,
        similarity_threshold=0.55,
        recent_window_size=10,
        min_chars=10,
    )
    embedding_provider = _DemoEmbeddingProvider()
    gate = EntropyGate(config, embedding_provider=embedding_provider)

    # Each turn is (user_input, assistant_response).
    # Turns 2 and 4 are near-duplicates of turns 1 and 3 respectively.
    turns = [
        ("I love hiking in the mountains and enjoy photography.",
         "That sounds great! Hiking and photography go well together."),
        ("I really enjoy hiking in the mountains and taking photos.",
         "Nice, hiking and photography are wonderful hobbies."),  # near-duplicate of turn 1
        ("My favorite programming language is Python and I work at Acme Corp.",
         "Python is a great choice! Acme Corp sounds interesting."),  # novel
        ("I work at Acme Corp and my favorite language is Python.",
         "Got it, you're at Acme and love Python."),  # near-duplicate of turn 3
        ("I'm planning a trip to Japan next month to visit Tokyo.",
         "Tokyo is amazing! You'll love it."),  # novel
    ]

    print_subheader("Processing Dialogue Turns")
    extract_count = 0
    skip_count = 0
    for i, (user_input, assistant_response) in enumerate(turns, 1):
        result: GateResult = await gate.evaluate(user_input, assistant_response)
        if result.should_extract:
            status = f"EXTRACT (reason: {result.reason})"
            extract_count += 1
        else:
            sim_str = f", similarity: {result.similarity_score:.2f}" if result.similarity_score else ""
            status = f"SKIP (reason: {result.reason}{sim_str})"
            skip_count += 1
        display = user_input[:60] + "..." if len(user_input) > 60 else user_input
        print(f"  Turn {i}: {status}")
        print(f"    \"{display}\"")

    print(f"\n  Extracted: {extract_count} | Skipped: {skip_count} / {len(turns)} total turns")


# =============================================================================
# Phase 2-3: Conflict-Aware Consolidation
# =============================================================================


async def demo_conflict_aware_consolidation() -> None:
    print_header("PHASE 2-3 - Conflict-Aware Consolidation")

    print("""
The ConflictAwareConsolidator detects and resolves conflicts between memories:
- Deduplication: merges near-identical memories
- Contradiction detection: flags conflicting facts (heuristic + optional LLM)
- Asymmetric keyword overlap: uses len(intersection) / max(len(new_kw), 1)
- Configurable contradiction prompt (overridable via constructor)
""")

    consolidator = ConflictAwareConsolidator()

    existing = [
        _make_memory("User works at Acme Corp as a software engineer", tags=["work", "acme"]),
        _make_memory("User likes hiking and outdoor activities", tags=["hiking", "outdoors"]),
        _make_memory("User prefers Python for backend development", tags=["python", "backend"]),
    ]

    new_items = [
        _make_memory("User works at Acme Corp as a senior engineer", tags=["work", "acme"]),  # update
        _make_memory("User hates hiking and prefers indoor activities", tags=["hiking", "indoors"]),  # contradiction
        _make_memory("User is learning Rust for systems programming", tags=["rust", "systems"]),  # novel
    ]

    print_subheader("Existing Memories")
    for m in existing:
        print(f"  - {m.content}")

    print_subheader("New Memories to Consolidate")
    for m in new_items:
        print(f"  - {m.content}")

    # Use decide_actions to get per-item decisions (more informative than consolidate)
    decisions = await consolidator.decide_actions(new_items=new_items, existing_items=existing)

    print_subheader("Consolidation Decisions")
    action_counts: Dict[str, int] = {}
    for d in decisions:
        action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1
        icon = {
            ConsolidationAction.ADD: "+",
            ConsolidationAction.MERGE: "~",
            ConsolidationAction.IGNORE: "=",
            ConsolidationAction.CONFLICT: "!",
        }.get(d.action, "?")
        sim_str = f" (sim: {d.similarity_score:.2f})" if d.similarity_score is not None else ""
        kw_str = f" (kw_overlap: {d.keyword_overlap:.2f})" if d.keyword_overlap is not None else ""
        contra_str = " [CONTRADICTION]" if d.is_contradiction else ""
        print(f"  [{icon}] {d.action.value.upper():8s} {d.new_item.content[:60]}")
        print(f"           reason: {d.reason}{sim_str}{kw_str}{contra_str}")
        if d.target_item:
            print(f"           target: {d.target_item.content[:60]}")

    print_subheader("Summary")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")

    # Also run full consolidate to show the final result
    result = await consolidator.consolidate(new_items=new_items, existing_items=existing)
    print(f"\n  Final consolidated items: {len(result)}")


# =============================================================================
# Phase 4: Enhanced Memory Indexing & Fast-Path Retrieval
# =============================================================================


async def demo_enhanced_indexing() -> None:
    print_header("PHASE 4 - Enhanced Memory Indexing & Fast-Path Retrieval")

    print("""
The Enhanced Memory Index aggregates entity-level statistics from memories:
- Entity aggregation: event counts, attribute sets, temporal sequences
- Relation triples: subject-predicate-object with confidence
- Temporal index: date -> memory ID mapping

The Fast-Path Retriever uses this index for O(1) lookups on common queries
(count, list, relation, attribute) before falling back to full retrieval.
""")

    # Build sample memories
    memories = [
        _make_memory("Alice joined Acme Corp in January 2024", tags=["Alice", "Acme"]),
        _make_memory("Alice likes coffee and hiking", tags=["Alice", "coffee", "hiking"]),
        _make_memory("Bob works at Globex as a data scientist", tags=["Bob", "Globex"]),
        _make_memory("Alice met Bob at a conference in Berlin", tags=["Alice", "Bob", "Berlin"]),
        _make_memory("Bob likes tea and cycling", tags=["Bob", "tea", "cycling"]),
    ]

    print_subheader("Source Memories")
    for m in memories:
        print(f"  - {m.content}")

    # Build the enhanced index
    builder = AggregationBuilder()
    index = builder.build_aggregations(memories)

    print_subheader("Enhanced Memory Index")
    print(f"  Entities: {len(index.entities)}")
    for name, agg in index.entities.items():
        print(f"    {name}:")
        print(f"      Events: {dict(agg.event_counts)}")
        if agg.attribute_sets:
            for k, v in agg.attribute_sets.items():
                print(f"      {k}: {v}")
    print(f"  Relations: {len(index.relations)}")
    for rel in index.relations:
        print(f"    {rel.subject} --[{rel.predicate}]--> {rel.object} (conf: {rel.confidence:.2f})")
    print(f"  Temporal entries: {len(index.temporal_index)}")

    # Fast-path retrieval
    print_subheader("Fast-Path Retrieval (O(1) Cache Lookups)")

    fp_config = RetrievalFastPathConfig(enabled=True, min_confidence=0.5)
    retriever = FastPathRetriever(config=fp_config)
    retriever.set_enhanced_index(index)

    queries = [
        "How many events for Alice?",
        "List all tags for Bob",
        "What does Alice like?",
    ]

    for q in queries:
        result = retriever.try_fast_path(q)
        if result.hit:
            print(f"  HIT  [{result.query_type}] \"{q}\"")
            for m in result.memories[:2]:
                print(f"    -> {m.content[:80]}")
        else:
            print(f"  MISS \"{q}\" (falls back to full retrieval)")


# =============================================================================
# Phase 5: Bridge Discovery & Multi-Hop Path Mining
# =============================================================================


async def demo_bridge_discovery_and_path_mining() -> None:
    print_header("PHASE 5 - Bridge Discovery & Multi-Hop Path Mining")

    print("""
Bridge Discovery finds intermediate nodes to connect disconnected regions
of the graph. Multi-Hop Path Mining uses DFS to enumerate reasoning paths
through the subgraph. Node scoring ranks nodes by relevance.
""")

    # Build a sample graph
    t1 = datetime(2024, 6, 1, 10, 0, 0)
    t2 = datetime(2024, 6, 1, 11, 0, 0)
    t3 = datetime(2024, 6, 15, 14, 0, 0)  # 2 weeks later (disconnected)
    t4 = datetime(2024, 6, 15, 15, 0, 0)

    nodes = [
        _make_graph_node("n1", "Alice", labels=["Person"], summary="Alice is a developer at Acme", attributes={"created_at": t1}),
        _make_graph_node("n2", "Bob", labels=["Person"], summary="Bob is a manager at Acme", attributes={"created_at": t2, "keywords": ["management"]}),
        _make_graph_node("n3", "TechCorp", labels=["Organization"], summary="TechCorp is a tech company", attributes={"created_at": t3}),
        _make_graph_node("n4", "Project-X", labels=["Project"], summary="Project-X is a collaboration between Acme and TechCorp", attributes={"created_at": t4}),
    ]

    edges = [
        _make_graph_edge("e1", "n1", "n2", "KNOWS", "Alice knows Bob"),
        _make_graph_edge("e2", "n2", "n4", "LEADS", "Bob leads Project-X"),
        _make_graph_edge("e3", "n3", "n4", "SPONSORS", "TechCorp sponsors Project-X"),
    ]

    print_subheader("Sample Graph")
    for n in nodes:
        print(f"  [{n.node_id}] {n.name} ({', '.join(n.labels)})")
    for e in edges:
        print(f"  {e.source_node_id} --[{e.edge_type}]--> {e.target_node_id}: {e.fact}")

    # Connection checking
    print_subheader("Connection Checking")
    pairs = [("n1", "n2"), ("n1", "n3"), ("n3", "n4")]
    node_map = {n.node_id: n for n in nodes}
    for a, b in pairs:
        conn = check_connection(node_map[a], node_map[b], edges)
        status = conn if conn else "DISCONNECTED"
        print(f"  {node_map[a].name} <-> {node_map[b].name}: {status}")

    # Path mining
    print_subheader("Multi-Hop Path Mining (DFS)")
    config = GraphPathMiningConfig(
        enabled=True,
        max_path_depth=4,
        min_path_length=2,
        max_paths=10,
    )
    paths = discover_reasoning_paths(nodes=nodes, edges=edges, config=config)
    print(f"  Discovered {len(paths)} reasoning paths:")
    for i, p in enumerate(paths[:8], 1):
        names = [node_map[nid].name for nid in p.node_ids if nid in node_map]
        edge_info = " -> ".join(names)
        print(f"    {i}. {edge_info} (hops: {len(p.node_ids) - 1})")

    # Node scoring
    print_subheader("Node Scoring")
    query_words = {"alice", "project", "techcorp"}
    for n in nodes:
        s = score_node(n, query_words=query_words, target_entity="Alice")
        print(f"  {n.name}: score = {s:.1f}")

    # Budget enforcement
    print_subheader("Node Budget Enforcement")
    ranked = rank_and_limit_nodes(
        nodes,
        query="Alice project TechCorp",
        target_entity="Alice",
        config=config,
    )
    print(f"  Input: {len(nodes)} nodes -> Output: {len(ranked)} nodes (after budget)")
    for n in ranked:
        print(f"    - {n.name}")


# =============================================================================
# Phase 6: Topology-Aware Context Serialization
# =============================================================================


async def demo_topology_aware_serialization() -> None:
    print_header("PHASE 6 - Topology-Aware Context Serialization")

    print("""
Instead of flat lists of facts/entities/evidence, the topology-aware renderer
produces structured text with:
- Labeled facts [F1], [F2] with inline evidence
- Annotated reasoning paths: F1 --[KNOWS]--> F2 --[LEADS]--> F3
- Bridge connection summaries explaining inferred links
""")

    # Build a GraphRetrievalResult
    rr = GraphRetrievalResult(
        plan_mode="hybrid",
        plan_reason="multi-hop query detected",
        nodes=[
            GraphNodeHit(node_id="n1", label="Alice", score=0.95, attrs={
                "labels": ["Person"], "summary": "Alice is a developer at Acme Corp",
                "created_at": "2024-06-01",
            }),
            GraphNodeHit(node_id="n2", label="Bob", score=0.88, attrs={
                "labels": ["Person"], "summary": "Bob is a manager who leads Project-X",
            }),
            GraphNodeHit(node_id="n3", label="TechCorp", score=0.72, attrs={
                "labels": ["Organization"], "summary": "TechCorp sponsors Project-X",
            }),
            GraphNodeHit(node_id="n4", label="Project-X", score=0.80, attrs={
                "labels": ["Project"], "summary": "A collaboration between Acme and TechCorp",
            }),
        ],
        edges=[
            GraphEdgeHit(edge_id="e1", source_id="n1", target_id="n2", relation="KNOWS", score=0.9,
                         attrs={"fact": "Alice knows Bob from the Acme engineering team"}),
            GraphEdgeHit(edge_id="e2", source_id="n2", target_id="n4", relation="LEADS", score=0.85,
                         attrs={"fact": "Bob leads Project-X"}),
            GraphEdgeHit(edge_id="e3", source_id="n3", target_id="n4", relation="SPONSORS", score=0.75,
                         attrs={"fact": "TechCorp sponsors Project-X"}),
        ],
        evidence=[
            EvidenceItem(source="episode", source_id="n1", content="Alice mentioned she met Bob at the Acme onboarding in January 2024",
                         score=0.9, metadata={"created_at": "2024-06-01"}),
            EvidenceItem(source="episode", source_id="n2", content="Bob presented the Project-X roadmap at the all-hands meeting",
                         score=0.8, metadata={"created_at": "2024-06-10"}),
        ],
        debug={},
        reasoning_paths=[
            ReasoningPath(node_ids=["n1", "n2", "n4"], edge_types=["KNOWS", "LEADS"]),
            ReasoningPath(node_ids=["n3", "n4"], edge_types=["SPONSORS"]),
            ReasoningPath(node_ids=["n1", "n2", "n4", "n3"], edge_types=["KNOWS", "LEADS", "SPONSORS"]),
        ],
        bridge_connections=[
            BridgeConnection(source_node_id="n1", bridge_node_id="n2", target_node_id="n4", bridge_type="inferred"),
        ],
    )

    # Render with legacy flat mode
    print_subheader("Legacy Flat Rendering (Before Phase 6)")
    flat_lines = []
    flat_lines.append("<FACTS>")
    for e in rr.edges:
        fact = e.attrs.get("fact", f"{e.relation} ({e.source_id} -> {e.target_id})")
        flat_lines.append(f"- {fact}")
    flat_lines.append("")
    flat_lines.append("<ENTITIES>")
    for n in rr.nodes:
        flat_lines.append(f"- {n.label} ({', '.join(n.attrs.get('labels', []))})")
    flat_lines.append("")
    flat_lines.append("<EVIDENCE>")
    for ev in rr.evidence:
        flat_lines.append(f"- [{ev.source_id}] {ev.metadata.get('created_at', '')} :: {ev.content[:80]}")
    flat_lines.append("")
    flat_lines.append("<REASONING_PATHS>")
    flat_lines.append("  1. E1 -> E2 -> E4")
    flat_lines.append("  2. E3 -> E4")
    flat_lines.append("")
    flat_lines.append("<BRIDGE_CONNECTIONS: 1 inferred links found>")
    print("\n".join(flat_lines))

    # Render with topology-aware mode
    print_subheader("Topology-Aware Rendering (Phase 6)")
    topo_config = TopologySerializationConfig(
        enabled=True,
        include_timestamps=True,
        include_edge_types_in_paths=True,
        max_evidence_per_fact=1,
    )
    renderer = TopologyAwareRenderer(topo_config)
    text = renderer.render(rr)
    print(text)

    # Show the structured view
    print_subheader("Structured TopologyView Object")
    view = renderer.build_view(rr)
    print(f"  Facts:   {len(view.facts)}")
    for f in view.facts:
        ev_count = len(f.evidence)
        print(f"    [{f.label}] {f.entity_name} - {f.content[:50]}... ({ev_count} evidence)")
    print(f"  Paths:   {len(view.paths)}")
    for p in view.paths:
        print(f"    {p.index}. {p.summary}")
    print(f"  Bridges: {len(view.bridges)}")
    for b in view.bridges:
        print(f"    - {b.description}")


# =============================================================================
# Feature Summary
# =============================================================================


def print_feature_summary() -> None:
    print_header("FEATURE SUMMARY")

    features = [
        ("Phase 1", "Entropy Gate", "Reduces redundant extraction calls via TF-IDF similarity"),
        ("Phase 2-3", "Conflict-Aware Consolidation", "Dedup, merge, contradiction detection with asymmetric keyword overlap"),
        ("Phase 4", "Enhanced Memory Indexing", "Entity aggregation, relation triples, temporal index"),
        ("Phase 4", "Fast-Path Retrieval", "O(1) cache lookups for count/list/relation/attribute queries"),
        ("Phase 5", "Bridge Discovery", "Finds intermediate nodes to connect disconnected graph regions"),
        ("Phase 5", "Multi-Hop Path Mining", "DFS-based reasoning path enumeration with scoring"),
        ("Phase 6", "Topology-Aware Serialization", "Labeled facts, annotated paths, bridge summaries for LLM context"),
    ]

    print("\nAll features are disabled by default (feature-flagged) and non-breaking.\n")
    for phase, name, desc in features:
        print(f"  [{phase:>10}]  {name}")
        print(f"               {desc}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    print("\n" + "=" * 70)
    print("  CTXFORGE IMPLEMENTATION PLAN FEATURES DEMO (Phases 0-6)")
    print("=" * 70)

    try:
        await demo_entropy_gate()
        await demo_conflict_aware_consolidation()
        await demo_enhanced_indexing()
        await demo_bridge_discovery_and_path_mining()
        await demo_topology_aware_serialization()
        print_feature_summary()

        print("All demos completed successfully!\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
