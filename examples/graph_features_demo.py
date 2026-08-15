#!/usr/bin/env python3
"""
End-to-end demo for graph-enhanced memory features.

Validates four features against live MySQL (session/memory) and Neo4j (graph)
backends:

  P0-A  Gist Extraction        -- two-phase atomic gist + fact extraction
  P0-B  Personalized PageRank  -- query-seeded PPR reranking of graph nodes
  P1-A  Entity Linking         -- KNN cosine SAME_AS edges between entities
  P1-B  Multi-Type Graph Nodes -- Passage / Fact node enrichment

Prerequisites:
  - MySQL running (default: localhost:3306, db=contextengine)
  - Neo4j running  (default: bolt://localhost:7687, user=neo4j)
  - OpenAI API key (or Azure equivalent) in env / examples/.env
  - ChromaDB (pip install chromadb)

Usage:
    export OPENAI_API_KEY=sk-...
    python -m ctxforge.examples.graph_features_demo
    python -m ctxforge.examples.graph_features_demo --skip-mysql   # in-memory stores
    python -m ctxforge.examples.graph_features_demo --skip-neo4j   # in-memory graph
    python -m ctxforge.examples.graph_features_demo --skip-mysql --skip-neo4j  # all in-memory
"""

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure ctxforge is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxforge.engine.factory import EngineFactory
from ctxforge.examples.config import load_config, print_config_summary
from ctxforge.extraction import EntityExtractor, HybridExtractor, PatternExtractor
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.storage import DeduplicatingMemoryStore, InMemoryMemoryStore, InMemorySessionStore
from ctxforge.storage.connection import MySQLConfig as MySQLCfg
from ctxforge.vectorstores import ChromaDBStore
from ctxforge.vectorstores.chroma_store import ChromaConfig

# ---------------------------------------------------------------------------
# Conversation script designed to exercise all four features
# ---------------------------------------------------------------------------

CONVERSATION_TURNS: List[Dict[str, str]] = [
    {
        "user": (
            "I met Alice Chen yesterday at the Python conference in San Francisco. "
            "She's the CTO of NovaTech and gave a great talk on graph databases."
        ),
        "assistant": (
            "That sounds like a great experience! Alice Chen from NovaTech giving a "
            "talk on graph databases at a Python conference in SF — very relevant to "
            "your interests."
        ),
    },
    {
        "user": (
            "Alice mentioned that NovaTech is hiring ML engineers. "
            "She also recommended the book 'Designing Data-Intensive Applications' "
            "by Martin Kleppmann."
        ),
        "assistant": (
            "Good to know about the NovaTech ML openings. And 'Designing "
            "Data-Intensive Applications' is an excellent book — great recommendation "
            "from Alice."
        ),
    },
    {
        "user": (
            "By the way, I also ran into A. Chen from Nova Technologies at the "
            "after-party. Turns out she went to MIT just like me."
        ),
        "assistant": (
            "Small world! So both you and Alice went to MIT. It's great to have "
            "that connection."
        ),
    },
]

# Queries that should trigger PPR and entity-linking benefits.
VALIDATION_QUERIES: List[str] = [
    "What do I know about Alice?",
    "Tell me about NovaTech",
    "What books were recommended at the conference?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}\n")


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Demo class
# ---------------------------------------------------------------------------

class GraphFeaturesDemo:
    def __init__(
        self,
        *,
        use_mysql: bool = True,
        use_neo4j: bool = True,
    ):
        self.use_mysql = use_mysql
        self.use_neo4j = use_neo4j

        self.session_id = str(uuid.uuid4())
        self.user_id = "graph-features-demo-user"

        self.engine: Any = None
        self.llm_provider: Any = None
        self._factory: Optional[EngineFactory] = None
        self._cleanup_stores: List[Any] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        _section("Setup: Storage Backends")

        config = load_config()
        print_config_summary(config)

        # -- session / memory stores --
        if self.use_mysql:
            from ctxforge.storage.mysql.memory import MySQLMemoryStore
            from ctxforge.storage.mysql.session import MySQLSessionStore

            mysql_cfg = MySQLCfg(
                host=os.getenv("MYSQL_HOST", config.mysql.host),
                port=int(os.getenv("MYSQL_PORT", str(config.mysql.port))),
                database=os.getenv("MYSQL_DATABASE", config.mysql.database),
                user=os.getenv("MYSQL_USER", config.mysql.user),
                password=os.getenv("MYSQL_PASSWORD", config.mysql.password),
            )
            session_store = MySQLSessionStore(mysql_cfg)
            memory_store_inner = MySQLMemoryStore(mysql_cfg)
            await session_store.initialize()
            await memory_store_inner.initialize()
            memory_store = DeduplicatingMemoryStore(memory_store_inner)
            self._cleanup_stores.extend([session_store, memory_store_inner])
            print(f"  MySQL stores ready  ({mysql_cfg.host}:{mysql_cfg.port}/{mysql_cfg.database})")
        else:
            session_store = InMemorySessionStore()
            memory_store = DeduplicatingMemoryStore(InMemoryMemoryStore())
            print("  In-memory session + memory stores")

        # -- vector store --
        chroma_cfg = ChromaConfig(
            collection_name="graph_features_demo",
            persist_directory=config.chroma.persist_directory,
            dimension=config.engine.storage.memory.vector.embedding.dimension,
        )
        vector_store = ChromaDBStore(chroma_cfg)
        await vector_store.initialize()
        print(f"  ChromaDB ready  (collection={chroma_cfg.collection_name})")

        # -- graph backend --
        graph_backend = "neo4j" if self.use_neo4j else "memory"
        neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_pass = os.getenv("NEO4J_PASSWORD", "contextengine_dev")
        neo4j_db = os.getenv("NEO4J_DATABASE")

        if self.use_neo4j:
            try:
                import importlib.util
                if importlib.util.find_spec("neo4j") is None:
                    raise ImportError("neo4j driver not installed")
                print(f"  Neo4j driver found  ({neo4j_url})")
            except ImportError:
                print("  neo4j package not installed; falling back to in-memory graph")
                graph_backend = "memory"

        # -- engine config with all four features enabled --
        emb_cfg = config.engine.storage.memory.vector.embedding
        engine_cfg = config.engine.merge_with({
            "retrieval": {
                "strategy": "semantic",
                "rerank_enabled": True,
                "reranker": "rrf",
                "rerank_top_k": 30,
            },
            "storage": {"memory": {"vector": {"backend": "chromadb"}}},
            # -- P0-A: Gist extraction --
            "extraction": {
                "enabled": True,
                "use_llm": True,
                "use_patterns": True,
                "async_processing": True,
                "extract_gists": True,
                "gist_enhanced_facts": True,
                "gist_model": None,
            },
            # -- Graph features --
            "graph": {
                "enabled": True,
                "store": {
                    "backend": graph_backend,
                    "neo4j": {
                        "url": neo4j_url,
                        "username": neo4j_user,
                        "password": neo4j_pass,
                        "database": neo4j_db,
                        "create_indexes": True,
                        "entity_label": "__Entity__",
                    },
                },
                "ontology": {
                    "module": "ctxforge.graph.default_ontology",
                    "attr_name": "GRAPH_ONTOLOGY",
                },
                "embeddings": {
                    "enabled": True,
                    "embedding": {
                        "provider": emb_cfg.provider,
                        "model": emb_cfg.model,
                        "api_key": emb_cfg.api_key,
                        "dimension": emb_cfg.dimension,
                        "batch_size": emb_cfg.batch_size,
                    },
                },
                "extraction": {"enabled": True, "model": None},
                "invalidation": {"enabled": True},
                "temporal": {"enabled": True},
                "communities": {
                    "enabled": True,
                    "rebuild_every_n_episodes": 2,
                    "min_cluster_size": 2,
                    "max_concurrency": 3,
                },
                # -- P1-A: Entity linking --
                "entity_linking": {
                    "enabled": True,
                    "similarity_threshold": 0.80,
                    "max_neighbors": 5,
                    "run_on_ingest": True,
                },
                # -- P0-B: PPR retrieval --
                "retrieval": {
                    "enabled": True,
                    "max_facts": 20,
                    "max_entities": 20,
                    "include_entities": True,
                    "methods": ["semantic", "keyword", "bfs"],
                    "seed_k": 8,
                    "bfs_max_depth": 2,
                    "bfs_edges_per_node": 12,
                    "rerank_enabled": True,
                    "reranker": "rrf",
                    "rerank_top_k": 30,
                    "ppr_enabled": True,
                    "ppr_damping": 0.5,
                    "ppr_seed_top_k": 20,
                    "ppr_result_top_k": 10,
                },
                "section_name": "Graph Memory",
            },
        })

        # -- build engine --
        _section("Building Engine")
        factory = EngineFactory()
        self._factory = factory

        embedding_provider = factory._create_embedding_provider(engine_cfg)
        extractor = HybridExtractor(
            extractors=[PatternExtractor(), EntityExtractor()]
        )

        self.engine = await factory.build(
            engine_cfg,
            session_store=session_store,
            memory_store=memory_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            extractor=extractor,
        )

        # -- LLM provider --
        self.llm_provider = factory._create_llm_provider(engine_cfg)
        if self.llm_provider is None:
            self.llm_provider = MockLLMProvider(latency_ms=0)
        print(f"  LLM provider: {self.llm_provider.name}")

        # -- wiring summary --
        gs = getattr(self.engine, "_graph_service", None)
        print("\n  Feature wiring:")
        print(f"    graph enabled       : {engine_cfg.graph.enabled}")
        print(f"    graph backend       : {engine_cfg.graph.store.backend}")
        print(f"    extract_gists       : {engine_cfg.extraction.extract_gists}")
        print(f"    gist_enhanced_facts : {engine_cfg.extraction.gist_enhanced_facts}")
        print(f"    ppr_enabled         : {engine_cfg.graph.retrieval.ppr_enabled}")
        print(f"    entity_linking      : {engine_cfg.graph.entity_linking.enabled}")
        print(f"    graph store wired   : {bool(getattr(gs, '_store', None)) if gs else False}")
        print(f"    entity linker wired : {bool(getattr(gs, '_entity_linker', None)) if gs else False}")

        # -- clean slate --
        await self._reset_scope()

    async def _reset_scope(self) -> None:
        """Delete previous demo data for a clean run."""
        try:
            deleted = await self.engine.delete_all_user_memories(
                self.user_id, include_inactive=True,
            )
            print(f"\n  Cleaned {deleted} old memories for {self.user_id}")
        except Exception as exc:
            print(f"  Could not clean memories: {exc}")

        try:
            gs = getattr(self.engine, "_graph_service", None)
            store = getattr(gs, "_store", None) if gs else None
            if store is not None:
                removed = await store.delete_scope(self.user_id)
                print(f"  Cleaned {removed} graph items (scope={self.user_id})")
        except Exception as exc:
            print(f"  Could not clean graph scope: {exc}")

    # ------------------------------------------------------------------
    # Conversation ingestion
    # ------------------------------------------------------------------

    async def ingest_conversation(self) -> None:
        _section("Phase 1: Ingesting Conversation Turns")

        for i, turn in enumerate(CONVERSATION_TURNS, 1):
            _sub(f"Turn {i}")
            print(f"  User: {turn['user'][:80]}...")

            # prepare_context creates the session and retrieves memories.
            await self.engine.prepare_context(
                session_id=self.session_id,
                user_id=self.user_id,
                user_input=turn["user"],
                include_history=True,
                include_memories=True,
            )

            # record_turn triggers extraction (gists + facts) and graph ingestion.
            await self.engine.record_turn(
                session_id=self.session_id,
                user_id=self.user_id,
                user_input=turn["user"],
                assistant_response=turn["assistant"],
            )
            print(f"  Recorded turn {i}")

            # Give async extraction a moment to complete.
            await asyncio.sleep(3)

        # Let background tasks (community rebuild, entity linking) settle.
        print("\n  Waiting for background tasks...")
        await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Validation: P0-A  Gist Extraction
    # ------------------------------------------------------------------

    async def validate_gist_extraction(self) -> None:
        _section("Validate P0-A: Gist Extraction")

        memories = await self.engine.get_user_memories(self.user_id, limit=100)

        gist_memories = [
            m for m in memories
            if "gist" in (getattr(m, "tags", None) or [])
        ]
        other_memories = [
            m for m in memories
            if "gist" not in (getattr(m, "tags", None) or [])
        ]

        print(f"  Total memories stored : {len(memories)}")
        print(f"  Gist memories         : {len(gist_memories)}")
        print(f"  Other memories        : {len(other_memories)}")

        if gist_memories:
            print("\n  Sample gists:")
            for g in gist_memories[:8]:
                meta = getattr(g, "metadata", {}) or {}
                ts = meta.get("resolved_timestamp", "")
                print(f"    - {g.content}")
                if ts:
                    print(f"      (timestamp: {ts})")
            print("\n  [PASS] Gist extraction produced atomic memories")
        else:
            print("\n  [INFO] No gist-tagged memories found.")
            print("         This is expected if LLM extraction is async and")
            print("         hasn't completed yet, or if the LLM provider is mocked.")

    # ------------------------------------------------------------------
    # Validation: P1-B  Multi-Type Graph Nodes (Passage / Fact)
    # ------------------------------------------------------------------

    async def validate_multi_type_nodes(self) -> None:
        _section("Validate P1-B: Multi-Type Graph Nodes")

        gs = getattr(self.engine, "_graph_service", None)
        store = getattr(gs, "_store", None) if gs else None
        if store is None:
            print("  [SKIP] No graph store available")
            return

        # Retrieve all nodes for this scope.
        try:
            result = await store.search(
                scope_id=self.user_id,
                query="",
                scope="nodes",
                limit=200,
            )
            nodes = result.nodes if result else []
        except Exception as exc:
            print(f"  [SKIP] Could not query nodes: {exc}")
            return

        entity_nodes = [n for n in nodes if "Passage" not in n.labels and "Fact" not in n.labels]
        passage_nodes = [n for n in nodes if "Passage" in n.labels]
        fact_nodes = [n for n in nodes if "Fact" in n.labels]

        print(f"  Total graph nodes  : {len(nodes)}")
        print(f"  Entity nodes       : {len(entity_nodes)}")
        print(f"  Passage nodes      : {len(passage_nodes)}")
        print(f"  Fact nodes         : {len(fact_nodes)}")

        if entity_nodes:
            print("\n  Sample entities:")
            for n in entity_nodes[:6]:
                print(f"    - {n.name}  labels={n.labels}")

        if passage_nodes:
            print("\n  Sample passages:")
            for n in passage_nodes[:3]:
                summary = (n.summary or "")[:80]
                print(f"    - {n.name}  summary={summary}...")
            print("\n  [PASS] Passage nodes created")
        else:
            print("\n  [INFO] No Passage nodes found (enrichment may have been skipped)")

        if fact_nodes:
            print("\n  Sample facts:")
            for n in fact_nodes[:5]:
                print(f"    - {n.name}")
            print("\n  [PASS] Fact nodes created")
        else:
            print("\n  [INFO] No Fact nodes found (enrichment may have been skipped)")

    # ------------------------------------------------------------------
    # Validation: P1-A  Entity Linking (SAME_AS edges)
    # ------------------------------------------------------------------

    async def validate_entity_linking(self) -> None:
        _section("Validate P1-A: Entity Linking (SAME_AS)")

        gs = getattr(self.engine, "_graph_service", None)
        store = getattr(gs, "_store", None) if gs else None
        if store is None:
            print("  [SKIP] No graph store available")
            return

        try:
            result = await store.search(
                scope_id=self.user_id,
                query="",
                scope="edges",
                limit=500,
            )
            edges = result.edges if result else []
        except Exception as exc:
            print(f"  [SKIP] Could not query edges: {exc}")
            return

        same_as_edges = [e for e in edges if e.edge_type == "SAME_AS"]
        other_edges = [e for e in edges if e.edge_type != "SAME_AS"]

        print(f"  Total graph edges : {len(edges)}")
        print(f"  SAME_AS edges     : {len(same_as_edges)}")
        print(f"  Other edges       : {len(other_edges)}")

        if same_as_edges:
            print("\n  SAME_AS links discovered:")
            for e in same_as_edges[:10]:
                sim = (e.attributes or {}).get("similarity", "?")
                print(f"    {e.source_node_id} <-> {e.target_node_id}")
                print(f"      fact: {e.fact}")
                print(f"      similarity: {sim}")
            print("\n  [PASS] Entity linking created SAME_AS edges")
        else:
            print("\n  [INFO] No SAME_AS edges found.")
            print("         Entities may not have had sufficiently similar embeddings,")
            print("         or entity linking ran before embeddings were computed.")

        if other_edges:
            edge_types = {}
            for e in other_edges:
                edge_types[e.edge_type] = edge_types.get(e.edge_type, 0) + 1
            print("\n  Edge type distribution:")
            for etype, count in sorted(edge_types.items()):
                print(f"    {etype}: {count}")

    # ------------------------------------------------------------------
    # Validation: P0-B  Personalized PageRank retrieval
    # ------------------------------------------------------------------

    async def validate_ppr_retrieval(self) -> None:
        _section("Validate P0-B: PPR-Reranked Graph Retrieval")

        for query in VALIDATION_QUERIES:
            _sub(f'Query: "{query}"')

            context = await self.engine.prepare_context(
                session_id=self.session_id,
                user_id=self.user_id,
                user_input=query,
                include_history=True,
                include_memories=True,
            )

            # Show retrieved memories.
            if context.memories:
                print("  Retrieved memories:")
                for m in context.memories[:5]:
                    mtype = getattr(m.type, "value", str(m.type))
                    score = getattr(m, "relevance_score", None) or ""
                    print(f"    [{mtype}] {m.content}  (score={score})")
            else:
                print("  (no memories retrieved)")

            # Show graph context section if present.
            if context.sections:
                for sec in context.sections:
                    sec_name = getattr(sec, "name", "") or getattr(sec, "section_name", "")
                    if "graph" in str(sec_name).lower():
                        content = getattr(sec, "content", "") or ""
                        lines = content.strip().splitlines()
                        print(f"\n  Graph context ({len(lines)} lines):")
                        for line in lines[:15]:
                            print(f"    {line}")
                        if len(lines) > 15:
                            print(f"    ... ({len(lines) - 15} more lines)")

            # Check context metadata for PPR debug info.
            meta = getattr(context, "metadata", {}) or {}
            ppr_meta = meta.get("graph_ppr") or meta.get("ppr_debug")
            if ppr_meta:
                print(f"\n  PPR debug: {ppr_meta}")
                print("  [PASS] PPR reranking was applied")
            else:
                print("\n  [INFO] No PPR debug metadata in context")
                print("         PPR may have been applied but debug info not propagated,")
                print("         or there were too few graph nodes to trigger PPR.")

    # ------------------------------------------------------------------
    # Standalone PPR unit validation (always works, no external deps)
    # ------------------------------------------------------------------

    async def validate_ppr_algorithm(self) -> None:
        _sub("PPR Algorithm (standalone check)")

        from ctxforge.graph.retrieval.pagerank import (
            compute_seed_scores,
            personalized_pagerank,
        )
        from ctxforge.protocols.graph import GraphEdge, GraphNode

        def _node(nid, emb=None, label="Entity"):
            return GraphNode(
                node_id=nid, scope_id="test", name=nid,
                labels=[label], name_embedding=emb,
            )

        def _edge(eid, src, tgt, etype="RELATED"):
            return GraphEdge(
                edge_id=eid, scope_id="test",
                source_node_id=src, target_node_id=tgt, edge_type=etype,
            )

        # Chain: A -> B -> C, seed at A
        nodes = [_node("A"), _node("B"), _node("C")]
        edges = [_edge("e1", "A", "B"), _edge("e2", "B", "C")]
        scores = personalized_pagerank(nodes, edges, {"A": 1.0}, damping=0.5)

        assert scores["A"] > scores["B"] > scores["C"], "Chain ranking violated"
        print(f"  Chain PPR scores: A={scores['A']:.4f} > B={scores['B']:.4f} > C={scores['C']:.4f}")
        print("  [PASS] PPR chain ranking correct")

        # Node type weights: hub H -> P(Passage), E(Person)
        nodes2 = [_node("H", label="Person"), _node("P", label="Passage"), _node("E", label="Person")]
        edges2 = [_edge("e1", "H", "P"), _edge("e2", "H", "E")]

        s_default = personalized_pagerank(nodes2, edges2, {"H": 1.0}, damping=0.8)
        s_weighted = personalized_pagerank(
            nodes2, edges2, {"H": 1.0}, damping=0.8,
            node_type_weights={"Person": 1.0, "Passage": 0.05},
        )

        ratio_default = s_default["P"] / s_default["E"]
        ratio_weighted = s_weighted["P"] / s_weighted["E"]
        assert ratio_weighted < ratio_default, "Node type weights had no effect"
        print(f"  Weight effect: P/E ratio default={ratio_default:.4f} weighted={ratio_weighted:.4f}")
        print("  [PASS] Node type weights reduce Passage node scores")

        # Seed scores via cosine similarity
        query = [1.0, 0.0, 0.0]
        nodes3 = [
            _node("X", emb=[1.0, 0.0, 0.0]),
            _node("Y", emb=[0.0, 1.0, 0.0]),
            _node("Z", emb=[0.7, 0.7, 0.0]),
        ]
        seeds = compute_seed_scores(query, nodes3, top_k=3)
        assert seeds["X"] > seeds.get("Z", 0), "Seed scoring failed"
        assert "Y" not in seeds, "Orthogonal node should not be seeded"
        print(f"  Seed scores: X={seeds['X']:.4f}, Z={seeds.get('Z', 0):.4f}, Y absent")
        print("  [PASS] compute_seed_scores correct")

    # ------------------------------------------------------------------
    # Standalone entity-linking validation
    # ------------------------------------------------------------------

    async def validate_entity_linking_algorithm(self) -> None:
        _sub("Entity Linking Algorithm (standalone check)")

        from ctxforge.graph.maintenance.entity_linking import EntityLinker
        from ctxforge.graph.stores.memory import InMemoryGraphStore
        from ctxforge.protocols.graph import GraphNode

        store = InMemoryGraphStore()
        linker = EntityLinker(similarity_threshold=0.85, max_neighbors=5)

        # Two nodes with near-identical embeddings (cosine ~ 1.0)
        nodes = [
            GraphNode(
                node_id="alice_chen", scope_id="test", name="Alice Chen",
                labels=["Person"], name_embedding=[1.0, 0.0, 0.0],
            ),
            GraphNode(
                node_id="a_chen", scope_id="test", name="A. Chen",
                labels=["Person"], name_embedding=[0.99, 0.1, 0.0],
            ),
            GraphNode(
                node_id="bob", scope_id="test", name="Bob Smith",
                labels=["Person"], name_embedding=[0.0, 1.0, 0.0],
            ),
        ]
        await store.upsert_nodes("test", nodes)

        new_edges = await linker.link_entities(nodes, store, "test")
        same_as = [e for e in new_edges if e.edge_type == "SAME_AS"]

        print(f"  Input nodes: {len(nodes)}")
        print(f"  SAME_AS edges created: {len(same_as)}")
        for e in same_as:
            sim = (e.attributes or {}).get("similarity", "?")
            print(f"    {e.source_node_id} <-> {e.target_node_id}  (sim={sim})")

        # alice_chen and a_chen should link; bob should not link to either
        linked_pairs = {(e.source_node_id, e.target_node_id) for e in same_as}
        linked_pairs |= {(t, s) for s, t in linked_pairs}
        assert ("alice_chen", "a_chen") in linked_pairs or ("a_chen", "alice_chen") in linked_pairs, \
            "Expected SAME_AS between alice_chen and a_chen"
        assert ("bob", "alice_chen") not in linked_pairs, \
            "Bob should not link to Alice"
        print("  [PASS] Entity linking correctly identifies similar entities")

    # ------------------------------------------------------------------
    # Run all validations
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.setup()

        # Standalone algorithm checks (no external services needed).
        _section("Standalone Algorithm Validation")
        await self.validate_ppr_algorithm()
        await self.validate_entity_linking_algorithm()

        # End-to-end with live backends.
        await self.ingest_conversation()
        await self.validate_gist_extraction()
        await self.validate_multi_type_nodes()
        await self.validate_entity_linking()
        await self.validate_ppr_retrieval()

        _section("Demo Complete")
        print("  All validation steps finished.")
        print("  Review [PASS] / [INFO] markers above for results.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        for store in self._cleanup_stores:
            close = getattr(store, "close", None) or getattr(store, "cleanup", None)
            if close and callable(close):
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end demo for graph-enhanced memory features",
    )
    parser.add_argument(
        "--skip-mysql",
        action="store_true",
        help="Use in-memory stores instead of MySQL",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Use in-memory graph store instead of Neo4j",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    demo = GraphFeaturesDemo(
        use_mysql=not args.skip_mysql,
        use_neo4j=not args.skip_neo4j,
    )
    try:
        await demo.run()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
