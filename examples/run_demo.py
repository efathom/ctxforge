#!/usr/bin/env python3
"""
CtxForge End-to-End Demo.

This demo exercises the "public" framework flow:
- EngineFactory.build() (validated DI entrypoint)
- ctxforge engine.prepare_context()/record_turn()
- config-driven pipelines (pipelines.prepare / pipelines.record)
- vectorstore-backed memory retrieval (semantic/hybrid/temporal when enabled)
- optional expertise retrieval via prepare_context(expertise_id=...)

Usage:
    export OPENAI_API_KEY=sk-...
    python -m ctxforge.examples.run_demo
    python -m ctxforge.examples.run_demo --expertise
    python -m ctxforge.examples.run_demo --postgres
    python -m ctxforge.examples.run_demo --mysql
"""

import argparse
import asyncio
import importlib.util
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure `ctxforge` is importable when running as a script (not `python -m ...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxforge.config.base import MiddlewareItemConfig
from ctxforge.core.alignment_types import CharSpan
from ctxforge.core.expertise import Expertise, ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.factory import EngineFactory
from ctxforge.examples.config import DemoConfig, load_config, print_config_summary
from ctxforge.expertise.curator import ExpertiseCurator
from ctxforge.expertise.prompt_validation import (
    ExpertiseExample,
    validate_expertise_examples,
)
from ctxforge.expertise.reflector import ExpertiseReflector
from ctxforge.extraction import EntityExtractor, HybridExtractor, PatternExtractor
from ctxforge.extraction.alignment import AlignmentStatus, WordAligner
from ctxforge.extraction.chunking import ChunkIterator, make_batches
from ctxforge.extraction.schema_constraints import (
    SchemaConfig,
    generate_graph_extraction_schema,
    generate_memory_extraction_schema,
)
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.graph import GraphSearchFilters
from ctxforge.protocols.llm import ChatMessage
from ctxforge.storage import DeduplicatingMemoryStore, InMemoryMemoryStore, InMemorySessionStore
from ctxforge.storage.connection import MySQLConfig as MySQLCfg
from ctxforge.storage.connection import PostgresConfig as PGConfig
from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
from ctxforge.storage.mysql.expertise import MySQLExpertiseStore
from ctxforge.storage.mysql.memory import MySQLMemoryStore
from ctxforge.storage.mysql.session import MySQLSessionStore
from ctxforge.storage.postgres.expertise import PostgresExpertiseStore
from ctxforge.storage.postgres.memory import PostgresMemoryStore
from ctxforge.storage.postgres.session import PostgresSessionStore
from ctxforge.vectorstores import ChromaDBStore
from ctxforge.vectorstores.chroma_store import ChromaConfig
from ctxforge.visualization import (
    save_visualization,
    visualize_memory_extractions,
)


def _openai_dict_messages_to_chat_messages(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    return [ChatMessage(role=m["role"], content=m.get("content") or "") for m in messages]


class ContextEngineDemo:
    def __init__(self, config: DemoConfig, use_postgres: bool, use_mysql: bool, use_expertise: bool):
        self.config = config
        self.use_postgres = use_postgres
        self.use_mysql = use_mysql
        self.use_expertise = use_expertise

        self.session_id = str(uuid.uuid4())
        self.user_id = "demo-user"

        self.engine = None
        # Demo LLM provider (may be OpenAI or mock depending on config and env).
        self.llm_provider: Optional[Any] = None
        self.expertise_id: Optional[str] = None
        self._factory: Optional[EngineFactory] = None
        
        # Store references for cleanup
        self._session_store: Optional[Any] = None
        self._memory_store_inner: Optional[Any] = None
        self._expertise_store: Optional[Any] = None

    async def setup(
        self,
        reset_chroma: bool = False,
        reset_memories: bool = False,
        *,
        exercise_plan_features: bool = True,
        graph_backend: str = "memory",
    ) -> None:
        print("=" * 60)
        print("🚀 Setting up CtxForge Demo")
        print("=" * 60)

        # 1) Stores
        print("\n📦 Initializing storage backends...")
        if self.use_postgres:
            pg_config = PGConfig(
                host=self.config.postgres.host,
                port=self.config.postgres.port,
                database=self.config.postgres.database,
                user=self.config.postgres.user,
                password=self.config.postgres.password,
            )
            session_store = PostgresSessionStore(pg_config)
            memory_store_inner = PostgresMemoryStore(pg_config)
            await session_store.initialize()
            await memory_store_inner.initialize()
            memory_store = DeduplicatingMemoryStore(memory_store_inner)
            self._session_store = session_store
            self._memory_store_inner = memory_store_inner
            print("   ✅ PostgreSQL session + memory stores")
        elif self.use_mysql:
            mysql_config = MySQLCfg(
                host=self.config.mysql.host,
                port=self.config.mysql.port,
                database=self.config.mysql.database,
                user=self.config.mysql.user,
                password=self.config.mysql.password,
            )
            session_store = MySQLSessionStore(mysql_config)
            memory_store_inner = MySQLMemoryStore(mysql_config)
            await session_store.initialize()
            await memory_store_inner.initialize()
            memory_store = DeduplicatingMemoryStore(memory_store_inner)
            self._session_store = session_store
            self._memory_store_inner = memory_store_inner
            print(f"   ✅ MySQL session + memory stores ({mysql_config.host}:{mysql_config.port}/{mysql_config.database})")
        else:
            session_store = InMemorySessionStore()
            memory_store = DeduplicatingMemoryStore(InMemoryMemoryStore())
            print("   ✅ In-memory session + memory stores")

        # 2) Vector store (ChromaDB)
        print("\n🔍 Initializing ChromaDB vector store...")
        if reset_chroma:
            persist_dir = Path(self.config.chroma.persist_directory)
            repo_root = Path(__file__).resolve().parents[2]
            resolved = persist_dir if persist_dir.is_absolute() else (repo_root / persist_dir).resolve()

            # Safety guard: only allow deleting directories inside this repo
            if not str(resolved).startswith(str(repo_root.resolve())):
                print(f"   ⚠️ Refusing to delete persist_directory outside repo: {resolved}")
            elif resolved.exists() and resolved.is_dir():
                print(f"   🧹 Resetting Chroma persist directory: {resolved}")
                shutil.rmtree(resolved)

        chroma_config = ChromaConfig(
            collection_name=self.config.chroma.collection_name,
            persist_directory=self.config.chroma.persist_directory,
            dimension=self.config.engine.storage.memory.vector.embedding.dimension,
        )
        vector_store = ChromaDBStore(chroma_config)
        await vector_store.initialize()
        print(f"   ✅ ChromaDB initialized ({vector_store.name})")

        # 3) Build engine config overrides for demo
        # NOTE: For the demo we keep extraction local (pattern/entity), but we can still
        # enable LLM-driven update planning and graph extraction (they use config.llm).
        engine_cfg = self.config.engine.merge_with(
            {
                # Ensure vectorstore-backed retrieval is exercised
                "retrieval": {
                    "strategy": "semantic",
                    "rerank_enabled": bool(exercise_plan_features),
                    "reranker": "llm",
                    "rerank_top_k": 10,
                },
                "storage": {"memory": {"vector": {"backend": "chromadb"}}},
                # Keep demo extraction local (no LLM extraction)
                "extraction": {
                    "use_llm": False,
                    "async_processing": True,
                    "update_planning_enabled": bool(exercise_plan_features),
                    "update_planning_candidates_per_item": 5,
                },
                "expertise": {
                    "enabled": bool(self.use_expertise),
                    "vectorstore": {"backend": "chromadb"},
                },
                "scopes": {
                    "enable_global": True,
                    "global_scope_id": "global",
                    "global_retrieval_limit": 5,
                    "global_score_weight": 0.8,
                    "allow_global_writes": False,
                },
            }
        )

        if exercise_plan_features:
            if graph_backend == "neo4j":
                try:
                    if importlib.util.find_spec("neo4j") is None:
                        raise ImportError("neo4j")
                except Exception:
                    print("   ⚠️ Python package 'neo4j' is not installed; falling back to in-memory graph store for this demo.")
                    graph_backend = "memory"

            # Exercise module plugin loading (adds a demo middleware).
            engine_cfg = engine_cfg.merge_with(
                {
                    "plugins": {
                        "modules": ["ctxforge.examples.demo_plugins"],
                        "registrations": [],
                    }
                }
            )
            # Add the plugin middleware to the prepare pipeline so it is observable in context metadata.
            engine_cfg.pipelines.prepare.chain.append(
                MiddlewareItemConfig(
                    type="demo_marker",
                    enabled=True,
                    priority=999,
                    phases=["prepare_input", "prepare_retrieval"],
                    config={"key": "demo_plugin_loaded", "value": "true"},
                )
            )

            # Graph memory
            neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
            neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "contextengine_dev")
            neo4j_database = os.getenv("NEO4J_DATABASE")

            engine_cfg = engine_cfg.merge_with(
                {
                    "graph": {
                        "enabled": True,
                        "embeddings": {
                            "enabled": True,
                            # Use the same embedding configuration as memory by default.
                            "embedding": {
                                "provider": engine_cfg.storage.memory.vector.embedding.provider,
                                "model": engine_cfg.storage.memory.vector.embedding.model,
                                "api_key": engine_cfg.storage.memory.vector.embedding.api_key,
                                "dimension": engine_cfg.storage.memory.vector.embedding.dimension,
                                "batch_size": engine_cfg.storage.memory.vector.embedding.batch_size,
                                "base_url": engine_cfg.storage.memory.vector.embedding.base_url,
                            },
                        },
                        "invalidation": {
                            "enabled": True,
                            "candidate_limit": 25,
                            "model": None,
                        },
                        "temporal": {
                            "enabled": True,
                            "model": None,
                        },
                        "communities": {
                            "enabled": True,
                            "max_communities": 5,
                            # Keep the demo responsive: rebuild more frequently on small graphs.
                            "rebuild_every_n_episodes": 3,
                            "min_cluster_size": 2,
                            "max_concurrency": 3,
                            "model": None,
                        },
                        "store": {
                            "backend": graph_backend,
                            "neo4j": {
                                "url": neo4j_url,
                                "username": neo4j_username,
                                "password": neo4j_password,
                                "database": neo4j_database,
                                "create_indexes": True,
                                "entity_label": "__Entity__",
                            },
                        },
                        # Use the built-in default ontology by default.
                        "ontology": {"module": "ctxforge.graph.default_ontology", "attr_name": "GRAPH_ONTOLOGY"},
                        "extraction": {"enabled": True, "model": None},
                        "retrieval": {
                            "enabled": True,
                            "max_facts": 20,
                            "max_entities": 20,
                            "max_episodes": 5,
                            "include_entities": True,
                            "include_episodes": False,
                            "valid_only": True,
                            # Hybrid recipe: seed entities via semantic+keyword, then expand via BFS,
                            # and use deterministic RRF to merge edge candidates.
                            "methods": ["semantic", "keyword", "bfs"],
                            "seed_k": 8,
                            "bfs_max_depth": 2,
                            "bfs_edges_per_node": 12,
                            "rerank_enabled": True,
                            "reranker": "rrf",
                            "rerank_top_k": 30,
                            # Planner/evidence + token budgeting demo knobs
                            "planner_mode": "auto",
                            "evidence_enabled": True,
                            "max_evidence_items": 5,
                            "max_entity_tokens": 250,
                            "max_relation_tokens": 350,
                            "max_evidence_tokens": 220,
                            "max_total_tokens": 900,
                        },
                        "section_name": "Graph Memory",
                    }
                }
            )

            # Enable two-step fusion (opt-in workflow; requires a provided llm provider at call-site).
            engine_cfg = engine_cfg.merge_with(
                {
                    "fusion": {
                        "enabled": True,
                        "mode": "two_step",
                        "synthesis_model": None,
                        "max_tokens": 800,
                    }
                }
            )

        if self.use_expertise:
            engine_cfg.pipelines.record.chain.append(
                MiddlewareItemConfig(
                    type="expertise_evolution",
                    enabled=True,
                    # High priority so it wraps the rest of the record chain:
                    # it calls next() first, then evolves after downstream middleware.
                    priority=1000,
                    phases=["record_persisted"],
                    config={
                        "auto_curate": True,
                        "evolve_on_success": True,
                        "evolve_on_failure": True,
                        "min_confidence": 0.2,
                    },
                )
            )

        # 4) Providers + engine wiring
        factory = EngineFactory()
        self._factory = factory
        embedding_provider = factory._create_embedding_provider(engine_cfg)  # demo-only shortcut

        if self.use_expertise:
            if self.use_postgres:
                expertise_store = PostgresExpertiseStore(pg_config)  # type: ignore[name-defined]
                await expertise_store.initialize()
                self._expertise_store = expertise_store
            elif self.use_mysql:
                expertise_store = MySQLExpertiseStore(mysql_config)  # type: ignore[name-defined]
                await expertise_store.initialize()
                self._expertise_store = expertise_store
            else:
                expertise_store = InMemoryExpertiseStore()
        else:
            expertise_store = None

        extractor = HybridExtractor(extractors=[PatternExtractor(), EntityExtractor()])

        # Full expertise evolution pipeline components (Reflector + Curator)
        reflector = None
        curator = None
        if self.use_expertise:
            # Use the same configured LLM provider for reflection/curation.
            llm_for_tools = factory._create_llm_provider(engine_cfg)  # demo-only shortcut
            if llm_for_tools is None:
                llm_for_tools = MockLLMProvider(latency_ms=0)
            reflector = ExpertiseReflector(llm_provider=llm_for_tools)
            curator = ExpertiseCurator(llm_provider=llm_for_tools)

        print("\n🧩 Building CtxForge via EngineFactory.build()...")
        self.engine = await factory.build(
            engine_cfg,
            session_store=session_store,
            memory_store=memory_store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            extractor=extractor,
            expertise_store=expertise_store,
            expertise_vector_store=vector_store if self.use_expertise else None,
            expertise_embedding_provider=embedding_provider if self.use_expertise else None,
            reflector=reflector,
            curator=curator,
        )
        print("   ✅ Engine ready")

        if exercise_plan_features:
            try:
                retriever_name = getattr(self.engine, "_retriever", None).name if getattr(self.engine, "_retriever", None) else "(none)"
                graph_service = getattr(self.engine, "_graph_service", None)
                graph_store_present = bool(getattr(graph_service, "store", None)) if graph_service is not None else False
                print("\n🧪 Feature wiring checks:")
                print(f"   - retrieval: {engine_cfg.retrieval.strategy.value} (rerank_enabled={engine_cfg.retrieval.rerank_enabled})")
                print(f"   - retriever name: {retriever_name}")
                print(f"   - update planning enabled: {engine_cfg.extraction.update_planning_enabled}")
                print(f"   - graph enabled: {engine_cfg.graph.enabled} (backend={engine_cfg.graph.store.backend})")
                print(f"   - graph store wired: {graph_store_present}")
                print(f"   - plugins.modules: {engine_cfg.plugins.modules}")
                print(f"   - fusion enabled: {getattr(engine_cfg, 'fusion', None).enabled if getattr(engine_cfg, 'fusion', None) else False}")
            except Exception as e:
                print(f"   ⚠️ Feature wiring check failed: {e}")

        # Optional: wipe all saved memories for this demo user before seeding
        if reset_memories:
            print(f"\n🧹 Resetting memories for user_id={self.user_id} ...")
            deleted = await self.engine.delete_all_user_memories(self.user_id, include_inactive=True)
            print(f"   ✅ Deleted {deleted} memories")
            # Also reset graph data for this scope so repeated demo runs are deterministic.
            try:
                graph_service = getattr(self.engine, "_graph_service", None)
                store = getattr(graph_service, "store", None) if graph_service is not None else None
                if store is not None:
                    removed = await store.delete_scope(self.user_id)
                    print(f"   ✅ Deleted {removed} graph items (scope_id={self.user_id})")
            except Exception as e:
                print(f"   ⚠️ Failed to reset graph data: {e}")

        # 5) LLM provider (separate from engine)
        self.llm_provider = factory._create_llm_provider(engine_cfg)  # demo-only shortcut
        if self.llm_provider is None:
            # Network-free demo fallback.
            self.llm_provider = MockLLMProvider(latency_ms=0)
            # Keep outputs predictable and short.
            self.llm_provider.set_responses(
                [
                    "OK (mock)",
                    "Noted (mock)",
                    "Here is a concise answer grounded in the provided context. (mock)",
                ]
            )
        print(f"\n🤖 LLM provider: {self.llm_provider.name} ({self.llm_provider.default_model})")

        # 6) Seed a couple memories for retrieval demo
        print("\n🧪 Seeding demo memories...")
        seed_user = [
            MemoryFactory.semantic_memory(self.user_id, "User likes spicy food"),
            MemoryFactory.semantic_memory(self.user_id, "User prefers concise answers"),
        ]
        # Seed one global memory to exercise user+global retrieval merge.
        global_scope_id = getattr(self.engine.config.scopes, "global_scope_id", "global")
        seed_global = [
            MemoryFactory.semantic_memory(global_scope_id, "Global tip: tofu stir-fry is a great spicy dinner option")
        ]

        # Optimization: when embeddings are flaky (e.g. intermittent Azure 5xx), precompute
        # embeddings in one batch for the seed items. This reduces the number of embedding
        # requests during setup and makes the demo more reliable.
        try:
            embedding_provider = factory._create_embedding_provider(engine_cfg)
            if embedding_provider is not None and self.engine.memory_indexer is not None:
                all_seed = seed_user + seed_global
                texts = [self.engine.memory_indexer._build_indexable_content(m) for m in all_seed]  # type: ignore[attr-defined]
                resp = await embedding_provider.embed(texts)
                for m, emb in zip(all_seed, resp.embeddings, strict=False):
                    m.embedding = emb
        except Exception:
            # Best-effort: if embedding precompute fails, fall back to the normal add_memory path.
            pass

        for m in seed_user:
            await self.engine.add_memory(m)
        # Seed one global memory to exercise user+global retrieval merge.
        try:
            for m in seed_global:
                await self.engine.add_memory(m)
            print(f"   ✅ Seeded global memory (scope_id={global_scope_id})")
        except Exception as e:
            print(f"   ⚠️ Failed to seed global memory: {e}")
        print("   ✅ Seeded memories")
        try:
            stored = await self.engine.get_user_memories(self.user_id, limit=25)
            print(f"   📦 Stored memories now: {len(stored)}")
            for m in stored[:10]:
                mtype = getattr(m.type, "value", str(m.type))
                print(f"   - [{mtype}] {m.content}")
        except Exception as e:
            print(f"   ⚠️ Failed to list stored memories: {e}")

        # 7) Seed expertise (if enabled) + index
        if self.use_expertise:
            expertise = await self._create_initial_expertise()
            self.expertise_id = expertise.expertise_id
            await self.engine.save_expertise(expertise)
            if self.engine.expertise_indexer is not None:
                await self.engine.expertise_indexer.index_all(expertise, only_active=False)
            print(f"   ✅ Seeded expertise: {self.expertise_id} (items={len(expertise.items)})")

        print("\n" + "=" * 60)
        print("✨ Setup complete! Type messages to chat (exit to quit).")
        print("=" * 60)

    async def process_turn(
        self,
        user_input: str,
        *,
        outcome: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> str:
        if self.engine is None or self.llm_provider is None:
            raise RuntimeError("Demo not set up; call setup() first")

        # Snapshot stored memories BEFORE this turn (so we can print delta after extraction)
        before_memories = await self.engine.get_user_memories(self.user_id, limit=500)
        before_ids = {m.memory_id for m in before_memories if m.memory_id}

        # 1) Prepare context (unified API — expertise_id is optional)
        context = await self.engine.prepare_context(
            session_id=self.session_id,
            user_id=self.user_id,
            user_input=user_input,
            include_history=True,
            include_memories=True,
            expertise_id=self.expertise_id if (self.use_expertise and self.expertise_id) else None,
            max_expertise_items=5,
        )

        # Print retrieval results for visibility
        print("\n🧠 Retrieved Memories")
        if context.memories:
            for m in context.memories[:10]:
                mtype = getattr(m.type, "value", str(m.type))
                print(f"- [{mtype}] {m.content}")
        else:
            print("- (none)")

        print("\n📚 Retrieved Expertise")
        if context.expertise_items:
            for item in context.expertise_items[:10]:
                section = getattr(item.section, "value", str(item.section))
                print(f"- [{section}] {item.content}  (id={item.item_id})")
        else:
            print("- (none)")

        # Print graph section if enabled
        section_name = None
        try:
            section_name = getattr(self.engine.config.graph, "section_name", None) if getattr(self.engine.config, "graph", None) else None
        except Exception:
            section_name = None
        if section_name:
            graph_section = context.get_section(section_name)
            print(f"\n🕸️ {section_name}")
            if graph_section and graph_section.content:
                print(graph_section.content)
            else:
                print("- (none)")

        # 2) Call the LLM
        chat_messages = _openai_dict_messages_to_chat_messages(context.to_openai_messages())
        llm_resp = await self.llm_provider.chat(chat_messages)
        assistant_text = llm_resp.content

        # 3) Record turn + (if expertise enabled) evolve expertise via record pipeline
        pipeline_metadata = None
        if self.use_expertise and self.expertise_id:
            pipeline_metadata = {
                "expertise_id": self.expertise_id,
                "expertise_items_used": list(context.expertise_items_used or []),
            }
            if outcome:
                pipeline_metadata["turn_outcome"] = outcome
            if ground_truth is not None:
                pipeline_metadata["ground_truth"] = ground_truth

        await self.engine.record_turn(
            session_id=self.session_id,
            user_id=self.user_id,
            user_input=user_input,
            assistant_response=assistant_text,
            pipeline_metadata=pipeline_metadata,
        )
        await self.engine.wait_for_background_tasks(timeout=5.0)

        # Snapshot stored memories AFTER this turn + background extraction, and print delta
        after_memories = await self.engine.get_user_memories(self.user_id, limit=500)
        after_ids = {m.memory_id for m in after_memories if m.memory_id}
        added_ids = after_ids - before_ids
        added_memories = [m for m in after_memories if m.memory_id in added_ids]
        print("\n🧾 New Memories Saved This Turn")
        if added_memories:
            print(f"- Added: {len(added_memories)} (total now: {len(after_memories)})")
            for m in added_memories[:10]:
                mtype = getattr(m.type, "value", str(m.type))
                print(f"  - [{mtype}] {m.content}")
            if len(added_memories) > 10:
                print(f"  - ... and {len(added_memories) - 10} more")
        else:
            print(f"- Added: 0 (total now: {len(after_memories)})")

        return assistant_text

    async def _create_initial_expertise(self) -> Expertise:
        expertise = Expertise(
            expertise_id=f"demo-expertise-{uuid.uuid4().hex[:8]}",
            name="Demo Expertise",
            domain="general",
            description="Seeded expertise for the CtxForge demo.",
        )

        expertise.items.extend(
            [
                ExpertiseItem(
                    section=ExpertiseSection.STRATEGIES,
                    content="Always confirm requirements and constraints before implementing.",
                ),
                ExpertiseItem(
                    section=ExpertiseSection.HEURISTICS,
                    content="Prefer simple solutions; optimize only when necessary.",
                ),
            ]
        )

        return expertise

    async def cleanup(self) -> None:
        """Clean up all resources including database connections."""
        if self.engine is not None:
            await self.engine.close()
        
        # Explicitly close MySQL/PostgreSQL connection pools to avoid
        # "Event loop is closed" warnings on shutdown
        if self._session_store is not None:
            try:
                if hasattr(self._session_store, 'disconnect'):
                    await self._session_store.disconnect()
            except Exception:
                pass
        
        if self._memory_store_inner is not None:
            try:
                if hasattr(self._memory_store_inner, 'disconnect'):
                    await self._memory_store_inner.disconnect()
            except Exception:
                pass
        
        if self._expertise_store is not None:
            try:
                if hasattr(self._expertise_store, 'close'):
                    await self._expertise_store.close()
                elif hasattr(self._expertise_store, 'disconnect'):
                    await self._expertise_store.disconnect()
            except Exception:
                pass


async def _demo_enhanced_extraction_features(demo: ContextEngineDemo) -> None:
    """
    Demonstrate the enhanced extraction features from ENHANCED_EXTRACTION_PLAN.md:
    - Text Alignment (WordAligner, CharSpan)
    - Text Chunking (ChunkIterator)
    - Schema Constraints (JSON schema generation)
    - Visualization (HTML output)
    - Expertise Prompt Validation
    """
    print("\n" + "=" * 60)
    print("🔬 ENHANCED EXTRACTION FEATURES DEMO")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. TEXT ALIGNMENT DEMO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("1️⃣  TEXT ALIGNMENT (WordAligner)")
    print("-" * 50)
    
    source_text = "Alice works at Acme Corp. She loves Python programming and machine learning."
    extractions = ["Alice", "Acme Corp", "Python programming", "deep learning"]  # Last one won't match
    
    aligner = WordAligner(fuzzy_threshold=0.75)
    
    print(f"\n📝 Source Text: \"{source_text}\"")
    print("\n🎯 Aligning extractions to source:")
    
    for extraction in extractions:
        result = aligner.align(extraction, source_text)
        status_icon = {
            AlignmentStatus.MATCH_EXACT: "✅",
            AlignmentStatus.MATCH_FUZZY: "🔶",
            AlignmentStatus.MATCH_PARTIAL: "⚠️",
            AlignmentStatus.UNALIGNED: "❌",
        }.get(result.status, "?")
        
        span_info = ""
        if result.char_span:
            span_info = f" @ chars [{result.char_span.start_pos}:{result.char_span.end_pos}]"
            matched = source_text[result.char_span.start_pos:result.char_span.end_pos]
            span_info += f" = \"{matched}\""
        
        print(f"   {status_icon} \"{extraction}\" → {result.status.value}{span_info} (conf: {result.confidence:.2f})")

    # -------------------------------------------------------------------------
    # 2. TEXT CHUNKING DEMO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("2️⃣  TEXT CHUNKING (ChunkIterator)")
    print("-" * 50)
    
    long_text = """
    This is the first sentence about artificial intelligence.
    Machine learning is a subset of AI that enables systems to learn.
    Deep learning uses neural networks with many layers.
    Natural language processing helps computers understand human language.
    Computer vision allows machines to interpret visual information.
    Reinforcement learning trains agents through rewards and penalties.
    """.strip()
    
    print(f"\n📝 Original Text ({len(long_text)} chars):")
    print(f"   \"{long_text[:80]}...\"")
    
    # Chunk with small buffer to demonstrate splitting
    chunks = list(ChunkIterator(long_text, max_char_buffer=150))
    
    print(f"\n📦 Chunked into {len(chunks)} chunks (max 150 chars each):")
    for i, chunk in enumerate(chunks):
        span = chunk.char_span
        preview = chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
        print(f"   Chunk {i+1}: chars [{span.start_pos}:{span.end_pos}] ({len(chunk.text)} chars)")
        print(f"            \"{preview}\"")
    
    # Demonstrate batching
    batches = list(make_batches(iter(chunks), batch_size=2))
    print(f"\n📊 Batched for parallel processing: {len(batches)} batches of up to 2 chunks")

    # -------------------------------------------------------------------------
    # 3. SCHEMA CONSTRAINTS DEMO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3️⃣  SCHEMA CONSTRAINTS (JSON Schema Generation)")
    print("-" * 50)
    
    # Memory extraction schema
    from ctxforge.core.memory import MemoryType as MT
    memory_schema = generate_memory_extraction_schema(
        allowed_types=[MT.SEMANTIC, MT.EPISODIC],
        config=SchemaConfig(),
    )
    print("\n📋 Memory Extraction Schema (for semantic/episodic):")
    print(f"   Type: {memory_schema.get('type')}")
    print(f"   Properties: {list(memory_schema.get('properties', {}).keys())}")
    if "items" in memory_schema:
        item_props = memory_schema["items"].get("properties", {})
        if "memory_type" in item_props and "enum" in item_props["memory_type"]:
            print(f"   Allowed memory_types: {item_props['memory_type']['enum']}")
    
    # Graph extraction schema
    from ctxforge.graph.ontology import GraphOntology
    demo_ontology = GraphOntology(
        entity_types={"Person": None, "Organization": None, "Location": None},
        edge_types={"works_at": None, "knows": None, "located_in": None},
        allowed_edges={
            "works_at": [("Person", "Organization")],
            "knows": [("Person", "Person")],
            "located_in": [("Person", "Location"), ("Organization", "Location")],
        },
    )
    graph_schema = generate_graph_extraction_schema(
        ontology=demo_ontology,
        config=SchemaConfig(),
    )
    print("\n📋 Graph Extraction Schema:")
    print(f"   Entity types: {list(demo_ontology.entity_types.keys())}")
    print(f"   Edge types: {list(demo_ontology.edge_types.keys())}")
    
    entity_props = graph_schema.get("properties", {}).get("entities", {}).get("items", {}).get("properties", {})
    if "entity_type" in entity_props and "enum" in entity_props["entity_type"]:
        print(f"   Entity type enum: {entity_props['entity_type']['enum']}")

    # -------------------------------------------------------------------------
    # 4. EXPERTISE PROMPT VALIDATION DEMO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("4️⃣  EXPERTISE PROMPT VALIDATION")
    print("-" * 50)
    
    # Create some example expertise examples (some valid, some invalid)
    examples = [
        ExpertiseExample(
            turn_input="How do I sort a list in Python?",
            turn_response="Use sorted() or list.sort() for in-place sorting.",
            expected_items=["sorting", "Python"],
            expected_feedback={"item-1": "helpful"},
        ),
        ExpertiseExample(
            turn_input="",  # Invalid: missing input
            turn_response="This won't work well.",
            expected_items=[],
            expected_feedback={},
        ),
        ExpertiseExample(
            turn_input="What is recursion?",
            turn_response="A function that calls itself.",
            expected_items=["recursion"],
            expected_feedback={"item-2": "invalid_feedback"},  # Invalid feedback value
        ),
    ]
    
    print(f"\n📝 Validating {len(examples)} expertise examples...")
    report = validate_expertise_examples(examples)
    
    print("\n📊 Validation Report:")
    print(f"   Has errors: {report.has_errors}")
    print(f"   Has warnings: {report.has_warnings}")
    print(f"   Is valid: {report.is_valid}")
    
    if report.issues:
        print(f"\n⚠️  Issues found ({len(report.issues)}):")
        for issue in report.issues:
            icon = "❌" if issue.issue_type == "error" else "⚠️"
            print(f"   {icon} {issue.short_msg()}")

    # -------------------------------------------------------------------------
    # 5. VISUALIZATION DEMO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("5️⃣  EXTRACTION VISUALIZATION")
    print("-" * 50)
    
    # Create mock extraction candidates with source grounding
    from ctxforge.core.memory import MemoryType
    from ctxforge.protocols.extractor import ExtractionCandidate
    
    viz_source = "John loves Python programming. He works at Google in California."
    
    # Align and create candidates
    candidates = []
    extractions_for_viz = [
        ("John", MemoryType.SEMANTIC),
        ("Python programming", MemoryType.SEMANTIC),
        ("Google", MemoryType.SEMANTIC),
        ("California", MemoryType.EPISODIC),
    ]
    
    for content, mem_type in extractions_for_viz:
        result = aligner.align(content, viz_source)
        if result.status != AlignmentStatus.UNALIGNED:
            candidates.append(ExtractionCandidate(
                content=content,
                memory_type=mem_type,
                confidence=result.confidence,
                source_text=viz_source,
                source_span=result.char_span,
                alignment_status=result.status,
                matched_text=result.matched_text,
            ))
    
    print(f"\n📝 Source: \"{viz_source}\"")
    print(f"✅ Created {len(candidates)} aligned extraction candidates")
    
    # Generate visualization HTML
    html_output = visualize_memory_extractions(
        source_text=viz_source,
        candidates=candidates,
        title="Enhanced Extraction Demo",
    )
    
    # Save to file
    viz_path = Path(__file__).parent / "extraction_viz_demo.html"
    try:
        save_visualization(html_output, str(viz_path))
        print(f"\n🌐 Visualization saved to: {viz_path}")
        print("   Open in browser to see highlighted extractions with tooltips!")
    except Exception as e:
        print(f"\n⚠️  Could not save visualization: {e}")
    
    # Show what's in the HTML
    print("\n📊 HTML Visualization Preview:")
    print("   - Document title: Enhanced Extraction Demo")
    print(f"   - Legend items: {len(set(c.memory_type.value for c in candidates))} types")
    print(f"   - Highlighted spans: {len(candidates)}")
    for c in candidates:
        span_info = f"[{c.source_span.start_pos}:{c.source_span.end_pos}]" if c.source_span else "N/A"
        print(f"     • \"{c.content}\" ({c.memory_type.value}) @ {span_info}")

    # -------------------------------------------------------------------------
    # 6. GRAPH EXTRACTION WITH SOURCE GROUNDING
    # -------------------------------------------------------------------------
    if demo.engine is not None and getattr(demo.engine.config.graph, "enabled", False):
        print("\n" + "-" * 50)
        print("6️⃣  GRAPH EXTRACTION WITH SOURCE GROUNDING")
        print("-" * 50)
        
        graph_service = getattr(demo.engine, "_graph_service", None)
        store = getattr(graph_service, "store", None) if graph_service else None
        
        if store is not None:
            try:
                # Get some nodes and edges with their source grounding info
                search_result = await store.search(
                    demo.user_id,
                    "",
                    scope="nodes",
                    limit=5,
                )
                
                print("\n📊 Graph Nodes with Source Grounding:")
                for node in search_result.nodes[:5]:
                    print(f"   • {node.name} (labels: {node.labels})")
                    if node.source_episode_ids:
                        print(f"     Source episodes: {node.source_episode_ids[:3]}")
                    if node.source_spans:
                        for ep_id, span in list(node.source_spans.items())[:2]:
                            if isinstance(span, CharSpan):
                                print(f"     Span in {ep_id}: [{span.start_pos}:{span.end_pos}]")
                    if node.alignment_status:
                        status_val = node.alignment_status.value if hasattr(node.alignment_status, 'value') else node.alignment_status
                        print(f"     Alignment: {status_val} (conf: {node.extraction_confidence:.2f})")
                
                # Get edges too
                edge_result = await store.search(
                    demo.user_id,
                    "",
                    scope="edges",
                    limit=5,
                )
                
                print("\n📊 Graph Edges with Source Grounding:")
                for edge in edge_result.edges[:5]:
                    print(f"   • {edge.edge_type}: {edge.fact or 'N/A'}")
                    if edge.source_spans:
                        for ep_id, span in list(edge.source_spans.items())[:2]:
                            if isinstance(span, CharSpan):
                                print(f"     Span in {ep_id}: [{span.start_pos}:{span.end_pos}]")
                    if edge.alignment_status:
                        status_val = edge.alignment_status.value if hasattr(edge.alignment_status, 'value') else edge.alignment_status
                        print(f"     Alignment: {status_val}")
                
            except Exception as e:
                print(f"\n⚠️  Graph source grounding demo skipped: {e}")

    print("\n" + "=" * 60)
    print("✅ ENHANCED EXTRACTION FEATURES DEMO COMPLETE")
    print("=" * 60)


async def _run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres", action="store_true", help="Use PostgreSQL stores (requires running DB)")
    parser.add_argument("--mysql", action="store_true", help="Use MySQL stores (requires running DB)")
    parser.add_argument("--expertise", action="store_true", help="Enable expertise features")
    parser.add_argument("--config", default=None, help="Path to engine_config.yaml")
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Delete Chroma persist_directory before running (useful if the on-disk index is corrupted)",
    )
    parser.add_argument(
        "--reset-memories",
        action="store_true",
        help="Delete all stored memories for the demo user before running (Postgres/Redis/In-memory)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive chat loop (default is scripted demo run)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Disable the additional features from docs/MEM0_ZEP_IMPLEMENTATION_PLAN.md (keeps the original demo behavior)",
    )
    parser.add_argument(
        "--graph-backend",
        default="memory",
        choices=["memory", "neo4j"],
        help="Graph backend to use when exercising graph memory (default: memory)",
    )
    args = parser.parse_args()

    # Best-effort: load demo environment variables from ctxforge/examples/.env.
    # This avoids requiring shell-specific "source" commands when running the demo.
    try:
        from dotenv import load_dotenv  # type: ignore

        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        # python-dotenv is optional; config may still be supplied via the shell or other means.
        pass

    cfg = load_config(args.config)
    print_config_summary(cfg)

    demo = ContextEngineDemo(cfg, use_postgres=bool(args.postgres), use_mysql=bool(args.mysql), use_expertise=bool(args.expertise))
    await demo.setup(
        reset_chroma=bool(args.reset_chroma),
        reset_memories=bool(args.reset_memories),
        exercise_plan_features=not bool(args.minimal),
        graph_backend=str(args.graph_backend),
    )

    if args.interactive:
        while True:
            user_input = input("\nYou> ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            response = await demo.process_turn(user_input)
            print(f"\nAssistant> {response}")
    else:
        # Use a per-run unique suffix so memory extraction produces non-duplicate facts
        # even if you don't pass --reset-memories.
        unique = uuid.uuid4().hex[:8]
        scripted_turns: List[Dict[str, Any]] = [
            # This turn should retrieve the seeded memories (semantic match)
            {
                "user": "Quickly suggest a spicy dinner idea. Keep it concise.",
                "outcome": "success",
            },
            # Global scope retrieval check (seeded global memory should be eligible)
            {
                "user": "Any global tips for spicy dinner ideas?",
                "outcome": "success",
            },
            # These turns are designed to reliably trigger memory extraction (name/email/location/preferences).
            {
                "user": (
                    f"My name is Demo User {unique}. "
                    f"My email is demo+{unique}@example.com. "
                    "I live in Berlin and prefer vegetarian food."
                ),
                "outcome": "success",
            },
            {
                "user": (
                    "More preferences: I hate cilantro, and I'm allergic to peanuts. "
                    "Please remember these."
                ),
                "outcome": "success",
            },
            # Graph-friendly turns (ontology-aligned) to exercise episode->extract->upsert->retrieve.
            # These are intentionally explicit about entity/edge types to maximize extraction reliability.
            {
                "user": (
                    "Graph seed. Use these exact ontology types:\n"
                    "- entity types: Person, Organization, Location\n"
                    "- edge types: WORKS_FOR, LIKES\n"
                    "\n"
                    "I am a Person. I started WORKS_FOR the Organization \"Acme\" on 2020-01-01. "
                    "I LIKES the Location \"Berlin\"."
                ),
                "outcome": "success",
            },
            {
                "user": (
                    "More graph facts (same ontology):\n"
                    "I LIKES the Location \"Tokyo\".\n"
                    "I WORKS_FOR the Organization \"Acme\".\n"
                ),
                "outcome": "success",
            },
            {
                "user": (
                    "Graph seed follow-up. Use the same ontology.\n"
                    "Entities:\n"
                    "- Organization: Acme\n"
                    "- Location: Berlin\n"
                    "\n"
                    "Relationship:\n"
                    "- Acme LIKES Berlin\n"
                ),
                "outcome": "success",
            },
            {
                "user": (
                    "Using the graph, answer: what Location does my employer LIKES?"
                ),
                "outcome": "success",
            },
            {
                "user": "Using the graph, answer: what Organization do I WORKS_FOR and what Location do I LIKES?",
                "outcome": "success",
            },
            {
                "user": (
                    f"Update: Demo User {unique} no longer WORKS_FOR the Organization \"Acme\". "
                    f"Demo User {unique} now WORKS_FOR the Organization \"Globex\"."
                ),
                "outcome": "success",
                "tag": "job_change",
            },
            {
                "user": "Using the graph, answer again: what Organization do I WORKS_FOR now?",
                "outcome": "success",
            },
            {
                "user": "We are building an expertise system. What should a good expertise item look like?",
                "outcome": "success",
            },
            {
                "user": "Extract 3 reusable 'rules of thumb' from this: 'I keep forgetting to add tests and I break stuff.'",
                "outcome": "success",
            },
            {
                # Provide ground truth + explicit outcome to drive reflection/curation -> new expertise items.
                # (Even if the model answers correctly, we can still demonstrate the pipeline behavior.)
                "user": "Answer with exactly '42' and nothing else.",
                "ground_truth": "42",
                "outcome": "failure",
            },
            {
                "user": "Given the failure above, propose one new expertise item that would prevent that mistake in the future.",
                "outcome": "success",
            },
            {
                # Another turn that often produces actionable curation suggestions.
                "user": "Summarize the expertise evolution rule in one sentence, then give one example.",
                "outcome": "success",
            },
        ]

        demo_job_change_ts: Optional[datetime] = None
        for turn in scripted_turns:
            user_text = turn["user"]
            print(f"\nYou> {user_text}")
            if turn.get("tag") == "job_change":
                demo_job_change_ts = datetime.now(timezone.utc)
            response = await demo.process_turn(
                user_text,
                outcome=turn.get("outcome"),
                ground_truth=turn.get("ground_truth"),
            )
            print(f"\nAssistant> {response}")

        # Optional: demonstrate two-step answer workflow (graph + memory → synthesis).
        if demo.engine is not None and demo.llm_provider is not None and getattr(demo.engine.config, "fusion", None) is not None:
            if getattr(demo.engine.config.fusion, "enabled", False):
                try:
                    from ctxforge.helpers import answer_two_step as helpers_answer_two_step

                    print("\n" + "=" * 60)
                    print("🧩 Two-step answer demo (graph + memory → synthesis)")
                    print("=" * 60)
                    fused = await helpers_answer_two_step(
                        demo.engine,
                        demo.llm_provider,
                        session_id=demo.session_id,
                        user_id=demo.user_id,
                        user_input="What organization do I work for now and which locations do I like?",
                    )
                    print(f"\nTwo-step Answer> {fused}")
                except Exception as e:
                    print(f"\n🧩 Two-step answer skipped: {e}")

        # Quick sanity check of graph filters (label-based node filtering).
        if demo.engine is not None and getattr(demo.engine.config.graph, "enabled", False):
            graph_service = getattr(demo.engine, "_graph_service", None)
            store = getattr(graph_service, "store", None) if graph_service is not None else None
            if store is not None:
                try:
                    out = await store.search(
                        demo.user_id,
                        "",
                        scope="nodes",
                        limit=10,
                        filters=GraphSearchFilters(node_labels=["Organization"]),
                    )
                    orgs = [n.name for n in out.nodes]
                    print(f"\n🔎 Graph filter check (Organization nodes): {orgs}")
                except Exception as e:
                    print(f"\n🔎 Graph filter check skipped: {e}")

            # Semantic node search check (best-effort; requires embeddings enabled + Neo4j vector support).
            embedder = getattr(graph_service, "_embed", None) if graph_service is not None else None
            if store is not None and embedder is not None:
                try:
                    query_text = "Acme company"
                    qv = await embedder.embed_single(query_text)
                    nodes = await store.search_nodes_semantic(
                        demo.user_id,
                        qv,
                        limit=10,
                        filters=GraphSearchFilters(node_labels=["Organization"]),
                    )
                    stop_names = {"system", "you", "user", "assistant"}
                    names = [n.name for n in nodes if (n.name or "").strip().lower() not in stop_names]
                    print(f"\n🧭 Graph node semantic search (query='{query_text}', label=Organization): {names[:5]}")
                except Exception as e:
                    print(f"\n🧭 Graph node semantic search skipped: {e}")

            # Temporal validity demo: compare WORKS_FOR edges as-of before the job change vs now.
            if store is not None and demo_job_change_ts is not None:
                try:
                    print("\n" + "=" * 60)
                    print("🕰️ Graph temporal validity demo (as_of)")
                    print("=" * 60)
                    out_before = await store.search(
                        demo.user_id,
                        "",
                        scope="edges",
                        limit=50,
                        filters=GraphSearchFilters(edge_types=["WORKS_FOR"], valid_only=True, as_of=demo_job_change_ts),
                    )
                    before_facts = [e.fact for e in out_before.edges if e.fact]
                    print(f"\nAs-of {demo_job_change_ts.isoformat()} valid WORKS_FOR facts:")
                    for f in before_facts[:10]:
                        print(f"- {f}")

                    out_now = await store.search(
                        demo.user_id,
                        "",
                        scope="edges",
                        limit=50,
                        filters=GraphSearchFilters(edge_types=["WORKS_FOR"], valid_only=True),
                    )
                    now_facts = [e.fact for e in out_now.edges if e.fact]
                    print("\nCurrent valid WORKS_FOR facts:")
                    for f in now_facts[:10]:
                        print(f"- {f}")
                except Exception as e:
                    print(f"\n🕰️ Graph temporal validity demo skipped: {e}")

        # =====================================================
        # ENHANCED EXTRACTION FEATURES DEMO
        # =====================================================
        if not args.minimal:
            await _demo_enhanced_extraction_features(demo)

        # Final summary of stored state
        if demo.engine is not None:
            print("\n" + "=" * 60)
            print("📦 Final Stored State")
            print("=" * 60)
            memories = await demo.engine.get_user_memories(demo.user_id, limit=25)
            print(f"\n🧠 Stored Memories (showing {min(len(memories), 25)}/{len(memories)})")
            for m in memories[:25]:
                mtype = getattr(m.type, "value", str(m.type))
                print(f"- [{mtype}] {m.content}")

            if demo.use_expertise and demo.expertise_id:
                exp = await demo.engine.load_expertise(demo.expertise_id)
                if exp is not None:
                    print(f"\n📚 Stored Expertise: {exp.expertise_id} ({exp.name})")
                    for item in exp.items[:25]:
                        section = getattr(item.section, "value", str(item.section))
                        print(f"- [{section}] {item.content}  (id={item.item_id})")

    await demo.cleanup()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()


