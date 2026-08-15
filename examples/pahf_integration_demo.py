"""
Memory Integration Demo (Real Infrastructure)
===============================================

Validates all four memory integration features end-to-end using real infrastructure:
- **MySQL** for session and memory storage
- **Neo4j** for graph storage (optional, falls back to in-memory)
- **Azure OpenAI** as the LLM provider

Credentials and connection details come from ``examples/.env`` via ``examples/config.py``.

Features demonstrated:
  Feature 1 - Multi-Level Memory Integration Pipeline (Detect -> Summarize -> Dedup -> Integrate -> Store)
  Feature 2 - Preference Evolution Tracking (change detection, supersession, history)
  Feature 3 - Personalization Effectiveness Metrics (hit rate, ACPE, feedback frequency)
  Feature 4 - Memory Context Summarization (narrative synthesis before LLM injection)

Run:
    cd ctxforge
    source venv/bin/activate
    python examples/pahf_integration_demo.py
"""

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

# Ensure ctxforge is importable from the examples/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxforge.core.memory import MemoryFactory, MemoryType
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.services.memory_synthesizer_service import (
    MemorySynthesizerService,
)
from ctxforge.engine.services.personalization_metrics_service import (
    PersonalizationMetricsService,
)
from ctxforge.engine.services.preference_evolution_service import (
    PreferenceEvolutionService,
)
from ctxforge.examples.config import DemoConfig, load_config, print_config_summary
from ctxforge.extraction.integration_config import (
    IntegrationConfig,
    PersonalizationMetricsConfig,
    PreferenceEvolutionConfig,
    SynthesizerConfig,
)
from ctxforge.extraction.integration_pipeline import (
    MemoryIntegrationPipeline,
)
from ctxforge.protocols.extractor import ExtractionCandidate
from ctxforge.storage import DeduplicatingMemoryStore
from ctxforge.storage.connection import MySQLConfig as MySQLCfg
from ctxforge.storage.mysql.memory import MySQLMemoryStore
from ctxforge.storage.mysql.session import MySQLSessionStore

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

OK = "[OK]"
WARN = "[WARN]"
WIDTH = 72


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def section(title: str) -> None:
    print(f"\n--- {title} ---\n")


def soft_check(label: str, condition: bool, detail: str = "") -> None:
    """Print result but never hard-fail (LLM output is non-deterministic)."""
    status = OK if condition else WARN
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status} {label}{suffix}")


def indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(text, prefix)


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

def _build_mysql_config(config: DemoConfig) -> MySQLCfg:
    return MySQLCfg(
        host=config.mysql.host,
        port=config.mysql.port,
        database=config.mysql.database,
        user=config.mysql.user,
        password=config.mysql.password,
    )


async def _init_mysql_stores(mysql_config: MySQLCfg):
    """Initialize MySQL session + memory stores, raising on failure."""
    session_store = MySQLSessionStore(mysql_config)
    memory_store_inner = MySQLMemoryStore(mysql_config)
    await session_store.initialize()
    await memory_store_inner.initialize()
    return session_store, memory_store_inner


def _create_llm_provider(engine_cfg):
    """Create a real LLM provider from config (Azure OpenAI or OpenAI)."""
    factory = EngineFactory()
    llm = factory._create_llm_provider(engine_cfg)
    if llm is None:
        raise RuntimeError(
            "Could not create LLM provider from config. "
            "Ensure AZURE_OPENAI_* or OPENAI_API_KEY env vars are set in examples/.env"
        )
    return llm


# ---------------------------------------------------------------------------
# Feature 1: Multi-Level Memory Integration Pipeline
# ---------------------------------------------------------------------------

async def demo_feature1_integration_pipeline(
    llm, memory_store_inner: MySQLMemoryStore,
) -> None:
    banner("Feature 1: Multi-Level Memory Integration Pipeline")

    print(
        "The integration pipeline runs every extraction candidate through five\n"
        "stages: Detect -> Summarize -> Dedup -> Integrate -> Store.\n"
        "Using real Azure OpenAI for detect/summarize/integrate stages.\n"
    )

    user_id = "demo-f1"

    # Clean up any leftover data from previous runs.
    try:
        existing = await memory_store_inner.get_by_user(user_id)
        for m in existing:
            await memory_store_inner.delete(m.memory_id)
    except Exception:
        pass

    # ----- 1a. Non-actionable candidate is filtered by Detect -----
    section("1a. Detect stage filters non-actionable content")

    config = IntegrationConfig(skip_detect_for_high_confidence=False)
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=memory_store_inner, config=config,
    )

    candidate_ack = ExtractionCandidate(
        content="ok thanks",
        memory_type=MemoryType.SEMANTIC,
        confidence=0.6,
        source_text="ok thanks",
    )
    results = await pipeline.process(
        candidates=[candidate_ack], user_id=user_id, query="thanks",
    )
    soft_check(
        "Acknowledgment 'ok thanks' filtered out",
        len(results) == 0,
        f"got {len(results)} results",
    )

    # ----- 1b. Actionable candidate passes through all stages -----
    section("1b. Full pipeline: Detect -> Summarize -> Dedup (new) -> Store")

    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=memory_store_inner, config=config,
    )

    candidate_veg = ExtractionCandidate(
        content="I'm a vegetarian and I never eat meat",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.85,
        source_text="I'm a vegetarian and I never eat meat",
    )
    results = await pipeline.process(
        candidates=[candidate_veg], user_id=user_id, query="What should I eat?",
    )

    soft_check("At least one result produced", len(results) >= 1)
    if results:
        soft_check("Operation is 'add' (no prior memory)", results[0].operation == "add")
        if results[0].memory_item:
            print(f"\n    Stored memory: {results[0].memory_item.content!r}")

    stored = await memory_store_inner.get_by_user(user_id)
    soft_check("Memory persisted in MySQL store", len(stored) >= 1, f"count={len(stored)}")

    # ----- 1c. Dedup detects similar existing memory -> UPDATE -----
    section("1c. Dedup detects similar memory -> merge via Integrate stage")

    config_low_thresh = IntegrationConfig(
        skip_detect_for_high_confidence=False,
        similarity_threshold=0.3,
    )
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=memory_store_inner, config=config_low_thresh,
    )

    candidate_veg2 = ExtractionCandidate(
        content="I'm a strict vegetarian; I also avoid eggs",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.9,
        source_text="I'm a strict vegetarian; I also avoid eggs",
    )
    results = await pipeline.process(
        candidates=[candidate_veg2], user_id=user_id, query="diet",
    )

    if results:
        soft_check("Operation is 'update' (dedup matched)", results[0].operation == "update")
        soft_check(
            "Similarity score > 0",
            results[0].similarity_score > 0,
            f"score={results[0].similarity_score:.3f}",
        )
        if results[0].memory_item:
            print(f"\n    Updated memory: {results[0].memory_item.content!r}")
    else:
        soft_check("Got results from dedup pipeline", False, "no results")

    # ----- 1d. High-confidence skip detect -----
    section("1d. High-confidence candidates skip Detect stage")

    # Use a fresh store section (different user_id) so no dedup match
    user_id_hi = "demo-f1-hi"
    try:
        existing = await memory_store_inner.get_by_user(user_id_hi)
        for m in existing:
            await memory_store_inner.delete(m.memory_id)
    except Exception:
        pass

    config_skip = IntegrationConfig(skip_detect_for_high_confidence=True)
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=memory_store_inner, config=config_skip,
    )

    candidate_hi = ExtractionCandidate(
        content="I love hiking in the mountains",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.95,
        source_text="I love hiking in the mountains",
    )
    results = await pipeline.process(
        candidates=[candidate_hi], user_id=user_id_hi, query="hobbies",
    )

    soft_check("High-confidence candidate produced result", len(results) >= 1)
    if results:
        soft_check(
            "detect_skipped metadata set",
            results[0].stage_metadata.get("detect_skipped") is True,
        )


# ---------------------------------------------------------------------------
# Feature 2: Preference Evolution Tracking
# ---------------------------------------------------------------------------

async def demo_feature2_preference_evolution(
    llm, memory_store_inner: MySQLMemoryStore,
) -> None:
    banner("Feature 2: Preference Evolution Tracking")

    print(
        "When a user changes a preference (e.g. dark mode -> light mode),\n"
        "the system detects the change via real LLM, supersedes the old memory,\n"
        "and maintains a preference changelog.\n"
    )

    user_id = "demo-f2"

    # Clean up
    try:
        existing = await memory_store_inner.get_by_user(user_id)
        for m in existing:
            await memory_store_inner.delete(m.memory_id)
    except Exception:
        pass

    pref_config = PreferenceEvolutionConfig(
        enabled=True,
        auto_supersede=True,
        importance_decay_on_supersede=0.1,
        track_history=True,
    )

    # ----- 2a. Detect a preference change -----
    section("2a. Detect preference change (real LLM)")

    old_mem = MemoryFactory.semantic_memory(
        user_id=user_id, content="User prefers dark mode",
    )
    old_mem.importance = 1.0
    await memory_store_inner.add(old_mem)
    print(f"    Old memory: {old_mem.content!r} (importance={old_mem.importance})")

    service = PreferenceEvolutionService(
        llm=llm, memory_store=memory_store_inner, config=pref_config,
    )
    change = await service.detect_preference_change(
        new_content="I switched to light mode, dark mode hurts my eyes now",
        existing_memory=old_mem,
        query="theme preference",
    )

    soft_check("Preference change detected", change is not None)
    if change:
        print(f"    Change type : {change.change_type}")
        print(f"    Old content : {change.old_content!r}")
        print(f"    New content : {change.new_content!r}")

    # ----- 2b. Apply the change -----
    section("2b. Apply preference change (supersede + decay + link)")

    new_mem = MemoryFactory.semantic_memory(
        user_id=user_id,
        content="User now prefers light mode because dark mode hurts their eyes",
    )

    if change:
        result = await service.apply_preference_change(
            change=change, new_memory=new_mem, old_memory=old_mem,
        )

        soft_check(
            "Old memory superseded",
            old_mem.superseded_by == new_mem.memory_id,
        )
        soft_check(
            "Old memory importance decayed",
            old_mem.importance < 0.2,
            f"importance={old_mem.importance:.3f}",
        )
        soft_check(
            "Preference history in new memory metadata",
            "preference_changes" in result.metadata,
        )

        if "preference_changes" in result.metadata:
            print("\n    Preference changelog:")
            for pc in result.metadata["preference_changes"]:
                print(f"      from: {pc['from']!r}")
                print(f"      to  : {pc['to']!r}")
                print(f"      type: {pc['change_type']}")
        print(f"    Preference version: {result.metadata.get('preference_version')}")
    else:
        print("    Skipping apply (no change detected by LLM).")

    # ----- 2c. Query preference history -----
    section("2c. Query preference history")

    await memory_store_inner.add(new_mem)

    history = await service.get_preference_history(user_id=user_id)
    soft_check("Preference history has entries", len(history) >= 1, f"count={len(history)}")
    if history:
        h = history[0]
        print(f"    From : {h.old_content!r}")
        print(f"    To   : {h.new_content!r}")
        print(f"    Type : {h.change_type}")

    # ----- 2d. No change detected when just adding new info -----
    section("2d. No change when adding new (non-contradictory) info")

    no_change = await service.detect_preference_change(
        new_content="I also like the solarized color scheme",
        existing_memory=new_mem,
        query="color preferences",
    )
    soft_check("No change detected for additive info", no_change is None)


# ---------------------------------------------------------------------------
# Feature 3: Personalization Effectiveness Metrics
# ---------------------------------------------------------------------------

async def demo_feature3_personalization_metrics() -> None:
    banner("Feature 3: Personalization Effectiveness Metrics")

    print(
        "Lightweight metrics track whether personalization improves over\n"
        "time: memory hit rate, feedback frequency, cumulative personalization\n"
        "score (ACPE-inspired), and memory utilization.\n"
        "(This feature is purely computational -- no LLM calls needed.)\n"
    )

    config = PersonalizationMetricsConfig(
        enabled=True,
        memory_hit_threshold=0.6,
    )
    service = PersonalizationMetricsService(config=config)

    section("3a. Simulate 5 conversation turns with retrieval + extraction")

    turns = [
        ([0.9, 0.8, 0.4], False, 2, 0),
        ([0.3, 0.2],       True,  1, 0),
        ([0.85, 0.7, 0.6], False, 1, 1),
        ([0.95],           False, 0, 0),
        ([0.7, 0.65, 0.5], False, 1, 0),
    ]

    for i, (confidences, feedback, extractions, pref_changes) in enumerate(turns, 1):
        memories = []
        for conf in confidences:
            m = MemoryFactory.semantic_memory(user_id="demo-f3", content=f"fact-{i}")
            m.confidence_score = conf
            memories.append(m)

        service.record_retrieval(
            session_id="session-1",
            user_id="demo-f3",
            memories=memories,
        )
        if feedback:
            service.record_feedback(session_id="session-1", feedback_occurred=True)
        if extractions:
            service.record_extraction(
                session_id="session-1",
                extraction_count=extractions,
                preference_changes=pref_changes,
            )

        sm = service.get_session_metrics("session-1")
        t = sm.turn_metrics[-1]
        hit_str = "HIT" if t.memory_hit else "miss"
        fb_str = " + feedback" if t.feedback_occurred else ""
        print(
            f"    Turn {i}: {t.memories_retrieved} memories "
            f"(avg_score={t.memory_avg_score:.2f}, {hit_str})"
            f"{fb_str}"
            f"  extractions={t.extraction_count} pref_changes={t.preference_changes}"
        )

    # ----- 3b. Aggregate metrics -----
    section("3b. Aggregate session metrics")

    sm = service.get_session_metrics("session-1")
    soft_check("5 turns recorded", len(sm.turn_metrics) == 5)

    print(f"\n    Memory hit rate          : {sm.memory_hit_rate:.1%}")
    print(f"    Avg memory relevance     : {sm.avg_memory_relevance:.3f}")
    print(f"    Feedback frequency (FF)  : {sm.feedback_frequency:.1%}")
    print(f"    Memory utilization       : {sm.memory_utilization:.1%}")

    cps = sm.cumulative_personalization_score
    print(f"    Cumulative pers. score   : {[f'{s:.2f}' for s in cps]}")

    soft_check(
        "Memory hit rate is 80% (4 of 5 turns had high-confidence memory)",
        sm.memory_hit_rate == 0.8,
    )
    soft_check(
        "Feedback frequency is 20% (1 of 5 turns)",
        abs(sm.feedback_frequency - 0.2) < 0.01,
    )

    # ----- 3c. Export as dict -----
    section("3c. Export metrics dict (for monitoring/logging)")

    d = service.to_dict("session-1")
    print(indent(json.dumps(d, indent=2, default=str)))

    # ----- 3d. User-level metrics -----
    section("3d. User-level metrics across sessions")

    m2 = MemoryFactory.semantic_memory(user_id="demo-f3", content="test")
    m2.confidence_score = 0.9
    service.record_retrieval(session_id="session-2", user_id="demo-f3", memories=[m2])

    user_metrics = service.get_user_metrics("demo-f3")
    soft_check("2 sessions for user", len(user_metrics) == 2)


# ---------------------------------------------------------------------------
# Feature 4: Memory Context Summarization
# ---------------------------------------------------------------------------

async def demo_feature4_memory_synthesis(llm) -> None:
    banner("Feature 4: Memory Context Summarization")

    print(
        "Instead of injecting individual bullet-point memories into the LLM\n"
        "context, the synthesizer consolidates them into a coherent narrative\n"
        "using real Azure OpenAI.\n"
    )

    # ----- 4a. Synthesize multiple memories into a narrative -----
    section("4a. Synthesize 5 memories into a coherent narrative (real LLM)")

    memories = [
        MemoryFactory.semantic_memory(user_id="demo-f4", content="Dan is vegetarian"),
        MemoryFactory.semantic_memory(user_id="demo-f4", content="Dan prefers spicy food"),
        MemoryFactory.semantic_memory(user_id="demo-f4", content="Dan is allergic to peanuts"),
        MemoryFactory.semantic_memory(user_id="demo-f4", content="Dan likes Thai and Indian cuisine"),
        MemoryFactory.semantic_memory(user_id="demo-f4", content="Dan lives in Portland, OR"),
    ]

    print("    Input memories:")
    for m in memories:
        print(f"      - {m.content}")

    synthesizer = MemorySynthesizerService(llm=llm)
    result = await synthesizer.synthesize(
        memories=memories,
        query="Recommend a restaurant for Dan",
        max_tokens=300,
    )

    soft_check("Synthesis produced a narrative", result is not None)
    if result:
        print("\n    Synthesized narrative:")
        print(indent(result))

    # ----- 4b. Synthesis with irrelevant memories -----
    section("4b. Synthesis with irrelevant memories")

    result_irrelevant = await synthesizer.synthesize(
        memories=[MemoryFactory.semantic_memory(user_id="demo-f4", content="random fact about weather")],
        query="unrelated query about weather",
    )
    soft_check(
        "Returns None for irrelevant memories",
        result_irrelevant is None,
        f"got: {result_irrelevant!r:.80}" if result_irrelevant else "",
    )

    # ----- 4c. Synthesis config threshold -----
    section("4c. Synthesis skipped when fewer than min_memories_to_synthesize")

    synth_config = SynthesizerConfig(min_memories_to_synthesize=5)
    print(
        f"    min_memories_to_synthesize = {synth_config.min_memories_to_synthesize}\n"
        f"    -> Engine would skip synthesis and use raw bullet-point format\n"
        f"       for memory sets smaller than this threshold."
    )
    soft_check("Config threshold works", synth_config.min_memories_to_synthesize == 5)


# ---------------------------------------------------------------------------
# Pipeline integration with preference evolution
# ---------------------------------------------------------------------------

async def demo_pipeline_with_evolution(
    llm, memory_store_inner: MySQLMemoryStore,
) -> None:
    banner("Feature 1+2 Combined: Pipeline with Preference Evolution")

    print(
        "The integration pipeline automatically detects preference changes\n"
        "during the Integrate stage and delegates to the evolution service.\n"
        "All powered by real Azure OpenAI.\n"
    )

    user_id = "demo-f12"

    # Clean up
    try:
        existing = await memory_store_inner.get_by_user(user_id)
        for m in existing:
            await memory_store_inner.delete(m.memory_id)
    except Exception:
        pass

    # Seed an existing preference
    old_pref = MemoryFactory.semantic_memory(
        user_id=user_id, content="User prefers dark mode",
    )
    await memory_store_inner.add(old_pref)
    print(f"    Existing memory: {old_pref.content!r}")

    pref_service = PreferenceEvolutionService(
        llm=llm,
        memory_store=memory_store_inner,
        config=PreferenceEvolutionConfig(enabled=True, auto_supersede=True),
    )

    int_config = IntegrationConfig(
        skip_detect_for_high_confidence=True,
        similarity_threshold=0.3,
    )
    pipeline = MemoryIntegrationPipeline(
        llm=llm,
        memory_store=memory_store_inner,
        config=int_config,
        preference_evolution_service=pref_service,
    )

    candidate = ExtractionCandidate(
        content="I changed my mind, light mode is much better for my eyes",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.95,
        source_text="I changed my mind, light mode is much better for my eyes",
    )

    results = await pipeline.process(
        candidates=[candidate], user_id=user_id, query="theme preference",
    )

    soft_check("At least one result produced", len(results) >= 1)
    if results:
        r = results[0]
        soft_check("Operation is 'update'", r.operation == "update")
        soft_check(
            "Preference change detected in metadata",
            r.stage_metadata.get("preference_change") is True,
        )
        if r.memory_item:
            print(f"\n    Final memory content: {r.memory_item.content!r}")

    stored = await memory_store_inner.get_by_user(user_id)
    print(f"\n    All memories for {user_id}:")
    for m in stored:
        sup = f" [superseded by {m.superseded_by[:8]}...]" if m.superseded_by else ""
        print(
            f"      - {m.content!r} "
            f"(importance={m.importance:.2f}{sup})"
        )


# ---------------------------------------------------------------------------
# End-to-End: Full Engine via EngineFactory.build()
# ---------------------------------------------------------------------------

async def demo_end_to_end(
    config: DemoConfig,
    session_store: MySQLSessionStore,
    memory_store: DeduplicatingMemoryStore,
    memory_store_inner: MySQLMemoryStore,
) -> None:
    banner("End-to-End: Full Engine Integration via EngineFactory.build()")

    print(
        "This section wires all four memory integration features into a CtxForge engine\n"
        "built via EngineFactory.build() with MySQL stores, optional Neo4j,\n"
        "and real Azure OpenAI.\n"
    )

    user_id = "demo-e2e"

    # Clean up previous run data
    try:
        existing = await memory_store_inner.get_by_user(user_id)
        for m in existing:
            await memory_store_inner.delete(m.memory_id)
    except Exception:
        pass

    # Build engine config with memory integration features enabled
    engine_cfg = config.engine.merge_with({
        "extraction": {
            "enabled": True,
            "integration_pipeline_enabled": True,
            "integration_similarity_threshold": 0.3,
            "preference_evolution_enabled": True,
            "synthesis_enabled": True,
            "synthesis_min_memories": 2,
            "personalization_metrics_enabled": True,
            "personalization_memory_hit_threshold": 0.5,
        },
        "retrieval": {
            "strategy": "semantic",
        },
    })

    # Optional: configure Neo4j graph store
    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "contextengine_dev")
    neo4j_database = os.getenv("NEO4J_DATABASE")

    graph_backend = "memory"
    try:
        import neo4j  # noqa: F401

        # Test connectivity
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))
        driver.verify_connectivity()
        driver.close()
        graph_backend = "neo4j"
        print(f"    Neo4j available at {neo4j_url} -- using neo4j backend")
    except Exception:
        print("    Neo4j not available -- falling back to in-memory graph")

    engine_cfg = engine_cfg.merge_with({
        "graph": {
            "enabled": True,
            "store": {
                "backend": graph_backend,
                "neo4j": {
                    "url": neo4j_url,
                    "username": neo4j_username,
                    "password": neo4j_password,
                    "database": neo4j_database,
                    "create_indexes": True,
                },
            },
            "extraction": {"enabled": True, "model": None},
            "retrieval": {"enabled": True, "max_facts": 10},
        },
    })

    # Use ChromaDB vector store (same pattern as run_demo.py)
    try:
        from ctxforge.vectorstores import ChromaDBStore
        from ctxforge.vectorstores.chroma_store import ChromaConfig as ChromaCfg

        chroma_config = ChromaCfg(
            collection_name="integration_demo",
            persist_directory=config.chroma.persist_directory,
            dimension=config.engine.storage.memory.vector.embedding.dimension,
        )
        vector_store = ChromaDBStore(chroma_config)
        await vector_store.initialize()
        print(f"    ChromaDB initialized ({vector_store.name})")
    except Exception as e:
        print(f"    ChromaDB unavailable ({e}) -- engine will use default vector store")
        vector_store = None

    # Build embedding provider
    factory = EngineFactory()
    embedding_provider = factory._create_embedding_provider(engine_cfg)

    section("Building engine via EngineFactory.build()")

    engine = await factory.build(
        engine_cfg,
        session_store=session_store,
        memory_store=memory_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )
    print("    Engine ready")

    # Seed an initial memory
    section("Turn 1: User states a preference")

    initial_mem = MemoryFactory.semantic_memory(
        user_id=user_id, content="User prefers Python for scripting",
    )
    initial_mem.confidence_score = 0.8

    # Pre-compute embedding if possible
    if embedding_provider is not None and engine.memory_indexer is not None:
        try:
            text = engine.memory_indexer._build_indexable_content(initial_mem)
            resp = await embedding_provider.embed([text])
            initial_mem.embedding = resp.embeddings[0]
        except Exception:
            pass

    await engine.add_memory(initial_mem)

    ctx = await engine.prepare_context(
        session_id="demo-s1",
        user_id=user_id,
        user_input="What language should I use for this script?",
    )

    soft_check("Context prepared successfully", ctx is not None)
    if ctx:
        soft_check(
            "Memory section present",
            any(s.name == "memories" for s in ctx.sections),
        )
        mem_section = next((s for s in ctx.sections if s.name == "memories"), None)
        if mem_section:
            is_synthesized = ctx.metadata.get("memory_synthesized", False)
            print("\n    Memory section content:")
            print(indent(mem_section.content[:300]))
            print(f"\n    Synthesized: {is_synthesized}")

    # Check metrics
    sm = engine.get_session_metrics("demo-s1")
    if sm:
        soft_check(
            "Retrieval metrics recorded",
            len(sm.turn_metrics) >= 1,
            f"turns={len(sm.turn_metrics)}",
        )

    section("Turn 1: Record the turn (triggers extraction + integration)")

    await engine.record_turn(
        session_id="demo-s1",
        user_id=user_id,
        user_input="What language should I use for this script?",
        assistant_response="I'd recommend Python since you prefer it for scripting tasks!",
    )

    # Brief wait for background extraction
    await asyncio.sleep(1.0)

    stored = await memory_store_inner.get_by_user(user_id)
    soft_check(
        "Memories in store after turn 1",
        len(stored) >= 1,
        f"count={len(stored)}",
    )
    print("\n    Stored memories:")
    for m in stored:
        print(f"      - [{m.type.value}] {m.content!r}")

    section("Turn 2: Prepare context again (should have more memories)")

    ctx2 = await engine.prepare_context(
        session_id="demo-s1",
        user_id=user_id,
        user_input="Set up my development environment",
    )
    soft_check("Context for turn 2 prepared", ctx2 is not None)

    if ctx2:
        mem_section2 = next((s for s in ctx2.sections if s.name == "memories"), None)
        if mem_section2:
            print("\n    Memory section:")
            print(indent(mem_section2.content[:300]))

    # Final metrics
    section("Final metrics snapshot")

    sm = engine.get_session_metrics("demo-s1")
    if sm:
        print(f"    Turns recorded       : {len(sm.turn_metrics)}")
        print(f"    Memory hit rate      : {sm.memory_hit_rate:.1%}")
        print(f"    Avg memory relevance : {sm.avg_memory_relevance:.3f}")
        print(f"    Feedback frequency   : {sm.feedback_frequency:.1%}")
        cps = sm.cumulative_personalization_score
        print(f"    Cumulative pers.     : {[f'{s:.2f}' for s in cps]}")
    else:
        print("    (No metrics available)")

    user_metrics = engine.get_user_metrics(user_id)
    soft_check(
        "User-level metrics available",
        user_metrics is not None and len(user_metrics) >= 1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print()
    print("*" * WIDTH)
    print("  Memory Integration Demo - Real Infrastructure")
    print("  MySQL + Neo4j (optional) + Azure OpenAI")
    print("*" * WIDTH)

    # ---- Load config ----
    section("Loading configuration")
    try:
        config = load_config()
    except ValueError as e:
        print(f"\n    Configuration error: {e}")
        print("\n    Ensure examples/.env has the required credentials.")
        sys.exit(1)

    print_config_summary(config)

    # ---- Initialize MySQL stores ----
    section("Initializing MySQL stores")
    mysql_config = _build_mysql_config(config)
    print(f"    Connecting to MySQL at {mysql_config.host}:{mysql_config.port}/{mysql_config.database}")

    try:
        session_store, memory_store_inner = await _init_mysql_stores(mysql_config)
    except Exception as e:
        print(f"\n    MySQL connection failed: {e}")
        print("    Ensure MySQL is running locally with:")
        print(f"      host={mysql_config.host} port={mysql_config.port}")
        print(f"      database={mysql_config.database} user={mysql_config.user}")
        sys.exit(1)

    memory_store = DeduplicatingMemoryStore(memory_store_inner)
    print("    MySQL session + memory stores initialized")

    # ---- Create real LLM provider ----
    section("Creating LLM provider")
    engine_cfg = config.engine
    try:
        llm = _create_llm_provider(engine_cfg)
    except Exception as e:
        print(f"\n    LLM provider creation failed: {e}")
        sys.exit(1)

    print(f"    LLM provider: {llm.name} ({llm.default_model})")

    # ---- Run demos ----
    try:
        await demo_feature1_integration_pipeline(llm, memory_store_inner)
        await demo_feature2_preference_evolution(llm, memory_store_inner)
        await demo_feature3_personalization_metrics()
        await demo_feature4_memory_synthesis(llm)
        await demo_pipeline_with_evolution(llm, memory_store_inner)
        await demo_end_to_end(config, session_store, memory_store, memory_store_inner)
    finally:
        # Resource cleanup
        section("Cleaning up resources")
        try:
            if hasattr(session_store, 'close'):
                await session_store.close()
            if hasattr(memory_store_inner, 'close'):
                await memory_store_inner.close()
            print("    MySQL connections closed")
        except Exception as e:
            print(f"    Cleanup warning: {e}")

    print()
    print("=" * WIDTH)
    print("  All demos completed!")
    print("=" * WIDTH)
    print()


if __name__ == "__main__":
    asyncio.run(main())
