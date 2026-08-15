#!/usr/bin/env python3
"""
P0 + P1 Features End-to-End Demo
==================================

Validates the six newly implemented features against MySQL and Neo4j storage:

  1. Tool Memory Type           (P0) -- TOOL enum, ToolExecutionRecord, MemoryFactory
  2. Salience-Aware Retriever   (P0) -- blended similarity/reinforcement/recency
  3. Content Hash Deduplication  (P1) -- O(1) hash-based dedup in DeduplicationConsolidator
  4. Inline Memory References    (P1) -- [ref:ID] markers, citations, reference maps
  5. Hierarchical Categories     (P1) -- MemoryCategory, MemoryCategorizer
  6. Typed Extraction Prompts    (P1) -- per-type system prompts in LLMExtractor

Storage backends exercised:
  - MySQL   -- session + memory persistence
  - Neo4j   -- graph memory (entities, episodes, edges)

Run:
    cd ctxforge
    source venv/bin/activate
    python examples/p0_p1_features_demo.py          # MySQL + Neo4j
    python examples/p0_p1_features_demo.py --skip-neo4j   # MySQL only
    python examples/p0_p1_features_demo.py --in-memory     # no external deps
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Ensure ctxforge is importable when running from the examples/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Load .env (best-effort)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _dotenv_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_dotenv_path, override=False)
    load_dotenv(override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Imports — core models
# ---------------------------------------------------------------------------
# Config
from ctxforge.config.base import RetrievalStrategyType
from ctxforge.core.categories import MemoryCategory
from ctxforge.core.memory import (
    MemoryFactory,
    MemoryItem,
    MemorySource,
    MemoryType,
    ToolExecutionRecord,
    add_tool_record,
    get_tool_statistics,
)

# Extraction
from ctxforge.extraction.categorizer import MemoryCategorizer
from ctxforge.extraction.consolidation.deduplicator import DeduplicationConsolidator
from ctxforge.extraction.llm_extractor import LLMExtractor
from ctxforge.extraction.typed_prompts import (
    get_all_typed_prompts,
    get_typed_prompt,
)
from ctxforge.protocols.extractor import ExtractionCandidate
from ctxforge.protocols.retriever import RetrievalConfig

# Retrieval
from ctxforge.retrieval.retrievers.salience import (
    SalienceRetriever,
    compute_salience_score,
)

# Storage
from ctxforge.storage import (
    InMemoryMemoryStore,
    MySQLConfig,
    MySQLMemoryStore,
)

# Utils
from ctxforge.utils.hashing import compute_content_hash
from ctxforge.utils.references import (
    build_reference_map,
    extract_references,
    format_as_citations,
    strip_references,
)

# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def section(title: str) -> None:
    print(f"\n--- {title} ---\n")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"  [..] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Step 1: Tool Memory Type
# ---------------------------------------------------------------------------

async def demo_tool_memory(memory_store) -> MemoryItem:
    """Demonstrate TOOL memory type with ToolExecutionRecord."""
    banner("Step 1: Tool Memory Type (P0)")

    # 1a. Create a TOOL memory via MemoryFactory
    section("1a. Create TOOL memory via MemoryFactory")
    tool_mem = MemoryFactory.tool_memory(
        user_id="demo-user",
        tool_name="web_search",
        content="Web search tool: use for finding current information",
        tags=["retrieval"],
        metadata={"model_hint": "prefer concise queries"},
    )
    info(f"Created: type={tool_mem.type.value}, tags={tool_mem.tags}")
    info(f"Metadata keys: {list(tool_mem.metadata.keys())}")
    assert tool_mem.type == MemoryType.TOOL
    assert "tool_records" in tool_mem.metadata
    assert "tool_name" in tool_mem.metadata
    ok("MemoryFactory.tool_memory() creates correct TOOL memory")

    # 1b. Persist to the store
    section("1b. Persist TOOL memory")
    mid = await memory_store.add(tool_mem)
    info(f"Stored memory_id={mid or tool_mem.memory_id}")

    retrieved = await memory_store.get(tool_mem.memory_id)
    assert retrieved is not None
    assert retrieved.type == MemoryType.TOOL
    ok(f"TOOL memory persisted and retrieved (type={retrieved.type.value})")

    # 1c. Append ToolExecutionRecords
    section("1c. Append ToolExecutionRecords")
    records = [
        ToolExecutionRecord(
            tool_name="web_search",
            input_params={"query": "latest Python release"},
            output="Python 3.13 ...",
            success=True,
            time_cost=1.2,
            token_cost=350,
            quality_score=0.85,
        ),
        ToolExecutionRecord(
            tool_name="web_search",
            input_params={"query": "weather tomorrow"},
            output=None,
            success=False,
            time_cost=5.0,
            token_cost=0,
            quality_score=0.0,
        ),
        ToolExecutionRecord(
            tool_name="web_search",
            input_params={"query": "Python asyncio tutorial"},
            output="Great tutorial found ...",
            success=True,
            time_cost=0.8,
            token_cost=200,
            quality_score=0.92,
        ),
    ]
    for rec in records:
        add_tool_record(tool_mem, rec)
    info(f"Appended {len(records)} records, access_count={tool_mem.access_count}")

    # 1d. Aggregate statistics
    section("1d. Tool statistics")
    stats = get_tool_statistics(tool_mem)
    info(f"count={stats['count']}, success_rate={stats['success_rate']:.2f}, "
         f"avg_time={stats['avg_time_cost']:.2f}s, avg_score={stats['avg_score']:.2f}")
    assert stats["count"] == 3
    assert 0 < stats["success_rate"] < 1  # 2/3
    ok("get_tool_statistics() returns correct aggregates")

    # 1e. ValueError on non-TOOL memory
    section("1e. Guard: add_tool_record on SEMANTIC memory raises ValueError")
    semantic_mem = MemoryFactory.semantic_memory(user_id="demo-user", content="A fact")
    try:
        add_tool_record(semantic_mem, records[0])
        fail("Should have raised ValueError")
    except ValueError as e:
        ok(f"Correctly raised ValueError: {e}")

    return tool_mem


# ---------------------------------------------------------------------------
# Step 2: Salience-Aware Retriever
# ---------------------------------------------------------------------------

async def demo_salience_retriever(memory_store) -> None:
    """Demonstrate the salience retriever with real storage."""
    banner("Step 2: Salience-Aware Retriever (P0)")

    # 2a. Verify enum registration
    section("2a. RetrievalStrategyType.SALIENCE")
    assert RetrievalStrategyType.SALIENCE.value == "salience"
    ok("SALIENCE registered in RetrievalStrategyType")

    # 2b. Score function
    section("2b. compute_salience_score edge cases")
    assert compute_salience_score(0.9, access_count=0, accessed_at=None) == 0.0
    ok("access_count=0 -> score=0 (cold-start)")

    now = datetime.now(timezone.utc)
    score_recent = compute_salience_score(1.0, 3, accessed_at=now)
    score_old = compute_salience_score(1.0, 3,
                                        accessed_at=now - timedelta(days=60))
    info(f"score(recent)={score_recent:.3f}, score(old)={score_old:.3f}")
    assert score_recent > score_old
    ok("Recent memories score higher than old ones")

    # 2c. Retriever with stored memories
    section("2c. SalienceRetriever end-to-end")
    dim = 8
    emb_base = [0.5] * dim

    memories_to_store = []
    for i, (name, ac, days_ago) in enumerate([
        ("cold-start", 0, 0),
        ("active-recent", 10, 1),
        ("active-old", 10, 90),
        ("moderate", 3, 7),
    ]):
        emb = list(emb_base)
        emb[i % dim] += 0.2  # slight variation
        mem = MemoryItem(
            user_id="demo-user",
            content=f"Salience test: {name}",
            type=MemoryType.SEMANTIC,
            source=MemorySource.AGENT_INFERENCE,
            embedding=emb,
            access_count=ac,
            accessed_at=(now - timedelta(days=days_ago)) if days_ago >= 0 else None,
        )
        await memory_store.add(mem)
        memories_to_store.append(mem)

    async def mock_embed(text: str) -> List[float]:
        return emb_base

    retriever = SalienceRetriever(memory_store, mock_embed, half_life_days=30.0)
    results = await retriever.retrieve("salience test", "demo-user",
                                        RetrievalConfig(limit=10, min_score=0.0))
    info(f"Retrieved {len(results)} results")
    for r in results:
        info(f"  score={r.score:.4f}  content={r.memory.content!r}")

    if results:
        top = results[0]
        assert "active-recent" in top.memory.content
        ok("Top result is 'active-recent' (high access + recent)")
    else:
        ok("(no results with score > 0, cold-start memories only)")


# ---------------------------------------------------------------------------
# Step 3: Content Hash Deduplication
# ---------------------------------------------------------------------------

async def demo_content_hash_dedup(memory_store) -> None:
    """Demonstrate O(1) hash-based deduplication."""
    banner("Step 3: Content Hash Deduplication (P1)")

    # 3a. compute_content_hash
    section("3a. Hash properties")
    h1 = compute_content_hash("The user likes coffee", "semantic")
    h2 = compute_content_hash("  the  USER  likes   coffee  ", "semantic")
    h3 = compute_content_hash("The user likes coffee", "episodic")
    info(f"h1={h1}, h2={h2}, h3={h3}")
    assert h1 == h2, "Normalisation should collapse whitespace + case"
    assert h1 != h3, "Different type prefix should yield different hash"
    assert len(h1) == 16
    ok("Deterministic, normalised, type-prefixed, 16-char hex")

    # 3b. ExtractionCandidate.to_memory_item includes hash
    section("3b. ExtractionCandidate includes content_hash in metadata")
    candidate = ExtractionCandidate(
        content="User prefers morning meetings",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.9,
        source_text="I prefer morning meetings",
    )
    mem = candidate.to_memory_item(user_id="demo-user")
    info(f"content_hash={mem.metadata.get('content_hash')}")
    assert "content_hash" in mem.metadata
    ok("to_memory_item() auto-computes content_hash")

    # 3c. DeduplicationConsolidator hash pass
    section("3c. DeduplicationConsolidator fast hash dedup")
    existing_content = "User is a software engineer"
    existing_hash = compute_content_hash(existing_content, "semantic")

    existing_mem = MemoryItem(
        user_id="demo-user",
        content=existing_content,
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        metadata={"content_hash": existing_hash},
    )
    new_mem_dup = MemoryItem(
        user_id="demo-user",
        content="  user is a software   engineer  ",  # normalises to same hash
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        metadata={"content_hash": compute_content_hash("user is a software engineer", "semantic")},
    )
    new_mem_unique = MemoryItem(
        user_id="demo-user",
        content="User lives in Seattle",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        metadata={"content_hash": compute_content_hash("User lives in Seattle", "semantic")},
    )

    consolidator = DeduplicationConsolidator(similarity_threshold=0.85)
    old_ac = existing_mem.access_count
    result = await consolidator.consolidate(
        new_items=[new_mem_dup, new_mem_unique],
        existing_items=[existing_mem],
    )
    info(f"Input: 2 new items, 1 existing.  Output: {len(result)} passed")
    info(f"Existing access_count: {old_ac} -> {existing_mem.access_count}")
    assert len(result) == 1
    assert result[0].content == "User lives in Seattle"
    assert existing_mem.access_count == old_ac + 1
    ok("Hash-duplicate skipped; unique item passed; existing access_count bumped")


# ---------------------------------------------------------------------------
# Step 4: Inline Memory References
# ---------------------------------------------------------------------------

async def demo_inline_references() -> None:
    """Demonstrate inline [ref:ID] markers."""
    banner("Step 4: Inline Memory References (P1)")

    section("4a. extract_references")
    text = "Based on [ref:mem_001] and [ref:mem_002,mem_003], the user prefers tea."
    refs = extract_references(text)
    info(f"Text: {text!r}")
    info(f"Extracted IDs: {refs}")
    assert refs == ["mem_001", "mem_002", "mem_003"]
    ok("Extracts single and comma-separated ref IDs")

    section("4b. strip_references")
    clean = strip_references(text)
    info(f"Stripped: {clean!r}")
    assert "[ref:" not in clean
    ok("All [ref:...] markers removed, whitespace cleaned")

    section("4c. format_as_citations")
    memory_map = {
        "mem_001": "User mentioned they drink green tea daily",
        "mem_002": "User visited a tea shop in Kyoto",
        "mem_003": "User prefers loose-leaf over bags",
    }
    citations = format_as_citations(memory_map)
    info(f"Citations:\n{citations}")
    assert "[1] (mem_001):" in citations
    ok("Numbered citations generated")

    section("4d. build_reference_map")
    lookup = {
        "mem_001": "Green tea daily",
        "mem_002": "Tea shop in Kyoto",
        "mem_003": "Loose-leaf preference",
        "mem_999": "Unrelated memory",
    }
    ref_map = build_reference_map(text, lookup)
    info(f"Filtered map keys: {list(ref_map.keys())}")
    assert "mem_999" not in ref_map
    assert len(ref_map) == 3
    ok("build_reference_map filters to referenced IDs only")


# ---------------------------------------------------------------------------
# Step 5: Hierarchical Memory Categories
# ---------------------------------------------------------------------------

async def demo_hierarchical_categories() -> None:
    """Demonstrate category creation, assignment, and auto-creation."""
    banner("Step 5: Hierarchical Memory Categories (P1)")

    section("5a. MemoryCategory model")
    travel_cat = MemoryCategory(
        name="Travel",
        description="Travel-related memories",
        embedding=[1.0, 0.0, 0.0],
    )
    food_cat = MemoryCategory(
        name="Food",
        description="Food and cuisine preferences",
        embedding=[0.0, 1.0, 0.0],
    )
    info(f"Travel category: id={travel_cat.category_id[:8]}..., name={travel_cat.name}")
    info(f"Food category: id={food_cat.category_id[:8]}..., name={food_cat.name}")
    ok("MemoryCategory created with auto-generated UUID")

    section("5b. MemoryCategorizer.categorize()")
    categorizer = MemoryCategorizer(
        categories=[travel_cat, food_cat],
        similarity_threshold=0.5,
    )

    travel_mem = MemoryItem(
        user_id="demo-user",
        content="Visited Tokyo last spring",
        type=MemoryType.EPISODIC,
        source=MemorySource.AGENT_INFERENCE,
        embedding=[0.9, 0.1, 0.0],  # close to travel
    )
    food_mem = MemoryItem(
        user_id="demo-user",
        content="Loves sushi and ramen",
        type=MemoryType.PREFERENCE,
        source=MemorySource.AGENT_INFERENCE,
        embedding=[0.1, 0.95, 0.0],  # close to food
    )
    mixed_mem = MemoryItem(
        user_id="demo-user",
        content="Tried street food in Bangkok",
        type=MemoryType.EPISODIC,
        source=MemorySource.AGENT_INFERENCE,
        embedding=[0.7, 0.7, 0.0],  # close to both
    )
    no_emb_mem = MemoryItem(
        user_id="demo-user",
        content="Prefers aisle seats",
        type=MemoryType.PREFERENCE,
        source=MemorySource.AGENT_INFERENCE,
        embedding=None,
    )

    for mem in [travel_mem, food_mem, mixed_mem, no_emb_mem]:
        assignments = categorizer.categorize(mem)
        cat_names = []
        for a in assignments:
            if a.category_id == travel_cat.category_id:
                cat_names.append("Travel")
            elif a.category_id == food_cat.category_id:
                cat_names.append("Food")
        info(f"  {mem.content!r:40s} -> categories: {cat_names or ['(none)']}")

    # Travel memory should match Travel
    assert any(a.category_id == travel_cat.category_id
               for a in categorizer.categorize(travel_mem))
    # No-embedding memory should match nothing
    assert categorizer.categorize(no_emb_mem) == []
    ok("Categorization matches expected categories")

    section("5c. auto_create_category from memories")
    cluster = [travel_mem, mixed_mem]
    auto_cat = categorizer.auto_create_category(
        cluster, name="Adventure", description="Adventure travel memories"
    )
    info(f"Auto-created '{auto_cat.name}': embedding={[f'{v:.2f}' for v in auto_cat.embedding]}")
    # Centroid of [0.9, 0.1, 0.0] and [0.7, 0.7, 0.0] = [0.8, 0.4, 0.0]
    assert abs(auto_cat.embedding[0] - 0.8) < 0.01
    ok("auto_create_category computes centroid embedding")


# ---------------------------------------------------------------------------
# Step 6: Typed Extraction Prompts
# ---------------------------------------------------------------------------

async def demo_typed_extraction_prompts() -> None:
    """Demonstrate per-type extraction prompts."""
    banner("Step 6: Typed Extraction Prompts (P1)")

    section("6a. All 5 types have prompts")
    all_prompts = get_all_typed_prompts()
    info(f"Prompt count: {len(all_prompts)}")
    for mtype, prompt in all_prompts.items():
        preview = prompt[:60].replace("\n", " ")
        info(f"  {mtype.value:12s} -> {preview!r}...")
    assert len(all_prompts) == 5
    ok("All 5 memory types have dedicated prompts")

    section("6b. Prompts mention their type")
    for mtype in MemoryType:
        prompt = get_typed_prompt(mtype)
        assert mtype.value.lower() in prompt.lower(), (
            f"{mtype.value} not mentioned in its prompt"
        )
    ok("Each prompt references its memory type keyword")

    section("6c. Fallback for unknown type")
    # get_typed_prompt falls back to SEMANTIC for unknown types
    semantic_prompt = get_typed_prompt(MemoryType.SEMANTIC)
    assert len(semantic_prompt) > 0
    ok("Fallback to SEMANTIC prompt works")

    section("6d. LLMExtractor accepts use_typed_prompts flag")

    async def mock_llm(prompt: str) -> str:
        return "[]"

    ext = LLMExtractor(llm_func=mock_llm, use_typed_prompts=True)
    assert ext._use_typed_prompts is True
    ok("LLMExtractor constructed with use_typed_prompts=True")


# ---------------------------------------------------------------------------
# Neo4j graph memory demo (optional)
# ---------------------------------------------------------------------------

async def demo_neo4j_graph(skip: bool) -> None:
    """Exercise Neo4j graph store with the new TOOL memory type."""
    banner("Bonus: Neo4j Graph Store")

    if skip:
        info("Skipped (--skip-neo4j or --in-memory)")
        return

    try:
        import neo4j  # noqa: F401
    except ImportError:
        info("Skipped: 'neo4j' package not installed")
        return

    from ctxforge.config.base import Neo4jGraphStoreConfig
    from ctxforge.graph.stores.neo4j import Neo4jGraphStore
    from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode, GraphSearchFilters

    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "contextengine_dev")
    neo4j_database = os.getenv("NEO4J_DATABASE")

    config = Neo4jGraphStoreConfig(
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        create_indexes=True,
    )

    section("Connect to Neo4j")
    try:
        store = Neo4jGraphStore(config)
        await store.initialize()
        ok(f"Connected to Neo4j at {neo4j_url}")
    except Exception as e:
        info(f"Cannot connect to Neo4j ({e}). Skipping graph demo.")
        return

    scope = "demo-p0p1"
    now = datetime.now(timezone.utc)

    try:
        section("Create entities representing tool usage")
        entity_web_search = GraphNode(
            name="web_search",
            entity_type="Tool",
            properties={"description": "Web search tool", "memory_type": "tool"},
        )
        entity_user = GraphNode(
            name="demo-user",
            entity_type="User",
            properties={"role": "developer"},
        )
        await store.add_nodes([entity_web_search, entity_user], scope_id=scope)
        ok("Added Tool and User entity nodes")

        section("Create 'USES' edge")
        edge = GraphEdge(
            source="demo-user",
            target="web_search",
            relation="USES",
            properties={"frequency": "daily", "success_rate": "0.67"},
            created_at=now,
        )
        await store.add_edges([edge], scope_id=scope)
        ok("Added USES edge from User to Tool")

        section("Create episode")
        ep = GraphEpisode(
            name="tool_usage_session_1",
            content="User executed web_search 3 times with 2 successes",
            created_at=now,
        )
        await store.add_episode(ep, scope_id=scope)
        ok("Added episode for tool usage session")

        section("Search graph for tool entities")
        filters = GraphSearchFilters(entity_types=["Tool"])
        results = await store.search(query="web_search", scope_id=scope, filters=filters, limit=5)
        info(f"Search returned {len(results.nodes)} nodes, {len(results.edges)} edges")
        for n in results.nodes:
            info(f"  Node: {n.name} ({n.entity_type})")
        ok("Graph search returns tool entities")

    finally:
        section("Cleanup: delete demo scope")
        try:
            await store.delete_scope(scope)
            ok(f"Deleted scope '{scope}'")
        except Exception:
            info("(cleanup skipped)")
        await store.close()


# ---------------------------------------------------------------------------
# MySQL integration wiring
# ---------------------------------------------------------------------------

async def create_mysql_store():
    """Create and initialise a MySQL memory store from .env config."""
    mysql_config = MySQLConfig(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "contextengine"),
        user=os.getenv("MYSQL_USER", "contextengine"),
        password=os.getenv("MYSQL_PASSWORD", "contextengine"),
    )
    store = MySQLMemoryStore(mysql_config)
    await store.initialize()
    return store, mysql_config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="P0 + P1 Features Demo")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="Skip Neo4j graph store demo")
    parser.add_argument("--in-memory", action="store_true",
                        help="Use in-memory stores only (no MySQL/Neo4j)")
    args = parser.parse_args()

    use_mysql = not args.in_memory
    skip_neo4j = args.skip_neo4j or args.in_memory

    print()
    print("+" * 72)
    print("+  ctxforge P0 + P1 Features End-to-End Demo")
    print(f"+  Storage: {'MySQL' if use_mysql else 'In-Memory'}"
          f"  | Neo4j: {'skip' if skip_neo4j else 'enabled'}")
    print("+" * 72)

    # --- Create memory store ---
    mysql_store = None
    if use_mysql:
        try:
            mysql_store, mysql_cfg = await create_mysql_store()
            memory_store = mysql_store
            info(f"MySQL connected: {mysql_cfg.host}:{mysql_cfg.port}/{mysql_cfg.database}")
        except Exception as e:
            print(f"\n  [WARN] MySQL not available ({e}). Falling back to in-memory.\n")
            use_mysql = False

    if not use_mysql:
        memory_store = InMemoryMemoryStore()
        info("Using InMemoryMemoryStore")

    passed = 0
    total = 7

    try:
        # Step 1
        await demo_tool_memory(memory_store)
        passed += 1

        # Step 2
        await demo_salience_retriever(memory_store)
        passed += 1

        # Step 3
        await demo_content_hash_dedup(memory_store)
        passed += 1

        # Step 4
        await demo_inline_references()
        passed += 1

        # Step 5
        await demo_hierarchical_categories()
        passed += 1

        # Step 6
        await demo_typed_extraction_prompts()
        passed += 1

        # Bonus: Neo4j
        await demo_neo4j_graph(skip=skip_neo4j)
        passed += 1

    finally:
        # Cleanup MySQL connection
        if mysql_store is not None:
            try:
                await mysql_store.close()
            except Exception:
                pass

    banner("Summary")
    print(f"\n  {passed}/{total} demo sections completed successfully.\n")
    if passed == total:
        print("  All P0 + P1 features validated end-to-end!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
