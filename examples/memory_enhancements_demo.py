"""
Memory Enhancements Demo
=========================

Demonstrates all 9 phases of the memory enhancement plan:

  Phase 1 – Lossless Restatement (coreference resolution + temporal anchoring)
  Phase 2 – Multi-View Indexing (keyword search + hybrid search + RRF)
  Phase 3 – Query Decomposition (intent-aware sub-queries)
  Phase 4 – Retrieval Reflection (coverage-based stopping + gap queries)
  Phase 5 – Token Budget Packing (greedy priority packing)
  Phase 6 – Memory Consolidation (decay + merge + prune)
  Phase 7 – Cross-Session Structured Observations
  Phase 8 – Sliding Window Batch Extraction
  Phase 9 – Parallel Processing (concurrent retrieval + extraction)

Run:
    cd ctxforge
    source venv/bin/activate
    python examples/memory_enhancements_demo.py
"""

import asyncio
import datetime
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

# Ensure ctxforge is importable when running from the examples/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxforge.config.base import ConsolidationConfig
from ctxforge.core.context import Context
from ctxforge.core.events import Event, EventType
from ctxforge.core.memory import MemoryFactory
from ctxforge.core.observation import ObservationType
from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.engine.services.consolidation_service import ConsolidationService
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.extraction.chunking import sliding_window
from ctxforge.extraction.llm_extractor import LLMExtractor
from ctxforge.extraction.observation_extractor import ObservationExtractor
from ctxforge.protocols.llm import (
    ChatMessage,
    EmbeddingResponse,
    LLMResponse,
)
from ctxforge.retrieval.ranking import reciprocal_rank_fusion
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore
from ctxforge.utils.budget_packer import budget_pack

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str) -> None:
    print(f"\n--- {title} ---\n")


class DemoLLMProvider:
    """Minimal LLM provider that returns pre-configured responses in order."""

    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = list(responses or [])
        self._idx = 0

    @property
    def name(self) -> str:
        return "demo"

    @property
    def default_model(self) -> str:
        return "demo-model"

    def set_responses(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            content = "[]"
        return LLMResponse(
            content=content,
            model=model or self.default_model,
            input_tokens=10,
            output_tokens=20,
            latency_ms=1.0,
        )


class DemoEmbeddingProvider:
    """Deterministic embedding provider for demos."""

    @property
    def name(self) -> str:
        return "demo-embedder"

    @property
    def default_model(self) -> str:
        return "demo-embed"

    @property
    def embedding_dimension(self) -> int:
        return 3

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        embeddings = []
        for t in texts:
            h = hash(t) % 10000
            embeddings.append([float(h), float(h % 100), float(h % 10)])
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model or "demo-embed",
            total_tokens=0,
            latency_ms=0.0,
        )

    async def embed_single(
        self, text: str, model: Optional[str] = None, **kwargs: Any
    ) -> List[float]:
        resp = await self.embed([text], model=model)
        return resp.embeddings[0]


# ===================================================================
# Phase 1: Lossless Restatement
# ===================================================================

async def demo_phase1_restatement() -> None:
    banner("Phase 1: Lossless Restatement")

    print("MemoryItem now supports a 'restatement' field that stores a")
    print("self-contained version of the fact with resolved pronouns and dates.\n")

    mem = MemoryFactory.semantic_memory(
        user_id="user_1",
        content="He said he'd go there tomorrow",
    )
    mem.restatement = "Bob said he will go to Seattle on 2026-02-16"
    mem.extracted_entities = {
        "persons": ["Bob"],
        "locations": ["Seattle"],
        "timestamps": ["2026-02-16"],
    }

    print(f"  Original content : {mem.content}")
    print(f"  Restatement      : {mem.restatement}")
    print(f"  Entities         : {mem.extracted_entities}")
    print(f"  display_content  : {mem.display_content}")
    print()

    prompt_text = mem.to_prompt_format()
    print("  to_prompt_format() output (prefers restatement):")
    for line in prompt_text.strip().split("\n"):
        print(f"    {line}")

    section("LLM Extraction with Restatement")

    llm = DemoLLMProvider([
        json.dumps([{
            "content": "User is moving to Portland next month",
            "restatement": "User is relocating to Portland, OR in March 2026",
            "type": "EPISODIC",
            "confidence": 0.9,
            "tags": ["relocation"],
            "keywords": ["portland", "moving", "relocation"],
            "topics": ["housing"],
            "entities": {
                "persons": [],
                "locations": ["Portland, OR"],
                "timestamps": ["March 2026"],
            },
        }])
    ])
    extractor = LLMExtractor(llm_provider=llm)
    result = await extractor.extract(
        user_input="I'm moving to Portland next month",
        agent_response="That's exciting!",
    )
    candidates = result.candidates

    for c in candidates:
        print(f"  Extracted content    : {c.content}")
        print(f"  Extracted restatement: {c.restatement}")
        print(f"  Extracted entities   : {c.extracted_entities}")

        mi = c.to_memory_item("user_1")
        print(f"  MemoryItem.restatement: {mi.restatement}")
        print(f"  MemoryItem.keywords   : {mi.keywords}")
        print(f"  MemoryItem.locations  : {mi.locations}")


# ===================================================================
# Phase 2: Multi-View Indexing
# ===================================================================

async def demo_phase2_multiview() -> None:
    banner("Phase 2: Multi-View Indexing")

    store = InMemoryMemoryStore()

    memories = [
        MemoryFactory.semantic_memory("u1", "User loves espresso coffee"),
        MemoryFactory.semantic_memory("u1", "User visited Paris last summer"),
        MemoryFactory.semantic_memory("u1", "User works at Acme Corp as a data scientist"),
        MemoryFactory.semantic_memory("u1", "User prefers dark mode in all editors"),
    ]
    memories[0].keywords = ["espresso", "coffee", "beverage"]
    memories[0].topics = ["food", "drinks"]
    memories[1].keywords = ["paris", "travel", "summer"]
    memories[1].locations = ["Paris"]
    memories[1].topics = ["travel"]
    memories[2].keywords = ["acme", "data scientist", "career"]
    memories[2].persons = ["User"]
    memories[2].topics = ["career"]
    memories[3].keywords = ["dark mode", "editor", "preferences"]
    memories[3].topics = ["preferences"]

    for m in memories:
        await store.add(m)

    section("Keyword Search")
    results = await store.keyword_search("u1", keywords=["coffee", "espresso"])
    print("  Query keywords: ['coffee', 'espresso']")
    print(f"  Found {len(results)} result(s):")
    for r in results:
        print(f"    - {r.content}  (keywords: {r.keywords})")

    section("Keyword Search with Filters")
    results = await store.keyword_search(
        "u1",
        keywords=["paris", "travel", "career"],
        filters={"topics": ["travel"]},
    )
    print("  Query keywords: ['paris', 'travel', 'career'] + filter topics=['travel']")
    print(f"  Found {len(results)} result(s):")
    for r in results:
        print(f"    - {r.content}")

    section("Reciprocal Rank Fusion (RRF)")
    list_a = [memories[0], memories[1], memories[2]]
    list_b = [memories[2], memories[0], memories[3]]
    fused = reciprocal_rank_fusion(
        ranked_lists=[list_a, list_b],
        key_fn=lambda m: m.memory_id,
        limit=3,
    )
    print("  List A: [espresso, paris, acme]")
    print("  List B: [acme, espresso, dark mode]")
    print("  RRF result (top 3):")
    for i, m in enumerate(fused, 1):
        print(f"    {i}. {m.content}")


# ===================================================================
# Phase 3: Query Decomposition (conceptual demo)
# ===================================================================

async def demo_phase3_decomposition() -> None:
    banner("Phase 3: Query Decomposition")

    print("The retrieval controller decomposes complex questions into sub-queries.")
    print("This happens automatically inside RetrievalControllerService._plan_queries().\n")

    llm = DemoLLMProvider([
        json.dumps({
            "sub_queries": [
                "What programming languages does the user know?",
                "Where does the user live?",
                "What is the user's job title?",
            ]
        })
    ])

    messages = [
        ChatMessage(
            role="system",
            content="You are a query planner. Decompose the question into sub-queries.",
        ),
        ChatMessage(
            role="user",
            content="Tell me about the user's skills, location, and career",
        ),
    ]
    resp = await llm.chat(messages)
    data = json.loads(resp.content)

    print("  Original question: 'Tell me about the user's skills, location, and career'")
    print("  Decomposed sub-queries:")
    for i, sq in enumerate(data["sub_queries"], 1):
        print(f"    {i}. {sq}")
    print("\n  Each sub-query retrieves independently, results are merged and deduped.")


# ===================================================================
# Phase 4: Retrieval Reflection (conceptual demo)
# ===================================================================

async def demo_phase4_reflection() -> None:
    banner("Phase 4: Retrieval Reflection")

    print("The router now evaluates coverage and generates gap-filling queries.\n")

    router_response = {
        "decision": "reflect",
        "evidence": ["User likes Python", "User lives in Seattle"],
        "gaps": "Missing information about user's work experience",
        "coverage_percentage": 0.45,
        "gap_queries": [
            "What is the user's job history?",
            "What companies has the user worked at?",
        ],
    }

    print("  After first retrieval round, the router returns:")
    print(f"    decision           : {router_response['decision']}")
    print(f"    coverage_percentage: {router_response['coverage_percentage']}")
    print(f"    gaps               : {router_response['gaps']}")
    print(f"    gap_queries        : {router_response['gap_queries']}")
    print()
    print("  The controller uses gap_queries[0] as the next retrieval query,")
    print("  promoting 'reflect' -> 'retrieve' automatically.")
    print()

    router_response_2 = {
        "decision": "answer",
        "evidence": [
            "User likes Python",
            "User lives in Seattle",
            "User worked at Acme Corp for 3 years",
        ],
        "gaps": "None",
        "coverage_percentage": 0.92,
        "draft_answer": "The user is a Python developer in Seattle who worked at Acme.",
    }
    print("  After second retrieval round:")
    print(f"    coverage_percentage: {router_response_2['coverage_percentage']}")
    print(f"    gaps               : {router_response_2['gaps']}")
    print(f"    decision           : {router_response_2['decision']}")
    print("  -> Coverage >= 0.7 and gaps is None => early stop.")


# ===================================================================
# Phase 5: Token Budget Packing
# ===================================================================

async def demo_phase5_budget_packing() -> None:
    banner("Phase 5: Token Budget Packing")

    section("budget_pack() utility")
    items = [
        ("system_prompt", "You are a helpful assistant."),
        ("memories", "User likes Python. User lives in Seattle. User works at Acme."),
        ("skills", "Available skills: code review, debugging, testing."),
        ("history", "User: Hi\nAssistant: Hello!\nUser: How are you?\nAssistant: Great!"),
        ("graph", "Entity: Python -> related_to -> Data Science"),
    ]

    packed = budget_pack(
        items=items,
        budget=20,
        text_fn=lambda x: x[1],
    )
    print("  Budget: 20 tokens (word-split heuristic)")
    print(f"  Items ({len(items)} total):")
    for name, text in items:
        word_count = len(text.split())
        print(f"    [{name}] {word_count} tokens: {text[:60]}...")
    print(f"\n  Packed ({len(packed)} items fit):")
    for name, text in packed:
        print(f"    [{name}] {text[:60]}...")

    section("Context.priority_pack_sections()")
    ctx = Context(
        system_prompt="You are helpful.",
        user_query="What do I like?",
        session_id="demo-session",
        user_id="demo-user",
    )
    ctx.add_section(name="memories", content="User likes Python.", priority=80, is_required=True)
    ctx.add_section(name="skills", content="Skills: code review, debugging, testing, deployment.", priority=60)
    ctx.add_section(name="graph", content="Python -> Data Science -> Machine Learning -> Deep Learning.", priority=50)
    ctx.add_section(name="history", content="Long conversation history " * 10, priority=40)

    packed_sections = ctx.priority_pack_sections(budget=25)
    print("  Budget: 25 tokens")
    print(f"  All sections: {[s.name for s in ctx.sections]}")
    print(f"  Packed sections: {[s.name for s in packed_sections]}")
    print("  (Required sections always included, then optional by priority)")


# ===================================================================
# Phase 6: Memory Consolidation
# ===================================================================

async def demo_phase6_consolidation() -> None:
    banner("Phase 6: Memory Consolidation")

    store = InMemoryMemoryStore()
    embedder = DemoEmbeddingProvider()
    config = ConsolidationConfig(
        enabled=True,
        decay_factor=0.5,
        max_age_days=7,
        merge_similarity_threshold=0.95,
        min_importance=0.2,
    )

    section("Setup: Create memories with varying ages and importance")

    old_mem = MemoryFactory.semantic_memory("u1", "User liked Java in 2020")
    old_mem.created_at = datetime.datetime.now() - datetime.timedelta(days=30)
    old_mem.importance = 0.6
    old_mem.embedding = [1.0, 0.0, 0.0]
    await store.add(old_mem)
    print(f"  Old memory (30 days): '{old_mem.content}' importance={old_mem.importance}")

    new_mem = MemoryFactory.semantic_memory("u1", "User now prefers Python")
    new_mem.importance = 1.0
    new_mem.embedding = [0.0, 1.0, 0.0]
    await store.add(new_mem)
    print(f"  New memory (today)  : '{new_mem.content}' importance={new_mem.importance}")

    low_mem = MemoryFactory.semantic_memory("u1", "User mentioned weather once")
    low_mem.importance = 0.05
    low_mem.embedding = [0.0, 0.0, 1.0]
    await store.add(low_mem)
    print(f"  Low-importance mem  : '{low_mem.content}' importance={low_mem.importance}")

    section("Run Consolidation")
    svc = ConsolidationService(
        memory_store=store,
        embedding_provider=embedder,
        config=config,
    )
    report = await svc.consolidate("u1")

    print("  Consolidation Report:")
    print(f"    Decayed : {report.decayed} (old memories had importance reduced)")
    print(f"    Merged  : {report.merged} (near-duplicates superseded)")
    print(f"    Pruned  : {report.pruned} (low-importance soft-deleted)")

    section("Post-Consolidation State")
    remaining = await store.get_by_user("u1", limit=100, include_inactive=False)
    for m in remaining:
        print(f"  [{m.memory_id[:8]}] '{m.content}' importance={m.importance:.2f} active={m.is_active}")


# ===================================================================
# Phase 7: Cross-Session Structured Observations
# ===================================================================

async def demo_phase7_observations() -> None:
    banner("Phase 7: Cross-Session Structured Observations")

    section("ObservationExtractor")
    llm = DemoLLMProvider([
        json.dumps([
            {
                "type": "decision",
                "summary": "Chose PostgreSQL over MySQL for the new service",
                "detail": "PostgreSQL was selected for its JSON support and extensions",
                "confidence": 0.95,
            },
            {
                "type": "bugfix",
                "summary": "Fixed race condition in auth token refresh",
                "confidence": 0.9,
            },
            {
                "type": "discovery",
                "summary": "Found that the API rate limit is 100 req/min, not 1000",
                "confidence": 0.85,
            },
        ])
    ])

    extractor = ObservationExtractor(llm)
    events = [
        Event(type=EventType.USER, content="Let's use PostgreSQL for the new service"),
        Event(type=EventType.AGENT, content="Good choice, PostgreSQL has great JSON support"),
        Event(type=EventType.USER, content="I found a race condition in the auth flow"),
        Event(type=EventType.AGENT, content="Let me help fix that"),
        Event(type=EventType.USER, content="The API rate limit is actually 100/min"),
    ]

    observations = await extractor.extract(events)
    print(f"  Extracted {len(observations)} observations from {len(events)} events:")
    for obs in observations:
        print(f"    [{obs.type.value:>10}] {obs.summary} (confidence={obs.confidence})")
        if obs.detail:
            print(f"               Detail: {obs.detail}")

    section("Session Summary Report")
    llm.set_responses([
        json.dumps({
            "request": "Set up database and fix auth bugs",
            "investigated": "Database options and auth token refresh flow",
            "learned": "PostgreSQL has better JSON support; API rate limit is 100/min",
            "completed": "Selected PostgreSQL; fixed auth race condition",
            "next_steps": "Implement database migration scripts",
        })
    ])
    report = await extractor.summarize_session(events)
    print(f"  Request     : {report.request}")
    print(f"  Investigated: {report.investigated}")
    print(f"  Learned     : {report.learned}")
    print(f"  Completed   : {report.completed}")
    print(f"  Next Steps  : {report.next_steps}")

    section("Saving Observations as Scoped Memories")
    scoped_store = InMemoryScopedMemoryStore()
    scoped_svc = ScopedMemoryService(scoped_store)

    scoped_memories = []
    category_map = {
        ObservationType.DECISION: MemoryCategory.DECISION,
        ObservationType.BUGFIX: MemoryCategory.BUGFIX,
        ObservationType.DISCOVERY: MemoryCategory.DISCOVERY,
    }
    for obs in observations:
        cat = category_map.get(obs.type, MemoryCategory.CONTEXT)
        obs_id = str(uuid.uuid4())
        scoped_memories.append(ScopedMemory(
            id=obs_id,
            scope=MemoryScope.PROJECT,
            scope_id="my-project",
            category=cat,
            key=f"obs_{obs.type.value}_{obs_id[:8]}",
            content=obs.summary,
            metadata={"confidence": obs.confidence},
        ))

    saved = await scoped_svc.save_observations(scoped_memories)
    print(f"  Saved {saved} observations as project-scoped memories")

    stored = await scoped_store.query(ScopedMemoryQuery(
        project_id="my-project",
        include_global=False,
        include_session=False,
    ))
    for sm in stored:
        display = MemoryCategory.get_display_name(sm.category)
        print(f"    [{display:>12}] {sm.content}")


# ===================================================================
# Phase 8: Sliding Window Batch Extraction
# ===================================================================

async def demo_phase8_batch_extraction() -> None:
    banner("Phase 8: Sliding Window Batch Extraction")

    section("sliding_window() utility")
    items = ["turn1", "turn2", "turn3", "turn4", "turn5", "turn6"]
    windows = sliding_window(items, window_size=3, stride=2)
    print(f"  Items: {items}")
    print("  window_size=3, stride=2")
    print(f"  Windows: {windows}")

    section("LLMExtractor.extract_batch()")
    responses = [
        json.dumps([
            {"content": "User likes hiking", "type": "SEMANTIC", "confidence": 0.9},
            {"content": "User has a dog named Max", "type": "SEMANTIC", "confidence": 0.85},
        ]),
        json.dumps([
            {"content": "User is training for a marathon", "type": "EPISODIC", "confidence": 0.8},
        ]),
    ]
    llm = DemoLLMProvider(responses)
    extractor = LLMExtractor(llm_provider=llm)

    turns = [
        ("I love hiking on weekends", "That sounds fun!"),
        ("My dog Max always comes with me", "Dogs are great hiking companions"),
        ("I'm training for a marathon next month", "Good luck with your training!"),
        ("Running 30 miles a week now", "That's impressive dedication"),
    ]

    results = await extractor.extract_batch(turns, window_size=2)
    print(f"  {len(turns)} conversation turns -> {len(results)} extracted facts")
    print(f"  (Processed in {len(sliding_window(turns, 2, 2))} windows of size 2)")
    for r in results:
        print(f"    - [{r.memory_type.value:>10}] {r.content} (conf={r.confidence})")


# ===================================================================
# Phase 9: Parallel Processing
# ===================================================================

async def demo_phase9_parallel() -> None:
    banner("Phase 9: Parallel Processing")

    section("Parallel Batch Extraction (max_concurrency)")

    call_count = {"n": 0}

    async def _slow_chat(messages, **kwargs):
        call_count["n"] += 1
        await asyncio.sleep(0.05)
        return LLMResponse(
            content=json.dumps([{
                "content": f"Fact from window {call_count['n']}",
                "type": "SEMANTIC",
                "confidence": 0.8,
            }]),
            model="demo",
            input_tokens=10,
            output_tokens=20,
            latency_ms=50.0,
        )

    llm = DemoLLMProvider()
    llm.chat = _slow_chat

    extractor = LLMExtractor(llm_provider=llm)
    turns = [(f"Turn {i}", f"Response {i}") for i in range(8)]

    # Sequential
    call_count["n"] = 0
    t0 = time.monotonic()
    seq_results = await extractor.extract_batch(turns, window_size=1, max_concurrency=1)
    seq_time = time.monotonic() - t0

    # Parallel
    call_count["n"] = 0
    t0 = time.monotonic()
    par_results = await extractor.extract_batch(turns, window_size=1, max_concurrency=4)
    par_time = time.monotonic() - t0

    print("  8 turns, window_size=1 (8 LLM calls, each ~50ms)")
    print(f"  Sequential (max_concurrency=1): {seq_time:.3f}s  ({len(seq_results)} facts)")
    print(f"  Parallel   (max_concurrency=4): {par_time:.3f}s  ({len(par_results)} facts)")
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print(f"  Speedup: {speedup:.1f}x")

    section("Parallel Sub-Query Retrieval")
    print("  In the retrieval controller, when query planning produces")
    print("  multiple sub-queries, they are executed in parallel using")
    print("  asyncio.gather with a semaphore (max_parallel_queries=4).")
    print()
    print("  Before (sequential):")
    print("    for sq in sub_queries:")
    print("        results += await memory.search(query=sq)")
    print()
    print("  After (parallel):")
    print("    sem = asyncio.Semaphore(max_parallel_queries)")
    print("    results = await asyncio.gather(*[search(sq) for sq in sub_queries])")


# ===================================================================
# Main
# ===================================================================

async def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           CtxForge Memory Enhancements Demo (9 Phases)             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    await demo_phase1_restatement()
    await demo_phase2_multiview()
    await demo_phase3_decomposition()
    await demo_phase4_reflection()
    await demo_phase5_budget_packing()
    await demo_phase6_consolidation()
    await demo_phase7_observations()
    await demo_phase8_batch_extraction()
    await demo_phase9_parallel()

    banner("All 9 Phases Demonstrated Successfully")
    print()


if __name__ == "__main__":
    asyncio.run(main())
