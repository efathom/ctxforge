"""Tests for the multi-stage memory integration pipeline."""

import pytest

from ctxforge.core.memory import MemoryFactory, MemoryType
from ctxforge.extraction.integration_config import IntegrationConfig, IntegrationResult
from ctxforge.extraction.integration_pipeline import (
    IntegrationContext,
    MemoryIntegrationPipeline,
)
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.extractor import ExtractionCandidate
from ctxforge.protocols.update_planner import MemoryOperationType
from ctxforge.storage.memory.memory import InMemoryMemoryStore


def _candidate(content: str, confidence: float = 0.8) -> ExtractionCandidate:
    """Helper to create a simple ExtractionCandidate."""
    return ExtractionCandidate(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        confidence=confidence,
        source_text=content,
    )


@pytest.mark.asyncio
async def test_pipeline_empty_candidates():
    llm = MockLLMProvider(latency_ms=0)
    store = InMemoryMemoryStore()
    pipeline = MemoryIntegrationPipeline(llm=llm, memory_store=store)
    results = await pipeline.process(candidates=[], user_id="u1", query="hello")
    assert results == []


@pytest.mark.asyncio
async def test_detect_stage_filters_non_actionable():
    """Non-actionable candidates are filtered out after detect stage."""
    llm = MockLLMProvider(latency_ms=0)
    # First call: detect returns 'No', second call: detect returns 'Yes',
    # then summarize returns summarized content
    llm.set_responses(["No", "Yes", "User prefers dark mode"])
    store = InMemoryMemoryStore()
    config = IntegrationConfig(skip_detect_for_high_confidence=False)
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=store, config=config,
    )

    candidates = [
        _candidate("ok thanks"),
        _candidate("I prefer dark mode"),
    ]
    results = await pipeline.process(
        candidates=candidates, user_id="u1", query="preferences",
    )

    # Only the actionable candidate should produce a result
    assert len(results) == 1
    assert results[0].memory_item is not None


@pytest.mark.asyncio
async def test_detect_stage_skips_high_confidence():
    """High-confidence candidates skip the detect stage."""
    llm = MockLLMProvider(latency_ms=0)
    # Only summarize call (no detect call for high-confidence)
    llm.set_responses(["User prefers dark mode"])
    store = InMemoryMemoryStore()
    config = IntegrationConfig(skip_detect_for_high_confidence=True)
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=store, config=config,
    )

    candidates = [_candidate("I prefer dark mode", confidence=0.95)]
    results = await pipeline.process(
        candidates=candidates, user_id="u1", query="preferences",
    )

    assert len(results) == 1
    assert results[0].was_actionable is True


@pytest.mark.asyncio
async def test_summarize_stage_produces_clean_content():
    """Summarize stage produces a clean preference statement."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["The user prefers to work in the evening."])
    store = InMemoryMemoryStore()
    config = IntegrationConfig(skip_detect_for_high_confidence=True)
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=store, config=config,
    )

    candidates = [_candidate("I like working evenings", confidence=0.95)]
    results = await pipeline.process(
        candidates=candidates, user_id="u1", query="work schedule",
    )

    assert len(results) == 1
    mem = results[0].memory_item
    assert mem is not None
    assert "evening" in mem.content.lower()


@pytest.mark.asyncio
async def test_dedup_detects_similar_existing_memory():
    """Dedup stage finds similar existing memories and marks for UPDATE."""
    llm = MockLLMProvider(latency_ms=0)
    # summarize, then integrate (merge)
    llm.set_responses([
        "User likes dark mode",
        "User likes dark mode with blue accent",
    ])
    store = InMemoryMemoryStore()

    existing = MemoryFactory.semantic_memory(
        user_id="u1", content="User likes dark mode",
    )
    await store.add(existing)

    config = IntegrationConfig(
        skip_detect_for_high_confidence=True,
        similarity_threshold=0.5,  # Low threshold so Jaccard match works
    )
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=store, config=config,
    )

    candidates = [_candidate("User likes dark mode with blue accent", confidence=0.95)]
    results = await pipeline.process(
        candidates=candidates, user_id="u1", query="preferences",
    )

    assert len(results) == 1
    assert results[0].operation == "update"
    assert results[0].similarity_score > 0


@pytest.mark.asyncio
async def test_dedup_adds_new_when_no_match():
    """Dedup stage marks for ADD when no similar memory exists."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["User enjoys hiking in the mountains"])
    store = InMemoryMemoryStore()

    config = IntegrationConfig(
        skip_detect_for_high_confidence=True,
        similarity_threshold=0.8,
    )
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=store, config=config,
    )

    candidates = [_candidate("I enjoy hiking in the mountains", confidence=0.95)]
    results = await pipeline.process(
        candidates=candidates, user_id="u1", query="hobbies",
    )

    assert len(results) == 1
    assert results[0].operation == "add"


@pytest.mark.asyncio
async def test_full_pipeline_add_path():
    """Full pipeline: detect -> summarize -> dedup(no match) -> add."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["Yes", "User is vegetarian"])
    store = InMemoryMemoryStore()

    config = IntegrationConfig(skip_detect_for_high_confidence=False)
    pipeline = MemoryIntegrationPipeline(
        llm=llm, memory_store=store, config=config,
    )

    candidates = [_candidate("I am vegetarian")]
    results = await pipeline.process(
        candidates=candidates, user_id="u1", query="food preferences",
    )

    assert len(results) == 1
    assert results[0].memory_item is not None
    assert results[0].operation == "add"

    # Memory should be in the store
    stored = await store.get_by_user("u1")
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_text_similarity_helper():
    """Test the Jaccard similarity helper."""
    assert MemoryIntegrationPipeline._text_similarity("", "") == 0.0
    assert MemoryIntegrationPipeline._text_similarity("hello world", "hello world") == 1.0
    score = MemoryIntegrationPipeline._text_similarity(
        "user likes dark mode",
        "user prefers dark mode",
    )
    assert 0.0 < score < 1.0


@pytest.mark.asyncio
async def test_integration_context_dataclass():
    """IntegrationContext can be created and has sensible defaults."""
    c = _candidate("test")
    ctx = IntegrationContext(candidate=c, user_id="u1", query="q")
    assert ctx.is_actionable is True
    assert ctx.operation == MemoryOperationType.ADD
    assert ctx.summarized_content is None
    assert ctx.similar_memory is None


@pytest.mark.asyncio
async def test_integration_result_dataclass():
    """IntegrationResult can be created with defaults."""
    r = IntegrationResult()
    assert r.memory_item is None
    assert r.operation == "add"
    assert r.was_actionable is True
    assert r.preference_changed is False
