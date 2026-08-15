"""Tests for content-hash deduplication."""


import pytest

from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.extraction.consolidation.deduplicator import DeduplicationConsolidator
from ctxforge.protocols.extractor import ExtractionCandidate
from ctxforge.utils.hashing import compute_content_hash


def test_compute_content_hash_deterministic():
    h1 = compute_content_hash("Hello world", "semantic")
    h2 = compute_content_hash("Hello world", "semantic")
    assert h1 == h2


def test_compute_content_hash_normalization():
    h1 = compute_content_hash("hello  world", "semantic")
    h2 = compute_content_hash("  Hello   World  ", "semantic")
    assert h1 == h2


def test_compute_content_hash_type_prefix():
    h1 = compute_content_hash("hello world", "semantic")
    h2 = compute_content_hash("hello world", "episodic")
    assert h1 != h2


def test_compute_content_hash_length():
    h = compute_content_hash("test content", "procedural")
    assert len(h) == 16


def _make_memory(content, memory_type=MemoryType.SEMANTIC, content_hash=None):
    mem = MemoryItem(
        user_id="u1",
        content=content,
        type=memory_type,
        source=MemorySource.AGENT_INFERENCE,
    )
    if content_hash:
        mem.metadata["content_hash"] = content_hash
    return mem


@pytest.mark.asyncio
async def test_deduplicator_skips_hash_match():
    h = compute_content_hash("some fact", "semantic")
    existing = _make_memory("some fact", content_hash=h)
    new_item = _make_memory("some fact", content_hash=h)
    old_count = existing.access_count

    consolidator = DeduplicationConsolidator()
    result = await consolidator.consolidate([new_item], [existing])

    # Hash match: new item should be skipped
    assert len(result) == 0
    # Existing item's access_count should be incremented
    assert existing.access_count == old_count + 1


@pytest.mark.asyncio
async def test_deduplicator_passes_hash_miss():
    h1 = compute_content_hash("fact A", "semantic")
    h2 = compute_content_hash("fact B", "semantic")
    existing = _make_memory("fact A", content_hash=h1)
    new_item = _make_memory("fact B", content_hash=h2)

    consolidator = DeduplicationConsolidator()
    result = await consolidator.consolidate([new_item], [existing])

    assert len(result) == 1
    assert result[0].content == "fact B"


def test_extraction_candidate_includes_hash():
    candidate = ExtractionCandidate(
        content="User likes coffee",
        memory_type=MemoryType.SEMANTIC,
        confidence=0.9,
        source_text="I like coffee",
    )
    memory = candidate.to_memory_item(user_id="u1")
    assert "content_hash" in memory.metadata
    assert len(memory.metadata["content_hash"]) == 16
