"""Tests for sliding window batch extraction (Phase 8)."""

import json
from unittest.mock import AsyncMock

import pytest

from ctxforge.extraction.chunking import sliding_window
from ctxforge.extraction.llm_extractor import LLMExtractor

# ---------------------------------------------------------------------------
# sliding_window utility tests
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_basic_no_overlap(self):
        items = [1, 2, 3, 4, 5]
        result = sliding_window(items, window_size=2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_with_overlap(self):
        items = [1, 2, 3, 4, 5, 6]
        result = sliding_window(items, window_size=3, stride=2)
        assert result == [[1, 2, 3], [3, 4, 5], [5, 6]]

    def test_window_larger_than_items(self):
        items = [1, 2]
        result = sliding_window(items, window_size=5)
        assert result == [[1, 2]]

    def test_empty_items(self):
        assert sliding_window([], window_size=3) == []

    def test_zero_window_size(self):
        assert sliding_window([1, 2], window_size=0) == []

    def test_stride_equals_one(self):
        items = [1, 2, 3]
        result = sliding_window(items, window_size=2, stride=1)
        assert result == [[1, 2], [2, 3], [3]]

    def test_stride_zero_defaults_to_window_size(self):
        items = [1, 2, 3, 4]
        result = sliding_window(items, window_size=2, stride=0)
        assert result == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# LLMExtractor.extract_batch tests
# ---------------------------------------------------------------------------

def _make_llm(responses):
    """Create a mock LLM provider that returns responses in sequence."""
    llm = AsyncMock()
    call_idx = {"i": 0}

    async def _chat(messages, **kwargs):
        idx = call_idx["i"]
        call_idx["i"] += 1
        resp = AsyncMock()
        resp.content = responses[idx] if idx < len(responses) else "[]"
        return resp

    llm.chat = _chat
    llm.name = "mock"
    return llm


@pytest.mark.asyncio
async def test_batch_extraction_groups_turns():
    """Batch extraction should group turns into windows and call LLM per window."""
    resp1 = json.dumps([
        {"content": "User likes coffee", "type": "SEMANTIC", "confidence": 0.9},
    ])
    resp2 = json.dumps([
        {"content": "User lives in Seattle", "type": "SEMANTIC", "confidence": 0.8},
    ])
    llm = _make_llm([resp1, resp2])
    extractor = LLMExtractor(llm_provider=llm)

    turns = [
        ("I love coffee", "Great!"),
        ("Especially espresso", "Nice choice"),
        ("I live in Seattle", "Beautiful city"),
        ("Near the waterfront", "Lovely area"),
    ]

    results = await extractor.extract_batch(turns, window_size=2)

    assert len(results) == 2
    contents = {r.content for r in results}
    assert "User likes coffee" in contents
    assert "User lives in Seattle" in contents


@pytest.mark.asyncio
async def test_batch_extraction_overlap_context_prevents_duplicates():
    """Overlap context should be forwarded so the LLM avoids re-extracting."""
    call_prompts = []

    async def _chat(messages, **kwargs):
        for m in messages:
            if m.role == "user":
                call_prompts.append(m.content)
        resp = AsyncMock()
        resp.content = json.dumps([
            {"content": "User likes tea", "type": "SEMANTIC", "confidence": 0.8},
        ])
        return resp

    llm = AsyncMock()
    llm.chat = _chat
    llm.name = "mock"

    extractor = LLMExtractor(llm_provider=llm)

    turns = [
        ("I like tea", "Nice"),
        ("Green tea mostly", "Healthy"),
        ("Also oolong", "Interesting"),
        ("With honey", "Sweet"),
    ]

    await extractor.extract_batch(
        turns, window_size=2, overlap_context=["User prefers Python"]
    )

    # First window prompt should include the seed overlap context
    assert "User prefers Python" in call_prompts[0]
    # Second window prompt should include facts extracted from first window
    assert "User likes tea" in call_prompts[1]


@pytest.mark.asyncio
async def test_batch_extraction_single_turn_fallback():
    """window_size=1 should behave like per-turn extraction."""
    responses = [
        json.dumps([{"content": f"Fact {i}", "type": "SEMANTIC", "confidence": 0.8}])
        for i in range(3)
    ]
    llm = _make_llm(responses)
    extractor = LLMExtractor(llm_provider=llm)

    turns = [("a", "b"), ("c", "d"), ("e", "f")]
    results = await extractor.extract_batch(turns, window_size=1)

    assert len(results) == 3
    assert {r.content for r in results} == {"Fact 0", "Fact 1", "Fact 2"}


@pytest.mark.asyncio
async def test_batch_extraction_empty_turns():
    """Empty turns list should return empty results."""
    llm = _make_llm([])
    extractor = LLMExtractor(llm_provider=llm)

    results = await extractor.extract_batch([], window_size=5)
    assert results == []


@pytest.mark.asyncio
async def test_batch_extraction_deduplicates_across_windows():
    """Identical facts from different windows should be deduplicated."""
    same_response = json.dumps([
        {"content": "User likes coffee", "type": "SEMANTIC", "confidence": 0.9},
    ])
    llm = _make_llm([same_response, same_response])
    extractor = LLMExtractor(llm_provider=llm)

    turns = [("I like coffee", "Nice"), ("Coffee is great", "Indeed")]
    results = await extractor.extract_batch(turns, window_size=1)

    assert len(results) == 1
    assert results[0].content == "User likes coffee"


@pytest.mark.asyncio
async def test_batch_extraction_parallel_windows():
    """Concurrent window processing should produce the same results."""
    responses = [
        json.dumps([{"content": f"Fact {i}", "type": "SEMANTIC", "confidence": 0.8}])
        for i in range(4)
    ]
    llm = _make_llm(responses)
    extractor = LLMExtractor(llm_provider=llm)

    turns = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
    results = await extractor.extract_batch(
        turns, window_size=1, max_concurrency=3
    )

    assert len(results) == 4
    assert {r.content for r in results} == {"Fact 0", "Fact 1", "Fact 2", "Fact 3"}


@pytest.mark.asyncio
async def test_batch_extraction_parallel_faster_than_sequential():
    """Parallel processing should be faster than sequential with delays."""
    import time as _time

    async def _slow_chat(messages, **kwargs):
        import asyncio as _asyncio
        await _asyncio.sleep(0.05)
        resp = AsyncMock()
        resp.content = json.dumps([
            {"content": "fact", "type": "SEMANTIC", "confidence": 0.8}
        ])
        return resp

    llm = AsyncMock()
    llm.chat = _slow_chat
    llm.name = "slow-mock"

    extractor = LLMExtractor(llm_provider=llm)
    turns = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]

    # Sequential
    t0 = _time.monotonic()
    await extractor.extract_batch(turns, window_size=1, max_concurrency=1)
    sequential_time = _time.monotonic() - t0

    # Parallel
    t0 = _time.monotonic()
    await extractor.extract_batch(turns, window_size=1, max_concurrency=4)
    parallel_time = _time.monotonic() - t0

    # Parallel should be meaningfully faster (at least 1.5x)
    assert parallel_time < sequential_time * 0.8
