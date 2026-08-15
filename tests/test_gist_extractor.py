"""Tests for gist extraction (atomic, timestamped memory extraction)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.memory import MemoryType
from ctxforge.extraction.gist_enhanced_extractor import GistEnhancedExtractor
from ctxforge.extraction.gist_extractor import GistExtractor
from ctxforge.extraction.gist_prompts import build_gist_prompt
from ctxforge.protocols.extractor import ExtractionCandidate, ExtractionConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(response_json: dict) -> MagicMock:
    """Create a mock ILLMProvider that returns ``response_json``."""
    llm = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps(response_json)
    llm.chat = AsyncMock(return_value=resp)
    return llm


def _make_fact_extractor(candidates: list[ExtractionCandidate]) -> MagicMock:
    """Create a mock BaseExtractor for the fact-extraction phase."""
    ext = MagicMock()
    ext._do_extract = AsyncMock(return_value=candidates)
    return ext


# ---------------------------------------------------------------------------
# GistExtractor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gist_extraction_basic():
    """Two gists extracted from a compound sentence."""
    llm = _make_llm({
        "gists": [
            {
                "content": "[2026-03-07T10:00:00] Alice fixed the fence on Monday (2026-03-03).",
                "timestamp": "2026-03-03",
                "confidence": 0.95,
            },
            {
                "content": "[2026-03-07T10:00:00] Alice bought 3 cows from Peter on March 5th (2026-03-05).",
                "timestamp": "2026-03-05",
                "confidence": 0.9,
            },
        ]
    })

    ext = GistExtractor(llm_provider=llm)
    config = ExtractionConfig()
    candidates = await ext._do_extract(
        "Alice: I fixed the fence on Monday, then bought 3 cows from Peter on March 5th",
        config,
    )

    assert len(candidates) == 2
    assert all(c.memory_type == MemoryType.EPISODIC for c in candidates)
    assert all("gist" in c.tags for c in candidates)
    assert all("atomic" in c.tags for c in candidates)


@pytest.mark.asyncio
async def test_gist_temporal_resolution():
    """Resolved timestamp stored in metadata."""
    llm = _make_llm({
        "gists": [
            {
                "content": "[2026-03-07] Bob met advisor last Thursday (2026-03-05).",
                "timestamp": "2026-03-05",
                "confidence": 0.9,
            }
        ]
    })

    ext = GistExtractor(llm_provider=llm)
    candidates = await ext._do_extract("Bob met advisor last Thursday", ExtractionConfig())

    assert len(candidates) == 1
    assert candidates[0].metadata["resolved_timestamp"] == "2026-03-05"
    assert candidates[0].metadata["source_type"] == "gist_extraction"


@pytest.mark.asyncio
async def test_gist_candidate_fields():
    """Verify memory_type, tags, confidence, restatement."""
    llm = _make_llm({
        "gists": [
            {"content": "Alice loves coffee.", "timestamp": "2026-03-07", "confidence": 0.8}
        ]
    })

    ext = GistExtractor(llm_provider=llm)
    candidates = await ext._do_extract("Alice loves coffee", ExtractionConfig())

    c = candidates[0]
    assert c.memory_type == MemoryType.EPISODIC
    assert c.confidence == 0.8
    assert c.restatement == c.content  # gists are already self-contained


@pytest.mark.asyncio
async def test_gist_empty_input():
    """Empty input produces no gists."""
    llm = _make_llm({"gists": []})
    ext = GistExtractor(llm_provider=llm)

    assert await ext._do_extract("", ExtractionConfig()) == []
    assert await ext._do_extract("   ", ExtractionConfig()) == []


@pytest.mark.asyncio
async def test_gist_llm_failure():
    """LLM failure returns empty list, doesn't raise."""
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("API down"))

    ext = GistExtractor(llm_provider=llm)
    result = await ext._do_extract("some text", ExtractionConfig())
    assert result == []


@pytest.mark.asyncio
async def test_gist_confidence_clamping():
    """Out-of-range confidence is clamped to [0, 1]."""
    llm = _make_llm({
        "gists": [
            {"content": "fact1", "timestamp": "2026-01-01", "confidence": 1.5},
            {"content": "fact2", "timestamp": "2026-01-01", "confidence": -0.2},
        ]
    })

    ext = GistExtractor(llm_provider=llm)
    candidates = await ext._do_extract("text", ExtractionConfig())

    assert candidates[0].confidence == 1.0
    assert candidates[1].confidence == 0.0


# ---------------------------------------------------------------------------
# GistEnhancedExtractor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gist_enhanced_two_phase():
    """Gist context is passed to fact extractor in phase 2."""
    gist_llm = _make_llm({
        "gists": [
            {"content": "Alice works at Apple.", "timestamp": "2026-03-07", "confidence": 0.9}
        ]
    })
    gist_ext = GistExtractor(llm_provider=gist_llm)

    fact_candidates = [
        ExtractionCandidate(
            content="Alice is employed by Apple",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            source_text="Alice works at Apple",
        )
    ]
    fact_ext = _make_fact_extractor(fact_candidates)

    enhanced = GistEnhancedExtractor(gist_extractor=gist_ext, fact_extractor=fact_ext)
    results = await enhanced._do_extract("Alice works at Apple", ExtractionConfig())

    # Should return 1 gist + 1 fact
    assert len(results) == 2
    gist_results = [r for r in results if "gist" in r.tags]
    fact_results = [r for r in results if "gist" not in r.tags]
    assert len(gist_results) == 1
    assert len(fact_results) == 1

    # Verify gist_context was passed to fact extractor
    call_args = fact_ext._do_extract.call_args
    context = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("context")
    assert "gist_context" in context
    assert "Alice works at Apple." in context["gist_context"]


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


def test_build_gist_prompt_structure():
    """build_gist_prompt returns system + user messages."""
    msgs = build_gist_prompt("hello world", current_time="2026-03-07T12:00:00")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "2026-03-07T12:00:00" in msgs[1]["content"]
    assert "hello world" in msgs[1]["content"]


def test_build_gist_prompt_default_time():
    """When current_time is None, a timestamp is still generated."""
    msgs = build_gist_prompt("test")
    assert "Current time:" in msgs[1]["content"]
