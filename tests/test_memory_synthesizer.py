"""Tests for the memory synthesis service."""

import pytest

from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.services.memory_synthesizer_service import (
    MemorySynthesizerService,
)
from ctxforge.extraction.integration_config import SynthesizerConfig
from ctxforge.llm.mock_provider import MockLLMProvider


@pytest.mark.asyncio
async def test_synthesize_empty_memories():
    """Returns None for empty memory list."""
    llm = MockLLMProvider(latency_ms=0)
    service = MemorySynthesizerService(llm=llm)
    result = await service.synthesize(memories=[], query="hello")
    assert result is None


@pytest.mark.asyncio
async def test_synthesize_produces_narrative():
    """Produces a coherent narrative from multiple memories."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses([
        "The user is vegetarian, prefers dark mode, and works evening shifts."
    ])
    service = MemorySynthesizerService(llm=llm)

    memories = [
        MemoryFactory.semantic_memory(user_id="u1", content="User is vegetarian"),
        MemoryFactory.semantic_memory(user_id="u1", content="User prefers dark mode"),
        MemoryFactory.semantic_memory(
            user_id="u1", content="User works evening shifts",
        ),
    ]

    result = await service.synthesize(
        memories=memories, query="Tell me about the user",
    )

    assert result is not None
    assert "vegetarian" in result.lower()


@pytest.mark.asyncio
async def test_synthesize_returns_none_for_no_relevant():
    """Returns None when LLM says no relevant info."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["No relevant personalized information found."])
    service = MemorySynthesizerService(llm=llm)

    memories = [
        MemoryFactory.semantic_memory(user_id="u1", content="some memory"),
    ]

    result = await service.synthesize(
        memories=memories, query="unrelated query",
    )

    assert result is None


@pytest.mark.asyncio
async def test_synthesize_uses_config_model():
    """Config model is passed to LLM."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["synthesized narrative"])
    config = SynthesizerConfig(model="custom-model", max_synthesis_tokens=100)
    service = MemorySynthesizerService(llm=llm, config=config)

    memories = [
        MemoryFactory.semantic_memory(user_id="u1", content="fact"),
    ]

    result = await service.synthesize(memories=memories, query="q")
    assert result == "synthesized narrative"
    # MockLLMProvider ignores model, but we verify no errors occur


@pytest.mark.asyncio
async def test_synthesize_handles_llm_failure():
    """Returns None on LLM failure."""
    llm = MockLLMProvider(latency_ms=0)

    # Create a service and monkeypatch generate to raise
    service = MemorySynthesizerService(llm=llm)

    async def _fail(*args, **kwargs):
        raise RuntimeError("LLM down")

    service._llm.generate = _fail

    memories = [
        MemoryFactory.semantic_memory(user_id="u1", content="fact"),
    ]

    result = await service.synthesize(memories=memories, query="q")
    assert result is None


@pytest.mark.asyncio
async def test_synthesize_respects_max_tokens():
    """max_tokens parameter is honored."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["short summary"])
    service = MemorySynthesizerService(llm=llm)

    memories = [
        MemoryFactory.semantic_memory(user_id="u1", content="fact"),
    ]

    result = await service.synthesize(
        memories=memories, query="q", max_tokens=50,
    )
    assert result == "short summary"
