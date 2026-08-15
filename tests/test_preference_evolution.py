"""Tests for the preference evolution tracking service."""

import pytest

from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.services.preference_evolution_service import (
    PreferenceChange,
    PreferenceEvolutionService,
)
from ctxforge.extraction.integration_config import PreferenceEvolutionConfig
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.storage.memory.memory import InMemoryMemoryStore


@pytest.mark.asyncio
async def test_detect_preference_change_yes():
    """Detects when a preference change occurs."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["Yes"])
    store = InMemoryMemoryStore()
    service = PreferenceEvolutionService(llm=llm, memory_store=store)

    existing = MemoryFactory.semantic_memory(
        user_id="u1", content="User prefers dark mode",
    )
    change = await service.detect_preference_change(
        new_content="I now prefer light mode",
        existing_memory=existing,
        query="theme preference",
    )

    assert change is not None
    assert change.change_type == "supersede"
    assert change.old_content == "User prefers dark mode"
    assert change.new_content == "I now prefer light mode"


@pytest.mark.asyncio
async def test_detect_preference_change_no():
    """Returns None when no preference change detected."""
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(["No"])
    store = InMemoryMemoryStore()
    service = PreferenceEvolutionService(llm=llm, memory_store=store)

    existing = MemoryFactory.semantic_memory(
        user_id="u1", content="User likes dark mode",
    )
    change = await service.detect_preference_change(
        new_content="User also likes minimalist design",
        existing_memory=existing,
        query="preferences",
    )

    assert change is None


@pytest.mark.asyncio
async def test_detect_preference_change_llm_failure():
    """Returns None on LLM failure (fail-safe)."""
    llm = MockLLMProvider(latency_ms=0)
    # MockLLMProvider without responses returns default, not an error.
    # We'll set an invalid response that doesn't start with 'yes'.
    llm.set_responses(["error occurred"])
    store = InMemoryMemoryStore()
    service = PreferenceEvolutionService(llm=llm, memory_store=store)

    existing = MemoryFactory.semantic_memory(user_id="u1", content="test")
    change = await service.detect_preference_change(
        new_content="new", existing_memory=existing, query="q",
    )
    assert change is None


@pytest.mark.asyncio
async def test_apply_preference_change_supersedes_old():
    """Apply marks old memory as superseded and links memories."""
    llm = MockLLMProvider(latency_ms=0)
    store = InMemoryMemoryStore()
    config = PreferenceEvolutionConfig(
        enabled=True,
        auto_supersede=True,
        importance_decay_on_supersede=0.1,
        track_history=True,
    )
    service = PreferenceEvolutionService(
        llm=llm, memory_store=store, config=config,
    )

    old_mem = MemoryFactory.semantic_memory(
        user_id="u1", content="User prefers dark mode",
    )
    old_mem.importance = 1.0
    await store.add(old_mem)

    new_mem = MemoryFactory.semantic_memory(
        user_id="u1", content="User now prefers light mode",
    )

    change = PreferenceChange(
        old_memory_id=old_mem.memory_id,
        old_content=old_mem.content,
        new_content=new_mem.content,
        change_type="supersede",
    )

    result = await service.apply_preference_change(
        change=change, new_memory=new_mem, old_memory=old_mem,
    )

    # Old memory should be superseded
    assert old_mem.superseded_by == new_mem.memory_id
    assert old_mem.importance == pytest.approx(0.1, abs=0.01)

    # New memory should have change history
    assert "preference_changes" in result.metadata
    assert len(result.metadata["preference_changes"]) == 1
    assert result.metadata["preference_version"] == 1

    # Memories should be linked
    assert old_mem.memory_id in result.related_memory_ids
    assert new_mem.memory_id in old_mem.related_memory_ids


@pytest.mark.asyncio
async def test_apply_preference_change_no_auto_supersede():
    """When auto_supersede is False, old memory importance is not decayed."""
    llm = MockLLMProvider(latency_ms=0)
    store = InMemoryMemoryStore()
    config = PreferenceEvolutionConfig(
        enabled=True,
        auto_supersede=False,
        track_history=True,
    )
    service = PreferenceEvolutionService(
        llm=llm, memory_store=store, config=config,
    )

    old_mem = MemoryFactory.semantic_memory(
        user_id="u1", content="User prefers dark mode",
    )
    old_mem.importance = 1.0
    await store.add(old_mem)

    new_mem = MemoryFactory.semantic_memory(
        user_id="u1", content="User now prefers light mode",
    )

    change = PreferenceChange(
        old_memory_id=old_mem.memory_id,
        old_content=old_mem.content,
        new_content=new_mem.content,
    )

    await service.apply_preference_change(
        change=change, new_memory=new_mem, old_memory=old_mem,
    )

    # Importance should NOT be decayed
    assert old_mem.importance == 1.0


@pytest.mark.asyncio
async def test_get_preference_history():
    """Retrieves preference change history from memory metadata."""
    llm = MockLLMProvider(latency_ms=0)
    store = InMemoryMemoryStore()
    service = PreferenceEvolutionService(llm=llm, memory_store=store)

    mem = MemoryFactory.semantic_memory(
        user_id="u1", content="User prefers light mode",
    )
    mem.metadata["preference_changes"] = [
        {
            "from": "dark mode",
            "to": "light mode",
            "changed_at": "2026-03-01T12:00:00",
            "old_memory_id": "old-1",
            "change_type": "supersede",
        }
    ]
    await store.add(mem)

    history = await service.get_preference_history(user_id="u1")
    assert len(history) == 1
    assert history[0].old_content == "dark mode"
    assert history[0].new_content == "light mode"


@pytest.mark.asyncio
async def test_get_preference_history_with_topic_filter():
    """Topic filter narrows preference history results."""
    llm = MockLLMProvider(latency_ms=0)
    store = InMemoryMemoryStore()
    service = PreferenceEvolutionService(llm=llm, memory_store=store)

    mem = MemoryFactory.semantic_memory(
        user_id="u1", content="User prefers light mode",
    )
    mem.metadata["preference_changes"] = [
        {
            "from": "dark mode",
            "to": "light mode",
            "changed_at": "2026-03-01T12:00:00",
            "old_memory_id": "old-1",
            "change_type": "supersede",
        }
    ]
    await store.add(mem)

    # Filter by topic that matches
    history = await service.get_preference_history(
        user_id="u1", topic="mode",
    )
    assert len(history) == 1

    # Filter by topic that doesn't match
    history = await service.get_preference_history(
        user_id="u1", topic="food",
    )
    assert len(history) == 0


@pytest.mark.asyncio
async def test_preference_change_dataclass():
    """PreferenceChange dataclass has sensible defaults."""
    change = PreferenceChange(
        old_memory_id="m1",
        old_content="old",
        new_content="new",
    )
    assert change.change_type == "supersede"
    assert change.rationale is None
    assert change.changed_at is not None
