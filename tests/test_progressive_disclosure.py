"""
Tests for Progressive Disclosure API.

Tests MemoryIndexEntry, MemoryIndex, DisclosureLevel, HeadlineService,
and progressive formatting in the context assembler.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.compaction.assembler import DefaultContextAssembler
from ctxforge.config.base import ProgressiveDisclosureConfig
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.core.memory_index import (
    DisclosureLevel,
    MemoryIndex,
    MemoryIndexEntry,
)
from ctxforge.engine.services.headline_service import HeadlineService

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_memory():
    """Create a sample memory item."""
    return MemoryItem(
        memory_id="mem-001",
        user_id="user-123",
        content="User prefers dark mode for all applications and IDEs.",
        type=MemoryType.PREFERENCE,
        source=MemorySource.USER_EXPLICIT,
        confidence_score=0.95,
        tags=["preference", "ui"],
    )


@pytest.fixture
def sample_memory_with_headline():
    """Create a sample memory item with pre-generated headline."""
    mem = MemoryItem(
        memory_id="mem-002",
        user_id="user-123",
        content="User uses pytest for all Python testing with coverage reports.",
        type=MemoryType.PROCEDURAL,
        source=MemorySource.USER_EXPLICIT,
        confidence_score=0.9,
        tags=["testing", "python"],
    )
    mem.headline = "Uses pytest for Python testing"
    mem.subtitle = "Prefers pytest with coverage reports for all Python projects"
    return mem


@pytest.fixture
def multiple_memories():
    """Create multiple memory items for testing."""
    memories = []
    for i in range(5):
        mem = MemoryItem(
            memory_id=f"mem-{i:03d}",
            user_id="user-123",
            content=f"This is memory content number {i} with detailed info.",
            type=MemoryType.SEMANTIC,
            source=MemorySource.AGENT_INFERENCE,
            confidence_score=0.8 + i * 0.02,
            tags=[f"tag-{i}"],
        )
        if i < 2:
            # First two have headlines
            mem.headline = f"Memory {i} headline"
            mem.subtitle = f"Subtitle for memory {i}"
        memories.append(mem)
    return memories


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=MagicMock(
        content="""<headline_response>
  <title>Prefers dark mode</title>
  <subtitle>User wants dark mode enabled for all applications</subtitle>
</headline_response>"""
    ))
    return provider


# =============================================================================
# DisclosureLevel Tests
# =============================================================================


class TestDisclosureLevel:
    """Tests for DisclosureLevel enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert DisclosureLevel.HEADLINE.value == "headline"
        assert DisclosureLevel.SUMMARY.value == "summary"
        assert DisclosureLevel.FULL.value == "full"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert DisclosureLevel.HEADLINE == DisclosureLevel.HEADLINE
        assert DisclosureLevel.HEADLINE != DisclosureLevel.FULL


# =============================================================================
# MemoryIndexEntry Tests
# =============================================================================


class TestMemoryIndexEntry:
    """Tests for MemoryIndexEntry."""

    def test_from_memory_without_headline(self, sample_memory):
        """Test creating entry from memory without headline."""
        entry = MemoryIndexEntry.from_memory(sample_memory)

        assert entry.memory_id == "mem-001"
        assert entry.memory_type == "preference"
        assert "User prefers dark mode" in entry.headline
        assert entry.subtitle is None
        assert entry.confidence == 0.95
        assert entry.tags == ["preference", "ui"]
        assert entry._full_content == sample_memory.content

    def test_from_memory_with_headline(self, sample_memory_with_headline):
        """Test creating entry from memory with headline."""
        entry = MemoryIndexEntry.from_memory(sample_memory_with_headline)

        assert entry.headline == "Uses pytest for Python testing"
        assert entry.subtitle == "Prefers pytest with coverage reports for all Python projects"

    def test_from_memory_with_override(self, sample_memory):
        """Test creating entry with headline override."""
        entry = MemoryIndexEntry.from_memory(
            sample_memory,
            headline="Custom headline",
            subtitle="Custom subtitle",
        )

        assert entry.headline == "Custom headline"
        assert entry.subtitle == "Custom subtitle"

    def test_to_prompt_headline_level(self, sample_memory_with_headline):
        """Test formatting at headline level."""
        entry = MemoryIndexEntry.from_memory(sample_memory_with_headline)
        prompt = entry.to_prompt(DisclosureLevel.HEADLINE)

        assert "procedural" in prompt
        assert "Uses pytest for Python testing" in prompt
        assert "coverage reports" not in prompt  # Subtitle not included

    def test_to_prompt_summary_level(self, sample_memory_with_headline):
        """Test formatting at summary level."""
        entry = MemoryIndexEntry.from_memory(sample_memory_with_headline)
        prompt = entry.to_prompt(DisclosureLevel.SUMMARY)

        assert "procedural" in prompt
        assert "Uses pytest for Python testing" in prompt
        assert "coverage reports" in prompt  # Subtitle included

    def test_to_prompt_full_level(self, sample_memory_with_headline):
        """Test formatting at full level."""
        entry = MemoryIndexEntry.from_memory(sample_memory_with_headline)
        prompt = entry.to_prompt(DisclosureLevel.FULL)

        assert "procedural" in prompt
        # Full content should be shown
        assert "pytest for all Python testing" in prompt

    def test_estimate_tokens(self, sample_memory_with_headline):
        """Test token estimation."""
        entry = MemoryIndexEntry.from_memory(sample_memory_with_headline)

        headline_tokens = entry.estimate_tokens(DisclosureLevel.HEADLINE)
        full_tokens = entry.estimate_tokens(DisclosureLevel.FULL)

        # Full should have more tokens than headline
        assert full_tokens > headline_tokens
        assert headline_tokens > 0


# =============================================================================
# MemoryIndex Tests
# =============================================================================


class TestMemoryIndex:
    """Tests for MemoryIndex."""

    def test_empty_index(self):
        """Test empty index behavior."""
        index = MemoryIndex()

        assert len(index) == 0
        assert index.to_prompt() == ""
        assert index.estimate_tokens() == 0

    def test_add_entries(self, multiple_memories):
        """Test adding entries to index."""
        index = MemoryIndex()

        for memory in multiple_memories:
            entry = MemoryIndexEntry.from_memory(memory)
            index.add(entry)

        assert len(index) == 5
        assert index.total_memories == 5

    def test_to_prompt_default(self, multiple_memories):
        """Test default prompt formatting."""
        index = MemoryIndex(total_memories=len(multiple_memories))
        for memory in multiple_memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        prompt = index.to_prompt()

        assert "User Context (5 relevant memories):" in prompt
        assert "Memory 0 headline" in prompt  # First has headline
        assert "Memory 1 headline" in prompt  # Second has headline

    def test_to_prompt_with_expansion(self, multiple_memories):
        """Test prompt with top N expanded."""
        index = MemoryIndex(total_memories=len(multiple_memories))
        for memory in multiple_memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        prompt = index.to_prompt(expand_top_n=2)

        # First two should show full content
        assert "memory content number 0" in prompt
        assert "memory content number 1" in prompt

    def test_to_prompt_max_entries(self, multiple_memories):
        """Test prompt with max entries limit."""
        index = MemoryIndex()  # Don't set total_memories, let add() track it
        for memory in multiple_memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        prompt = index.to_prompt(max_entries=3)

        # 5 total, showing 3, so "... and 2 more"
        assert "... and 2 more" in prompt

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty index."""
        index = MemoryIndex()
        assert index.estimate_tokens() == 0

    def test_estimate_tokens_with_expansion(self, multiple_memories):
        """Test token estimation with expansion."""
        index = MemoryIndex(total_memories=len(multiple_memories))
        for memory in multiple_memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        headline_tokens = index.estimate_tokens(expand_top_n=0)
        expanded_tokens = index.estimate_tokens(expand_top_n=3)

        assert expanded_tokens > headline_tokens

    def test_iteration(self, multiple_memories):
        """Test iteration over index."""
        index = MemoryIndex(total_memories=len(multiple_memories))
        for memory in multiple_memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        entries = list(index)
        assert len(entries) == 5
        assert all(isinstance(e, MemoryIndexEntry) for e in entries)


# =============================================================================
# HeadlineService Tests
# =============================================================================


class TestHeadlineService:
    """Tests for HeadlineService."""

    @pytest.mark.asyncio
    async def test_generate_headline_success(self, sample_memory, mock_llm_provider):
        """Test successful headline generation."""
        service = HeadlineService(mock_llm_provider)

        headline, subtitle = await service.generate_headline(sample_memory)

        assert headline == "Prefers dark mode"
        assert subtitle == "User wants dark mode enabled for all applications"
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_headline_fallback(self, sample_memory):
        """Test fallback when LLM fails."""
        failing_provider = AsyncMock()
        failing_provider.generate = AsyncMock(side_effect=Exception("LLM error"))

        service = HeadlineService(failing_provider)
        headline, subtitle = await service.generate_headline(sample_memory)

        # Should fallback to first sentence
        assert "User prefers dark mode" in headline
        assert subtitle == ""

    @pytest.mark.asyncio
    async def test_generate_and_update(self, sample_memory, mock_llm_provider):
        """Test generate and update memory in place."""
        service = HeadlineService(mock_llm_provider)

        assert sample_memory.headline is None
        result = await service.generate_and_update(sample_memory)

        assert result.headline == "Prefers dark mode"
        assert result.subtitle == "User wants dark mode enabled for all applications"
        assert sample_memory.headline == "Prefers dark mode"  # Updated in place

    @pytest.mark.asyncio
    async def test_generate_batch(self, multiple_memories, mock_llm_provider):
        """Test batch headline generation."""
        service = HeadlineService(mock_llm_provider)

        # First two already have headlines
        result = await service.generate_batch(multiple_memories, skip_existing=True)

        # Only 3 should have been processed (skip first 2)
        assert mock_llm_provider.generate.call_count == 3
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_generate_batch_no_skip(self, multiple_memories, mock_llm_provider):
        """Test batch generation without skipping existing."""
        service = HeadlineService(mock_llm_provider)

        await service.generate_batch(multiple_memories, skip_existing=False)

        # All 5 should be processed
        assert mock_llm_provider.generate.call_count == 5

    @pytest.mark.asyncio
    async def test_build_index(self, multiple_memories, mock_llm_provider):
        """Test building MemoryIndex from memories."""
        service = HeadlineService(mock_llm_provider)

        index = await service.build_index(multiple_memories)

        assert len(index) == 5
        # All should have headlines now
        for entry in index:
            assert entry.headline

    def test_parse_response_valid(self, mock_llm_provider):
        """Test parsing valid XML response."""
        service = HeadlineService(mock_llm_provider)

        response = """<headline_response>
  <title>Test Title</title>
  <subtitle>Test subtitle here</subtitle>
</headline_response>"""

        headline, subtitle = service._parse_response(response)

        assert headline == "Test Title"
        assert subtitle == "Test subtitle here"

    def test_parse_response_missing_title(self, mock_llm_provider):
        """Test parsing response without title raises error."""
        service = HeadlineService(mock_llm_provider)

        response = """<headline_response>
  <subtitle>Only subtitle</subtitle>
</headline_response>"""

        with pytest.raises(ValueError, match="No headline found"):
            service._parse_response(response)

    def test_fallback_headline(self, mock_llm_provider):
        """Test fallback headline extraction."""
        service = HeadlineService(mock_llm_provider)

        content = "This is the first sentence. This is the second."
        headline = service._fallback_headline(content)

        assert headline == "This is the first sentence"

    def test_fallback_headline_long_content(self, mock_llm_provider):
        """Test fallback with very long first sentence."""
        service = HeadlineService(mock_llm_provider, max_headline_chars=30)

        content = "This is a very long first sentence that exceeds the limit."
        headline = service._fallback_headline(content)

        assert len(headline) <= 30
        assert headline.endswith("...")


# =============================================================================
# MemoryItem Headline Methods Tests
# =============================================================================


class TestMemoryItemHeadlineMethods:
    """Tests for headline-related methods on MemoryItem."""

    def test_has_headline_false(self, sample_memory):
        """Test has_headline returns False when no headline."""
        assert not sample_memory.has_headline()

    def test_has_headline_true(self, sample_memory_with_headline):
        """Test has_headline returns True when headline exists."""
        assert sample_memory_with_headline.has_headline()

    def test_to_headline_format_with_headline(self, sample_memory_with_headline):
        """Test headline format when headline exists."""
        result = sample_memory_with_headline.to_headline_format()

        assert "procedural" in result
        assert "Uses pytest for Python testing" in result

    def test_to_headline_format_fallback(self, sample_memory):
        """Test headline format fallback to truncated content."""
        result = sample_memory.to_headline_format()

        assert "preference" in result
        assert "User prefers dark mode" in result

    def test_to_summary_format_with_both(self, sample_memory_with_headline):
        """Test summary format with headline and subtitle."""
        result = sample_memory_with_headline.to_summary_format()

        assert "Uses pytest for Python testing" in result
        assert "coverage reports" in result

    def test_to_summary_format_headline_only(self, sample_memory):
        """Test summary format with headline only."""
        sample_memory.headline = "Test headline"
        result = sample_memory.to_summary_format()

        assert "Test headline" in result


# =============================================================================
# DefaultContextAssembler Progressive Disclosure Tests
# =============================================================================


class TestAssemblerProgressiveDisclosure:
    """Tests for progressive disclosure in DefaultContextAssembler."""

    def test_init_with_progressive_options(self):
        """Test assembler initialization with progressive disclosure options."""
        assembler = DefaultContextAssembler(
            use_progressive_disclosure=True,
            progressive_expand_top_n=5,
        )

        assert assembler._use_progressive is True
        assert assembler._expand_top_n == 5

    def test_format_memories_progressive(self, multiple_memories):
        """Test progressive memory formatting."""
        assembler = DefaultContextAssembler(use_progressive_disclosure=True)

        result = assembler._format_memories(multiple_memories)

        assert "User Context" in result
        assert "relevant memories" in result

    def test_format_memories_progressive_via_format(self, multiple_memories):
        """Test progressive formatting via memory_format option."""
        assembler = DefaultContextAssembler(memory_format="progressive")

        result = assembler._format_memories(multiple_memories)

        assert "User Context" in result

    def test_format_memories_from_index(self, multiple_memories):
        """Test formatting from pre-built index."""
        assembler = DefaultContextAssembler(progressive_expand_top_n=2)

        # Build index
        index = MemoryIndex(total_memories=len(multiple_memories))
        for memory in multiple_memories:
            index.add(MemoryIndexEntry.from_memory(memory))

        result = assembler.format_memories_from_index(index)

        assert "User Context" in result

    def test_format_memories_budget_adjustment(self, multiple_memories):
        """Test that expansion adjusts to token budget."""
        assembler = DefaultContextAssembler(
            use_progressive_disclosure=True,
            progressive_expand_top_n=5,
        )

        # Very low budget should reduce expansion
        result = assembler._format_memories_progressive(
            multiple_memories,
            token_budget=50,
        )

        # Should still produce output
        assert len(result) > 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestProgressiveDisclosureConfig:
    """Tests for ProgressiveDisclosureConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProgressiveDisclosureConfig()

        assert config.enabled is False
        assert config.max_headline_chars == 80
        assert config.max_subtitle_chars == 150
        assert config.expand_top_n == 3
        assert config.use_llm_headlines is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProgressiveDisclosureConfig(
            enabled=True,
            max_headline_chars=100,
            max_subtitle_chars=200,
            expand_top_n=5,
            use_llm_headlines=False,
        )

        assert config.enabled is True
        assert config.max_headline_chars == 100
        assert config.max_subtitle_chars == 200
        assert config.expand_top_n == 5
        assert config.use_llm_headlines is False

    def test_validation_min_values(self):
        """Test validation of minimum values."""
        # Should not raise with valid min values
        config = ProgressiveDisclosureConfig(
            max_headline_chars=20,
            max_subtitle_chars=50,
            expand_top_n=0,
        )
        assert config.max_headline_chars == 20
