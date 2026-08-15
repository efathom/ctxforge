"""
Tests for entropy-aware extraction gating.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock

from ctxforge.config.base import EntropyGateConfig
from ctxforge.extraction.entropy_gate import EntropyGate, GateResult


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, embeddings: List[List[float]] = None):
        """
        Initialize with optional predefined embeddings.

        If embeddings is provided, they are returned in order.
        Otherwise, generates simple hash-based embeddings.
        """
        self._embeddings = embeddings or []
        self._call_count = 0
        self.name = "mock"
        self.default_model = "mock-embedding"
        self.embedding_dimension = 3

    async def embed_single(self, text: str) -> List[float]:
        """Return next predefined embedding or generate one."""
        if self._call_count < len(self._embeddings):
            result = self._embeddings[self._call_count]
            self._call_count += 1
            return result
        # Generate simple embedding based on text hash
        h = hash(text) % 1000
        return [h / 1000, (h * 2) % 1000 / 1000, (h * 3) % 1000 / 1000]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [await self.embed_single(t) for t in texts]


class TestEntropyGate:
    """Tests for EntropyGate."""

    def test_disabled_gate_always_allows(self):
        """Disabled gate should always allow extraction."""
        config = EntropyGateConfig(enabled=False)
        gate = EntropyGate(config, embedding_provider=None)

        result = asyncio.run(gate.evaluate("hello", "world"))

        assert result.should_extract is True
        assert result.reason == "gate_disabled"

    def test_no_provider_disables_gate(self):
        """Gate without embedding provider should be disabled."""
        config = EntropyGateConfig(enabled=True)
        gate = EntropyGate(config, embedding_provider=None)

        assert gate.enabled is False
        result = asyncio.run(gate.evaluate("hello", "world"))
        assert result.should_extract is True

    def test_below_min_chars_skips(self):
        """Turns below min_chars should be skipped."""
        config = EntropyGateConfig(enabled=True, min_chars=50)
        provider = MockEmbeddingProvider()
        gate = EntropyGate(config, embedding_provider=provider)

        result = asyncio.run(gate.evaluate("hi", "hello"))

        assert result.should_extract is False
        assert result.reason == "below_min_chars"

    def test_first_turn_always_novel(self):
        """First turn should always be considered novel."""
        config = EntropyGateConfig(enabled=True, min_chars=5)
        provider = MockEmbeddingProvider()
        gate = EntropyGate(config, embedding_provider=provider)

        result = asyncio.run(gate.evaluate("Hello there!", "Hi, how can I help?"))

        assert result.should_extract is True
        assert result.reason == "novel"
        assert result.similarity_score == 0.0  # No previous turns

    def test_identical_turns_blocked(self):
        """Identical turns should be blocked as high similarity."""
        config = EntropyGateConfig(
            enabled=True,
            min_chars=5,
            similarity_threshold=0.9,
        )
        # Use identical embeddings for identical text
        provider = MockEmbeddingProvider(embeddings=[
            [1.0, 0.0, 0.0],  # First turn
            [1.0, 0.0, 0.0],  # Second turn (identical)
        ])
        gate = EntropyGate(config, embedding_provider=provider)

        # First turn - novel
        result1 = asyncio.run(gate.evaluate("Hello!", "Hi there!"))
        assert result1.should_extract is True

        # Second turn - identical embedding, should be blocked
        result2 = asyncio.run(gate.evaluate("Hello!", "Hi there!"))
        assert result2.should_extract is False
        assert result2.reason == "high_similarity"
        assert result2.similarity_score >= 0.9

    def test_different_turns_allowed(self):
        """Different turns should be allowed."""
        config = EntropyGateConfig(
            enabled=True,
            min_chars=5,
            similarity_threshold=0.9,
        )
        # Use orthogonal embeddings for different text
        provider = MockEmbeddingProvider(embeddings=[
            [1.0, 0.0, 0.0],  # First turn
            [0.0, 1.0, 0.0],  # Second turn (orthogonal)
        ])
        gate = EntropyGate(config, embedding_provider=provider)

        # First turn
        result1 = asyncio.run(gate.evaluate("Hello!", "Hi there!"))
        assert result1.should_extract is True

        # Second turn - different embedding
        result2 = asyncio.run(gate.evaluate("What's the weather?", "It's sunny."))
        assert result2.should_extract is True
        assert result2.reason == "novel"

    def test_window_size_respected(self):
        """Recent window size should limit comparison scope."""
        config = EntropyGateConfig(
            enabled=True,
            min_chars=5,
            similarity_threshold=0.9,
            recent_window_size=2,
        )
        # Generate unique embeddings for each turn
        provider = MockEmbeddingProvider(embeddings=[
            [1.0, 0.0, 0.0],  # Turn 1
            [0.0, 1.0, 0.0],  # Turn 2
            [0.0, 0.0, 1.0],  # Turn 3
            [1.0, 0.0, 0.0],  # Turn 4 - same as Turn 1, but Turn 1 evicted
        ])
        gate = EntropyGate(config, embedding_provider=provider)

        # Add 3 turns (window size is 2, so Turn 1 gets evicted)
        asyncio.run(gate.evaluate("Turn 1", "Response 1"))
        asyncio.run(gate.evaluate("Turn 2", "Response 2"))
        asyncio.run(gate.evaluate("Turn 3", "Response 3"))

        # Turn 4 has same embedding as Turn 1, but Turn 1 is evicted
        # Window now contains Turn 2 and Turn 3 embeddings
        result = asyncio.run(gate.evaluate("Turn 4", "Response 4"))
        assert result.should_extract is True  # Novel compared to window

    def test_embedding_error_allows_extraction(self):
        """Embedding errors should allow extraction to proceed."""
        config = EntropyGateConfig(enabled=True, min_chars=5)

        # Create provider that raises an error
        provider = MockEmbeddingProvider()
        provider.embed_single = AsyncMock(side_effect=Exception("API error"))

        gate = EntropyGate(config, embedding_provider=provider)

        result = asyncio.run(gate.evaluate("Hello!", "Hi there!"))

        assert result.should_extract is True
        assert result.reason == "embedding_error"

    def test_reset_clears_window(self):
        """Reset should clear the recent embeddings window."""
        config = EntropyGateConfig(enabled=True, min_chars=5, similarity_threshold=0.9)
        provider = MockEmbeddingProvider(embeddings=[
            [1.0, 0.0, 0.0],  # First turn
            [1.0, 0.0, 0.0],  # After reset - same embedding
        ])
        gate = EntropyGate(config, embedding_provider=provider)

        # First turn
        asyncio.run(gate.evaluate("Hello!", "Hi!"))

        # Reset
        gate.reset()

        # Same embedding should now be novel (window is empty)
        result = asyncio.run(gate.evaluate("Hello!", "Hi!"))
        assert result.should_extract is True
        assert result.reason == "novel"

    def test_enabled_property(self):
        """Test enabled property reflects config and provider state."""
        config_enabled = EntropyGateConfig(enabled=True)
        config_disabled = EntropyGateConfig(enabled=False)
        provider = MockEmbeddingProvider()

        gate_with_provider = EntropyGate(config_enabled, provider)
        gate_without_provider = EntropyGate(config_enabled, None)
        gate_disabled = EntropyGate(config_disabled, provider)

        assert gate_with_provider.enabled is True
        assert gate_without_provider.enabled is False
        assert gate_disabled.enabled is False


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_gate_result_fields(self):
        """Test GateResult has expected fields."""
        result = GateResult(
            should_extract=True,
            reason="novel",
            similarity_score=0.5,
            matched_turn_index=2,
        )

        assert result.should_extract is True
        assert result.reason == "novel"
        assert result.similarity_score == 0.5
        assert result.matched_turn_index == 2

    def test_gate_result_optional_fields(self):
        """Test GateResult optional fields default to None."""
        result = GateResult(should_extract=False, reason="gate_disabled")

        assert result.similarity_score is None
        assert result.matched_turn_index is None


class TestEntropyGateIntegration:
    """Integration tests for entropy gate with config."""

    def test_gate_from_engine_config(self):
        """Test creating gate from EngineConfig."""
        from ctxforge.config.base import EngineConfig

        config = EngineConfig.model_validate({
            "memory_quality": {
                "entropy_gate": {
                    "enabled": True,
                    "similarity_threshold": 0.95,
                    "min_chars": 30,
                    "recent_window_size": 10,
                }
            }
        })

        provider = MockEmbeddingProvider()
        gate = EntropyGate(
            config=config.memory_quality.entropy_gate,
            embedding_provider=provider,
        )

        assert gate.enabled is True
        assert gate._config.similarity_threshold == 0.95
        assert gate._config.min_chars == 30
        assert gate._config.recent_window_size == 10

    def test_gate_disabled_by_default(self):
        """Entropy gate is disabled by default in EngineConfig."""
        from ctxforge.config.base import EngineConfig

        config = EngineConfig()

        assert config.memory_quality.entropy_gate.enabled is False
