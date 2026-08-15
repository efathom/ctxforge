"""
Entropy-aware gating for memory extraction.

Filters low-value turns before expensive extraction by measuring novelty
against recent conversation history using embedding similarity.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from ctxforge.config.base import EntropyGateConfig
from ctxforge.protocols.llm import IEmbeddingProvider
from ctxforge.utils.math import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of entropy gate evaluation."""

    should_extract: bool
    reason: str
    similarity_score: Optional[float] = None
    matched_turn_index: Optional[int] = None


class EntropyGate:
    """
    Pre-extraction gate that filters low-novelty turns.

    Uses embedding similarity to detect repetitive or low-information turns
    that don't warrant expensive extraction. Maintains a sliding window of
    recent turn embeddings for comparison.

    Gate logic:
    1. Skip turns below min_chars threshold
    2. Compute embedding for combined user_input + assistant_response
    3. Compare against recent turn embeddings
    4. If max similarity >= threshold, skip extraction (redundant)
    5. Otherwise, allow extraction and add to recent window

    Example:
        gate = EntropyGate(config, embedding_provider)
        result = await gate.evaluate(user_input, assistant_response)
        if result.should_extract:
            # proceed with extraction
    """

    def __init__(
        self,
        config: EntropyGateConfig,
        embedding_provider: Optional[IEmbeddingProvider] = None,
    ):
        """
        Initialize the entropy gate.

        Args:
            config: Gate configuration.
            embedding_provider: Provider for computing embeddings. If None,
                                gate will always allow extraction.
        """
        self._config = config
        self._embedding_provider = embedding_provider
        self._enabled = config.enabled and embedding_provider is not None

        # Sliding window of recent turn embeddings
        self._recent_embeddings: Deque[List[float]] = deque(
            maxlen=config.recent_window_size
        )

        # Simple LRU cache for embeddings (text hash -> embedding)
        self._embedding_cache: dict = {}
        self._cache_order: Deque[str] = deque(maxlen=config.embedding_cache_size)

    @property
    def enabled(self) -> bool:
        """Whether the gate is enabled and functional."""
        return self._enabled

    async def evaluate(
        self,
        user_input: str,
        assistant_response: str,
    ) -> GateResult:
        """
        Evaluate whether a turn should proceed to extraction.

        Args:
            user_input: The user's input text.
            assistant_response: The assistant's response text.

        Returns:
            GateResult indicating whether to extract and why.
        """
        if not self._enabled:
            return GateResult(
                should_extract=True,
                reason="gate_disabled",
            )

        combined_text = self._combine_text(user_input, assistant_response)

        # Check minimum length
        if len(combined_text) < self._config.min_chars:
            return GateResult(
                should_extract=False,
                reason="below_min_chars",
            )

        # Get embedding for current turn
        try:
            current_embedding = await self._get_embedding(combined_text)
        except Exception as e:
            logger.warning(f"Entropy gate embedding failed: {e}")
            # On error, allow extraction to proceed
            return GateResult(
                should_extract=True,
                reason="embedding_error",
            )

        # Compare against recent turns
        max_similarity = 0.0
        matched_index = None

        for idx, recent_embedding in enumerate(self._recent_embeddings):
            similarity = cosine_similarity(current_embedding, recent_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_index = idx

        # Check if too similar to recent turn
        if max_similarity >= self._config.similarity_threshold:
            return GateResult(
                should_extract=False,
                reason="high_similarity",
                similarity_score=max_similarity,
                matched_turn_index=matched_index,
            )

        # Novel enough - add to window and allow extraction
        self._recent_embeddings.append(current_embedding)

        return GateResult(
            should_extract=True,
            reason="novel",
            similarity_score=max_similarity,
        )

    def reset(self) -> None:
        """Clear the recent embeddings window and cache."""
        self._recent_embeddings.clear()
        self._embedding_cache.clear()
        self._cache_order.clear()

    def _combine_text(self, user_input: str, assistant_response: str) -> str:
        """Combine user input and response for embedding."""
        return f"{user_input.strip()} {assistant_response.strip()}"

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        cache_key = hash(text)

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if self._embedding_provider is None:
            raise ValueError("No embedding provider available")

        embedding = await self._embedding_provider.embed_single(text)

        # Add to cache with LRU eviction
        if len(self._cache_order) >= self._config.embedding_cache_size:
            oldest_key = self._cache_order.popleft()
            self._embedding_cache.pop(oldest_key, None)

        self._embedding_cache[cache_key] = embedding
        self._cache_order.append(cache_key)

        return embedding
