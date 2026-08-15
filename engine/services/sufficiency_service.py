"""
Sufficiency Judging Service.

Evaluates if retrieved content is sufficient to answer a query,
enabling progressive retrieval strategies.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from ctxforge.core.sufficiency import (
    ProgressiveRetrievalStats,
    SufficiencyResult,
    SufficiencyVerdict,
)
from ctxforge.prompts.sufficiency_judge import SUFFICIENCY_JUDGE_PROMPT
from ctxforge.protocols.llm import ILLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SufficiencyConfig:
    """Configuration for sufficiency judging."""

    def __init__(
        self,
        enabled: bool = True,
        max_iterations: int = 3,
        min_confidence: float = 0.7,
        fallback_sources: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.fallback_sources = fallback_sources or ["memories", "graph", "expertise"]


class SufficiencyService:
    """
    Service for judging if retrieval results are sufficient.

    Uses LLM to evaluate whether retrieved content can adequately
    answer the user's query. Supports progressive retrieval.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        config: Optional[SufficiencyConfig] = None,
    ):
        self._llm = llm_provider
        self._config = config or SufficiencyConfig()

    @property
    def config(self) -> SufficiencyConfig:
        """Get current configuration."""
        return self._config

    async def judge(
        self,
        query: str,
        retrieved_content: str,
        context: Optional[str] = None,
    ) -> SufficiencyResult:
        """
        Judge if retrieved content is sufficient to answer the query.

        Args:
            query: The user's query
            retrieved_content: The content retrieved so far
            context: Optional additional context

        Returns:
            SufficiencyResult with verdict and reasoning
        """
        if not self._config.enabled:
            return SufficiencyResult.enough("Sufficiency checking disabled")

        # Quick heuristic checks
        if not retrieved_content or not retrieved_content.strip():
            return SufficiencyResult.need_more(
                reasoning="No content retrieved",
                suggested_sources=self._config.fallback_sources,
            )

        # Use LLM to judge
        try:
            result = await self._llm_judge(query, retrieved_content, context)
            return result
        except Exception as e:
            logger.warning(f"LLM sufficiency judgment failed: {e}")
            return SufficiencyResult.uncertain(
                reasoning=f"Judgment failed: {str(e)}",
                confidence=0.3,
            )

    async def _llm_judge(
        self,
        query: str,
        content: str,
        context: Optional[str],
    ) -> SufficiencyResult:
        """Use LLM to judge sufficiency."""
        prompt = SUFFICIENCY_JUDGE_PROMPT.format(
            query=query,
            content=content[:5000],  # Limit content sent to LLM
            context=context or "No additional context",
        )

        response = await self._llm.generate(
            prompt=prompt,
            max_tokens=400,
            temperature=0.2,  # Low temperature for consistency
        )

        return self._parse_response(response.content)

    def _parse_response(self, response: str) -> SufficiencyResult:
        """Parse LLM response XML to extract sufficiency result."""
        # Extract fields from XML
        verdict_match = re.search(
            r'<verdict>(.*?)</verdict>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        reasoning_match = re.search(
            r'<consideration>(.*?)</consideration>',
            response,
            re.DOTALL
        )
        confidence_match = re.search(
            r'<confidence>(.*?)</confidence>',
            response,
            re.DOTALL
        )
        missing_match = re.search(
            r'<missing_aspects>(.*?)</missing_aspects>',
            response,
            re.DOTALL
        )
        sources_match = re.search(
            r'<suggested_sources>(.*?)</suggested_sources>',
            response,
            re.DOTALL
        )

        # Parse verdict
        verdict_str = verdict_match.group(1).strip().upper() if verdict_match else ""
        verdict = self._parse_verdict(verdict_str)

        # Parse reasoning
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"

        # Parse confidence
        confidence = 1.0
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1).strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        # Parse missing aspects
        missing_aspects: List[str] = []
        if missing_match:
            aspects_str = missing_match.group(1).strip()
            if aspects_str and aspects_str.lower() not in ("none", "empty", ""):
                missing_aspects = [a.strip() for a in aspects_str.split(",") if a.strip()]

        # Parse suggested sources
        suggested_sources: List[str] = []
        if sources_match:
            sources_str = sources_match.group(1).strip()
            if sources_str and sources_str.lower() not in ("none", "empty", ""):
                suggested_sources = [s.strip() for s in sources_str.split(",") if s.strip()]

        return SufficiencyResult(
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            missing_aspects=missing_aspects,
            suggested_sources=suggested_sources,
        )

    def _parse_verdict(self, verdict_str: str) -> SufficiencyVerdict:
        """Parse verdict string to enum."""
        verdict_map = {
            "ENOUGH": SufficiencyVerdict.ENOUGH,
            "MORE": SufficiencyVerdict.MORE,
            "UNCERTAIN": SufficiencyVerdict.UNCERTAIN,
        }
        return verdict_map.get(verdict_str, SufficiencyVerdict.UNCERTAIN)

    async def progressive_retrieve(
        self,
        query: str,
        retriever_func: Callable[[int], Any],
        formatter_func: Optional[Callable[[List[Any]], str]] = None,
        initial_limit: int = 5,
        max_limit: int = 20,
    ) -> Tuple[List[Any], SufficiencyResult, ProgressiveRetrievalStats]:
        """
        Progressively retrieve until sufficient or max iterations reached.

        Args:
            query: The user's query
            retriever_func: Async function(limit) -> List[results]
            formatter_func: Optional function to format results as string
            initial_limit: Starting number of results
            max_limit: Maximum results to fetch

        Returns:
            Tuple of (results, final_sufficiency_result, stats)
        """
        stats = ProgressiveRetrievalStats()
        current_limit = initial_limit
        results: List[Any] = []

        for iteration in range(1, self._config.max_iterations + 1):
            stats.total_iterations = iteration

            # Retrieve
            results = await retriever_func(current_limit)

            if iteration == 1:
                stats.initial_results = len(results)

            # Format for judging
            if formatter_func:
                content = formatter_func(results)
            else:
                content = self._default_format(results)

            # Judge sufficiency
            judgment = await self.judge(query, content)
            judgment.iteration = iteration

            if judgment.verdict == SufficiencyVerdict.ENOUGH:
                stats.final_results = len(results)
                stats.final_verdict = SufficiencyVerdict.ENOUGH
                return results, judgment, stats

            if judgment.verdict == SufficiencyVerdict.UNCERTAIN:
                # Uncertain - return what we have
                stats.final_results = len(results)
                stats.final_verdict = SufficiencyVerdict.UNCERTAIN
                return results, judgment, stats

            # MORE - increase limit and try again
            if judgment.suggested_sources:
                stats.sources_tried.extend(judgment.suggested_sources)

            current_limit = min(current_limit * 2, max_limit)

            if current_limit >= max_limit:
                break

        # Max iterations reached
        stats.final_results = len(results)
        stats.final_verdict = SufficiencyVerdict.UNCERTAIN

        return results, SufficiencyResult.uncertain(
            reasoning="Max retrieval iterations reached",
        ), stats

    def _default_format(self, results: List[Any]) -> str:
        """Default formatter for results."""
        if not results:
            return ""

        formatted = []
        for i, item in enumerate(results, 1):
            if hasattr(item, "content"):
                formatted.append(f"{i}. {item.content}")
            elif isinstance(item, dict) and "content" in item:
                formatted.append(f"{i}. {item['content']}")
            else:
                formatted.append(f"{i}. {str(item)}")

        return "\n".join(formatted)

    def is_content_empty(self, content: str) -> bool:
        """Check if content is effectively empty."""
        if not content:
            return True
        stripped = content.strip()
        return not stripped or stripped.lower() in ("none", "empty", "no results")


class SufficiencyServiceFactory:
    """Factory for creating SufficiencyService instances."""

    @staticmethod
    def create(
        llm_provider: Optional[ILLMProvider] = None,
        config: Optional[SufficiencyConfig] = None,
    ) -> Optional[SufficiencyService]:
        """
        Create a SufficiencyService if LLM provider is available.

        Returns None if no LLM provider is given.
        """
        if llm_provider is None:
            return None

        return SufficiencyService(
            llm_provider=llm_provider,
            config=config or SufficiencyConfig(),
        )
