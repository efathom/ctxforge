"""
Conflict-aware consolidation for memory extraction.

Decides merge/add/ignore based on similarity, keyword overlap,
and contradiction signals. Supports both heuristic and LLM-based
contradiction detection.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

from ctxforge.config.base import ConsolidationQualityConfig
from ctxforge.core.memory import MemoryItem
from ctxforge.extraction.consolidation.base import BaseConsolidator
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.utils.similarity import ISimilarityCalculator, TextSimilarityCalculator

logger = logging.getLogger(__name__)


# Default LLM prompt for contradiction detection
DEFAULT_CONTRADICTION_PROMPT = """You are a contradiction detector for memory entries.

Given two memory entries about the same topic, determine if they contradict each other.

Memory 1: {memory1}
Memory 2: {memory2}

Analyze whether these memories contain conflicting information. Consider:
- Opposite sentiments (likes vs dislikes)
- Different numeric values for the same attribute
- Contradictory states or facts
- Time-based changes that invalidate older information

Respond with JSON only:
{{
  "is_contradiction": true/false,
  "reason": "brief explanation",
  "confidence": 0.0-1.0,
  "prefer_newer": true/false
}}"""


class ConsolidationAction(str, Enum):
    """Action to take for a new memory item."""

    ADD = "add"
    MERGE = "merge"
    IGNORE = "ignore"
    CONFLICT = "conflict"


@dataclass
class ConsolidationDecision:
    """Result of consolidation decision for a single item."""

    action: ConsolidationAction
    reason: str
    new_item: MemoryItem
    target_item: Optional[MemoryItem] = None
    similarity_score: Optional[float] = None
    keyword_overlap: Optional[float] = None
    is_contradiction: bool = False


class ConflictAwareConsolidator(BaseConsolidator):
    """
    Consolidator that decides merge/add/ignore based on multiple signals.

    Decision logic:
    1. If similarity >= merge_threshold AND keyword_overlap >= kw_threshold:
       - If contradiction detected: apply contradiction_policy
       - Else: MERGE
    2. If similarity >= dedup_threshold (very high): IGNORE (duplicate)
    3. Otherwise: ADD

    Keyword overlap uses asymmetric formula (AriadneMem style):
        overlap = len(intersection) / max(len(new_keywords), 1)

    Contradiction detection supports two modes:
    - Heuristic: Fast pattern-based detection (default)
    - LLM: More accurate but slower/costlier

    Contradiction policies:
    - preserve_both: Keep both (ADD new, don't modify existing)
    - prefer_new: MERGE (overwrite existing with new)
    - prefer_existing: IGNORE (keep existing, discard new)

    Example:
        consolidator = ConflictAwareConsolidator(config)
        decisions = await consolidator.decide_actions(new_items, existing_items)
        result = await consolidator.consolidate(new_items, existing_items)
    """

    def __init__(
        self,
        config: Optional[ConsolidationQualityConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
        llm_provider: Optional[ILLMProvider] = None,
        contradiction_prompt: Optional[str] = None,
    ):
        """
        Initialize the conflict-aware consolidator.

        Args:
            config: Consolidation configuration. Uses defaults if None.
            similarity_calculator: Calculator for text similarity.
            llm_provider: Optional LLM provider for contradiction detection.
            contradiction_prompt: Custom prompt for LLM contradiction detection.
                Uses DEFAULT_CONTRADICTION_PROMPT if not provided.
        """
        self._config = config or ConsolidationQualityConfig()
        super().__init__(
            similarity_threshold=self._config.semantic_merge_threshold,
            similarity_calculator=similarity_calculator or TextSimilarityCalculator(),
        )

        # Thresholds
        self._merge_threshold = self._config.semantic_merge_threshold
        self._keyword_threshold = self._config.keyword_overlap_threshold
        self._dedup_threshold = 0.98  # Very high similarity = duplicate

        # Contradiction settings
        self._check_contradictions = self._config.contradiction_check_enabled
        self._contradiction_policy = self._config.contradiction_policy

        # LLM contradiction detection
        self._use_llm_contradiction = self._config.use_llm_contradiction_check
        self._llm_provider = llm_provider
        self._llm_contradiction_model = self._config.llm_contradiction_model
        self._contradiction_prompt = contradiction_prompt or DEFAULT_CONTRADICTION_PROMPT

    @property
    def name(self) -> str:
        """The name of this consolidator."""
        return "conflict_aware"

    @property
    def enabled(self) -> bool:
        """Whether conflict-aware consolidation is enabled."""
        return self._config.enabled

    async def consolidate(
        self,
        new_items: List[MemoryItem],
        existing_items: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Consolidate new items with existing ones.

        Args:
            new_items: New items to add
            existing_items: Existing items in the store

        Returns:
            Items to store (filtered/merged based on decisions)
        """
        if not new_items:
            return []

        decisions = await self.decide_actions(new_items, existing_items)
        result: List[MemoryItem] = []

        for decision in decisions:
            if decision.action == ConsolidationAction.ADD:
                result.append(decision.new_item)
            elif decision.action == ConsolidationAction.MERGE:
                if decision.target_item:
                    merged = await self.merge_memories([decision.new_item, decision.target_item])
                    result.append(merged)
                else:
                    result.append(decision.new_item)
            elif decision.action == ConsolidationAction.CONFLICT:
                # Handle based on policy
                if self._contradiction_policy == "preserve_both":
                    # Mark as conflicting but add
                    decision.new_item.metadata["conflict_with"] = (
                        decision.target_item.memory_id if decision.target_item else None
                    )
                    result.append(decision.new_item)
                elif self._contradiction_policy == "prefer_new":
                    if decision.target_item:
                        merged = await self.merge_memories([decision.new_item, decision.target_item])
                        result.append(merged)
                    else:
                        result.append(decision.new_item)
                # prefer_existing: don't add new item
            # IGNORE: don't add

        return result

    async def decide_actions(
        self,
        new_items: List[MemoryItem],
        existing_items: List[MemoryItem],
    ) -> List[ConsolidationDecision]:
        """
        Decide consolidation action for each new item.

        Args:
            new_items: New items to evaluate
            existing_items: Existing items to compare against

        Returns:
            List of decisions, one per new item
        """
        decisions: List[ConsolidationDecision] = []

        for new_item in new_items:
            decision = await self._decide_single(new_item, existing_items)
            decisions.append(decision)

        return decisions

    async def _decide_single(
        self,
        new_item: MemoryItem,
        existing_items: List[MemoryItem],
    ) -> ConsolidationDecision:
        """Decide action for a single new item."""
        if not existing_items:
            return ConsolidationDecision(
                action=ConsolidationAction.ADD,
                reason="no_existing_items",
                new_item=new_item,
            )

        # Find best match
        best_match: Optional[MemoryItem] = None
        best_similarity = 0.0
        best_kw_overlap = 0.0

        new_keywords = self._get_keywords(new_item)

        for existing in existing_items:
            similarity = await self._calculate_similarity(new_item, existing)

            if similarity > best_similarity:
                existing_keywords = self._get_keywords(existing)
                kw_overlap = self._keyword_overlap(new_keywords, existing_keywords)

                best_match = existing
                best_similarity = similarity
                best_kw_overlap = kw_overlap

        # Decision logic
        if best_similarity >= self._dedup_threshold:
            return ConsolidationDecision(
                action=ConsolidationAction.IGNORE,
                reason="duplicate",
                new_item=new_item,
                target_item=best_match,
                similarity_score=best_similarity,
                keyword_overlap=best_kw_overlap,
            )

        if best_similarity >= self._merge_threshold and best_kw_overlap >= self._keyword_threshold:
            # Check for contradiction
            if self._check_contradictions and best_match:
                is_conflict = await self._check_contradiction(new_item, best_match)
                if is_conflict:
                    return ConsolidationDecision(
                        action=ConsolidationAction.CONFLICT,
                        reason="contradiction_detected",
                        new_item=new_item,
                        target_item=best_match,
                        similarity_score=best_similarity,
                        keyword_overlap=best_kw_overlap,
                        is_contradiction=True,
                    )

            return ConsolidationDecision(
                action=ConsolidationAction.MERGE,
                reason="high_similarity_and_overlap",
                new_item=new_item,
                target_item=best_match,
                similarity_score=best_similarity,
                keyword_overlap=best_kw_overlap,
            )

        return ConsolidationDecision(
            action=ConsolidationAction.ADD,
            reason="novel",
            new_item=new_item,
            target_item=best_match,
            similarity_score=best_similarity,
            keyword_overlap=best_kw_overlap,
        )

    def _get_keywords(self, item: MemoryItem) -> Set[str]:
        """
        Get keywords for a memory item.

        Follows AriadneMem approach: prefer LLM-extracted keywords stored
        with the memory item, fall back to runtime extraction if not available.

        Keyword sources (in priority order):
        1. item.metadata["keywords"] - explicit LLM-extracted keywords
        2. item.tags - tags can serve as keywords
        3. Runtime extraction from content (fallback)

        Args:
            item: The memory item to get keywords for

        Returns:
            Set of keywords (lowercase)
        """
        # Try LLM-extracted keywords from metadata first
        if "keywords" in item.metadata:
            keywords = item.metadata["keywords"]
            if isinstance(keywords, (list, set)):
                extracted = {str(k).lower() for k in keywords if k}
                if extracted:
                    return extracted

        # Try tags as keywords (common pattern)
        if item.tags:
            # Filter out system tags (often prefixed with underscore or contain colons)
            user_tags = {
                t.lower() for t in item.tags
                if t and not t.startswith("_") and ":" not in t
            }
            if user_tags:
                return user_tags

        # Fall back to runtime extraction from content
        return self._extract_keywords_from_text(item.content)

    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """
        Extract keywords from text using stopword filtering.

        This is the fallback when LLM-extracted keywords are not available.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of keywords (lowercase)
        """
        # Simple keyword extraction: lowercase words, filter stopwords
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "that",
            "which", "who", "whom", "this", "these", "those", "am", "i", "my",
            "me", "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "it", "its", "they", "them", "their", "what",
        }

        # Extract words (alphanumeric sequences)
        words = re.findall(r"\b[a-z]+\b", text.lower())

        # Filter stopwords and short words
        keywords = {w for w in words if w not in stopwords and len(w) > 2}

        return keywords

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Alias for _extract_keywords_from_text for backward compatibility.

        Used by _heuristic_conflict_check which operates on raw text strings.
        """
        return self._extract_keywords_from_text(text)

    def _keyword_overlap(
        self,
        new_keywords: Set[str],
        existing_keywords: Set[str],
    ) -> float:
        """
        Calculate asymmetric keyword overlap ratio.

        Uses AriadneMem-style formula:
            overlap = len(intersection) / max(len(new_keywords), 1)

        This is asymmetric because it measures what fraction of the NEW
        item's keywords are covered by the existing item. A new item with
        fewer keywords that are all present in an existing item will have
        high overlap, indicating it's likely a duplicate or refinement.

        Args:
            new_keywords: Keywords from the new memory item
            existing_keywords: Keywords from the existing memory item

        Returns:
            Overlap ratio from 0.0 to 1.0
        """
        if not new_keywords:
            return 0.0

        intersection = new_keywords & existing_keywords

        return len(intersection) / max(len(new_keywords), 1)

    async def _check_contradiction(
        self,
        new_item: MemoryItem,
        existing_item: MemoryItem,
    ) -> bool:
        """
        Check if two items contradict each other.

        Uses LLM-based detection if configured and available,
        otherwise falls back to heuristic detection.

        Args:
            new_item: The new memory item
            existing_item: The existing memory item

        Returns:
            True if contradiction detected
        """
        # Try LLM-based detection if enabled and provider available
        if self._use_llm_contradiction and self._llm_provider:
            try:
                return await self._llm_contradiction_check(new_item, existing_item)
            except Exception as e:
                logger.warning(
                    "LLM contradiction check failed, falling back to heuristic: %s",
                    e,
                )
                # Fall through to heuristic

        # Use heuristic detection
        return self._heuristic_conflict_check(new_item, existing_item)

    async def _llm_contradiction_check(
        self,
        item1: MemoryItem,
        item2: MemoryItem,
    ) -> bool:
        """
        Use LLM to detect contradictions between memories.

        More accurate than heuristics but slower and costlier.
        Delegates conflict resolution to the LLM similar to AriadneMem's
        approach of trusting later-dated information.

        Args:
            item1: First memory item (typically the new one)
            item2: Second memory item (typically existing)

        Returns:
            True if LLM detects a contradiction
        """
        if not self._llm_provider:
            return False

        prompt = self._contradiction_prompt.format(
            memory1=item1.content,
            memory2=item2.content,
        )

        messages = [
            ChatMessage(role="user", content=prompt),
        ]

        response = await self._llm_provider.chat(
            messages=messages,
            model=self._llm_contradiction_model,
            temperature=0.0,
            max_tokens=200,
        )

        json_str = extract_json_from_text(response.content or "")
        if not json_str:
            # Invalid response format, return False (no contradiction detected)
            logger.debug("LLM returned non-JSON response for contradiction check")
            return False

        try:
            import json
            result = json.loads(json_str)
            is_contradiction = result.get("is_contradiction", False)

            if is_contradiction:
                logger.debug(
                    "LLM detected contradiction: %s (confidence: %.2f)",
                    result.get("reason", "unknown"),
                    result.get("confidence", 0.0),
                )

            return is_contradiction

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM contradiction JSON: %s", e)
            return False

    def _heuristic_conflict_check(
        self,
        item1: MemoryItem,
        item2: MemoryItem,
    ) -> bool:
        """
        Check if two items potentially conflict using heuristics.

        Fast pattern-based detection that checks for:
        - Positive/negative sentiment contradictions
        - Numeric value contradictions
        - Boolean contradictions (is/is not, has/doesn't have)

        Args:
            item1: First memory item
            item2: Second memory item

        Returns:
            True if heuristic detects potential conflict
        """
        # Different types rarely conflict
        if item1.type != item2.type:
            return False

        content1 = item1.content.lower()
        content2 = item2.content.lower()

        # Check for sentiment contradictions
        positive = ["likes", "loves", "prefers", "enjoys", "always", "wants", "favorite"]
        negative = ["dislikes", "hates", "avoids", "never", "doesn't like", "doesn't want"]

        has_positive_1 = any(p in content1 for p in positive)
        has_negative_1 = any(n in content1 for n in negative)
        has_positive_2 = any(p in content2 for p in positive)
        has_negative_2 = any(n in content2 for n in negative)

        if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
            # Check if they're about the same thing
            kw1 = self._extract_keywords(content1)
            kw2 = self._extract_keywords(content2)
            if self._keyword_overlap(kw1, kw2) > 0.3:
                return True

        # Check for boolean contradictions
        affirmative = ["is", "has", "does", "can", "will"]
        negations = ["is not", "isn't", "has not", "hasn't", "does not", "doesn't", "cannot", "can't", "won't"]

        has_affirmative_1 = any(a in content1 for a in affirmative)
        has_negation_1 = any(n in content1 for n in negations)
        has_affirmative_2 = any(a in content2 for a in affirmative)
        has_negation_2 = any(n in content2 for n in negations)

        if (has_affirmative_1 and has_negation_2) or (has_negation_1 and has_affirmative_2):
            kw1 = self._extract_keywords(content1)
            kw2 = self._extract_keywords(content2)
            if self._keyword_overlap(kw1, kw2) > 0.4:
                return True

        # Check for numeric contradictions (e.g., "age is 25" vs "age is 30")
        numbers1 = set(re.findall(r"\b\d+\b", content1))
        numbers2 = set(re.findall(r"\b\d+\b", content2))

        if numbers1 and numbers2 and numbers1 != numbers2:
            # Check if they're about the same attribute
            kw1 = self._extract_keywords(content1)
            kw2 = self._extract_keywords(content2)
            if self._keyword_overlap(kw1, kw2) > 0.5:
                return True

        return False
