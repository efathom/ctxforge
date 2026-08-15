"""
Skill Matcher.

This module provides trigger-based and semantic skill matching
for finding relevant skills based on user queries.

Supports composite scoring that incorporates trigger matching,
semantic similarity, evaluation quality, effectiveness metrics,
and relationship boosts for comprehensive skill ranking.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Set

from ctxforge.core.skill import (
    SkillMatch,
    SkillMetadata,
    SkillRelationship,
    SkillRelationType,
)
from ctxforge.utils.math import cosine_similarity

logger = logging.getLogger(__name__)

# Default weights for composite scoring
DEFAULT_WEIGHTS = {
    "trigger": 0.35,
    "semantic": 0.25,
    "evaluation": 0.15,
    "effectiveness": 0.15,
    "relationship": 0.10,
}


class SkillMatcher:
    """
    Matches user queries to relevant skills.

    Supports multiple matching strategies:
    1. Trigger pattern matching (substring, regex)
    2. Semantic matching via embeddings (if provider available)
    3. LLM-based matching (if provider available)
    4. Evaluation quality scoring
    5. Effectiveness-based scoring
    6. Relationship boost for companion skills
    """

    def __init__(
        self,
        embedding_provider: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the matcher.

        Args:
            embedding_provider: Optional embedding provider for semantic matching
            llm_provider: Optional LLM provider for intelligent matching
            weights: Optional custom weights for composite scoring factors.
                     Keys: trigger, semantic, evaluation, effectiveness, relationship.
        """
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        # Cache for skill description embeddings
        self._embedding_cache: Dict[str, List[float]] = {}

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    async def match(
        self,
        query: str,
        available_skills: List[SkillMetadata],
        threshold: float = 0.7,
        active_skill_names: Optional[List[str]] = None,
        relationships: Optional[List[SkillRelationship]] = None,
    ) -> List[SkillMatch]:
        """
        Match query to skills using composite scoring.

        The final score is computed as a weighted combination of:
        - trigger_score (pattern matching on triggers)
        - semantic_score (embedding similarity, if provider available)
        - evaluation_score (from skill evaluation overall_score)
        - effectiveness_score (from skill effectiveness success_rate)
        - relationship_boost (bonus for compose_with/depend_on active skills)

        Args:
            query: The user query to match against
            available_skills: List of available skills
            threshold: Minimum confidence threshold (0.0 - 1.0)
            active_skill_names: Names of currently active skills (for relationship boost)
            relationships: Known skill relationships (for relationship boost)

        Returns:
            List of SkillMatch objects sorted by confidence (descending)
        """
        if not available_skills:
            return []

        active_names: Set[str] = set(active_skill_names or [])
        rels = relationships or []

        # 1. Pattern matching on triggers
        trigger_scores = self._compute_trigger_scores(query, available_skills)

        # 2. Semantic matching (if embedding provider available)
        semantic_scores: Dict[str, float] = {}
        if self._embedding_provider:
            semantic_scores = await self._compute_semantic_scores(
                query, available_skills
            )

        # 3. Build composite scores
        matches: List[SkillMatch] = []
        for skill in available_skills:
            t_score = trigger_scores.get(skill.name, 0.0)
            s_score = semantic_scores.get(skill.name, 0.0)
            e_score = self._get_evaluation_score(skill)
            eff_score = self._get_effectiveness_score(skill)
            r_boost = self._get_relationship_boost(skill.name, active_names, rels)

            # Compute weighted sum and normalize by the total weight of
            # factors that actually contributed a non-zero score, so that
            # trigger-only matches are not penalized when other signals
            # are unavailable.
            weighted_sum = 0.0
            active_weight = 0.0
            factors = [
                (t_score, "trigger"),
                (s_score, "semantic"),
                (e_score, "evaluation"),
                (eff_score, "effectiveness"),
                (r_boost, "relationship"),
            ]
            for score, key in factors:
                w = self._weights.get(key, 0.0)
                weighted_sum += score * w
                if score > 0:
                    active_weight += w

            if active_weight > 0:
                final_score = weighted_sum / active_weight
            else:
                final_score = 0.0

            if final_score >= threshold:
                reason_parts = []
                if t_score > 0:
                    reason_parts.append(f"trigger={t_score:.2f}")
                if s_score > 0:
                    reason_parts.append(f"semantic={s_score:.2f}")
                if e_score > 0:
                    reason_parts.append(f"eval={e_score:.2f}")
                if eff_score > 0:
                    reason_parts.append(f"eff={eff_score:.2f}")
                if r_boost > 0:
                    reason_parts.append(f"rel_boost={r_boost:.2f}")

                matches.append(SkillMatch(
                    skill=skill,
                    confidence=min(1.0, final_score),
                    matched_trigger=trigger_scores.get(
                        f"_trigger_{skill.name}"
                    ),
                    match_reason=f"Composite({', '.join(reason_parts)})",
                ))

        matches.sort(key=lambda m: -m.confidence)
        return matches

    def _compute_trigger_scores(
        self,
        query: str,
        skills: List[SkillMetadata],
    ) -> Dict[str, float]:
        """Compute trigger match scores for each skill.

        Returns a dict mapping skill name -> score, plus
        '_trigger_<name>' -> matched trigger string.
        """
        scores: Dict[str, Any] = {}
        pattern_matches = self._match_triggers(query, skills)
        for m in pattern_matches:
            scores[m.skill.name] = m.confidence
            scores[f"_trigger_{m.skill.name}"] = m.matched_trigger
        return scores

    async def _compute_semantic_scores(
        self,
        query: str,
        skills: List[SkillMetadata],
    ) -> Dict[str, float]:
        """Compute semantic similarity scores for each skill."""
        scores: Dict[str, float] = {}
        semantic_matches = await self._semantic_match(query, skills)
        for m in semantic_matches:
            scores[m.skill.name] = m.confidence
        return scores

    def _get_evaluation_score(self, skill: SkillMetadata) -> float:
        """Extract evaluation overall_score from skill metadata.

        The evaluation data is stored on the full Skill object. SkillMetadata
        carries it via the metadata dict when available.
        """
        meta = getattr(skill, "metadata", None)
        if isinstance(meta, dict):
            eval_data = meta.get("evaluation")
            if isinstance(eval_data, dict):
                score = eval_data.get("overall_score", 0.0)
                if isinstance(score, (int, float)):
                    return float(score)
        return 0.0

    def _get_effectiveness_score(self, skill: SkillMetadata) -> float:
        """Extract effectiveness success_rate from skill metadata."""
        meta = getattr(skill, "metadata", None)
        if isinstance(meta, dict):
            eff_data = meta.get("effectiveness")
            if isinstance(eff_data, dict):
                rate = eff_data.get("success_rate", 0.0)
                if isinstance(rate, (int, float)):
                    return float(rate)
        return 0.0

    def _get_relationship_boost(
        self,
        skill_name: str,
        active_names: Set[str],
        relationships: List[SkillRelationship],
    ) -> float:
        """Compute relationship boost for a skill.

        A skill gets a boost of 1.0 if it has a COMPOSE_WITH or DEPEND_ON
        relationship with any currently active skill.
        """
        if not active_names or not relationships:
            return 0.0

        for rel in relationships:
            if rel.relation_type not in (
                SkillRelationType.COMPOSE_WITH,
                SkillRelationType.DEPEND_ON,
            ):
                continue
            if rel.source == skill_name and rel.target in active_names:
                return 1.0
            if rel.target == skill_name and rel.source in active_names:
                return 1.0
        return 0.0

    def _match_triggers(
        self,
        query: str,
        skills: List[SkillMetadata]
    ) -> List[SkillMatch]:
        """
        Match query against skill triggers using pattern matching.

        Args:
            query: The user query
            skills: List of skills to match against

        Returns:
            List of matches with confidence scores
        """
        matches: List[SkillMatch] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for skill in skills:
            best_trigger = None
            best_confidence = 0.0
            best_reason = ""

            for trigger in skill.triggers:
                trigger_lower = trigger.lower()
                confidence = 0.0
                reason = ""

                # Strategy 1: Exact substring match
                if trigger_lower in query_lower:
                    # Calculate confidence based on how much of the query the trigger covers
                    coverage = len(trigger_lower) / len(query_lower)
                    confidence = min(0.9, 0.6 + coverage * 0.3)
                    reason = f"Trigger '{trigger}' found in query"

                # Strategy 2: Word overlap
                else:
                    trigger_words = set(trigger_lower.split())
                    overlap = query_words & trigger_words
                    if overlap:
                        word_confidence = len(overlap) / len(trigger_words)
                        if word_confidence >= 0.5:
                            confidence = 0.5 + word_confidence * 0.2
                            reason = f"Words {overlap} match trigger '{trigger}'"

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_trigger = trigger
                    best_reason = reason

            if best_confidence > 0:
                matches.append(SkillMatch(
                    skill=skill,
                    confidence=best_confidence,
                    matched_trigger=best_trigger,
                    match_reason=best_reason,
                ))

        return matches

    async def _semantic_match(
        self,
        query: str,
        skills: List[SkillMetadata]
    ) -> List[SkillMatch]:
        """
        Match query against skills using semantic similarity.

        Args:
            query: The user query
            skills: List of skills to match against

        Returns:
            List of matches with semantic similarity scores
        """
        if not self._embedding_provider:
            return []

        matches: List[SkillMatch] = []

        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)

            for skill in skills:
                # Create a searchable text from skill
                skill_text = f"{skill.name} {skill.description} {' '.join(skill.triggers)}"
                skill_embedding = await self._get_embedding(skill_text, cache_key=skill.name)

                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, skill_embedding)

                if similarity > 0.3:  # Lower threshold for semantic matches
                    matches.append(SkillMatch(
                        skill=skill,
                        confidence=similarity,
                        matched_trigger=None,
                        match_reason=f"Semantic similarity: {similarity:.2f}",
                    ))

        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")

        return matches

    async def _get_embedding(
        self,
        text: str,
        cache_key: Optional[str] = None
    ) -> List[float]:
        """
        Get embedding for text, with optional caching.

        Args:
            text: Text to embed
            cache_key: Optional key for caching

        Returns:
            Embedding vector
        """
        if cache_key and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        embedding = await self._embedding_provider.embed(text)

        if cache_key:
            self._embedding_cache[cache_key] = embedding

        return embedding

    def _merge_matches(
        self,
        pattern_matches: List[SkillMatch],
        semantic_matches: List[SkillMatch]
    ) -> List[SkillMatch]:
        """
        Merge pattern and semantic matches, combining confidence scores.

        Args:
            pattern_matches: Matches from trigger patterns
            semantic_matches: Matches from semantic similarity

        Returns:
            Merged matches with combined confidence
        """
        # Build lookup by skill name
        merged: Dict[str, SkillMatch] = {}

        # Add pattern matches (higher weight)
        for match in pattern_matches:
            merged[match.skill.name] = match

        # Add/combine semantic matches
        for semantic_match in semantic_matches:
            name = semantic_match.skill.name
            if name in merged:
                # Combine scores (pattern gets 60%, semantic 40%)
                existing = merged[name]
                combined_confidence = (
                    existing.confidence * 0.6 +
                    semantic_match.confidence * 0.4
                )
                merged[name] = SkillMatch(
                    skill=existing.skill,
                    confidence=min(1.0, combined_confidence),
                    matched_trigger=existing.matched_trigger,
                    match_reason=(
                        f"{existing.match_reason}; "
                        f"Semantic boost: {semantic_match.confidence:.2f}"
                    ),
                )
            else:
                # Pure semantic match (lower base confidence)
                merged[name] = SkillMatch(
                    skill=semantic_match.skill,
                    confidence=semantic_match.confidence * 0.8,
                    matched_trigger=None,
                    match_reason=semantic_match.match_reason,
                )

        return list(merged.values())


class RegexSkillMatcher(SkillMatcher):
    """
    Skill matcher that supports regex triggers.

    Triggers starting with ^ or containing regex metacharacters
    are treated as regex patterns.
    """

    # Regex metacharacters (excluding common punctuation)
    REGEX_CHARS = re.compile(r'[\\^$*+?{}[\]|()]')

    def _is_regex_trigger(self, trigger: str) -> bool:
        """Check if a trigger is a regex pattern."""
        return bool(self.REGEX_CHARS.search(trigger))

    def _match_triggers(
        self,
        query: str,
        skills: List[SkillMetadata]
    ) -> List[SkillMatch]:
        """Override to support regex triggers."""
        matches: List[SkillMatch] = []
        query_lower = query.lower()

        for skill in skills:
            best_trigger = None
            best_confidence = 0.0
            best_reason = ""

            for trigger in skill.triggers:
                confidence = 0.0
                reason = ""

                if self._is_regex_trigger(trigger):
                    # Regex matching
                    try:
                        pattern = re.compile(trigger, re.IGNORECASE)
                        match = pattern.search(query)
                        if match:
                            # Confidence based on match length
                            match_len = match.end() - match.start()
                            confidence = min(0.95, 0.7 + match_len / len(query) * 0.25)
                            reason = f"Regex '{trigger}' matched: {match.group()}"
                    except re.error:
                        logger.warning(f"Invalid regex trigger: {trigger}")
                        continue
                else:
                    # Standard substring matching
                    trigger_lower = trigger.lower()
                    if trigger_lower in query_lower:
                        coverage = len(trigger_lower) / len(query_lower)
                        confidence = min(0.9, 0.6 + coverage * 0.3)
                        reason = f"Trigger '{trigger}' found in query"

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_trigger = trigger
                    best_reason = reason

            if best_confidence > 0:
                matches.append(SkillMatch(
                    skill=skill,
                    confidence=best_confidence,
                    matched_trigger=best_trigger,
                    match_reason=best_reason,
                ))

        return matches
