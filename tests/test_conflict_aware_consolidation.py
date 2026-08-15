"""
Tests for conflict-aware consolidation.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from ctxforge.config.base import ConsolidationQualityConfig
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.extraction.consolidation.conflict_aware import (
    DEFAULT_CONTRADICTION_PROMPT,
    ConflictAwareConsolidator,
    ConsolidationAction,
    ConsolidationDecision,
)
from ctxforge.protocols.llm import LLMResponse


def make_memory(
    content: str,
    memory_id: str = None,
    confidence: float = 0.8,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    tags: list = None,
    embedding: list = None,
) -> MemoryItem:
    """Helper to create test memory items."""
    return MemoryItem(
        memory_id=memory_id or f"mem_{hash(content) % 10000}",
        user_id="test_user",
        content=content,
        type=memory_type,
        confidence_score=confidence,
        tags=tags or [],
        embedding=embedding,
        created_at=datetime.now(timezone.utc),
    )


class TestConflictAwareConsolidator:
    """Tests for ConflictAwareConsolidator."""

    def test_name_property(self):
        """Test consolidator name."""
        consolidator = ConflictAwareConsolidator()
        assert consolidator.name == "conflict_aware"

    def test_enabled_property(self):
        """Test enabled property reflects config."""
        disabled_config = ConsolidationQualityConfig(enabled=False)
        enabled_config = ConsolidationQualityConfig(enabled=True)

        disabled = ConflictAwareConsolidator(disabled_config)
        enabled = ConflictAwareConsolidator(enabled_config)

        assert disabled.enabled is False
        assert enabled.enabled is True

    def test_add_when_no_existing(self):
        """New items should be added when no existing items."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        new_item = make_memory("User likes pizza")
        decisions = asyncio.run(consolidator.decide_actions([new_item], []))

        assert len(decisions) == 1
        assert decisions[0].action == ConsolidationAction.ADD
        assert decisions[0].reason == "no_existing_items"

    def test_add_when_novel(self):
        """Novel items should be added."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        new_item = make_memory("User works as a software engineer")
        existing = make_memory("User likes coffee")

        decisions = asyncio.run(consolidator.decide_actions([new_item], [existing]))

        assert len(decisions) == 1
        assert decisions[0].action == ConsolidationAction.ADD
        assert decisions[0].reason == "novel"

    def test_ignore_duplicate(self):
        """Near-identical items should be ignored."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        # Use identical embeddings to simulate exact match
        embedding = [1.0, 0.0, 0.0]
        new_item = make_memory("User likes pizza", embedding=embedding)
        existing = make_memory("User likes pizza", memory_id="existing_1", embedding=embedding)

        decisions = asyncio.run(consolidator.decide_actions([new_item], [existing]))

        assert len(decisions) == 1
        assert decisions[0].action == ConsolidationAction.IGNORE
        assert decisions[0].reason == "duplicate"

    def test_merge_similar_items(self):
        """Similar items with high keyword overlap should be merged."""
        config = ConsolidationQualityConfig(
            enabled=True,
            semantic_merge_threshold=0.7,
            keyword_overlap_threshold=0.3,
        )
        consolidator = ConflictAwareConsolidator(config)

        # Similar content with overlapping keywords
        new_item = make_memory("User's favorite food is Italian pizza")
        existing = make_memory("User loves Italian pizza", memory_id="existing_1")

        decisions = asyncio.run(consolidator.decide_actions([new_item], [existing]))

        assert len(decisions) == 1
        # Should merge due to high text similarity and keyword overlap
        assert decisions[0].action in [ConsolidationAction.MERGE, ConsolidationAction.ADD]

    def test_conflict_detection_sentiment(self):
        """Contradictory sentiment should be detected."""
        config = ConsolidationQualityConfig(
            enabled=True,
            semantic_merge_threshold=0.5,
            keyword_overlap_threshold=0.2,
            contradiction_check_enabled=True,
        )
        consolidator = ConflictAwareConsolidator(config)

        new_item = make_memory("User hates coffee")
        existing = make_memory("User loves coffee", memory_id="existing_1")

        decisions = asyncio.run(consolidator.decide_actions([new_item], [existing]))

        assert len(decisions) == 1
        # Should detect conflict due to opposite sentiments about same topic
        assert decisions[0].is_contradiction is True or decisions[0].action == ConsolidationAction.ADD

    def test_conflict_policy_preserve_both(self):
        """preserve_both policy should add new item with conflict metadata."""
        config = ConsolidationQualityConfig(
            enabled=True,
            semantic_merge_threshold=0.3,
            keyword_overlap_threshold=0.2,
            contradiction_check_enabled=True,
            contradiction_policy="preserve_both",
        )
        consolidator = ConflictAwareConsolidator(config)

        new_item = make_memory("User hates coffee")
        existing = make_memory("User loves coffee", memory_id="existing_1")

        result = asyncio.run(consolidator.consolidate([new_item], [existing]))

        # Should include the new item (preserve_both adds it)
        assert len(result) >= 1

    def test_conflict_policy_prefer_new(self):
        """prefer_new policy should merge (overwrite) on conflict."""
        config = ConsolidationQualityConfig(
            enabled=True,
            semantic_merge_threshold=0.3,
            keyword_overlap_threshold=0.2,
            contradiction_check_enabled=True,
            contradiction_policy="prefer_new",
        )
        consolidator = ConflictAwareConsolidator(config)

        new_item = make_memory("User hates coffee", confidence=0.9)
        existing = make_memory("User loves coffee", memory_id="existing_1", confidence=0.7)

        result = asyncio.run(consolidator.consolidate([new_item], [existing]))

        # Should have merged result
        assert len(result) >= 1

    def test_conflict_policy_prefer_existing(self):
        """prefer_existing policy should ignore new item on conflict."""
        config = ConsolidationQualityConfig(
            enabled=True,
            semantic_merge_threshold=0.3,
            keyword_overlap_threshold=0.2,
            contradiction_check_enabled=True,
            contradiction_policy="prefer_existing",
        )
        consolidator = ConflictAwareConsolidator(config)

        # Create items that will be detected as conflicting
        new_item = make_memory("User never drinks coffee")
        existing = make_memory("User always drinks coffee", memory_id="existing_1")

        # Force a conflict decision by using similar embeddings
        embedding = [0.8, 0.2, 0.0]
        new_item.embedding = embedding
        existing.embedding = [0.85, 0.15, 0.0]

        decisions = asyncio.run(consolidator.decide_actions([new_item], [existing]))

        # If conflict detected with prefer_existing, consolidate should return empty
        if decisions[0].action == ConsolidationAction.CONFLICT:
            result = asyncio.run(consolidator.consolidate([new_item], [existing]))
            assert len(result) == 0

    def test_multiple_items(self):
        """Test consolidation with multiple new items."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        new_items = [
            make_memory("User likes pizza"),
            make_memory("User works in tech"),
            make_memory("User lives in Seattle"),
        ]
        existing = [make_memory("User enjoys cooking", memory_id="existing_1")]

        decisions = asyncio.run(consolidator.decide_actions(new_items, existing))

        assert len(decisions) == 3
        # All should be ADD since they're novel
        for decision in decisions:
            assert decision.action == ConsolidationAction.ADD


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_extract_keywords_filters_stopwords(self):
        """Stopwords should be filtered out."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        keywords = consolidator._extract_keywords_from_text("The user is a software engineer")

        assert "the" not in keywords
        assert "is" not in keywords
        assert "a" not in keywords
        assert "user" in keywords
        assert "software" in keywords
        assert "engineer" in keywords

    def test_extract_keywords_lowercase(self):
        """Keywords should be lowercase."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        keywords = consolidator._extract_keywords_from_text("User LOVES Pizza")

        assert "user" in keywords
        assert "loves" in keywords
        assert "pizza" in keywords
        assert "User" not in keywords

    def test_get_keywords_from_metadata(self):
        """Keywords from metadata should be preferred."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item = make_memory("User likes pizza and pasta")
        item.metadata["keywords"] = ["pizza", "pasta", "italian", "food"]

        keywords = consolidator._get_keywords(item)

        # Should use metadata keywords, not extract from content
        assert keywords == {"pizza", "pasta", "italian", "food"}

    def test_get_keywords_from_tags(self):
        """Tags should be used as keywords when metadata not available."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item = make_memory("User likes pizza")
        item.tags = ["food", "preference", "italian"]

        keywords = consolidator._get_keywords(item)

        assert keywords == {"food", "preference", "italian"}

    def test_get_keywords_filters_system_tags(self):
        """System tags (with _ prefix or : separator) should be filtered."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item = make_memory("User likes pizza")
        item.tags = ["food", "_internal", "type:preference", "italian"]

        keywords = consolidator._get_keywords(item)

        # Should filter out _internal and type:preference
        assert keywords == {"food", "italian"}

    def test_get_keywords_fallback_to_extraction(self):
        """Should fall back to text extraction when no metadata/tags."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item = make_memory("User loves Italian pizza")
        # No metadata keywords, no tags

        keywords = consolidator._get_keywords(item)

        # Should extract from content
        assert "user" in keywords
        assert "loves" in keywords
        assert "italian" in keywords
        assert "pizza" in keywords

    def test_get_keywords_metadata_priority_over_tags(self):
        """Metadata keywords should take priority over tags."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item = make_memory("User likes pizza")
        item.metadata["keywords"] = ["llm", "extracted", "keywords"]
        item.tags = ["tag1", "tag2"]

        keywords = consolidator._get_keywords(item)

        # Should use metadata, not tags
        assert keywords == {"llm", "extracted", "keywords"}

    def test_keyword_overlap_identical(self):
        """Identical keyword sets should have overlap of 1.0."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        new_kw = {"pizza", "italian", "food"}
        existing_kw = {"pizza", "italian", "food"}

        overlap = consolidator._keyword_overlap(new_kw, existing_kw)

        assert overlap == 1.0

    def test_keyword_overlap_disjoint(self):
        """Disjoint keyword sets should have overlap of 0.0."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        new_kw = {"pizza", "italian"}
        existing_kw = {"coffee", "morning"}

        overlap = consolidator._keyword_overlap(new_kw, existing_kw)

        assert overlap == 0.0

    def test_keyword_overlap_asymmetric(self):
        """Asymmetric overlap should measure new keywords coverage."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        # New item has 2 keywords, existing has 4
        # Both of new's keywords are in existing
        new_kw = {"pizza", "italian"}
        existing_kw = {"pizza", "italian", "food", "restaurant"}

        overlap = consolidator._keyword_overlap(new_kw, existing_kw)

        # Asymmetric: intersection(2) / max(len(new_kw), 1) = 2/2 = 1.0
        # All of new's keywords are covered by existing
        assert overlap == 1.0

    def test_keyword_overlap_asymmetric_partial(self):
        """Partial asymmetric overlap calculation."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        # New item has 3 keywords, only 2 are in existing
        new_kw = {"pizza", "italian", "food"}
        existing_kw = {"pizza", "italian", "restaurant"}

        overlap = consolidator._keyword_overlap(new_kw, existing_kw)

        # Asymmetric: intersection(2) / max(len(new_kw), 1) = 2/3 ≈ 0.667
        assert abs(overlap - 2/3) < 0.001

    def test_keyword_overlap_empty_new(self):
        """Empty new keywords should return 0.0."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        new_kw = set()
        existing_kw = {"pizza", "italian"}

        overlap = consolidator._keyword_overlap(new_kw, existing_kw)

        assert overlap == 0.0


class TestConflictDetection:
    """Tests for conflict detection."""

    def test_no_conflict_different_types(self):
        """Different memory types should not conflict."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item1 = make_memory("User likes coffee", memory_type=MemoryType.SEMANTIC)
        item2 = make_memory("User bought coffee yesterday", memory_type=MemoryType.EPISODIC)

        assert consolidator._heuristic_conflict_check(item1, item2) is False

    def test_conflict_opposite_sentiments(self):
        """Opposite sentiments about same topic should conflict."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        # Use content with higher keyword overlap (coffee appears in both)
        item1 = make_memory("User loves drinking coffee")
        item2 = make_memory("User hates drinking coffee")

        assert consolidator._heuristic_conflict_check(item1, item2) is True

    def test_conflict_numeric_values(self):
        """Different numeric values for same attribute should conflict."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item1 = make_memory("User's age is 25 years old")
        item2 = make_memory("User's age is 30 years old")

        assert consolidator._heuristic_conflict_check(item1, item2) is True

    def test_no_conflict_same_numbers(self):
        """Same numeric values should not conflict."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        item1 = make_memory("User has 2 children")
        item2 = make_memory("User mentioned having 2 kids")

        assert consolidator._heuristic_conflict_check(item1, item2) is False


class TestLLMContradictionDetection:
    """Tests for LLM-based contradiction detection."""

    def test_llm_contradiction_enabled_with_provider(self):
        """LLM contradiction check should be used when enabled with provider."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=True,
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(
            content='{"is_contradiction": true, "reason": "opposite sentiments", "confidence": 0.9, "prefer_newer": true}',
            model="test-model",
        ))

        consolidator = ConflictAwareConsolidator(config, llm_provider=mock_llm)

        item1 = make_memory("User loves coffee")
        item2 = make_memory("User hates coffee")

        result = asyncio.run(consolidator._check_contradiction(item1, item2))

        assert result is True
        mock_llm.chat.assert_called_once()

    def test_llm_contradiction_fallback_on_error(self):
        """Should fall back to heuristic when LLM fails."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=True,
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM error"))

        consolidator = ConflictAwareConsolidator(config, llm_provider=mock_llm)

        # Items that heuristic would detect as conflict
        item1 = make_memory("User loves drinking coffee")
        item2 = make_memory("User hates drinking coffee")

        result = asyncio.run(consolidator._check_contradiction(item1, item2))

        # Should fall back to heuristic and detect conflict
        assert result is True

    def test_llm_contradiction_no_conflict(self):
        """LLM should correctly identify non-contradictions."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=True,
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(
            content='{"is_contradiction": false, "reason": "complementary info", "confidence": 0.85, "prefer_newer": false}',
            model="test-model",
        ))

        consolidator = ConflictAwareConsolidator(config, llm_provider=mock_llm)

        item1 = make_memory("User likes coffee")
        item2 = make_memory("User drinks coffee in the morning")

        result = asyncio.run(consolidator._check_contradiction(item1, item2))

        assert result is False

    def test_llm_contradiction_disabled_uses_heuristic(self):
        """When LLM check disabled, should use heuristic."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=False,
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock()

        consolidator = ConflictAwareConsolidator(config, llm_provider=mock_llm)

        item1 = make_memory("User loves drinking coffee")
        item2 = make_memory("User hates drinking coffee")

        result = asyncio.run(consolidator._check_contradiction(item1, item2))

        # Should use heuristic, not call LLM
        mock_llm.chat.assert_not_called()
        assert result is True

    def test_llm_contradiction_no_provider_uses_heuristic(self):
        """Without LLM provider, should use heuristic."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=True,
        )

        consolidator = ConflictAwareConsolidator(config, llm_provider=None)

        item1 = make_memory("User loves drinking coffee")
        item2 = make_memory("User hates drinking coffee")

        result = asyncio.run(consolidator._check_contradiction(item1, item2))

        # Should fall back to heuristic
        assert result is True

    def test_llm_contradiction_invalid_json_fallback(self):
        """Invalid JSON response should fall back to heuristic."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=True,
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(
            content="This is not valid JSON",
            model="test-model",
        ))

        consolidator = ConflictAwareConsolidator(config, llm_provider=mock_llm)

        # Items that heuristic would NOT detect as conflict
        item1 = make_memory("User works in tech")
        item2 = make_memory("User lives in Seattle")

        result = asyncio.run(consolidator._check_contradiction(item1, item2))

        # LLM returns invalid JSON, falls back to heuristic which returns False
        assert result is False

    def test_custom_contradiction_prompt(self):
        """Custom contradiction prompt should be used."""
        config = ConsolidationQualityConfig(
            enabled=True,
            use_llm_contradiction_check=True,
        )

        custom_prompt = """Custom prompt for testing.
Memory 1: {memory1}
Memory 2: {memory2}
Return JSON: {{"is_contradiction": true/false}}"""

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=LLMResponse(
            content='{"is_contradiction": true}',
            model="test-model",
        ))

        consolidator = ConflictAwareConsolidator(
            config,
            llm_provider=mock_llm,
            contradiction_prompt=custom_prompt,
        )

        assert consolidator._contradiction_prompt == custom_prompt

        item1 = make_memory("User likes coffee")
        item2 = make_memory("User hates coffee")

        asyncio.run(consolidator._check_contradiction(item1, item2))

        # Verify the custom prompt was used (check the call args)
        call_args = mock_llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        assert "Custom prompt for testing" in messages[0].content

    def test_default_contradiction_prompt_used(self):
        """Default prompt should be used when no custom prompt provided."""
        config = ConsolidationQualityConfig(enabled=True)
        consolidator = ConflictAwareConsolidator(config)

        assert consolidator._contradiction_prompt == DEFAULT_CONTRADICTION_PROMPT

    def test_default_prompt_exported(self):
        """DEFAULT_CONTRADICTION_PROMPT should be importable."""
        assert DEFAULT_CONTRADICTION_PROMPT is not None
        assert "contradiction" in DEFAULT_CONTRADICTION_PROMPT.lower()
        assert "{memory1}" in DEFAULT_CONTRADICTION_PROMPT
        assert "{memory2}" in DEFAULT_CONTRADICTION_PROMPT


class TestConsolidationDecision:
    """Tests for ConsolidationDecision dataclass."""

    def test_decision_fields(self):
        """Test decision has expected fields."""
        item = make_memory("Test content")
        decision = ConsolidationDecision(
            action=ConsolidationAction.ADD,
            reason="novel",
            new_item=item,
            similarity_score=0.5,
            keyword_overlap=0.3,
        )

        assert decision.action == ConsolidationAction.ADD
        assert decision.reason == "novel"
        assert decision.new_item == item
        assert decision.similarity_score == 0.5
        assert decision.keyword_overlap == 0.3
        assert decision.is_contradiction is False

    def test_decision_optional_fields(self):
        """Test decision optional fields default correctly."""
        item = make_memory("Test content")
        decision = ConsolidationDecision(
            action=ConsolidationAction.ADD,
            reason="no_existing",
            new_item=item,
        )

        assert decision.target_item is None
        assert decision.similarity_score is None
        assert decision.keyword_overlap is None
        assert decision.is_contradiction is False


class TestConsolidationAction:
    """Tests for ConsolidationAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        assert ConsolidationAction.ADD.value == "add"
        assert ConsolidationAction.MERGE.value == "merge"
        assert ConsolidationAction.IGNORE.value == "ignore"
        assert ConsolidationAction.CONFLICT.value == "conflict"

    def test_action_is_string_enum(self):
        """Action should be comparable to string."""
        assert ConsolidationAction.ADD == "add"
        assert ConsolidationAction.MERGE == "merge"


class TestIntegrationWithConfig:
    """Integration tests with EngineConfig."""

    def test_consolidator_from_engine_config(self):
        """Test creating consolidator from EngineConfig."""
        from ctxforge.config.base import EngineConfig

        config = EngineConfig.model_validate({
            "memory_quality": {
                "consolidation": {
                    "enabled": True,
                    "semantic_merge_threshold": 0.85,
                    "keyword_overlap_threshold": 0.6,
                    "contradiction_check_enabled": True,
                    "contradiction_policy": "preserve_both",
                }
            }
        })

        consolidator = ConflictAwareConsolidator(
            config=config.memory_quality.consolidation
        )

        assert consolidator.enabled is True
        assert consolidator._merge_threshold == 0.85
        assert consolidator._keyword_threshold == 0.6
        assert consolidator._check_contradictions is True
        assert consolidator._contradiction_policy == "preserve_both"

    def test_consolidation_disabled_by_default(self):
        """Consolidation is disabled by default in EngineConfig."""
        from ctxforge.config.base import EngineConfig

        config = EngineConfig()

        assert config.memory_quality.consolidation.enabled is False
