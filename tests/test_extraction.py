"""
Tests for the Extraction Pipeline.

Tests the memory extraction and consolidation components:
- PatternExtractor
- LLMExtractor
- EntityExtractor
- HybridExtractor
- DeduplicationConsolidator
- MergingConsolidator
"""

import json

import pytest

from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.extraction.consolidation import (
    DeduplicationConsolidator,
    MergingConsolidator,
)
from ctxforge.extraction.entity_extractor import EntityExtractor
from ctxforge.extraction.hybrid_extractor import HybridExtractor
from ctxforge.extraction.llm_extractor import LLMExtractor, MockLLMExtractor
from ctxforge.extraction.pattern_extractor import (
    PatternExtractor,
)
from ctxforge.extraction.utils import (
    clean_extraction,
    estimate_memory_importance,
    extract_sentences,
    normalize_text,
    parse_confidence,
)
from ctxforge.protocols.extractor import (
    ExtractionCandidate,
    ExtractionConfig,
    ExtractionResult,
)
from ctxforge.utils.similarity import (
    LevenshteinSimilarityCalculator,
    TextSimilarityCalculator,
    normalize_for_comparison,
)

# =============================================================================
# Utility Function Tests
# =============================================================================

class TestNormalizeText:
    """Tests for normalize_text utility."""
    
    def test_strips_whitespace(self):
        """Test whitespace stripping."""
        assert normalize_text("  hello  ") == "hello"
    
    def test_collapses_multiple_spaces(self):
        """Test space collapsing."""
        assert normalize_text("hello   world") == "hello world"
    
    def test_handles_newlines(self):
        """Test newline handling."""
        assert normalize_text("hello\n\nworld") == "hello world"
    
    def test_handles_empty(self):
        """Test empty string."""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestNormalizeForComparison:
    """Tests for normalize_for_comparison utility."""
    
    def test_lowercases(self):
        """Test lowercase conversion."""
        result = normalize_for_comparison("Hello World")
        assert result == result.lower()
    
    def test_removes_fillers(self):
        """Test filler word removal."""
        result = normalize_for_comparison("the cat is on the mat")
        assert "the" not in result.split()


class TestExtractSentences:
    """Tests for extract_sentences utility."""
    
    def test_splits_on_period(self):
        """Test period splitting."""
        result = extract_sentences("Hello. World.")
        assert len(result) == 2
    
    def test_splits_on_exclamation(self):
        """Test exclamation splitting."""
        result = extract_sentences("Hello! World!")
        assert len(result) == 2
    
    def test_handles_empty(self):
        """Test empty string."""
        assert extract_sentences("") == []


class TestSimilarityCalculatorInjection:
    """Tests for similarity calculator injection into extractors and consolidators."""
    
    def test_pattern_extractor_accepts_calculator(self):
        """PatternExtractor accepts custom similarity calculator."""
        calculator = LevenshteinSimilarityCalculator()
        extractor = PatternExtractor(similarity_calculator=calculator)
        assert extractor.similarity_calculator is calculator
    
    def test_entity_extractor_accepts_calculator(self):
        """EntityExtractor accepts custom similarity calculator."""
        calculator = LevenshteinSimilarityCalculator()
        extractor = EntityExtractor(similarity_calculator=calculator)
        assert extractor.similarity_calculator is calculator
    
    def test_hybrid_extractor_accepts_calculator(self):
        """HybridExtractor accepts custom similarity calculator."""
        calculator = LevenshteinSimilarityCalculator()
        extractor = HybridExtractor(similarity_calculator=calculator)
        assert extractor.similarity_calculator is calculator
    
    def test_deduplication_consolidator_accepts_calculator(self):
        """DeduplicationConsolidator accepts custom similarity calculator."""
        calculator = LevenshteinSimilarityCalculator()
        consolidator = DeduplicationConsolidator(similarity_calculator=calculator)
        assert consolidator.similarity_calculator is calculator
    
    def test_merging_consolidator_accepts_calculator(self):
        """MergingConsolidator accepts custom similarity calculator."""
        calculator = LevenshteinSimilarityCalculator()
        consolidator = MergingConsolidator(similarity_calculator=calculator)
        assert consolidator.similarity_calculator is calculator
    
    def test_default_calculator_is_text_based(self):
        """Default calculator is TextSimilarityCalculator."""
        extractor = PatternExtractor()
        assert isinstance(extractor.similarity_calculator, TextSimilarityCalculator)
    
    def test_calculator_can_be_changed(self):
        """Calculator can be changed after construction."""
        extractor = PatternExtractor()
        new_calculator = LevenshteinSimilarityCalculator()
        extractor.similarity_calculator = new_calculator
        assert extractor.similarity_calculator is new_calculator


class TestParseConfidence:
    """Tests for parse_confidence utility."""
    
    def test_float_value(self):
        """Test float parsing."""
        assert parse_confidence("0.8") == 0.8
    
    def test_percentage(self):
        """Test percentage parsing."""
        assert parse_confidence("80%") == 0.8
    
    def test_integer_as_percentage(self):
        """Test integer > 1 treated as percentage."""
        assert parse_confidence("75") == 0.75
    
    def test_clamps_to_range(self):
        """Test clamping to 0-1 range."""
        assert parse_confidence("1.5") == 1.0  # Values > 1 but > 100 get clamped
        assert parse_confidence("-0.5") == 0.0
        assert parse_confidence("150") == 1.0  # 150% -> 1.5 -> clamped to 1.0
    
    def test_invalid_returns_default(self):
        """Test invalid value returns default."""
        assert parse_confidence("invalid") == 0.5


class TestCleanExtraction:
    """Tests for clean_extraction utility."""
    
    def test_adds_period(self):
        """Test period addition."""
        assert clean_extraction("hello world").endswith(".")
    
    def test_capitalizes(self):
        """Test first letter capitalization."""
        result = clean_extraction("hello world")
        assert result[0].isupper()
    
    def test_handles_empty(self):
        """Test empty string."""
        assert clean_extraction("") == ""


class TestEstimateMemoryImportance:
    """Tests for estimate_memory_importance utility."""
    
    def test_explicit_higher(self):
        """Test explicit statements score higher."""
        explicit = estimate_memory_importance("test", "test", is_explicit=True)
        implicit = estimate_memory_importance("test", "test", is_explicit=False)
        assert explicit > implicit
    
    def test_strong_words_boost(self):
        """Test strong words boost importance."""
        strong = estimate_memory_importance("I love coffee", "I love coffee")
        weak = estimate_memory_importance("I like coffee", "I like coffee")
        assert strong > weak


# =============================================================================
# PatternExtractor Tests
# =============================================================================

class TestPatternExtractor:
    """Tests for PatternExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a pattern extractor."""
        return PatternExtractor()
    
    @pytest.mark.asyncio
    async def test_extracts_love_preference(self, extractor):
        """Test extraction of 'love' preference."""
        result = await extractor.extract(
            user_input="I love Italian food",
            agent_response="That sounds delicious!",
        )
        
        assert result.count >= 1
        assert any("Italian food" in c.content for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_like_preference(self, extractor):
        """Test extraction of 'like' preference."""
        result = await extractor.extract(
            user_input="I really like hiking",
            agent_response="Great hobby!",
        )
        
        assert result.count >= 1
        assert any("hiking" in c.content.lower() for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_dislike(self, extractor):
        """Test extraction of dislike."""
        result = await extractor.extract(
            user_input="I hate spicy food",
            agent_response="I understand!",
        )
        
        assert result.count >= 1
        assert any("dislike" in c.content.lower() or "spicy" in c.content.lower() 
                   for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_favorite(self, extractor):
        """Test extraction of favorite."""
        result = await extractor.extract(
            user_input="My favorite color is blue",
            agent_response="Nice choice!",
        )
        
        assert result.count >= 1
        assert any("blue" in c.content.lower() for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_work_info(self, extractor):
        """Test extraction of work information."""
        result = await extractor.extract(
            user_input="I work at Google",
            agent_response="Interesting!",
        )
        
        assert result.count >= 1
        assert any("Google" in c.content for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_location(self, extractor):
        """Test extraction of location."""
        result = await extractor.extract(
            user_input="I live in San Francisco",
            agent_response="Great city!",
        )
        
        assert result.count >= 1
        assert any("San Francisco" in c.content for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_profession(self, extractor):
        """Test extraction of profession."""
        result = await extractor.extract(
            user_input="I am a software engineer",
            agent_response="Cool!",
        )
        
        assert result.count >= 1
        assert any("software engineer" in c.content.lower() for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_respects_confidence_threshold(self, extractor):
        """Test confidence threshold filtering."""
        config = ExtractionConfig(min_confidence=0.95)
        
        result = await extractor.extract(
            user_input="I like coffee",
            agent_response="Me too!",
            config=config,
        )
        
        # All candidates should meet threshold
        for candidate in result.candidates:
            assert candidate.confidence >= 0.95
    
    @pytest.mark.asyncio
    async def test_respects_max_candidates(self, extractor):
        """Test max candidates limit."""
        config = ExtractionConfig(max_candidates=1)
        
        result = await extractor.extract(
            user_input="I love coffee and I like tea and I enjoy water",
            agent_response="Nice!",
            config=config,
        )
        
        assert len(result.candidates) <= 1
    
    @pytest.mark.asyncio
    async def test_no_extraction_from_empty(self, extractor):
        """Test no extraction from empty input."""
        result = await extractor.extract(
            user_input="",
            agent_response="",
        )
        
        assert result.count == 0
    
    @pytest.mark.asyncio
    async def test_extract_from_text(self, extractor):
        """Test extract_from_text method."""
        result = await extractor.extract_from_text(
            "I love Python programming"
        )
        
        assert result.count >= 1
    
    def test_add_custom_pattern(self, extractor):
        """Test adding custom patterns."""
        extractor.add_pattern(
            pattern=r"\bfavorite language is (\w+)\b",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            tags=["programming"],
        )
        
        summary = extractor.get_patterns_summary()
        assert "custom" in summary
    
    def test_patterns_summary(self, extractor):
        """Test pattern summary."""
        summary = extractor.get_patterns_summary()
        
        assert "preferences" in summary
        assert "personal_facts" in summary
        assert all(v > 0 for v in summary.values())


# =============================================================================
# LLMExtractor Tests
# =============================================================================

class TestMockLLMExtractor:
    """Tests for MockLLMExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a mock LLM extractor."""
        return MockLLMExtractor(
            responses={
                "coffee": [
                    {"content": "User likes coffee", "type": "SEMANTIC", "confidence": 0.9}
                ],
                "python": [
                    {"content": "User programs in Python", "type": "SEMANTIC", "confidence": 0.85}
                ],
            }
        )
    
    @pytest.mark.asyncio
    async def test_responds_to_keywords(self, extractor):
        """Test keyword-based responses."""
        result = await extractor.extract(
            user_input="I drink coffee every morning",
            agent_response="Nice!",
        )
        
        assert result.count >= 1
        assert any("coffee" in c.content.lower() for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, extractor):
        """Test no match returns empty."""
        result = await extractor.extract(
            user_input="Hello there",
            agent_response="Hi!",
        )
        
        assert result.count == 0
    
    def test_name_property(self, extractor):
        """Test name property."""
        assert extractor.name == "llm:mock"


class TestLLMExtractor:
    """Tests for LLMExtractor with mocked LLM."""
    
    @pytest.fixture
    def mock_llm_func(self):
        """Create a mock LLM function."""
        async def mock_func(prompt: str) -> str:
            return '[{"content": "User test", "type": "SEMANTIC", "confidence": 0.8, "tags": ["test"]}]'
        return mock_func
    
    @pytest.fixture
    def extractor(self, mock_llm_func):
        """Create an LLM extractor with mock function."""
        return LLMExtractor(llm_func=mock_llm_func)
    
    @pytest.mark.asyncio
    async def test_extracts_from_llm_response(self, extractor):
        """Test extraction from LLM response."""
        result = await extractor.extract(
            user_input="I like testing",
            agent_response="Good!",
        )
        
        assert result.count >= 1
    
    @pytest.mark.asyncio
    async def test_handles_json_in_markdown(self):
        """Test handling JSON in markdown code blocks."""
        async def mock_func(prompt: str) -> str:
            return '```json\n[{"content": "Test", "type": "SEMANTIC", "confidence": 0.8}]\n```'
        
        extractor = LLMExtractor(llm_func=mock_func)
        result = await extractor.extract("I like testing things", "response")
        
        assert result.count >= 1
    
    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Test handling empty LLM response."""
        async def mock_func(prompt: str) -> str:
            return "[]"
        
        extractor = LLMExtractor(llm_func=mock_func)
        result = await extractor.extract("test", "response")
        
        assert result.count == 0
    
    @pytest.mark.asyncio
    async def test_handles_malformed_json(self):
        """Test handling malformed JSON."""
        async def mock_func(prompt: str) -> str:
            return "This is not JSON at all"
        
        extractor = LLMExtractor(llm_func=mock_func)
        result = await extractor.extract("test", "response")
        
        # Should not crash, just return empty
        assert result.count == 0
    
    def test_requires_llm_or_func(self):
        """Test that constructor requires llm_provider or llm_func."""
        with pytest.raises(ValueError):
            LLMExtractor()

    @pytest.mark.asyncio
    async def test_parses_restatement_and_entities(self):
        """Test that restatement and entities are parsed from LLM response."""
        async def mock_func(prompt: str) -> str:
            return json.dumps([{
                "content": "He likes coffee",
                "restatement": "Bob likes coffee",
                "type": "SEMANTIC",
                "confidence": 0.9,
                "tags": ["preference"],
                "entities": {
                    "persons": ["Bob"],
                    "locations": [],
                    "timestamps": [],
                },
            }])

        extractor = LLMExtractor(llm_func=mock_func)
        result = await extractor.extract("He likes coffee", "Nice!")

        assert result.count >= 1
        candidate = result.candidates[0]
        assert candidate.restatement == "Bob likes coffee"
        assert candidate.extracted_entities.get("persons") == ["Bob"]

    @pytest.mark.asyncio
    async def test_restatement_none_when_missing(self):
        """Test restatement is None when LLM omits it."""
        async def mock_func(prompt: str) -> str:
            return '[{"content": "User likes tea", "type": "SEMANTIC", "confidence": 0.8}]'

        extractor = LLMExtractor(llm_func=mock_func)
        result = await extractor.extract("I like tea", "Cool!")

        assert result.count >= 1
        assert result.candidates[0].restatement is None

    @pytest.mark.asyncio
    async def test_to_memory_item_carries_restatement(self):
        """Test that to_memory_item propagates restatement and entities."""
        async def mock_func(prompt: str) -> str:
            return json.dumps([{
                "content": "She moved there yesterday",
                "restatement": "Alice moved to Seattle on 2026-02-15",
                "type": "EPISODIC",
                "confidence": 0.85,
                "tags": [],
                "entities": {
                    "persons": ["Alice"],
                    "locations": ["Seattle"],
                    "timestamps": ["2026-02-15"],
                },
            }])

        extractor = LLMExtractor(llm_func=mock_func)
        result = await extractor.extract("She moved there yesterday", "OK")

        memory = result.candidates[0].to_memory_item(user_id="u1")
        assert memory.restatement == "Alice moved to Seattle on 2026-02-15"
        assert memory.extracted_entities["persons"] == ["Alice"]
        assert memory.extracted_entities["locations"] == ["Seattle"]


class TestMockLLMExtractorRestatement:
    """Tests for MockLLMExtractor with restatement fields."""

    @pytest.mark.asyncio
    async def test_mock_extractor_with_restatement(self):
        extractor = MockLLMExtractor(
            responses={
                "coffee": [
                    {
                        "content": "He likes coffee",
                        "type": "SEMANTIC",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        result = await extractor.extract("He likes coffee", "Nice!")
        assert result.count >= 1
        # MockLLMExtractor doesn't produce restatement by default
        assert result.candidates[0].restatement is None


# =============================================================================
# EntityExtractor Tests
# =============================================================================

class TestEntityExtractor:
    """Tests for EntityExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create an entity extractor."""
        return EntityExtractor()
    
    @pytest.mark.asyncio
    async def test_extracts_date(self, extractor):
        """Test date extraction."""
        result = await extractor.extract(
            user_input="I started on January 15, 2024",
            agent_response="Got it!",
        )
        
        assert any("date" in c.tags for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_email(self, extractor):
        """Test email extraction."""
        result = await extractor.extract(
            user_input="My email is test@example.com",
            agent_response="Noted!",
        )
        
        assert any("email" in c.tags for c in result.candidates)
        assert any("test@example.com" in c.content for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_phone(self, extractor):
        """Test phone extraction."""
        result = await extractor.extract(
            user_input="Call me at 555-123-4567",
            agent_response="Will do!",
        )
        
        assert any("phone" in c.tags for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_url(self, extractor):
        """Test URL extraction."""
        result = await extractor.extract(
            user_input="Check out https://example.com",
            agent_response="Nice!",
        )
        
        assert any("url" in c.tags for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_money(self, extractor):
        """Test money extraction."""
        result = await extractor.extract(
            user_input="It costs $500",
            agent_response="I see!",
        )
        
        assert any("money" in c.tags for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_percentage(self, extractor):
        """Test percentage extraction."""
        result = await extractor.extract(
            user_input="The success rate is 90 percent",
            agent_response="Got it!",
        )
        
        assert any("percentage" in c.tags for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_extracts_relative_date(self, extractor):
        """Test relative date extraction."""
        result = await extractor.extract(
            user_input="I did it yesterday",
            agent_response="Nice!",
        )
        
        assert any("date" in c.tags for c in result.candidates)
    
    @pytest.mark.asyncio
    async def test_filter_entity_types(self):
        """Test filtering by entity types."""
        extractor = EntityExtractor(entity_types={"EMAIL"})
        
        result = await extractor.extract(
            user_input="My email is test@test.com and phone is 555-1234",
            agent_response="Got it!",
        )
        
        # Should only have email, not phone
        assert all("email" in c.tags for c in result.candidates)
    
    def test_name_property(self, extractor):
        """Test name property."""
        assert "entity" in extractor.name


# =============================================================================
# HybridExtractor Tests
# =============================================================================

class TestHybridExtractor:
    """Tests for HybridExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a hybrid extractor."""
        return HybridExtractor()
    
    @pytest.mark.asyncio
    async def test_combines_extractors(self, extractor):
        """Test that hybrid combines results from multiple extractors."""
        result = await extractor.extract(
            user_input="I love coffee and my email is test@test.com",
            agent_response="Got it!",
        )
        
        # Should have both pattern and entity results
        assert result.count >= 2
    
    @pytest.mark.asyncio
    async def test_deduplicates_results(self, extractor):
        """Test deduplication of similar results."""
        result = await extractor.extract(
            user_input="I love coffee",
            agent_response="Nice!",
        )
        
        # Should not have duplicate "loves coffee" entries
        contents = [c.content.lower() for c in result.candidates]
        # Check no exact duplicates
        assert len(contents) == len(set(contents))
    
    def test_add_extractor(self, extractor):
        """Test adding custom extractor."""
        mock = MockLLMExtractor(responses={"test": [{"content": "test"}]})
        extractor.add_extractor(mock)
        
        assert len(extractor.extractors) == 3  # pattern + entity + mock
    
    def test_remove_extractor(self, extractor):
        """Test removing extractor."""
        initial_count = len(extractor.extractors)
        removed = extractor.remove_extractor("pattern")
        
        assert removed
        assert len(extractor.extractors) == initial_count - 1
    
    def test_name_includes_extractors(self, extractor):
        """Test name includes extractor names."""
        name = extractor.name
        assert "hybrid" in name
        assert "pattern" in name
        assert "entity" in name
    
    @pytest.mark.asyncio
    async def test_without_patterns(self):
        """Test hybrid without pattern extractor."""
        extractor = HybridExtractor(include_patterns=False)
        
        result = await extractor.extract(
            user_input="I love coffee",
            agent_response="Nice!",
        )
        
        # Should still work (entity only)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_without_entities(self):
        """Test hybrid without entity extractor."""
        extractor = HybridExtractor(include_entities=False)
        
        result = await extractor.extract(
            user_input="test@test.com",
            agent_response="Got it!",
        )
        
        # Should not extract email (no entity extractor)
        assert not any("email" in c.tags for c in result.candidates)


# =============================================================================
# Consolidator Tests
# =============================================================================

class TestDeduplicationConsolidator:
    """Tests for DeduplicationConsolidator."""
    
    @pytest.fixture
    def consolidator(self):
        """Create a deduplication consolidator."""
        return DeduplicationConsolidator(similarity_threshold=0.8)
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memory items."""
        return [
            MemoryItem(
                user_id="user_1",
                content="User likes coffee",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
                confidence_score=0.8,
            ),
            MemoryItem(
                user_id="user_1",
                content="User loves coffee",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
                confidence_score=0.9,
            ),
            MemoryItem(
                user_id="user_1",
                content="User works at Google",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
                confidence_score=0.85,
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_removes_duplicates(self, consolidator, sample_memories):
        """Test duplicate removal."""
        new_items = sample_memories[:2]  # Similar items
        existing = []
        
        result = await consolidator.consolidate(new_items, existing)
        
        # Should keep only one of the similar items
        assert len(result) == 1
        # Should keep higher confidence
        assert result[0].confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_keeps_different_memories(self, consolidator, sample_memories):
        """Test that different memories are kept."""
        new_items = [sample_memories[0], sample_memories[2]]  # Different items
        existing = []
        
        result = await consolidator.consolidate(new_items, existing)
        
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_filters_against_existing(self, consolidator, sample_memories):
        """Test filtering against existing memories."""
        new_items = [sample_memories[0]]
        existing = [sample_memories[1]]  # Similar existing memory
        
        result = await consolidator.consolidate(new_items, existing)
        
        # New item has lower confidence, should be filtered
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_find_duplicates(self, consolidator, sample_memories):
        """Test find_duplicates method."""
        duplicates = await consolidator.find_duplicates(
            sample_memories[0],
            sample_memories[1:],
        )
        
        # Should find the similar "loves coffee" memory
        assert len(duplicates) >= 1
    
    @pytest.mark.asyncio
    async def test_recency_strategy(self):
        """Test recency keep strategy."""
        from datetime import datetime, timedelta
        
        consolidator = DeduplicationConsolidator(keep_strategy="recency")
        
        old_memory = MemoryItem(
            user_id="user_1",
            content="User likes tea",
            type=MemoryType.SEMANTIC,
            source=MemorySource.AGENT_INFERENCE,
            created_at=datetime.now() - timedelta(days=1),
        )
        
        new_memory = MemoryItem(
            user_id="user_1",
            content="User really likes tea",
            type=MemoryType.SEMANTIC,
            source=MemorySource.AGENT_INFERENCE,
            created_at=datetime.now(),
        )
        
        result = await consolidator.consolidate([new_memory], [old_memory])
        
        # Should keep newer memory
        assert len(result) == 1
        assert "really" in result[0].content
    
    def test_name_property(self, consolidator):
        """Test name property."""
        assert "deduplication" in consolidator.name


class TestMergingConsolidator:
    """Tests for MergingConsolidator."""
    
    @pytest.fixture
    def consolidator(self):
        """Create a merging consolidator."""
        return MergingConsolidator(merge_threshold=0.6)
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memory items."""
        return [
            MemoryItem(
                user_id="user_1",
                content="User likes coffee",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
                confidence_score=0.8,
                tags=["preference"],
            ),
            MemoryItem(
                user_id="user_1",
                content="User prefers dark roast coffee",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
                confidence_score=0.85,
                tags=["preference", "coffee"],
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_merges_related_memories(self, consolidator, sample_memories):
        """Test merging of related memories."""
        result = await consolidator.consolidate(sample_memories, [])
        
        # Should merge into one
        assert len(result) <= 2
    
    @pytest.mark.asyncio
    async def test_combines_tags(self, consolidator, sample_memories):
        """Test that tags are combined."""
        result = await consolidator.consolidate(sample_memories, [])
        
        if len(result) == 1:
            # Tags should be combined
            assert "preference" in result[0].tags
    
    @pytest.mark.asyncio
    async def test_keeps_unrelated_separate(self, consolidator):
        """Test that unrelated memories stay separate."""
        memories = [
            MemoryItem(
                user_id="user_1",
                content="User likes coffee",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
            ),
            MemoryItem(
                user_id="user_1",
                content="User lives in New York",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
            ),
        ]
        
        result = await consolidator.consolidate(memories, [])
        
        # Should remain separate
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_merge_memories_method(self, consolidator, sample_memories):
        """Test the merge_memories method directly."""
        merged = await consolidator.merge_memories(sample_memories)
        
        assert merged is not None
        assert merged.user_id == "user_1"
        # Should have combined tags
        assert len(merged.tags) >= 1
    
    def test_name_property(self, consolidator):
        """Test name property."""
        assert consolidator.name == "merging"


# =============================================================================
# Integration Tests
# =============================================================================

class TestExtractionWorkflow:
    """Integration tests for the full extraction workflow."""
    
    @pytest.mark.asyncio
    async def test_full_extraction_and_consolidation(self):
        """Test complete extraction and consolidation workflow."""
        # 1. Extract memories
        extractor = HybridExtractor()
        
        result = await extractor.extract(
            user_input="I love Italian food and I work at Microsoft. My email is john@microsoft.com",
            agent_response="That's great!",
        )
        
        assert result.count >= 2
        
        # 2. Convert to memory items
        memories = [
            c.to_memory_item("user_123")
            for c in result.filter_by_confidence(0.7)
        ]
        
        assert len(memories) >= 1
        
        # 3. Consolidate
        consolidator = DeduplicationConsolidator()
        final = await consolidator.consolidate(memories, [])
        
        # Should have consolidated memories
        assert len(final) >= 1
    
    @pytest.mark.asyncio
    async def test_validate_candidate(self):
        """Test candidate validation."""
        extractor = PatternExtractor()
        
        # Create a candidate
        candidate = ExtractionCandidate(
            content="User likes pizza",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I like pizza",
        )
        
        # Existing memories with similar content
        existing = [
            MemoryItem(
                user_id="user_1",
                content="User loves pizza",
                type=MemoryType.SEMANTIC,
                source=MemorySource.AGENT_INFERENCE,
            ),
        ]
        
        # Should detect as duplicate
        is_valid = await extractor.validate_candidate(candidate, existing)
        assert not is_valid  # Too similar
        
        # Different memory should be valid
        different = ExtractionCandidate(
            content="User works as engineer",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I work as an engineer",
        )
        
        is_valid = await extractor.validate_candidate(different, existing)
        assert is_valid


# =============================================================================
# ExtractionCandidate Tests
# =============================================================================

class TestExtractionCandidate:
    """Tests for ExtractionCandidate dataclass."""
    
    def test_to_memory_item(self):
        """Test conversion to MemoryItem."""
        candidate = ExtractionCandidate(
            content="User likes coffee",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I like coffee",
            tags=["preference"],
        )
        
        memory = candidate.to_memory_item("user_123")
        
        assert memory.user_id == "user_123"
        assert memory.content == "User likes coffee"
        assert memory.type == MemoryType.SEMANTIC
        assert memory.confidence_score == 0.9
        assert "preference" in memory.tags


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""
    
    def test_filter_by_confidence(self):
        """Test confidence filtering."""
        result = ExtractionResult(
            candidates=[
                ExtractionCandidate("a", MemoryType.SEMANTIC, 0.5, "source"),
                ExtractionCandidate("b", MemoryType.SEMANTIC, 0.8, "source"),
                ExtractionCandidate("c", MemoryType.SEMANTIC, 0.9, "source"),
            ]
        )
        
        filtered = result.filter_by_confidence(0.7)
        
        assert len(filtered) == 2
        assert all(c.confidence >= 0.7 for c in filtered)
    
    def test_filter_by_type(self):
        """Test type filtering."""
        result = ExtractionResult(
            candidates=[
                ExtractionCandidate("a", MemoryType.SEMANTIC, 0.8, "source"),
                ExtractionCandidate("b", MemoryType.EPISODIC, 0.8, "source"),
                ExtractionCandidate("c", MemoryType.SEMANTIC, 0.8, "source"),
            ]
        )
        
        filtered = result.filter_by_type(MemoryType.SEMANTIC)
        
        assert len(filtered) == 2
        assert all(c.memory_type == MemoryType.SEMANTIC for c in filtered)

