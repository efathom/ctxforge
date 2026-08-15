"""
Tests for the ctxforge.utils module.

Tests shared utilities:
- Math utilities (cosine_similarity, euclidean_distance, etc.)
- Similarity calculators (TextSimilarityCalculator, etc.)
"""

import math
from typing import List

import pytest

from ctxforge.utils.math import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    normalize_vector,
)
from ctxforge.utils.similarity import (
    AsyncToSyncAdapter,
    EmbeddingSimilarityCalculator,
    IAsyncSimilarityCalculator,
    ISimilarityCalculator,
    LevenshteinSimilarityCalculator,
    TextSimilarityCalculator,
    calculate_text_similarity,
    normalize_for_comparison,
)

# =============================================================================
# Tests for utils/math.py
# =============================================================================

class TestCosineSimilarity:
    """Tests for cosine_similarity function."""
    
    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        result = cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 1.0
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        result = cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert result == 0.0
    
    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == -1.0
    
    def test_similar_vectors(self):
        """Similar vectors have high positive similarity."""
        result = cosine_similarity([1.0, 1.0], [1.0, 0.9])
        assert result > 0.9
    
    def test_empty_vectors(self):
        """Empty vectors return 0.0."""
        assert cosine_similarity([], []) == 0.0
    
    def test_different_length_vectors(self):
        """Different length vectors return 0.0."""
        assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0
    
    def test_zero_vectors(self):
        """Zero vectors return 0.0."""
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    
    def test_one_zero_vector(self):
        """One zero vector returns 0.0."""
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
    
    def test_normalized_vectors(self):
        """Normalized vectors give expected results."""
        # 45 degree angle
        v1 = [1.0, 0.0]
        v2 = [math.sqrt(2)/2, math.sqrt(2)/2]
        result = cosine_similarity(v1, v2)
        assert abs(result - math.sqrt(2)/2) < 0.0001


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""
    
    def test_identical_vectors(self):
        """Identical vectors have distance 0.0."""
        result = euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result == 0.0
    
    def test_unit_distance(self):
        """Unit distance calculation."""
        result = euclidean_distance([0.0, 0.0], [1.0, 0.0])
        assert result == 1.0
    
    def test_3d_distance(self):
        """3D distance calculation."""
        # Distance from origin to (1, 1, 1) = sqrt(3)
        result = euclidean_distance([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert abs(result - math.sqrt(3)) < 0.0001
    
    def test_empty_vectors(self):
        """Empty vectors return infinity."""
        assert euclidean_distance([], []) == float('inf')
    
    def test_different_length_vectors(self):
        """Different length vectors return infinity."""
        assert euclidean_distance([1.0], [1.0, 2.0]) == float('inf')
    
    def test_negative_coordinates(self):
        """Distance works with negative coordinates."""
        result = euclidean_distance([-1.0, -1.0], [1.0, 1.0])
        assert abs(result - math.sqrt(8)) < 0.0001


class TestDotProduct:
    """Tests for dot_product function."""
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors have dot product 0."""
        result = dot_product([1.0, 0.0], [0.0, 1.0])
        assert result == 0.0
    
    def test_parallel_vectors(self):
        """Parallel vectors dot product."""
        result = dot_product([2.0, 0.0], [3.0, 0.0])
        assert result == 6.0
    
    def test_general_case(self):
        """General dot product calculation."""
        result = dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert result == 32.0
    
    def test_different_length_vectors(self):
        """Different length vectors return 0."""
        assert dot_product([1.0, 2.0], [1.0]) == 0.0
    
    def test_empty_vectors(self):
        """Empty vectors return 0."""
        assert dot_product([], []) == 0.0
    
    def test_negative_values(self):
        """Dot product with negative values."""
        result = dot_product([1.0, -1.0], [-1.0, 1.0])
        # 1*(-1) + (-1)*1 = -1 - 1 = -2
        assert result == -2.0


class TestNormalizeVector:
    """Tests for normalize_vector function."""
    
    def test_unit_vector(self):
        """Unit vector remains unchanged."""
        result = normalize_vector([1.0, 0.0, 0.0])
        assert result == [1.0, 0.0, 0.0]
    
    def test_scales_to_unit(self):
        """Vector is scaled to unit length."""
        result = normalize_vector([3.0, 4.0])
        # Magnitude is 5, so result should be [0.6, 0.8]
        assert abs(result[0] - 0.6) < 0.0001
        assert abs(result[1] - 0.8) < 0.0001
        
        # Verify magnitude is 1
        magnitude = math.sqrt(sum(x*x for x in result))
        assert abs(magnitude - 1.0) < 0.0001
    
    def test_zero_vector(self):
        """Zero vector returns zero vector."""
        result = normalize_vector([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0]
    
    def test_empty_vector(self):
        """Empty vector returns empty."""
        assert normalize_vector([]) == []
    
    def test_negative_values(self):
        """Normalization with negative values."""
        result = normalize_vector([-3.0, 4.0])
        assert abs(result[0] - (-0.6)) < 0.0001
        assert abs(result[1] - 0.8) < 0.0001


# =============================================================================
# Tests for utils/similarity.py
# =============================================================================

class TestNormalizeForComparison:
    """Tests for normalize_for_comparison function."""
    
    def test_lowercases(self):
        """Text is lowercased."""
        result = normalize_for_comparison("HELLO WORLD")
        assert result == "hello world"
    
    def test_removes_filler_words(self):
        """Filler words are removed."""
        result = normalize_for_comparison("the quick brown fox")
        assert "the" not in result
        assert "quick" in result
    
    def test_collapses_whitespace(self):
        """Multiple spaces are collapsed."""
        result = normalize_for_comparison("hello   world")
        assert "  " not in result
    
    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        result = normalize_for_comparison("  hello  ")
        assert result == "hello"
    
    def test_empty_string(self):
        """Empty string returns empty."""
        assert normalize_for_comparison("") == ""
    
    def test_only_fillers(self):
        """String with only fillers returns empty."""
        result = normalize_for_comparison("the a an is are")
        assert result == ""


class TestISimilarityCalculatorProtocol:
    """Tests for ISimilarityCalculator protocol compliance."""
    
    def test_text_calculator_implements_protocol(self):
        """TextSimilarityCalculator implements ISimilarityCalculator."""
        calculator = TextSimilarityCalculator()
        assert isinstance(calculator, ISimilarityCalculator)
    
    def test_levenshtein_calculator_implements_protocol(self):
        """LevenshteinSimilarityCalculator implements ISimilarityCalculator."""
        calculator = LevenshteinSimilarityCalculator()
        assert isinstance(calculator, ISimilarityCalculator)
    
    def test_protocol_has_calculate_method(self):
        """Protocol requires calculate method."""
        calculator = TextSimilarityCalculator()
        assert hasattr(calculator, 'calculate')
        assert callable(calculator.calculate)


class TestTextSimilarityCalculator:
    """Tests for TextSimilarityCalculator."""
    
    def test_identical_texts(self):
        """Identical texts return 1.0."""
        calculator = TextSimilarityCalculator()
        assert calculator.calculate("hello world", "hello world") == 1.0
    
    def test_similar_texts(self):
        """Similar texts return high score."""
        calculator = TextSimilarityCalculator()
        score = calculator.calculate("I love coffee", "I really love coffee")
        assert score > 0.7
    
    def test_different_texts(self):
        """Different texts return low score."""
        calculator = TextSimilarityCalculator()
        score = calculator.calculate("abc", "xyz")
        assert score < 0.5
    
    def test_empty_returns_zero(self):
        """Empty strings return 0.0."""
        calculator = TextSimilarityCalculator()
        assert calculator.calculate("", "hello") == 0.0
        assert calculator.calculate("hello", "") == 0.0
        assert calculator.calculate("", "") == 0.0
    
    def test_normalization_enabled(self):
        """With normalization, filler words are ignored."""
        calculator = TextSimilarityCalculator(normalize=True)
        score = calculator.calculate("love coffee", "love the coffee")
        assert score > 0.9
    
    def test_normalization_disabled(self):
        """Without normalization, case matters."""
        calculator = TextSimilarityCalculator(normalize=False)
        score_without = calculator.calculate("HELLO", "hello")
        
        calculator_with = TextSimilarityCalculator(normalize=True)
        score_with = calculator_with.calculate("HELLO", "hello")
        
        # With normalization, they're equal
        assert score_with > score_without
    
    def test_case_insensitive_with_normalization(self):
        """Normalization makes comparison case-insensitive."""
        calculator = TextSimilarityCalculator(normalize=True)
        score = calculator.calculate("Hello World", "hello world")
        assert score == 1.0


class TestLevenshteinSimilarityCalculator:
    """Tests for LevenshteinSimilarityCalculator."""
    
    def test_identical_texts(self):
        """Identical texts return 1.0."""
        calculator = LevenshteinSimilarityCalculator()
        assert calculator.calculate("hello", "hello") == 1.0
    
    def test_one_char_difference(self):
        """Single character difference gives high but not perfect score."""
        calculator = LevenshteinSimilarityCalculator()
        score = calculator.calculate("hello", "hallo")
        assert 0.7 < score < 1.0
    
    def test_empty_returns_zero(self):
        """Empty strings return 0.0."""
        calculator = LevenshteinSimilarityCalculator()
        assert calculator.calculate("", "hello") == 0.0
        assert calculator.calculate("hello", "") == 0.0
    
    def test_completely_different(self):
        """Completely different texts return low score."""
        calculator = LevenshteinSimilarityCalculator()
        score = calculator.calculate("abc", "xyz")
        assert score < 0.5
    
    def test_normalization_works(self):
        """Normalization is applied."""
        calculator = LevenshteinSimilarityCalculator(normalize=True)
        score = calculator.calculate("HELLO", "hello")
        assert score == 1.0


class TestEmbeddingSimilarityCalculator:
    """Tests for EmbeddingSimilarityCalculator."""
    
    @pytest.mark.asyncio
    async def test_similar_embeddings(self):
        """Similar embeddings return high score."""
        async def mock_embed(text: str) -> List[float]:
            if "coffee" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "tea" in text.lower():
                return [0.95, 0.05, 0.0]  # Similar to coffee
            return [0.0, 0.0, 1.0]
        
        calculator = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        score = await calculator.calculate("I love coffee", "I love tea")
        assert score > 0.9
    
    @pytest.mark.asyncio
    async def test_different_embeddings(self):
        """Different embeddings return low score."""
        async def mock_embed(text: str) -> List[float]:
            if "coffee" in text.lower():
                return [1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0]
        
        calculator = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        score = await calculator.calculate("coffee", "something else")
        assert score < 0.5
    
    @pytest.mark.asyncio
    async def test_caches_embeddings(self):
        """Embeddings are cached to avoid redundant calls."""
        call_count = 0
        
        async def mock_embed(text: str) -> List[float]:
            nonlocal call_count
            call_count += 1
            return [1.0, 0.0, 0.0]
        
        calculator = EmbeddingSimilarityCalculator(
            embedding_func=mock_embed,
            cache_embeddings=True,
        )
        
        await calculator.calculate("hello", "world")
        await calculator.calculate("hello", "world")
        
        # Each unique text should only be embedded once
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Cache can be disabled."""
        call_count = 0
        
        async def mock_embed(text: str) -> List[float]:
            nonlocal call_count
            call_count += 1
            return [1.0, 0.0, 0.0]
        
        calculator = EmbeddingSimilarityCalculator(
            embedding_func=mock_embed,
            cache_embeddings=False,
        )
        
        await calculator.calculate("hello", "world")
        await calculator.calculate("hello", "world")
        
        # Without cache, each call embeds both texts
        assert call_count == 4
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Cache can be cleared."""
        call_count = 0
        
        async def mock_embed(text: str) -> List[float]:
            nonlocal call_count
            call_count += 1
            return [1.0, 0.0, 0.0]
        
        calculator = EmbeddingSimilarityCalculator(
            embedding_func=mock_embed,
            cache_embeddings=True,
        )
        
        await calculator.calculate("hello", "world")
        assert call_count == 2
        
        calculator.clear_cache()
        
        await calculator.calculate("hello", "world")
        assert call_count == 4  # Re-embedded after cache clear
    
    @pytest.mark.asyncio
    async def test_empty_returns_zero(self):
        """Empty strings return 0.0."""
        async def mock_embed(text: str) -> List[float]:
            return [1.0, 0.0, 0.0]
        
        calculator = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        
        assert await calculator.calculate("", "hello") == 0.0
        assert await calculator.calculate("hello", "") == 0.0
    
    @pytest.mark.asyncio
    async def test_clamps_to_zero_one(self):
        """Result is clamped to [0, 1] range."""
        async def mock_embed(text: str) -> List[float]:
            # Opposite vectors have cosine similarity of -1
            if text == "a":
                return [1.0, 0.0]
            return [-1.0, 0.0]
        
        calculator = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        score = await calculator.calculate("a", "b")
        
        # Should be clamped to 0, not -1
        assert score == 0.0


class TestAsyncToSyncAdapter:
    """Tests for AsyncToSyncAdapter."""
    
    def test_implements_sync_protocol(self):
        """Adapter implements ISimilarityCalculator."""
        async def mock_embed(text: str) -> List[float]:
            return [1.0, 0.0, 0.0]
        
        async_calc = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        sync_calc = AsyncToSyncAdapter(async_calc)
        
        assert isinstance(sync_calc, ISimilarityCalculator)
    
    def test_has_calculate_method(self):
        """Adapter has synchronous calculate method."""
        async def mock_embed(text: str) -> List[float]:
            return [1.0, 0.0, 0.0]
        
        async_calc = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        sync_calc = AsyncToSyncAdapter(async_calc)
        
        assert hasattr(sync_calc, 'calculate')
        assert callable(sync_calc.calculate)


class TestCalculateTextSimilarityFunction:
    """Tests for calculate_text_similarity convenience function."""
    
    def test_identical_texts(self):
        """Identical texts return 1.0."""
        assert calculate_text_similarity("hello", "hello") == 1.0
    
    def test_similar_texts(self):
        """Similar texts return high score."""
        score = calculate_text_similarity("I love coffee", "I really love coffee")
        assert score > 0.7
    
    def test_empty_texts(self):
        """Empty texts return 0.0."""
        assert calculate_text_similarity("", "hello") == 0.0
    
    def test_uses_default_calculator(self):
        """Function uses TextSimilarityCalculator by default."""
        # This should behave identically to TextSimilarityCalculator
        calculator = TextSimilarityCalculator()
        
        text1 = "hello world"
        text2 = "hello there"
        
        assert calculate_text_similarity(text1, text2) == calculator.calculate(text1, text2)


class TestIAsyncSimilarityCalculatorProtocol:
    """Tests for IAsyncSimilarityCalculator protocol."""
    
    def test_embedding_calculator_implements_protocol(self):
        """EmbeddingSimilarityCalculator implements IAsyncSimilarityCalculator."""
        async def mock_embed(text: str) -> List[float]:
            return [1.0, 0.0, 0.0]
        
        calculator = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        assert isinstance(calculator, IAsyncSimilarityCalculator)


# =============================================================================
# Integration tests
# =============================================================================

class TestUtilsIntegration:
    """Integration tests for utils module."""
    
    def test_import_from_utils_package(self):
        """Can import from ctxforge.utils package."""
        from ctxforge.utils import (
            TextSimilarityCalculator,
            cosine_similarity,
        )
        
        assert callable(cosine_similarity)
        assert TextSimilarityCalculator is not None
    
    def test_math_and_similarity_work_together(self):
        """Math utilities work with similarity calculators."""
        # Normalize vectors
        v1 = normalize_vector([3.0, 4.0])
        v2 = normalize_vector([4.0, 3.0])
        
        # Calculate cosine similarity
        sim = cosine_similarity(v1, v2)
        
        # These vectors at ~37 degree angle should have similarity ~0.96
        assert 0.9 < sim < 1.0
    
    @pytest.mark.asyncio
    async def test_embedding_calculator_uses_cosine_similarity(self):
        """EmbeddingSimilarityCalculator uses cosine_similarity internally."""
        # Create embeddings that we know the cosine similarity of
        async def mock_embed(text: str) -> List[float]:
            if text == "a":
                return [1.0, 0.0]
            return [0.0, 1.0]  # Orthogonal
        
        calculator = EmbeddingSimilarityCalculator(embedding_func=mock_embed)
        score = await calculator.calculate("a", "b")
        
        # Orthogonal vectors have cosine similarity 0
        assert score == 0.0

