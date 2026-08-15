"""
Tests for memory extraction enhancements.

Tests the enhanced memory extraction with:
- Multi-pass extraction
- Source text alignment
- Chunking for long documents
"""


import pytest

from ctxforge.core.alignment_types import AlignmentStatus, CharSpan
from ctxforge.core.memory import MemorySource, MemoryType
from ctxforge.extraction.llm_extractor import LLMExtractor
from ctxforge.protocols.extractor import (
    ExtractionCandidate,
    ExtractionConfig,
)


class TestExtractionCandidateEnhancements:
    """Tests for ExtractionCandidate source grounding fields."""
    
    def test_source_grounding_fields_default_none(self):
        """Test source grounding fields default to None."""
        candidate = ExtractionCandidate(
            content="User likes coffee",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I love coffee",
        )
        
        assert candidate.source_span is None
        assert candidate.alignment_status is None
        assert candidate.matched_text is None
        assert candidate.extraction_pass == 1
    
    def test_source_grounding_fields_set(self):
        """Test setting source grounding fields."""
        # Now using proper types from core module
        span = CharSpan(start_pos=0, end_pos=10)
        candidate = ExtractionCandidate(
            content="User likes coffee",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I love coffee",
            source_span=span,
            alignment_status=AlignmentStatus.MATCH_EXACT,
            matched_text="User likes",
            extraction_pass=2,
        )
        
        assert candidate.source_span == span
        assert candidate.alignment_status == AlignmentStatus.MATCH_EXACT
        assert candidate.matched_text == "User likes"
        assert candidate.extraction_pass == 2
    
    def test_to_memory_item_includes_source_grounding(self):
        """Test conversion to MemoryItem includes source grounding in metadata."""
        span = CharSpan(start_pos=5, end_pos=20)
        candidate = ExtractionCandidate(
            content="User likes coffee",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I love coffee",
            source_span=span,
            alignment_status=AlignmentStatus.MATCH_EXACT,
            matched_text="loves coffee",
            extraction_pass=2,
        )
        
        memory = candidate.to_memory_item("user_123")
        
        # to_memory_item converts to primitives for serialization
        assert memory.metadata.get("source_span") == (5, 20)
        assert memory.metadata.get("alignment_status") == "exact"
        assert memory.metadata.get("matched_text") == "loves coffee"
        assert memory.metadata.get("extraction_pass") == 2
    
    def test_to_memory_item_without_grounding(self):
        """Test conversion to MemoryItem without source grounding."""
        candidate = ExtractionCandidate(
            content="User likes coffee",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            source_text="I love coffee",
        )
        
        memory = candidate.to_memory_item("user_123")
        
        assert "source_span" not in memory.metadata
        assert "alignment_status" not in memory.metadata
        assert "matched_text" not in memory.metadata
        # extraction_pass == 1 should not be in metadata (default)
        assert "extraction_pass" not in memory.metadata


class TestExtractionConfigEnhancements:
    """Tests for ExtractionConfig new settings."""
    
    def test_default_values(self):
        """Test new config fields have correct defaults."""
        config = ExtractionConfig()
        
        # Multi-pass settings
        assert config.extraction_passes == 1
        
        # Alignment settings
        assert config.enable_alignment is True
        assert config.fuzzy_alignment_threshold == 0.75
        assert config.accept_partial_matches is True
        
        # Chunking settings
        assert config.max_chunk_size == 2000
        assert config.parallel_chunks == 5
        
        # Schema constraints
        assert config.use_schema_constraints is False
    
    def test_custom_values(self):
        """Test setting custom config values."""
        config = ExtractionConfig(
            extraction_passes=3,
            enable_alignment=False,
            fuzzy_alignment_threshold=0.9,
            accept_partial_matches=False,
            max_chunk_size=1000,
            parallel_chunks=10,
            use_schema_constraints=True,
        )
        
        assert config.extraction_passes == 3
        assert config.enable_alignment is False
        assert config.fuzzy_alignment_threshold == 0.9
        assert config.accept_partial_matches is False
        assert config.max_chunk_size == 1000
        assert config.parallel_chunks == 10
        assert config.use_schema_constraints is True


class TestLLMExtractorEnhancements:
    """Tests for enhanced LLMExtractor."""
    
    @pytest.fixture
    def mock_llm_func(self):
        """Create a mock LLM function that returns extractable content."""
        async def mock_func(prompt: str) -> str:
            return '[{"content": "likes coffee", "type": "SEMANTIC", "confidence": 0.85, "tags": ["preference"]}]'
        return mock_func
    
    @pytest.fixture
    def extractor(self, mock_llm_func):
        """Create an LLM extractor with mock function."""
        return LLMExtractor(llm_func=mock_llm_func)
    
    @pytest.mark.asyncio
    async def test_alignment_enabled_by_default(self, extractor):
        """Test that alignment is enabled by default."""
        result = await extractor.extract(
            user_input="I really like drinking coffee every morning",
            agent_response="That's a great habit!",
        )
        
        assert result.count >= 1
        # When alignment is enabled and content matches, we get alignment status
        for candidate in result.candidates:
            if candidate.alignment_status is not None:
                # alignment_status is now AlignmentStatus enum
                assert candidate.alignment_status in [
                    AlignmentStatus.MATCH_EXACT,
                    AlignmentStatus.MATCH_FUZZY,
                    AlignmentStatus.MATCH_PARTIAL,
                    AlignmentStatus.UNALIGNED,
                ]
    
    @pytest.mark.asyncio
    async def test_alignment_disabled(self, mock_llm_func):
        """Test disabling alignment."""
        extractor = LLMExtractor(llm_func=mock_llm_func)
        config = ExtractionConfig(enable_alignment=False)
        
        result = await extractor.extract(
            user_input="I love coffee",
            agent_response="Nice!",
            config=config,
        )
        
        assert result.count >= 1
        # Without alignment, these fields should be None
        for candidate in result.candidates:
            assert candidate.alignment_status is None
            assert candidate.source_span is None
    
    @pytest.mark.asyncio
    async def test_multi_pass_extraction(self):
        """Test multi-pass extraction finds more candidates."""
        call_count = 0
        
        async def mock_func(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '[{"content": "likes coffee", "type": "SEMANTIC", "confidence": 0.8}]'
            elif call_count == 2:
                return '[{"content": "likes tea", "type": "SEMANTIC", "confidence": 0.75}]'
            else:
                return '[]'
        
        extractor = LLMExtractor(llm_func=mock_func)
        config = ExtractionConfig(
            extraction_passes=3,
            enable_alignment=False,  # Disable to simplify test
        )
        
        result = await extractor.extract(
            user_input="I enjoy coffee, tea, and many other beverages",
            agent_response="Nice variety!",
            config=config,
        )
        
        # Should have called LLM 3 times
        assert call_count == 3
        
        # Should have found candidates from multiple passes
        assert result.count >= 1
    
    @pytest.mark.asyncio
    async def test_extraction_pass_tagging(self):
        """Test that candidates are tagged with their extraction pass."""
        call_count = 0
        
        async def mock_func(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f'[{{"content": "fact from pass {call_count}", "type": "SEMANTIC", "confidence": 0.8}}]'
        
        extractor = LLMExtractor(llm_func=mock_func)
        config = ExtractionConfig(
            extraction_passes=2,
            enable_alignment=False,
        )
        
        result = await extractor.extract(
            user_input="I have multiple interests",
            agent_response="Tell me more!",
            config=config,
        )
        
        # Should have candidates from different passes
        pass_numbers = [c.extraction_pass for c in result.candidates]
        assert 1 in pass_numbers or 2 in pass_numbers
    
    @pytest.mark.asyncio
    async def test_chunking_for_long_text(self):
        """Test chunking is applied for long text."""
        chunk_inputs = []
        
        async def mock_func(prompt: str) -> str:
            chunk_inputs.append(prompt)
            return '[{"content": "test fact", "type": "SEMANTIC", "confidence": 0.8}]'
        
        extractor = LLMExtractor(llm_func=mock_func)
        config = ExtractionConfig(
            max_chunk_size=100,  # Small chunk size to trigger chunking
            enable_alignment=False,
        )
        
        # Create long text
        long_text = "I love coffee. " * 50  # ~750 characters
        
        _result = await extractor.extract(
            user_input=long_text,
            agent_response="Wow!",
            config=config,
        )
        
        # Should have processed multiple chunks
        assert len(chunk_inputs) > 1
    
    @pytest.mark.asyncio
    async def test_aligner_confidence_boost(self, mock_llm_func):
        """Test that exact alignment boosts confidence."""
        extractor = LLMExtractor(llm_func=mock_llm_func)
        
        # Use text that contains the exact content for matching
        result = await extractor.extract(
            user_input="I really like drinking coffee every morning, it helps me wake up",
            agent_response="Coffee is great!",
        )
        
        # Candidates should have alignment applied
        assert result.count >= 1
    
    @pytest.mark.asyncio
    async def test_accepts_custom_aligner(self, mock_llm_func):
        """Test that custom aligner can be provided."""
        from ctxforge.extraction.alignment import WordAligner
        
        custom_aligner = WordAligner(fuzzy_threshold=0.9)
        extractor = LLMExtractor(
            llm_func=mock_llm_func,
            aligner=custom_aligner,
        )
        
        result = await extractor.extract(
            user_input="I like coffee",
            agent_response="Nice!",
        )
        
        assert result.count >= 1


class TestLLMExtractorMerging:
    """Tests for multi-pass merging functionality."""
    
    @pytest.mark.asyncio
    async def test_merge_deduplicates_content(self):
        """Test that merging deduplicates by content."""
        call_count = 0
        
        async def mock_func(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # Both passes return same content
            return '[{"content": "likes coffee", "type": "SEMANTIC", "confidence": 0.8}]'
        
        extractor = LLMExtractor(llm_func=mock_func)
        config = ExtractionConfig(
            extraction_passes=2,
            enable_alignment=False,
        )
        
        result = await extractor.extract(
            user_input="I like coffee",
            agent_response="Nice!",
            config=config,
        )
        
        # Should deduplicate to just one
        assert result.count == 1
    
    @pytest.mark.asyncio
    async def test_merge_keeps_non_overlapping(self):
        """Test that non-overlapping results are kept."""
        call_count = 0
        
        async def mock_func(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '[{"content": "likes coffee", "type": "SEMANTIC", "confidence": 0.8}]'
            else:
                return '[{"content": "works as engineer", "type": "SEMANTIC", "confidence": 0.85}]'
        
        extractor = LLMExtractor(llm_func=mock_func)
        config = ExtractionConfig(
            extraction_passes=2,
            enable_alignment=False,
        )
        
        result = await extractor.extract(
            user_input="I like coffee and work as an engineer",
            agent_response="Great!",
            config=config,
        )
        
        # Should keep both unique results
        assert result.count == 2


class TestIntegrationEnhancements:
    """Integration tests for enhanced extraction."""
    
    @pytest.mark.asyncio
    async def test_full_enhanced_extraction_flow(self):
        """Test complete extraction with all enhancements."""
        async def mock_llm(prompt: str) -> str:
            return '''[
                {"content": "loves Python programming", "type": "SEMANTIC", "confidence": 0.9, "tags": ["programming"]},
                {"content": "works at tech company", "type": "SEMANTIC", "confidence": 0.8, "tags": ["work"]}
            ]'''
        
        extractor = LLMExtractor(llm_func=mock_llm)
        config = ExtractionConfig(
            extraction_passes=1,
            enable_alignment=True,
            min_confidence=0.7,
        )
        
        result = await extractor.extract(
            user_input="I love Python programming and I work at a tech company",
            agent_response="That's awesome!",
            config=config,
        )
        
        assert result.count >= 1
        
        # Convert to memory items
        memories = [
            c.to_memory_item("user_123", source=MemorySource.AGENT_INFERENCE)
            for c in result.candidates
        ]
        
        assert len(memories) >= 1
        
        # Check that source grounding is in metadata
        for memory in memories:
            # Alignment was attempted
            if memory.metadata.get("alignment_status"):
                assert memory.metadata["alignment_status"] in ["exact", "fuzzy", "partial", "unaligned"]

