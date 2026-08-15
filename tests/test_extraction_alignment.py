"""Tests for text alignment module."""


from ctxforge.extraction.alignment import (
    AlignmentStatus,
    CharSpan,
    TextTokenizer,
    TokenizedText,
    TokenSpan,
    WordAligner,
    merge_non_overlapping_spans,
)


class TestCharSpan:
    """Tests for CharSpan class."""
    
    def test_length(self):
        """Test length property."""
        span = CharSpan(start_pos=5, end_pos=15)
        assert span.length == 10
    
    def test_overlaps_true(self):
        """Test overlapping spans."""
        span1 = CharSpan(0, 10)
        span2 = CharSpan(5, 15)
        assert span1.overlaps(span2) is True
        assert span2.overlaps(span1) is True
    
    def test_overlaps_false_adjacent(self):
        """Test adjacent spans don't overlap."""
        span1 = CharSpan(0, 10)
        span2 = CharSpan(10, 20)
        assert span1.overlaps(span2) is False
        assert span2.overlaps(span1) is False
    
    def test_overlaps_false_separate(self):
        """Test separate spans don't overlap."""
        span1 = CharSpan(0, 10)
        span2 = CharSpan(15, 25)
        assert span1.overlaps(span2) is False
    
    def test_to_tuple(self):
        """Test to_tuple conversion."""
        span = CharSpan(5, 15)
        assert span.to_tuple() == (5, 15)


class TestTextTokenizer:
    """Tests for TextTokenizer class."""
    
    def test_tokenize_simple(self):
        """Test basic tokenization."""
        tokenizer = TextTokenizer()
        result = tokenizer.tokenize("Hello world!")
        
        assert len(result.tokens) == 3
        assert result.tokens[0].text == "hello"
        assert result.tokens[1].text == "world"
        assert result.tokens[2].text == "!"
    
    def test_tokenize_positions(self):
        """Test that token positions are correct."""
        tokenizer = TextTokenizer()
        result = tokenizer.tokenize("The quick brown fox")
        
        assert result.tokens[0].char_span.start_pos == 0
        assert result.tokens[0].char_span.end_pos == 3
        assert result.tokens[1].char_span.start_pos == 4
        assert result.tokens[1].char_span.end_pos == 9
    
    def test_tokenize_newline_detection(self):
        """Test newline detection between tokens."""
        tokenizer = TextTokenizer()
        result = tokenizer.tokenize("Hello\nWorld")
        
        assert len(result.tokens) == 2
        assert result.tokens[0].is_after_newline is False
        assert result.tokens[1].is_after_newline is True
    
    def test_tokenize_numbers(self):
        """Test number tokenization."""
        tokenizer = TextTokenizer()
        result = tokenizer.tokenize("The year is 2024")
        
        assert len(result.tokens) == 4
        assert result.tokens[3].text == "2024"
    
    def test_tokenize_punctuation(self):
        """Test punctuation handling."""
        tokenizer = TextTokenizer()
        result = tokenizer.tokenize("Hello, world!")
        
        assert len(result.tokens) == 4
        assert result.tokens[1].text == ","
        assert result.tokens[3].text == "!"
    
    def test_tokenize_empty_string(self):
        """Test empty string tokenization."""
        tokenizer = TextTokenizer()
        result = tokenizer.tokenize("")
        
        assert len(result.tokens) == 0


class TestTokenizedText:
    """Tests for TokenizedText class."""
    
    def test_get_text_for_span(self):
        """Test extracting text for a token span."""
        tokenizer = TextTokenizer()
        tokenized = tokenizer.tokenize("The quick brown fox jumps")
        
        # Get "quick brown"
        span = TokenSpan(start_index=1, end_index=3)
        text = tokenized.get_text_for_span(span)
        assert text == "quick brown"
    
    def test_get_text_for_span_empty_tokens(self):
        """Test with empty tokens."""
        tokenized = TokenizedText(text="", tokens=[])
        span = TokenSpan(start_index=0, end_index=1)
        assert tokenized.get_text_for_span(span) == ""


class TestWordAligner:
    """Tests for WordAligner class."""
    
    def test_exact_match(self):
        """Test exact token sequence matching."""
        aligner = WordAligner()
        result = aligner.align(
            "quick brown",
            "The quick brown fox jumps"
        )
        
        assert result.status == AlignmentStatus.MATCH_EXACT
        assert result.char_span is not None
        assert result.char_span.start_pos == 4
        assert result.confidence == 1.0
        assert result.matched_text == "quick brown"
    
    def test_exact_match_at_start(self):
        """Test exact match at beginning of text."""
        aligner = WordAligner()
        result = aligner.align(
            "The quick",
            "The quick brown fox"
        )
        
        assert result.status == AlignmentStatus.MATCH_EXACT
        assert result.char_span is not None
        assert result.char_span.start_pos == 0
    
    def test_exact_match_at_end(self):
        """Test exact match at end of text."""
        aligner = WordAligner()
        result = aligner.align(
            "fox jumps",
            "The quick brown fox jumps"
        )
        
        assert result.status == AlignmentStatus.MATCH_EXACT
        assert result.matched_text == "fox jumps"
    
    def test_fuzzy_match(self):
        """Test fuzzy matching with minor differences."""
        aligner = WordAligner(fuzzy_threshold=0.7)
        result = aligner.align(
            "quick browns",  # 's' added
            "The quick brown fox jumps"
        )
        
        # Should still find a fuzzy match
        assert result.status in [AlignmentStatus.MATCH_FUZZY, AlignmentStatus.MATCH_EXACT, AlignmentStatus.MATCH_PARTIAL]
    
    def test_no_match(self):
        """Test when no match is found."""
        aligner = WordAligner()
        result = aligner.align(
            "completely different text",
            "The quick brown fox"
        )
        
        assert result.status == AlignmentStatus.UNALIGNED
    
    def test_partial_match(self):
        """Test partial matching."""
        aligner = WordAligner(accept_partial=True)
        result = aligner.align(
            "quick brown fox jumps over the lazy dog",  # extends beyond source
            "The quick brown fox"
        )
        
        # Should find partial match
        assert result.status in [AlignmentStatus.MATCH_PARTIAL, AlignmentStatus.MATCH_EXACT]
    
    def test_empty_extraction(self):
        """Test with empty extraction text."""
        aligner = WordAligner()
        result = aligner.align("", "The quick brown fox")
        
        assert result.status == AlignmentStatus.UNALIGNED
    
    def test_empty_source(self):
        """Test with empty source text."""
        aligner = WordAligner()
        result = aligner.align("quick brown", "")
        
        assert result.status == AlignmentStatus.UNALIGNED
    
    def test_char_offset(self):
        """Test character offset adjustment."""
        aligner = WordAligner()
        result = aligner.align(
            "quick brown",
            "The quick brown fox",
            char_offset=100
        )
        
        assert result.status == AlignmentStatus.MATCH_EXACT
        assert result.char_span is not None
        assert result.char_span.start_pos == 104  # 100 + 4
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        aligner = WordAligner()
        result = aligner.align(
            "QUICK BROWN",
            "The quick brown fox"
        )
        
        assert result.status == AlignmentStatus.MATCH_EXACT
    
    def test_normalize_token_plural_stemming(self):
        """Test light plural stemming in normalization."""
        assert WordAligner._normalize_token("cats") == "cat"
        assert WordAligner._normalize_token("dogs") == "dog"
        assert WordAligner._normalize_token("bass") == "bass"  # ends with 'ss'
        assert WordAligner._normalize_token("a") == "a"  # too short


class TestMergeNonOverlappingSpans:
    """Tests for merge_non_overlapping_spans function."""
    
    def test_empty_input(self):
        """Test with empty input."""
        result = merge_non_overlapping_spans([])
        assert result == []
    
    def test_single_pass(self):
        """Test with single pass."""
        spans = [(CharSpan(0, 10), "a"), (CharSpan(20, 30), "b")]
        result = merge_non_overlapping_spans([spans])
        
        assert len(result) == 2
    
    def test_merge_non_overlapping(self):
        """Test merging non-overlapping spans from multiple passes."""
        pass1 = [(CharSpan(0, 10), "a")]
        pass2 = [(CharSpan(20, 30), "b")]
        
        result = merge_non_overlapping_spans([pass1, pass2])
        
        assert len(result) == 2
    
    def test_first_pass_wins(self):
        """Test that first pass wins for overlapping spans."""
        pass1 = [(CharSpan(0, 10), "first")]
        pass2 = [(CharSpan(5, 15), "second")]
        
        result = merge_non_overlapping_spans([pass1, pass2])
        
        assert len(result) == 1
        assert result[0][1] == "first"
    
    def test_complex_merge(self):
        """Test complex merging scenario."""
        pass1 = [(CharSpan(0, 10), "a"), (CharSpan(30, 40), "c")]
        pass2 = [(CharSpan(5, 15), "overlap"), (CharSpan(20, 25), "b")]
        
        result = merge_non_overlapping_spans([pass1, pass2])
        
        # Should have: a (0-10), b (20-25), c (30-40)
        # "overlap" should be excluded because it overlaps with "a"
        assert len(result) == 3
        extractions = [r[1] for r in result]
        assert "a" in extractions
        assert "b" in extractions
        assert "c" in extractions
        assert "overlap" not in extractions

