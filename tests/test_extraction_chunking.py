"""Tests for text chunking module."""


from ctxforge.extraction.alignment import CharSpan, TokenSpan
from ctxforge.extraction.chunking import (
    ChunkIterator,
    TextChunk,
    make_batches,
)


class TestTextChunk:
    """Tests for TextChunk class."""
    
    def test_length_property(self):
        """Test length property."""
        chunk = TextChunk(
            text="Hello world",
            char_span=CharSpan(0, 11),
            token_span=TokenSpan(0, 2),
        )
        assert chunk.length == 11
    
    def test_chunk_index(self):
        """Test chunk index."""
        chunk = TextChunk(
            text="Hello",
            char_span=CharSpan(0, 5),
            token_span=TokenSpan(0, 1),
            chunk_index=5,
        )
        assert chunk.chunk_index == 5
    
    def test_document_id(self):
        """Test document ID."""
        chunk = TextChunk(
            text="Hello",
            char_span=CharSpan(0, 5),
            token_span=TokenSpan(0, 1),
            document_id="doc-123",
        )
        assert chunk.document_id == "doc-123"


class TestChunkIterator:
    """Tests for ChunkIterator class."""
    
    def test_single_sentence(self):
        """Test text with a single short sentence."""
        text = "This is a test sentence."
        chunks = list(ChunkIterator(text, max_char_buffer=100))
        
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_respects_buffer_limit(self):
        """Test that chunks respect buffer limit."""
        text = "Short sentence. " * 20
        chunks = list(ChunkIterator(text, max_char_buffer=50))
        
        for chunk in chunks:
            assert chunk.length <= 50
    
    def test_respects_sentence_boundaries(self):
        """Test that chunks break at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = list(ChunkIterator(text, max_char_buffer=40))
        
        # Each chunk should end with a sentence
        for chunk in chunks:
            assert chunk.text.rstrip().endswith(".")
    
    def test_chunk_positions(self):
        """Test that chunk positions are correct."""
        text = "First. Second. Third."
        chunks = list(ChunkIterator(text, max_char_buffer=10))
        
        # Verify first chunk starts at 0
        assert chunks[0].char_span.start_pos == 0
        
        # Verify positions are sequential (non-overlapping)
        for i in range(1, len(chunks)):
            # The start of this chunk should be after the previous chunk's content
            prev_end = chunks[i-1].char_span.end_pos
            curr_start = chunks[i].char_span.start_pos
            assert curr_start >= prev_end
    
    def test_chunk_indices(self):
        """Test that chunk indices are sequential."""
        text = "First. Second. Third. Fourth. Fifth."
        chunks = list(ChunkIterator(text, max_char_buffer=15))
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_document_id_propagation(self):
        """Test that document ID is propagated to chunks."""
        text = "First. Second."
        chunks = list(ChunkIterator(text, max_char_buffer=10, document_id="my-doc"))
        
        for chunk in chunks:
            assert chunk.document_id == "my-doc"
    
    def test_empty_text(self):
        """Test with empty text."""
        text = ""
        chunks = list(ChunkIterator(text, max_char_buffer=100))
        
        assert len(chunks) == 0
    
    def test_long_sentence_break(self):
        """Test breaking a very long sentence."""
        # Create a long sentence that exceeds buffer
        text = "word " * 100  # ~500 characters
        chunks = list(ChunkIterator(text, max_char_buffer=50))
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be within limit
        for chunk in chunks:
            assert chunk.length <= 50
    
    def test_newline_handling(self):
        """Test handling of newlines."""
        text = "First paragraph text\nSecond paragraph text"
        chunks = list(ChunkIterator(text, max_char_buffer=30))
        
        # Should handle newlines properly
        assert len(chunks) >= 1
    
    def test_multiple_sentences_fit(self):
        """Test combining multiple sentences that fit."""
        text = "A. B. C."
        chunks = list(ChunkIterator(text, max_char_buffer=100))
        
        # All should fit in one chunk
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_unicode_text(self):
        """Test with Unicode text."""
        text = "你好世界。这是测试。"
        chunks = list(ChunkIterator(text, max_char_buffer=50))
        
        assert len(chunks) >= 1


class TestMakeBatches:
    """Tests for make_batches function."""
    
    def test_exact_batch_size(self):
        """Test when chunks divide evenly into batches."""
        text = "A. B. C. D."
        chunks = ChunkIterator(text, max_char_buffer=5)
        batches = list(make_batches(chunks, batch_size=2))
        
        # Should have 2 batches of 2
        for batch in batches[:-1]:  # All but possibly last
            assert len(batch) <= 2
    
    def test_partial_final_batch(self):
        """Test when final batch is partial."""
        text = "A. B. C."
        chunks = ChunkIterator(text, max_char_buffer=5)
        batches = list(make_batches(chunks, batch_size=2))
        
        # Should have at least one batch
        assert len(batches) >= 1
    
    def test_single_item_batches(self):
        """Test batch size of 1."""
        text = "A. B. C."
        chunks = ChunkIterator(text, max_char_buffer=5)
        batches = list(make_batches(chunks, batch_size=1))
        
        for batch in batches:
            assert len(batch) == 1
    
    def test_large_batch_size(self):
        """Test when batch size exceeds chunk count."""
        text = "A. B."
        chunks = ChunkIterator(text, max_char_buffer=5)
        batches = list(make_batches(chunks, batch_size=100))
        
        # Should have just one batch
        assert len(batches) == 1
    
    def test_empty_chunks(self):
        """Test with empty chunk iterator."""
        text = ""
        chunks = ChunkIterator(text, max_char_buffer=100)
        batches = list(make_batches(chunks, batch_size=10))
        
        assert len(batches) == 0

