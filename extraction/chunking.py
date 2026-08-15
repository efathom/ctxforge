"""
Intelligent text chunking for long document processing.

Splits text at sentence boundaries while respecting max buffer sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, TypeVar

from ctxforge.extraction.alignment import CharSpan, TextTokenizer, TokenSpan

T = TypeVar("T")


@dataclass
class TextChunk:
    """A chunk of text with position metadata."""
    
    text: str
    char_span: CharSpan
    token_span: TokenSpan
    document_id: Optional[str] = None
    chunk_index: int = 0
    
    @property
    def length(self) -> int:
        return len(self.text)


# Sentence-ending patterns
_SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？'}
_CLOSING_PUNCTUATION = {'"', "'", '»', ')', ']', '}'}


class ChunkIterator:
    """
    Iterates through text in optimally-sized chunks.
    
    Chunks:
    - Respect sentence boundaries when possible
    - Stay within max_char_buffer
    - Handle long sentences by breaking at newlines or tokens
    
    Usage:
        for chunk in ChunkIterator(text, max_char_buffer=2000):
            process(chunk)
    """
    
    def __init__(
        self,
        text: str,
        max_char_buffer: int = 2000,
        tokenizer: Optional[TextTokenizer] = None,
        document_id: Optional[str] = None,
    ):
        """
        Initialize chunk iterator.
        
        Args:
            text: The text to chunk
            max_char_buffer: Maximum characters per chunk
            tokenizer: Tokenizer to use
            document_id: Optional document identifier
        """
        self._tokenizer = tokenizer or TextTokenizer()
        self._tokenized = self._tokenizer.tokenize(text)
        self._max_buffer = max_char_buffer
        self._document_id = document_id
        self._chunk_index = 0
        self._current_token_idx = 0
    
    def __iter__(self) -> Iterator[TextChunk]:
        return self
    
    def __next__(self) -> TextChunk:
        if self._current_token_idx >= len(self._tokenized.tokens):
            raise StopIteration
        
        chunk = self._get_next_chunk()
        self._chunk_index += 1
        return chunk
    
    def _get_next_chunk(self) -> TextChunk:
        """Get the next chunk respecting sentence boundaries."""
        start_token_idx = self._current_token_idx
        tokens = self._tokenized.tokens
        text = self._tokenized.text
        
        if start_token_idx >= len(tokens):
            raise StopIteration
        
        # Find sentence end from current position
        sentence_end_idx = self._find_sentence_end(start_token_idx)
        
        # Check if sentence fits in buffer
        start_char = tokens[start_token_idx].char_span.start_pos
        end_char = tokens[sentence_end_idx - 1].char_span.end_pos
        
        if end_char - start_char <= self._max_buffer:
            # Try to add more sentences
            current_end_idx = sentence_end_idx
            
            while current_end_idx < len(tokens):
                next_sentence_end = self._find_sentence_end(current_end_idx)
                next_end_char = tokens[next_sentence_end - 1].char_span.end_pos
                
                if next_end_char - start_char > self._max_buffer:
                    break
                
                current_end_idx = next_sentence_end
            
            self._current_token_idx = current_end_idx
            end_token_idx = current_end_idx
        else:
            # Sentence too long - break at newline or buffer limit
            end_token_idx = self._break_long_sentence(start_token_idx, sentence_end_idx)
            self._current_token_idx = end_token_idx
        
        # Build chunk
        start_pos = tokens[start_token_idx].char_span.start_pos
        end_pos = tokens[end_token_idx - 1].char_span.end_pos
        
        return TextChunk(
            text=text[start_pos:end_pos],
            char_span=CharSpan(start_pos=start_pos, end_pos=end_pos),
            token_span=TokenSpan(start_index=start_token_idx, end_index=end_token_idx),
            document_id=self._document_id,
            chunk_index=self._chunk_index,
        )
    
    def _find_sentence_end(self, start_idx: int) -> int:
        """Find the end of the sentence starting at start_idx."""
        tokens = self._tokenized.tokens
        text = self._tokenized.text
        
        for i in range(start_idx, len(tokens)):
            token = tokens[i]
            token_text = text[token.char_span.start_pos:token.char_span.end_pos]
            
            # Check for sentence ending punctuation
            if any(token_text.endswith(p) for p in _SENTENCE_ENDINGS):
                # Include following closing punctuation
                end_idx = i + 1
                while end_idx < len(tokens):
                    next_text = text[tokens[end_idx].char_span.start_pos:tokens[end_idx].char_span.end_pos]
                    if next_text in _CLOSING_PUNCTUATION:
                        end_idx += 1
                    else:
                        break
                return end_idx
            
            # Check for newline break
            if i + 1 < len(tokens) and tokens[i + 1].is_after_newline:
                next_text = text[tokens[i + 1].char_span.start_pos:tokens[i + 1].char_span.end_pos]
                if next_text and next_text[0].isupper():
                    return i + 1
        
        return len(tokens)
    
    def _break_long_sentence(self, start_idx: int, end_idx: int) -> int:
        """Break a long sentence at appropriate points."""
        tokens = self._tokenized.tokens
        start_char = tokens[start_idx].char_span.start_pos
        
        # Try to break at newline first
        last_newline_idx = None
        
        for i in range(start_idx, min(end_idx, len(tokens))):
            end_char = tokens[i].char_span.end_pos
            
            if tokens[i].is_after_newline and i > start_idx:
                if end_char - start_char <= self._max_buffer:
                    last_newline_idx = i
            
            if end_char - start_char > self._max_buffer:
                if last_newline_idx:
                    return last_newline_idx
                # No good break point, just break at buffer limit
                return max(start_idx + 1, i)
        
        return end_idx


def make_batches(
    chunks: Iterator[TextChunk],
    batch_size: int = 10,
) -> Iterator[List[TextChunk]]:
    """
    Group chunks into batches for parallel processing.
    
    Args:
        chunks: Iterator of text chunks
        batch_size: Number of chunks per batch
        
    Yields:
        Lists of chunks
    """
    batch: List[TextChunk] = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch


def sliding_window(
    items: List[T],
    window_size: int,
    stride: Optional[int] = None,
) -> List[List[T]]:
    """Split a list into overlapping windows.

    Args:
        items: The items to window over.
        window_size: Maximum number of items per window.
        stride: How far to advance between windows.  Defaults to
            ``window_size`` (no overlap).  A stride smaller than
            ``window_size`` produces overlapping windows.

    Returns:
        List of windows, each a sub-list of *items*.
    """
    if not items or window_size <= 0:
        return []

    effective_stride = stride if stride is not None else window_size
    if effective_stride <= 0:
        effective_stride = window_size

    windows: List[List[T]] = []
    start = 0
    while start < len(items):
        windows.append(items[start : start + window_size])
        start += effective_stride
    return windows

