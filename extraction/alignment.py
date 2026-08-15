"""
Text alignment utilities for source grounding.

Provides exact and fuzzy matching to locate extracted content
within source text for provenance tracking.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, List, Optional, Tuple

# Import core types from the core module (no circular dependency)
from ctxforge.core.alignment_types import (
    AlignmentResult,
    AlignmentStatus,
    CharSpan,
    TokenSpan,
)

# Re-export for backwards compatibility
__all__ = [
    "AlignmentStatus",
    "CharSpan", 
    "TokenSpan",
    "AlignmentResult",
    "Token",
    "TokenizedText",
    "TextTokenizer",
    "WordAligner",
    "merge_non_overlapping_spans",
]


@dataclass
class Token:
    """A token with position information."""
    
    index: int
    text: str
    char_span: CharSpan
    is_after_newline: bool = False


@dataclass
class TokenizedText:
    """Text broken into tokens with position tracking."""
    
    text: str
    tokens: List[Token] = field(default_factory=list)
    
    def get_text_for_span(self, token_span: TokenSpan) -> str:
        """Get the original text for a token span."""
        if not self.tokens:
            return ""
        start_token = self.tokens[token_span.start_index]
        end_token = self.tokens[token_span.end_index - 1]
        return self.text[start_token.char_span.start_pos:end_token.char_span.end_pos]


class TextTokenizer:
    """
    Tokenizer for alignment purposes.
    
    Splits text into word-level tokens while preserving position information.
    Supports both regex-based (fast, English-optimized) and Unicode-aware modes.
    """
    
    def __init__(self, unicode_aware: bool = False):
        """
        Initialize tokenizer.
        
        Args:
            unicode_aware: If True, use Unicode-aware tokenization for
                          better CJK/Arabic/Thai support. Slower.
        """
        self._unicode_aware = unicode_aware
        # Pattern for word/number/punctuation tokens
        self._pattern = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+", re.UNICODE)
    
    def tokenize(self, text: str) -> TokenizedText:
        """
        Tokenize text into tokens with position information.
        
        Args:
            text: The text to tokenize
            
        Returns:
            TokenizedText with tokens and their positions
        """
        result = TokenizedText(text=text, tokens=[])
        
        prev_end = 0
        for idx, match in enumerate(self._pattern.finditer(text)):
            start, end = match.span()
            
            # Check for newline in gap
            is_after_newline = '\n' in text[prev_end:start] or '\r' in text[prev_end:start]
            
            token = Token(
                index=idx,
                text=match.group().lower(),  # Lowercase for matching
                char_span=CharSpan(start_pos=start, end_pos=end),
                is_after_newline=is_after_newline if idx > 0 else False,
            )
            result.tokens.append(token)
            prev_end = end
        
        return result


class WordAligner:
    """
    Aligns extracted text to source text positions.
    
    Uses difflib.SequenceMatcher for exact matching with fuzzy fallback.
    """
    
    def __init__(
        self,
        tokenizer: Optional[TextTokenizer] = None,
        fuzzy_threshold: float = 0.75,
        enable_fuzzy: bool = True,
        accept_partial: bool = True,
    ):
        """
        Initialize the aligner.
        
        Args:
            tokenizer: Tokenizer to use. Defaults to TextTokenizer.
            fuzzy_threshold: Minimum ratio for fuzzy match (0.0-1.0)
            enable_fuzzy: Whether to attempt fuzzy matching on failure
            accept_partial: Whether to accept partial matches
        """
        self._tokenizer = tokenizer or TextTokenizer()
        self._fuzzy_threshold = fuzzy_threshold
        self._enable_fuzzy = enable_fuzzy
        self._accept_partial = accept_partial
        self._matcher = difflib.SequenceMatcher(autojunk=False)
    
    def align(
        self,
        extraction_text: str,
        source_text: str,
        char_offset: int = 0,
    ) -> AlignmentResult:
        """
        Align an extraction to its position in source text.
        
        Args:
            extraction_text: The extracted text to align
            source_text: The source text to search in
            char_offset: Character offset for the source text chunk
            
        Returns:
            AlignmentResult with status and positions
        """
        if not extraction_text or not source_text:
            return AlignmentResult(status=AlignmentStatus.UNALIGNED)
        
        # Tokenize both texts
        source_tokenized = self._tokenizer.tokenize(source_text)
        extraction_tokens = [t.text for t in self._tokenizer.tokenize(extraction_text).tokens]
        source_tokens = [t.text for t in source_tokenized.tokens]
        
        if not extraction_tokens or not source_tokens:
            return AlignmentResult(status=AlignmentStatus.UNALIGNED)
        
        # Try exact matching first
        result = self._exact_align(
            extraction_tokens, source_tokens, source_tokenized, char_offset
        )
        
        if result.status != AlignmentStatus.UNALIGNED:
            return result
        
        # Try fuzzy matching if enabled
        if self._enable_fuzzy:
            result = self._fuzzy_align(
                extraction_tokens, source_tokens, source_tokenized, char_offset
            )
        
        return result
    
    def _exact_align(
        self,
        extraction_tokens: List[str],
        source_tokens: List[str],
        source_tokenized: TokenizedText,
        char_offset: int,
    ) -> AlignmentResult:
        """Attempt exact token sequence matching using difflib."""
        self._matcher.set_seqs(source_tokens, extraction_tokens)
        
        for src_idx, ext_idx, match_len in self._matcher.get_matching_blocks()[:-1]:
            # Check if we matched from the start of extraction
            if ext_idx == 0 and match_len > 0:
                # Check if it's an exact or partial match
                is_exact = (match_len == len(extraction_tokens))
                
                if is_exact or self._accept_partial:
                    start_token = source_tokenized.tokens[src_idx]
                    end_token = source_tokenized.tokens[src_idx + match_len - 1]
                    
                    char_span = CharSpan(
                        start_pos=char_offset + start_token.char_span.start_pos,
                        end_pos=char_offset + end_token.char_span.end_pos,
                    )
                    token_span = TokenSpan(
                        start_index=src_idx,
                        end_index=src_idx + match_len,
                    )
                    matched_text = source_tokenized.get_text_for_span(token_span)
                    
                    status = AlignmentStatus.MATCH_EXACT if is_exact else AlignmentStatus.MATCH_PARTIAL
                    
                    return AlignmentResult(
                        status=status,
                        char_span=char_span,
                        token_span=token_span,
                        confidence=1.0 if is_exact else (match_len / len(extraction_tokens)),
                        matched_text=matched_text,
                    )
        
        return AlignmentResult(status=AlignmentStatus.UNALIGNED)
    
    def _fuzzy_align(
        self,
        extraction_tokens: List[str],
        source_tokens: List[str],
        source_tokenized: TokenizedText,
        char_offset: int,
    ) -> AlignmentResult:
        """Attempt fuzzy alignment using sliding window matching."""
        extraction_normalized = [self._normalize_token(t) for t in extraction_tokens]
        len_e = len(extraction_tokens)
        
        if len_e == 0:
            return AlignmentResult(status=AlignmentStatus.UNALIGNED)
        
        extraction_counts = Counter(extraction_normalized)
        min_overlap = int(len_e * self._fuzzy_threshold)
        
        best_ratio = 0.0
        best_span: Optional[Tuple[int, int]] = None
        
        matcher = difflib.SequenceMatcher(autojunk=False, b=extraction_normalized)
        
        # Try windows of different sizes
        for window_size in range(len_e, min(len(source_tokens) + 1, len_e * 2)):
            if window_size > len(source_tokens):
                break
            
            # Sliding window
            window = deque([self._normalize_token(t) for t in source_tokens[:window_size]])
            window_counts = Counter(window)
            
            for start_idx in range(len(source_tokens) - window_size + 1):
                # Quick overlap check
                if (extraction_counts & window_counts).total() >= min_overlap:
                    matcher.set_seq1(list(window))
                    matches = sum(size for _, _, size in matcher.get_matching_blocks())
                    ratio = matches / len_e
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_span = (start_idx, window_size)
                
                # Slide window
                if start_idx + window_size < len(source_tokens):
                    old_token = window.popleft()
                    old_norm = self._normalize_token(old_token) if isinstance(old_token, str) else old_token
                    window_counts[old_norm] -= 1
                    if window_counts[old_norm] == 0:
                        del window_counts[old_norm]
                    
                    new_token = self._normalize_token(source_tokens[start_idx + window_size])
                    window.append(new_token)
                    window_counts[new_token] += 1
        
        if best_span and best_ratio >= self._fuzzy_threshold:
            start_idx, window_size = best_span
            start_token = source_tokenized.tokens[start_idx]
            end_token = source_tokenized.tokens[start_idx + window_size - 1]
            
            char_span = CharSpan(
                start_pos=char_offset + start_token.char_span.start_pos,
                end_pos=char_offset + end_token.char_span.end_pos,
            )
            token_span = TokenSpan(
                start_index=start_idx,
                end_index=start_idx + window_size,
            )
            matched_text = source_tokenized.get_text_for_span(token_span)
            
            return AlignmentResult(
                status=AlignmentStatus.MATCH_FUZZY,
                char_span=char_span,
                token_span=token_span,
                confidence=best_ratio,
                matched_text=matched_text,
            )
        
        return AlignmentResult(status=AlignmentStatus.UNALIGNED)
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def _normalize_token(token: str) -> str:
        """Normalize token for comparison (lowercase, light stemming)."""
        token = token.lower()
        # Light plural stemming
        if len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
            token = token[:-1]
        return token


def merge_non_overlapping_spans(
    spans_by_pass: List[List[Tuple[CharSpan, Any]]],
) -> List[Tuple[CharSpan, Any]]:
    """
    Merge extraction spans from multiple passes, keeping first-pass wins.
    
    Args:
        spans_by_pass: List of lists of (CharSpan, extraction) tuples per pass
        
    Returns:
        Merged list of non-overlapping (CharSpan, extraction) tuples
    """
    if not spans_by_pass:
        return []
    
    if len(spans_by_pass) == 1:
        return list(spans_by_pass[0])
    
    merged = list(spans_by_pass[0])
    
    for pass_spans in spans_by_pass[1:]:
        for span, extraction in pass_spans:
            overlaps = False
            for existing_span, _ in merged:
                if span.overlaps(existing_span):
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append((span, extraction))
    
    return merged

