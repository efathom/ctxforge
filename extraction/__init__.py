"""
Extraction Pipeline for the ctxforge framework.

Provides memory extraction capabilities to analyze conversations
and identify facts, preferences, and experiences worth storing.

Available extractors:
- PatternExtractor: Rule-based extraction using regex patterns
- LLMExtractor: LLM-powered intelligent extraction
- EntityExtractor: Named entity recognition
- HybridExtractor: Combines multiple extraction strategies

Available consolidators:
- DeduplicationConsolidator: Removes duplicate memories
- MergingConsolidator: Merges similar memories

Usage:
    from ctxforge.extraction import PatternExtractor, HybridExtractor
    
    # Create an extractor
    extractor = PatternExtractor()
    
    # Extract memories from a conversation
    result = await extractor.extract(
        user_input="I really love Italian food",
        agent_response="That's great! Do you have a favorite dish?"
    )
    
    # Process candidates
    for candidate in result.filter_by_confidence(0.7):
        print(f"{candidate.content} ({candidate.confidence})")
    
    # With custom similarity calculator (from utils module)
    from ctxforge.utils import LevenshteinSimilarityCalculator
    calculator = LevenshteinSimilarityCalculator()
    extractor = HybridExtractor(similarity_calculator=calculator)
"""

from ctxforge.extraction.alignment import (
    AlignmentResult,
    AlignmentStatus,
    CharSpan,
    TextTokenizer,
    Token,
    TokenizedText,
    TokenSpan,
    WordAligner,
    merge_non_overlapping_spans,
)
from ctxforge.extraction.base import BaseExtractor
from ctxforge.extraction.chunking import (
    ChunkIterator,
    TextChunk,
    make_batches,
)
from ctxforge.extraction.consolidation import (
    BaseConsolidator,
    ConflictAwareConsolidator,
    ConsolidationAction,
    ConsolidationDecision,
    DeduplicationConsolidator,
    MergingConsolidator,
)
from ctxforge.extraction.entity_extractor import EntityExtractor
from ctxforge.extraction.entropy_gate import EntropyGate, GateResult
from ctxforge.extraction.hybrid_extractor import HybridExtractor
from ctxforge.extraction.llm_extractor import LLMExtractor
from ctxforge.extraction.pattern_extractor import PatternExtractor
from ctxforge.extraction.schema_constraints import (
    SchemaConfig,
    generate_graph_extraction_schema,
    generate_memory_extraction_schema,
    generate_reflection_schema,
)
from ctxforge.extraction.typed_prompts import (
    TYPED_EXTRACTION_PROMPTS,
    get_all_typed_prompts,
    get_typed_prompt,
)
from ctxforge.extraction.utils import (
    extract_sentences,
    normalize_text,
)

__all__ = [
    # Extractors
    "BaseExtractor",
    "PatternExtractor",
    "LLMExtractor",
    "EntityExtractor",
    "HybridExtractor",
    # Consolidators
    "BaseConsolidator",
    "DeduplicationConsolidator",
    "MergingConsolidator",
    "ConflictAwareConsolidator",
    "ConsolidationAction",
    "ConsolidationDecision",
    # Utilities
    "normalize_text",
    "extract_sentences",
    # Alignment
    "AlignmentStatus",
    "AlignmentResult",
    "CharSpan",
    "TokenSpan",
    "Token",
    "TokenizedText",
    "TextTokenizer",
    "WordAligner",
    "merge_non_overlapping_spans",
    # Chunking
    "TextChunk",
    "ChunkIterator",
    "make_batches",
    # Schema Constraints
    "SchemaConfig",
    "generate_memory_extraction_schema",
    "generate_graph_extraction_schema",
    "generate_reflection_schema",
    # Entropy Gate
    "EntropyGate",
    "GateResult",
    # Typed Prompts
    "TYPED_EXTRACTION_PROMPTS",
    "get_typed_prompt",
    "get_all_typed_prompts",
]
