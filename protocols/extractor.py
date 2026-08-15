"""
Memory Extractor Protocol Interface.

Defines the contract for extracting memories from conversations.
Extractors analyze user input and agent responses to identify
facts, preferences, and other information worth remembering.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ctxforge.core.alignment_types import AlignmentStatus, CharSpan
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.core.session import Session
from ctxforge.utils.hashing import compute_content_hash


@dataclass
class ExtractionCandidate:
    """A candidate memory extracted from conversation."""
    
    content: str
    memory_type: MemoryType
    confidence: float  # 0.0 to 1.0
    source_text: str  # The text this was extracted from
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Lossless restatement: disambiguated, self-contained version of content.
    restatement: Optional[str] = None
    # Structured entities extracted alongside the memory.
    extracted_entities: Dict[str, Any] = field(default_factory=dict)

    # Source grounding fields (types from core module - no circular dependency)
    source_span: Optional[CharSpan] = None  # Position in source text
    alignment_status: Optional[AlignmentStatus] = None  # How well it aligned
    matched_text: Optional[str] = None  # Actual matched text from source
    extraction_pass: int = 1  # Which pass found this (for multi-pass)
    
    def to_memory_item(
        self,
        user_id: str,
        source: MemorySource = MemorySource.AGENT_INFERENCE,
    ) -> MemoryItem:
        """Convert to a MemoryItem."""
        # Include source grounding in metadata if available
        metadata = dict(self.metadata)
        if self.source_span:
            metadata["source_span"] = self.source_span.to_tuple()
        if self.alignment_status:
            metadata["alignment_status"] = self.alignment_status.value
        if self.matched_text:
            metadata["matched_text"] = self.matched_text
        if self.extraction_pass > 1:
            metadata["extraction_pass"] = self.extraction_pass

        metadata["content_hash"] = compute_content_hash(self.content, self.memory_type.value)

        # Propagate multi-view indexing fields from metadata / entities
        keywords = metadata.pop("keywords", [])
        topics = metadata.pop("topics", [])
        persons = list(self.extracted_entities.get("persons", []))
        locations = list(self.extracted_entities.get("locations", []))

        return MemoryItem(
            user_id=user_id,
            content=self.content,
            type=self.memory_type,
            source=source,
            confidence_score=self.confidence,
            tags=self.tags,
            metadata=metadata,
            restatement=self.restatement,
            extracted_entities=self.extracted_entities,
            keywords=keywords,
            persons=persons,
            locations=locations,
            topics=topics,
        )


@dataclass
class ExtractionResult:
    """Result from an extraction operation."""
    
    candidates: List[ExtractionCandidate]
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Number of candidates extracted."""
        return len(self.candidates)
    
    def filter_by_confidence(self, min_confidence: float) -> List[ExtractionCandidate]:
        """Get candidates above a confidence threshold."""
        return [c for c in self.candidates if c.confidence >= min_confidence]
    
    def filter_by_type(self, memory_type: MemoryType) -> List[ExtractionCandidate]:
        """Get candidates of a specific type."""
        return [c for c in self.candidates if c.memory_type == memory_type]


@dataclass
class ExtractionConfig:
    """Configuration for extraction operations."""
    
    # What to extract
    extract_semantic: bool = True
    extract_episodic: bool = True
    extract_procedural: bool = False
    extract_preference: bool = True
    extract_tool: bool = False

    # Quality thresholds
    min_confidence: float = 0.5
    max_candidates: int = 10
    
    # Model settings (for LLM-based extraction)
    model: Optional[str] = None
    temperature: float = 0.3
    
    # Pattern settings (for pattern-based extraction)
    custom_patterns: Optional[Dict[str, str]] = None
    
    # Multi-pass settings
    extraction_passes: int = 1  # Number of extraction passes for improved recall
    
    # Alignment settings
    enable_alignment: bool = True  # Whether to align extractions to source
    fuzzy_alignment_threshold: float = 0.75  # Threshold for fuzzy matching
    accept_partial_matches: bool = True  # Accept partial text matches
    
    # Chunking settings (for long text)
    max_chunk_size: int = 2000  # Max characters per chunk
    parallel_chunks: int = 5  # Max parallel chunk processing
    
    # Schema constraints
    use_schema_constraints: bool = False  # Use JSON schema for structured output


@runtime_checkable
class IMemoryExtractor(Protocol):
    """
    Protocol for memory extraction.
    
    Implementations analyze conversations to extract information
    worth storing in long-term memory.
    
    Example implementations:
    - LLMExtractor: Uses LLM for intelligent extraction
    - PatternExtractor: Uses regex/NLP patterns
    - HybridExtractor: Combines LLM and patterns
    - EntityExtractor: Extracts named entities
    """
    
    @property
    def name(self) -> str:
        """The name of this extractor."""
        ...
    
    async def extract(
        self,
        user_input: str,
        agent_response: str,
        session: Optional[Session] = None,
        config: Optional[ExtractionConfig] = None,
    ) -> ExtractionResult:
        """
        Extract memories from a conversation turn.
        
        Args:
            user_input: The user's input
            agent_response: The agent's response
            session: Optional session for additional context
            config: Optional extraction configuration
            
        Returns:
            ExtractionResult with candidates
        """
        ...
    
    async def extract_from_text(
        self,
        text: str,
        config: Optional[ExtractionConfig] = None,
    ) -> ExtractionResult:
        """
        Extract memories from raw text.
        
        Useful for batch processing or importing.
        
        Args:
            text: The text to extract from
            config: Optional extraction configuration
            
        Returns:
            ExtractionResult with candidates
        """
        ...
    
    async def validate_candidate(
        self,
        candidate: ExtractionCandidate,
        existing_memories: List[MemoryItem],
    ) -> bool:
        """
        Validate a candidate against existing memories.
        
        Checks for duplicates and conflicts.
        
        Args:
            candidate: The candidate to validate
            existing_memories: Existing memories for the user
            
        Returns:
            True if the candidate should be stored
        """
        ...


@runtime_checkable
class IConsolidator(Protocol):
    """
    Protocol for memory consolidation.
    
    Consolidators merge, deduplicate, and update memories
    to maintain a clean and consistent memory store.
    
    Example implementations:
    - DeduplicationConsolidator: Removes duplicates
    - MergingConsolidator: Merges similar memories
    - ConflictResolver: Resolves contradictory memories
    """
    
    @property
    def name(self) -> str:
        """The name of this consolidator."""
        ...
    
    async def consolidate(
        self,
        new_items: List[MemoryItem],
        existing_items: List[MemoryItem],
    ) -> List[MemoryItem]:
        """
        Consolidate new items with existing ones.
        
        Returns the items that should be stored (may include
        updates to existing items).
        
        Args:
            new_items: New items to add
            existing_items: Existing items in the store
            
        Returns:
            Items to store (new + updated existing)
        """
        ...
    
    async def find_duplicates(
        self,
        item: MemoryItem,
        candidates: List[MemoryItem],
        threshold: float = 0.9,
    ) -> List[MemoryItem]:
        """
        Find potential duplicates of an item.
        
        Args:
            item: The item to check
            candidates: Potential duplicate candidates
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of potential duplicates
        """
        ...
    
    async def merge_memories(
        self,
        memories: List[MemoryItem],
    ) -> MemoryItem:
        """
        Merge multiple memories into one.
        
        Args:
            memories: Memories to merge
            
        Returns:
            The merged memory
        """
        ...

