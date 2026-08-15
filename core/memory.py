"""
Long-term Memory System - Persistent knowledge storage.

Memory items represent facts, experiences, and procedures that persist
across sessions. They are retrieved based on relevance to provide
personalized context to the agent.
"""

import datetime
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ctxforge.utils.math import cosine_similarity


class MemoryType(str, Enum):
    """Types of long-term memories."""
    
    SEMANTIC = "semantic"       # Facts (e.g., "User is vegetarian")
    EPISODIC = "episodic"       # Events (e.g., "User went to Paris last summer")
    PROCEDURAL = "procedural"   # Skills/Procedures (e.g., Playbooks, workflows)
    PREFERENCE = "preference"   # User preferences (e.g., "User prefers dark mode")
    TOOL = "tool"               # Tool execution patterns


class MemorySource(str, Enum):
    """Sources of memory creation."""
    
    USER_EXPLICIT = "user_explicit"     # User explicitly stated
    USER_IMPLICIT = "user_implicit"     # Inferred from user behavior
    AGENT_INFERENCE = "agent_inference" # Agent-generated inference
    SYSTEM = "system"                   # System-generated
    EXTERNAL = "external"               # External data import


class MemoryItem(BaseModel):
    """
    Long-Term Memory Unit.
    
    Represents a single piece of information stored in long-term memory.
    Includes provenance information for trust and verification.
    
    Attributes:
        memory_id: Unique identifier for the memory
        user_id: The user this memory belongs to
        content: The actual memory content
        type: Type of memory (semantic, episodic, procedural)
        source: How this memory was created
        confidence_score: Trust level (0.0 to 1.0)
        created_at: When the memory was created
        updated_at: When the memory was last updated
        accessed_at: When the memory was last accessed
        access_count: How many times the memory has been accessed
        embedding: Optional vector embedding for semantic search
        metadata: Additional metadata
        tags: Tags for categorization
        expires_at: Optional expiration time
        is_active: Whether the memory is active
    """
    
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    type: MemoryType
    source: MemorySource = MemorySource.AGENT_INFERENCE
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    accessed_at: Optional[datetime.datetime] = None
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime.datetime] = None
    is_active: bool = True
    
    # Relationships
    related_memory_ids: List[str] = Field(default_factory=list)
    source_event_id: Optional[str] = None  # Link to the event that created this memory

    # Lossless restatement: disambiguated, self-contained version of content.
    # Pronouns resolved to proper nouns, relative times to absolute dates.
    restatement: Optional[str] = None
    # Structured entities extracted during memory creation.
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)

    # Consolidation fields: importance decays over time, superseded_by chains soft deletes.
    importance: float = Field(default=1.0, ge=0.0)
    superseded_by: Optional[str] = None

    # Multi-view indexing: structured metadata for lexical and symbolic search.
    keywords: List[str] = Field(default_factory=list)
    persons: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    event_timestamp: Optional[datetime.datetime] = None

    # Progressive disclosure fields (LLM-generated, stored persistently)
    headline: Optional[str] = None      # ~20-30 tokens, short title
    subtitle: Optional[str] = None      # ~50 tokens, one sentence explanation

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("Memory content cannot be empty")
        return v.strip()
    
    @property
    def item_id(self) -> str:
        """
        Alias for memory_id to conform to IContextItem protocol.
        
        This allows MemoryItem to be used in generic context operations
        that expect an item_id property.
        """
        return self.memory_id

    def record_access(self) -> None:
        """Record that this memory was accessed."""
        self.accessed_at = datetime.datetime.now()
        self.access_count += 1
    
    def update_content(self, new_content: str, source: Optional[MemorySource] = None) -> None:
        """Update the memory content."""
        self.content = new_content.strip()
        self.updated_at = datetime.datetime.now()
        if source:
            self.source = source
    
    def update_confidence(self, new_score: float) -> None:
        """Update the confidence score."""
        self.confidence_score = max(0.0, min(1.0, new_score))
        self.updated_at = datetime.datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the memory."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.datetime.now()
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the memory. Returns True if tag existed."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.datetime.now()
            return True
        return False
    
    def add_related_memory(self, memory_id: str) -> None:
        """Add a related memory reference."""
        if memory_id not in self.related_memory_ids:
            self.related_memory_ids.append(memory_id)
            self.updated_at = datetime.datetime.now()
    
    def is_expired(self) -> bool:
        """Check if the memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.datetime.now() > self.expires_at
    
    def deactivate(self) -> None:
        """Deactivate the memory (soft delete)."""
        self.is_active = False
        self.updated_at = datetime.datetime.now()
    
    def activate(self) -> None:
        """Reactivate the memory."""
        self.is_active = True
        self.updated_at = datetime.datetime.now()
    
    @property
    def display_content(self) -> str:
        """Return the best available content: restatement if present, else raw content."""
        return self.restatement if self.restatement else self.content

    def to_prompt_format(self) -> str:
        """Convert memory to a format suitable for prompt inclusion.

        Prefers the disambiguated ``restatement`` over raw ``content``
        so that the LLM receives self-contained facts.
        """
        type_label = self.type.value.capitalize()
        confidence_indicator = ""
        if self.confidence_score < 0.7:
            confidence_indicator = " (uncertain)"
        elif self.confidence_score < 0.9:
            confidence_indicator = " (likely)"
        return f"[{type_label}]{confidence_indicator}: {self.display_content}"

    def has_headline(self) -> bool:
        """Check if headline has been generated."""
        return self.headline is not None

    def to_headline_format(self) -> str:
        """Format for headline-only display (progressive disclosure tier 1)."""
        if self.headline:
            return f"• [{self.type.value}] {self.headline}"
        # Fallback to truncated content
        truncated = self.content[:80]
        if len(self.content) > 80:
            truncated += "..."
        return f"• [{self.type.value}] {truncated}"

    def to_summary_format(self) -> str:
        """Format for summary display (headline + subtitle, tier 2)."""
        if self.headline and self.subtitle:
            return f"• [{self.type.value}] {self.headline}: {self.subtitle}"
        elif self.headline:
            return f"• [{self.type.value}] {self.headline}"
        return self.to_headline_format()
    
    def similarity_score(self, other_embedding: List[float]) -> float:
        """
        Calculate cosine similarity with another embedding.
        Returns 0.0 if this memory has no embedding.
        """
        if self.embedding is None or not other_embedding:
            return 0.0
        
        return cosine_similarity(self.embedding, other_embedding)


class ToolExecutionRecord(BaseModel):
    """Record of a single tool execution."""

    tool_name: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    success: bool = True
    time_cost: float = 0.0
    token_cost: int = 0
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)


def add_tool_record(memory: MemoryItem, record: ToolExecutionRecord) -> None:
    """Append a tool execution record to a TOOL memory item.

    Args:
        memory: A MemoryItem with type TOOL.
        record: The execution record to append.

    Raises:
        ValueError: If the memory is not of type TOOL.
    """
    if memory.type != MemoryType.TOOL:
        raise ValueError(f"Expected TOOL memory, got {memory.type.value}")
    if "tool_records" not in memory.metadata:
        memory.metadata["tool_records"] = []
    memory.metadata["tool_records"].append(record.model_dump(mode="json"))
    memory.record_access()


def get_tool_statistics(memory: MemoryItem) -> Dict[str, Any]:
    """Compute aggregate statistics from a TOOL memory's execution records.

    Returns:
        Dict with keys: count, success_rate, avg_time_cost, avg_score.
    """
    records = memory.metadata.get("tool_records", [])
    if not records:
        return {"count": 0, "success_rate": 0.0, "avg_time_cost": 0.0, "avg_score": 0.0}
    count = len(records)
    successes = sum(1 for r in records if r.get("success", True))
    avg_time = sum(r.get("time_cost", 0.0) for r in records) / count
    avg_score = sum(r.get("quality_score", 0.0) for r in records) / count
    return {
        "count": count,
        "success_rate": successes / count,
        "avg_time_cost": avg_time,
        "avg_score": avg_score,
    }


class MemoryFactory:
    """Factory for creating common memory types."""
    
    @staticmethod
    def semantic_memory(
        user_id: str,
        content: str,
        source: MemorySource = MemorySource.AGENT_INFERENCE,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> MemoryItem:
        """Create a semantic (fact-based) memory."""
        return MemoryItem(
            user_id=user_id,
            content=content,
            type=MemoryType.SEMANTIC,
            source=source,
            confidence_score=confidence,
            tags=tags or [],
        )
    
    @staticmethod
    def episodic_memory(
        user_id: str,
        content: str,
        source: MemorySource = MemorySource.AGENT_INFERENCE,
        source_event_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryItem:
        """Create an episodic (event-based) memory."""
        return MemoryItem(
            user_id=user_id,
            content=content,
            type=MemoryType.EPISODIC,
            source=source,
            source_event_id=source_event_id,
            tags=tags or [],
        )
    
    @staticmethod
    def procedural_memory(
        user_id: str,
        content: str,
        source: MemorySource = MemorySource.SYSTEM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Create a procedural (skill/workflow) memory."""
        return MemoryItem(
            user_id=user_id,
            content=content,
            type=MemoryType.PROCEDURAL,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )

    @staticmethod
    def tool_memory(
        user_id: str,
        tool_name: str,
        content: str,
        source: MemorySource = MemorySource.SYSTEM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Create a tool execution pattern memory."""
        meta = metadata.copy() if metadata else {}
        meta["tool_records"] = []
        meta["tool_name"] = tool_name
        default_tags = ["tool", tool_name]
        if tags:
            for t in tags:
                if t not in default_tags:
                    default_tags.append(t)
        return MemoryItem(
            user_id=user_id,
            content=content,
            type=MemoryType.TOOL,
            source=source,
            tags=default_tags,
            metadata=meta,
        )


class MemoryQuery(BaseModel):
    """Query parameters for searching memories."""
    
    user_id: str
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    min_confidence: float = 0.0
    include_inactive: bool = False
    include_expired: bool = False
    limit: int = 10
    offset: int = 0
    
    # Sorting options
    sort_by: str = "relevance"  # relevance, recency, access_count, confidence
    sort_descending: bool = True

