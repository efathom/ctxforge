"""
Core data models for the Expertise system.

This module defines the data structures for managing structured, evolving
knowledge bases inspired by ACE framework's "Playbook" concept.

These models are placed in core/ to allow import by other core modules
(context.py, etc.) without circular dependencies.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ExpertiseSection(str, Enum):
    """
    Categories for organizing expertise items.
    
    Based on ACE framework's playbook sections, these provide logical
    groupings for different types of knowledge.
    """
    STRATEGIES = "strategies_and_insights"
    FORMULAS = "formulas_and_calculations"
    CODE_SNIPPETS = "code_snippets_and_templates"
    COMMON_MISTAKES = "common_mistakes_to_avoid"
    HEURISTICS = "problem_solving_heuristics"
    CONTEXT_CLUES = "context_clues_and_indicators"
    CUSTOM = "custom"
    
    @classmethod
    def from_string(cls, value: str) -> "ExpertiseSection":
        """Parse section from string, with normalization."""
        normalized = value.lower().replace(" ", "_").replace("&", "and")
        for section in cls:
            if section.value == normalized:
                return section
        return cls.CUSTOM
    
    def to_display_name(self) -> str:
        """Convert to human-readable display name."""
        return self.value.upper().replace("_", " ")
    
    def to_slug(self) -> str:
        """Get short slug for item ID generation."""
        slugs = {
            ExpertiseSection.STRATEGIES: "strat",
            ExpertiseSection.FORMULAS: "form",
            ExpertiseSection.CODE_SNIPPETS: "code",
            ExpertiseSection.COMMON_MISTAKES: "mist",
            ExpertiseSection.HEURISTICS: "heur",
            ExpertiseSection.CONTEXT_CLUES: "clue",
            ExpertiseSection.CUSTOM: "cust",
        }
        return slugs.get(self, "item")


class UsageFeedback(str, Enum):
    """
    Feedback on expertise item usage in a turn.
    
    The reflector assigns these tags to items based on whether
    they contributed positively or negatively to the outcome.
    """
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"


class TurnOutcome(str, Enum):
    """
    Outcome of a conversation turn.
    
    Used to evaluate whether the expertise was effective.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class CuratorOperation(str, Enum):
    """
    Operations the curator can perform on expertise.
    
    Based on ACE framework's curation operations.
    """
    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"


class ExpertiseItem(BaseModel):
    """
    Single expertise entry with usage tracking.
    
    Represents a single piece of domain knowledge with provenance
    and effectiveness tracking. Format inspired by ACE's bullet points:
    [id] helpful=X harmful=Y :: content
    
    Attributes:
        item_id: Unique identifier (e.g., "strat-00001")
        section: Category for organization
        content: The actual knowledge content
        helpful_count: Times marked helpful by reflector
        harmful_count: Times marked harmful by reflector
        source: Origin of this item (reflection, manual, import, etc.)
        is_active: Whether the item is currently active
        embedding: Optional vector embedding for semantic search
        metadata: Additional metadata
        created_at: When the item was created
        updated_at: When the item was last updated
    """
    
    item_id: str = Field(default_factory=lambda: f"item-{uuid.uuid4().hex[:8]}")
    section: ExpertiseSection = ExpertiseSection.CUSTOM
    content: str
    helpful_count: int = Field(default=0, ge=0)
    harmful_count: int = Field(default=0, ge=0)
    source: Optional[str] = None
    is_active: bool = True
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("Expertise item content cannot be empty")
        return v.strip()
    
    @property
    def effectiveness_score(self) -> float:
        """
        Calculate effectiveness score: helpful / (helpful + harmful).
        
        Returns 0.5 (neutral) for unused items.
        """
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total
    
    @property
    def total_usage(self) -> int:
        """Total times this item has been used (helpful + harmful)."""
        return self.helpful_count + self.harmful_count
    
    @property
    def is_high_performing(self) -> bool:
        """Check if item is high performing (helpful > 5, harmful < 2)."""
        return self.helpful_count > 5 and self.harmful_count < 2
    
    @property
    def is_problematic(self) -> bool:
        """Check if item is problematic (harmful >= helpful and harmful > 0)."""
        return self.harmful_count >= self.helpful_count and self.harmful_count > 0
    
    @property
    def is_unused(self) -> bool:
        """Check if item has never been used."""
        return self.total_usage == 0
    
    def increment_helpful(self) -> None:
        """Increment helpful count."""
        self.helpful_count += 1
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    def increment_harmful(self) -> None:
        """Increment harmful count."""
        self.harmful_count += 1
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    def update_content(self, new_content: str) -> None:
        """Update the content."""
        self.content = new_content.strip()
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    def deactivate(self) -> None:
        """Deactivate the item (soft delete)."""
        self.is_active = False
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    def activate(self) -> None:
        """Reactivate the item."""
        self.is_active = True
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    def to_prompt_format(self) -> str:
        """
        Convert to a format suitable for LLM prompt inclusion.
        
        Conforms to IContextItem protocol.
        
        Returns:
            A string representation optimized for prompt context.
        """
        section_label = self.section.to_display_name()
        effectiveness = ""
        if self.total_usage > 0:
            if self.effectiveness_score >= 0.8:
                effectiveness = " (proven effective)"
            elif self.effectiveness_score <= 0.3:
                effectiveness = " (use with caution)"
        return f"[{section_label}]{effectiveness}: {self.content}"
    
    def to_ace_format(self) -> str:
        """
        Convert to ACE playbook format.
        
        Returns:
            String in format: [id] helpful=X harmful=Y :: content
        """
        return (
            f"[{self.item_id}] "
            f"helpful={self.helpful_count} harmful={self.harmful_count} :: "
            f"{self.content}"
        )
    
    @classmethod
    def from_ace_format(cls, line: str, section: ExpertiseSection = ExpertiseSection.CUSTOM) -> Optional["ExpertiseItem"]:
        """
        Parse from ACE playbook format.
        
        Args:
            line: String in format: [id] helpful=X harmful=Y :: content
            section: Section to assign to the item
            
        Returns:
            ExpertiseItem or None if parsing fails
        """
        import re
        
        pattern = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
        match = re.match(pattern, line.strip())
        
        if not match:
            return None
        
        item_id, helpful, harmful, content = match.groups()
        return cls(
            item_id=item_id,
            section=section,
            content=content.strip(),
            helpful_count=int(helpful),
            harmful_count=int(harmful),
        )


class Expertise(BaseModel):
    """
    Complete expertise knowledge base.
    
    Represents a collection of expertise items organized by sections,
    with versioning and token budget management.
    
    Attributes:
        expertise_id: Unique identifier
        name: Human-readable name
        domain: Optional domain/topic area
        items: List of expertise items
        version: Version number for tracking changes
        token_budget: Maximum tokens allowed for this expertise
        next_item_id: Counter for generating unique item IDs
        metadata: Additional metadata
        created_at: When the expertise was created
        updated_at: When the expertise was last updated
    """
    
    expertise_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    domain: Optional[str] = None
    description: str = ""
    items: List[ExpertiseItem] = Field(default_factory=list)
    version: int = 1
    token_budget: int = 80000
    next_item_id: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Expertise name cannot be empty")
        return v.strip()
    
    @property
    def active_items(self) -> List[ExpertiseItem]:
        """Get only active items."""
        return [item for item in self.items if item.is_active]
    
    @property
    def item_count(self) -> int:
        """Total number of items."""
        return len(self.items)
    
    @property
    def active_item_count(self) -> int:
        """Number of active items."""
        return len(self.active_items)
    
    def get_item(self, item_id: str) -> Optional[ExpertiseItem]:
        """Get an item by ID."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None
    
    def get_items_by_section(self, section: ExpertiseSection) -> List[ExpertiseItem]:
        """Get all items in a specific section."""
        return [item for item in self.active_items if item.section == section]
    
    def add_item(
        self,
        section: ExpertiseSection,
        content: str,
        source: Optional[str] = None,
    ) -> ExpertiseItem:
        """
        Add a new item to the expertise.
        
        Args:
            section: Section to add the item to
            content: Item content
            source: Origin of the item
            
        Returns:
            The created ExpertiseItem
        """
        item_id = f"{section.to_slug()}-{self.next_item_id:05d}"
        self.next_item_id += 1
        
        item = ExpertiseItem(
            item_id=item_id,
            section=section,
            content=content,
            source=source,
        )
        self.items.append(item)
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        
        return item
    
    def remove_item(self, item_id: str, soft_delete: bool = True) -> bool:
        """
        Remove an item by ID.
        
        Args:
            item_id: ID of the item to remove
            soft_delete: If True, deactivate; if False, remove entirely
            
        Returns:
            True if item was found and removed/deactivated
        """
        for item in self.items:
            if item.item_id == item_id:
                if soft_delete:
                    item.deactivate()
                else:
                    self.items.remove(item)
                self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                return True
        return False
    
    def update_item_counts(
        self,
        item_id: str,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
    ) -> bool:
        """
        Update helpful/harmful counts for an item.
        
        Args:
            item_id: ID of the item to update
            helpful_delta: Amount to add to helpful count
            harmful_delta: Amount to add to harmful count
            
        Returns:
            True if item was found and updated
        """
        item = self.get_item(item_id)
        if item:
            item.helpful_count += helpful_delta
            item.harmful_count += harmful_delta
            item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            return True
        return False
    
    def increment_version(self) -> None:
        """Increment the version number."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    def to_ace_format(self) -> str:
        """
        Convert to ACE playbook format.
        
        Returns:
            String in ACE playbook format with sections and items
        """
        lines = []
        
        # Group items by section
        sections_order = [
            ExpertiseSection.STRATEGIES,
            ExpertiseSection.FORMULAS,
            ExpertiseSection.CODE_SNIPPETS,
            ExpertiseSection.COMMON_MISTAKES,
            ExpertiseSection.HEURISTICS,
            ExpertiseSection.CONTEXT_CLUES,
            ExpertiseSection.CUSTOM,
        ]
        
        for section in sections_order:
            section_items = self.get_items_by_section(section)
            if section_items or section != ExpertiseSection.CUSTOM:
                lines.append(f"## {section.to_display_name()}")
                for item in section_items:
                    lines.append(item.to_ace_format())
                lines.append("")
        
        return "\n".join(lines)
    
    def estimate_tokens(self) -> int:
        """
        Estimate token count for the expertise.
        
        Uses a rough approximation of 4 characters per token.
        """
        text = self.to_ace_format()
        return len(text) // 4


class CompletedTurn(BaseModel):
    """
    A completed conversation turn for reflection.
    
    Contains the information needed for the reflector to analyze
    whether the expertise was effective.
    """
    
    user_input: str
    assistant_response: str
    expected_output: Optional[str] = None
    actual_outcome: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExpertiseUsageLog(BaseModel):
    """
    Log of expertise item usage in a turn.
    
    Tracks which items were used and how they were rated.
    """
    
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    expertise_id: str
    items_used: List[str] = Field(default_factory=list)
    feedback: Dict[str, UsageFeedback] = Field(default_factory=dict)
    outcome: Optional[TurnOutcome] = None
    context_summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ReflectionResult(BaseModel):
    """
    Result of expertise reflection.
    
    Contains feedback on items and suggestions for improvement.
    """
    
    item_feedback: Dict[str, UsageFeedback] = Field(default_factory=dict)
    insights: str = ""
    suggested_additions: List[str] = Field(default_factory=list)
    suggested_removals: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @property
    def has_suggestions(self) -> bool:
        """Check if there are any suggestions."""
        return bool(self.suggested_additions or self.suggested_removals)
    
    @property
    def helpful_items(self) -> List[str]:
        """Get IDs of items marked helpful."""
        return [
            item_id for item_id, feedback in self.item_feedback.items()
            if feedback == UsageFeedback.HELPFUL
        ]
    
    @property
    def harmful_items(self) -> List[str]:
        """Get IDs of items marked harmful."""
        return [
            item_id for item_id, feedback in self.item_feedback.items()
            if feedback == UsageFeedback.HARMFUL
        ]


class CurationOp(BaseModel):
    """
    Single curation operation.
    
    Represents one change to be made to the expertise.
    """
    
    type: CuratorOperation
    section: Optional[ExpertiseSection] = None
    item_ids: List[str] = Field(default_factory=list)
    content: Optional[str] = None
    reason: str = ""


class CurationPlan(BaseModel):
    """
    Plan for expertise modifications.
    
    Contains a list of operations and the reasoning behind them.
    """
    
    operations: List[CurationOp] = Field(default_factory=list)
    reasoning: str = ""
    
    @property
    def operation_count(self) -> int:
        """Total number of operations."""
        return len(self.operations)
    
    @property
    def has_operations(self) -> bool:
        """Check if there are any operations."""
        return len(self.operations) > 0
    
    def get_operations_by_type(self, op_type: CuratorOperation) -> List[CurationOp]:
        """Get operations of a specific type."""
        return [op for op in self.operations if op.type == op_type]


class ExpertiseStats(BaseModel):
    """
    Statistics about expertise quality.
    
    Provides insights into the health and effectiveness of the expertise.
    """
    
    total_items: int = 0
    active_items: int = 0
    items_by_section: Dict[str, int] = Field(default_factory=dict)
    high_performing: int = 0
    problematic: int = 0
    unused: int = 0
    average_effectiveness: float = 0.0
    total_helpful: int = 0
    total_harmful: int = 0
    estimated_tokens: int = 0
    
    @classmethod
    def from_expertise(cls, expertise: Expertise) -> "ExpertiseStats":
        """
        Calculate statistics from an expertise.
        
        Args:
            expertise: The expertise to analyze
            
        Returns:
            ExpertiseStats with calculated values
        """
        stats = cls(
            total_items=expertise.item_count,
            active_items=expertise.active_item_count,
            estimated_tokens=expertise.estimate_tokens(),
        )
        
        effectiveness_scores = []
        
        for item in expertise.active_items:
            # Count by section
            section_key = item.section.value
            stats.items_by_section[section_key] = stats.items_by_section.get(section_key, 0) + 1
            
            # Track counts
            stats.total_helpful += item.helpful_count
            stats.total_harmful += item.harmful_count
            
            # Track categories
            if item.is_high_performing:
                stats.high_performing += 1
            if item.is_problematic:
                stats.problematic += 1
            if item.is_unused:
                stats.unused += 1
            
            # Track effectiveness
            effectiveness_scores.append(item.effectiveness_score)
        
        # Calculate average effectiveness
        if effectiveness_scores:
            stats.average_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        
        return stats


class SimilarGroup(BaseModel):
    """
    Group of similar expertise items.
    
    Used for deduplication and merging operations.
    """
    
    items: List[ExpertiseItem] = Field(default_factory=list)
    similarity_scores: List[float] = Field(default_factory=list)
    
    @property
    def item_count(self) -> int:
        """Number of items in the group."""
        return len(self.items)
    
    @property
    def item_ids(self) -> List[str]:
        """Get IDs of all items in the group."""
        return [item.item_id for item in self.items]
    
    @property
    def primary_item(self) -> Optional[ExpertiseItem]:
        """Get the first (primary) item in the group."""
        return self.items[0] if self.items else None
    
    @property
    def total_helpful(self) -> int:
        """Sum of helpful counts across all items."""
        return sum(item.helpful_count for item in self.items)
    
    @property
    def total_harmful(self) -> int:
        """Sum of harmful counts across all items."""
        return sum(item.harmful_count for item in self.items)

