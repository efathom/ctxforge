"""
Scoped Memory Models for Hierarchical Memory System.

This module defines the core data structures for scoped memories that
support GLOBAL, PROJECT, and SESSION namespaces with hierarchical override.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryScope(Enum):
    """Memory scope hierarchy (higher = more specific, overrides lower)."""
    GLOBAL = "global"      # User-level, applies everywhere
    PROJECT = "project"    # Project-specific context
    SESSION = "session"    # Ephemeral, current conversation only

    @classmethod
    def priority(cls, scope: "MemoryScope") -> int:
        """Get priority for scope (higher = overrides lower)."""
        priorities = {
            cls.GLOBAL: 0,
            cls.PROJECT: 1,
            cls.SESSION: 2,
        }
        return priorities.get(scope, 0)


class MemoryCategory(Enum):
    """Categories for organizing scoped memories."""
    PREFERENCE = "preference"      # User/project preferences
    CONVENTION = "convention"      # Coding conventions, patterns
    ARCHITECTURE = "architecture"  # System design decisions
    INSTRUCTION = "instruction"    # How agent should behave
    CONTEXT = "context"           # Domain-specific knowledge
    GOTCHA = "gotcha"             # Warnings and pitfalls
    # Structured observation categories (auto-extracted from sessions)
    DECISION = "decision"          # Decisions made during a session
    BUGFIX = "bugfix"              # Bugs found and fixed
    DISCOVERY = "discovery"        # New discoveries or insights
    FEATURE = "feature"            # Features implemented
    REFACTOR = "refactor"          # Refactoring work done

    @classmethod
    def get_display_name(cls, category: "MemoryCategory") -> str:
        """Get human-readable display name for category."""
        display_names = {
            cls.PREFERENCE: "Preferences",
            cls.CONVENTION: "Conventions",
            cls.ARCHITECTURE: "Architecture",
            cls.INSTRUCTION: "Instructions",
            cls.CONTEXT: "Context",
            cls.GOTCHA: "Gotchas & Warnings",
            cls.DECISION: "Decisions",
            cls.BUGFIX: "Bug Fixes",
            cls.DISCOVERY: "Discoveries",
            cls.FEATURE: "Features",
            cls.REFACTOR: "Refactoring",
        }
        return display_names.get(category, category.value.title())


@dataclass
class ScopedMemory:
    """A memory entry with scope and category."""
    id: str
    scope: MemoryScope
    scope_id: str              # user_id, project_id, or session_id
    category: MemoryCategory
    key: str                   # Unique identifier within scope
    content: str               # The memory content
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0          # Higher = loaded first in prompt
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "category": self.category.value,
            "key": self.key,
            "content": self.content,
            "metadata": self.metadata,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScopedMemory":
        """Create from dictionary representation."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        return cls(
            id=data["id"],
            scope=MemoryScope(data["scope"]),
            scope_id=data["scope_id"],
            category=MemoryCategory(data["category"]),
            key=data["key"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            priority=data.get("priority", 0),
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class ScopedMemoryQuery:
    """Query parameters for retrieving scoped memories."""
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    categories: Optional[List[MemoryCategory]] = None
    include_global: bool = True
    include_project: bool = True
    include_session: bool = True

    def get_scope_ids(self) -> Dict[MemoryScope, Optional[str]]:
        """Get mapping of scopes to their IDs."""
        result = {}
        if self.include_global and self.user_id:
            result[MemoryScope.GLOBAL] = self.user_id
        if self.include_project and self.project_id:
            result[MemoryScope.PROJECT] = self.project_id
        if self.include_session and self.session_id:
            result[MemoryScope.SESSION] = self.session_id
        return result


@dataclass
class MergedMemoryResult:
    """Result of merging memories across scopes."""
    memories: List[ScopedMemory]
    scope_counts: Dict[MemoryScope, int]
    override_count: int  # Number of memories that overrode lower-scope ones

    @property
    def total_count(self) -> int:
        """Get total number of memories."""
        return len(self.memories)

    def by_category(self) -> Dict[MemoryCategory, List[ScopedMemory]]:
        """Group memories by category."""
        result: Dict[MemoryCategory, List[ScopedMemory]] = {}
        for memory in self.memories:
            if memory.category not in result:
                result[memory.category] = []
            result[memory.category].append(memory)
        return result

    def format_for_prompt(self) -> str:
        """Format memories for prompt injection."""
        if not self.memories:
            return ""

        lines = ["## Context & Memories", ""]
        by_category = self.by_category()

        for category in MemoryCategory:
            if category in by_category:
                category_memories = by_category[category]
                # Sort by priority (higher first)
                category_memories.sort(key=lambda m: -m.priority)

                display_name = MemoryCategory.get_display_name(category)
                lines.append(f"### {display_name}")
                for memory in category_memories:
                    scope_label = f"[{memory.scope.value.upper()}]"
                    lines.append(f"- {scope_label} {memory.content}")
                lines.append("")

        return "\n".join(lines)
