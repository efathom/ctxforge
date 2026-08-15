"""
Protocol for Skill Storage.

Defines the interface for skill stores that support
progressive disclosure with base, user, and project scopes.
"""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ctxforge.core.skill import (
    Skill,
    SkillMatch,
    SkillMetadata,
    SkillRelationship,
    SkillScope,
)


@runtime_checkable
class ISkillStore(Protocol):
    """Protocol for skill storage."""

    async def initialize(self) -> None:
        """Initialize the store (create tables, etc.)."""
        ...

    async def save(self, skill: Skill) -> None:
        """Save a skill. Updates if name already exists in scope."""
        ...

    async def get(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> Optional[Skill]:
        """Get full skill by name, scope, and scope_id."""
        ...

    async def get_metadata(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> Optional[SkillMetadata]:
        """Get only skill metadata (for progressive disclosure)."""
        ...

    async def list_metadata(
        self,
        scope: SkillScope,
        scope_id: str
    ) -> List[SkillMetadata]:
        """List all skill metadata for a given scope."""
        ...

    async def list_all_metadata(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[SkillMetadata]:
        """List metadata for all available skills with scope layering."""
        ...

    async def delete(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> bool:
        """Delete a skill. Returns True if deleted."""
        ...

    async def search_by_trigger(
        self,
        query: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[SkillMatch]:
        """Find skills that match a query based on triggers."""
        ...

    async def count(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count skills, optionally filtered by scope."""
        ...

    async def clear(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear skills, optionally filtered by scope. Returns count deleted."""
        ...

    async def save_relationships(
        self, relationships: List[SkillRelationship]
    ) -> int:
        """Save skill relationships. Returns count saved."""
        ...

    async def get_relationships(
        self, skill_name: str
    ) -> List[SkillRelationship]:
        """Get all relationships for a skill (as source or target)."""
        ...

    async def get_all_relationships(self) -> List[SkillRelationship]:
        """Get all stored relationships."""
        ...

    async def search_by_category(
        self,
        category: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Find skills matching a category."""
        ...

    async def search_by_tags(
        self,
        tags: List[str],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Find skills matching any of the provided tags."""
        ...

    async def update_effectiveness(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str,
        metrics: Dict[str, Any],
    ) -> bool:
        """Update effectiveness metrics for a skill. Returns True if updated."""
        ...
