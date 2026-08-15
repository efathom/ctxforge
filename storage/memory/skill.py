"""
In-memory skill store implementation.

Suitable for testing and single-instance deployments.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from ctxforge.core.skill import (
    Skill,
    SkillMatch,
    SkillMetadata,
    SkillRelationship,
    SkillScope,
)
from ctxforge.engine.registry import registry


@registry.register_skill_store("memory")
class InMemorySkillStore:
    """
    In-memory skill store.

    Skills are stored in a dictionary keyed by (name, scope, scope_id).
    Suitable for testing and single-instance deployments.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the store.

        Args:
            config: Optional configuration (unused for in-memory)
        """
        # Key: (name, scope, scope_id) -> Skill
        self._store: Dict[tuple, Skill] = {}
        self._relationships: List[SkillRelationship] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the store (no-op for in-memory)."""
        pass

    async def save(self, skill: Skill) -> None:
        """Save a skill. Updates if name already exists in scope."""
        async with self._lock:
            key = (skill.name, skill.scope, skill.scope_id)
            skill.updated_at = datetime.now()
            self._store[key] = skill

    async def get(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> Optional[Skill]:
        """Get full skill by name, scope, and scope_id."""
        key = (name, scope, scope_id)
        return self._store.get(key)

    async def get_metadata(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> Optional[SkillMetadata]:
        """Get only skill metadata (for progressive disclosure)."""
        skill = await self.get(name, scope, scope_id)
        if skill:
            return skill.skill_metadata
        return None

    async def list_metadata(
        self,
        scope: SkillScope,
        scope_id: str
    ) -> List[SkillMetadata]:
        """List all skill metadata for a given scope."""
        skills = [
            s for s in self._store.values()
            if s.scope == scope and s.scope_id == scope_id
        ]
        # Sort by name
        skills.sort(key=lambda s: s.name)
        return [s.skill_metadata for s in skills]

    async def list_all_metadata(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[SkillMetadata]:
        """
        List metadata for all available skills with scope layering.

        Returns skills from BASE, then USER (if user_id provided),
        then PROJECT (if project_id provided). Later scopes override
        earlier ones by name.
        """
        # Collect skills by name (later scopes override)
        skills_by_name: Dict[str, SkillMetadata] = {}

        # 1. Base skills (scope_id = "system" or empty)
        for skill in self._store.values():
            if skill.scope == SkillScope.BASE:
                skills_by_name[skill.name] = skill.skill_metadata

        # 2. User skills (override base)
        if user_id:
            for skill in self._store.values():
                if skill.scope == SkillScope.USER and skill.scope_id == user_id:
                    skills_by_name[skill.name] = skill.skill_metadata

        # 3. Project skills (override user and base)
        if project_id:
            for skill in self._store.values():
                if skill.scope == SkillScope.PROJECT and skill.scope_id == project_id:
                    skills_by_name[skill.name] = skill.skill_metadata

        # Return sorted by name
        result = list(skills_by_name.values())
        result.sort(key=lambda s: s.name)
        return result

    async def delete(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> bool:
        """Delete a skill. Returns True if deleted."""
        async with self._lock:
            key = (name, scope, scope_id)
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def search_by_trigger(
        self,
        query: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[SkillMatch]:
        """Find skills that match a query based on triggers."""
        # Get all available skills with layering
        available = await self.list_all_metadata(user_id, project_id)

        matches: List[SkillMatch] = []
        query_lower = query.lower()

        for skill_meta in available:
            matched_trigger = skill_meta.matches_trigger(query)
            if matched_trigger:
                # Calculate confidence based on match quality
                # Full word match = higher confidence
                trigger_lower = matched_trigger.lower()
                if trigger_lower == query_lower:
                    confidence = 1.0
                elif (query_lower.startswith(trigger_lower) or
                      query_lower.endswith(trigger_lower)):
                    confidence = 0.9
                else:
                    confidence = 0.7

                matches.append(SkillMatch(
                    skill=skill_meta,
                    confidence=confidence,
                    matched_trigger=matched_trigger,
                    match_reason=f"Trigger '{matched_trigger}' matched query",
                ))

        # Sort by confidence (descending)
        matches.sort(key=lambda m: -m.confidence)
        return matches

    async def count(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count skills, optionally filtered by scope."""
        if scope is None and scope_id is None:
            return len(self._store)

        count = 0
        for s in self._store.values():
            if scope is not None and s.scope != scope:
                continue
            if scope_id is not None and s.scope_id != scope_id:
                continue
            count += 1
        return count

    async def clear(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear skills, optionally filtered by scope. Returns count deleted."""
        async with self._lock:
            if scope is None and scope_id is None:
                count = len(self._store)
                self._store.clear()
                return count

            # Find skills to delete
            to_delete = []
            for key, s in self._store.items():
                if scope is not None and s.scope != scope:
                    continue
                if scope_id is not None and s.scope_id != scope_id:
                    continue
                to_delete.append(key)

            # Delete them
            for key in to_delete:
                del self._store[key]

            return len(to_delete)

    async def save_relationships(
        self, relationships: List[SkillRelationship]
    ) -> int:
        """Save skill relationships. Returns count saved."""
        async with self._lock:
            saved = 0
            for rel in relationships:
                # Avoid exact duplicates
                is_dup = any(
                    r.source == rel.source
                    and r.target == rel.target
                    and r.relation_type == rel.relation_type
                    for r in self._relationships
                )
                if not is_dup:
                    self._relationships.append(rel)
                    saved += 1
            return saved

    async def get_relationships(
        self, skill_name: str
    ) -> List[SkillRelationship]:
        """Get all relationships for a skill (as source or target)."""
        return [
            r for r in self._relationships
            if r.source == skill_name or r.target == skill_name
        ]

    async def get_all_relationships(self) -> List[SkillRelationship]:
        """Get all stored relationships."""
        return list(self._relationships)

    async def search_by_category(
        self,
        category: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Find skills matching a category."""
        all_meta = await self.list_all_metadata(user_id, project_id)
        return [m for m in all_meta if m.category == category]

    async def search_by_tags(
        self,
        tags: List[str],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Find skills matching any of the provided tags."""
        all_meta = await self.list_all_metadata(user_id, project_id)
        tag_set = set(t.lower() for t in tags)
        return [
            m for m in all_meta
            if tag_set & set(t.lower() for t in m.tags)
        ]

    async def update_effectiveness(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str,
        metrics: Dict[str, Any],
    ) -> bool:
        """Update effectiveness metrics for a skill. Returns True if updated."""
        async with self._lock:
            key = (name, scope, scope_id)
            skill = self._store.get(key)
            if skill is None:
                return False
            if skill.effectiveness is None:
                skill.effectiveness = {}
            skill.effectiveness.update(metrics)
            skill.updated_at = datetime.now()
            return True
