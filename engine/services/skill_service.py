"""
Skill Service.

This service manages skills with progressive disclosure,
supporting base, user, and project scope layering.

Provides search-by-category, search-by-tags, and recommended-skills
queries that leverage relationship and metadata information.
"""
import logging
from typing import Any, Dict, List, Optional

from ctxforge.core.skill import (
    Skill,
    SkillMatch,
    SkillMetadata,
    SkillRelationType,
    SkillScope,
    SkillsIndex,
)
from ctxforge.protocols.skill import ISkillStore

logger = logging.getLogger(__name__)


class SkillService:
    """
    Service for skill management with progressive disclosure.

    Provides methods for registering skills, getting available skills
    with scope layering, and matching skills to user queries.
    """

    def __init__(
        self,
        store: ISkillStore,
        matcher: Optional[Any] = None,  # SkillMatcher, optional for enhanced matching
        relationship_service: Optional[Any] = None,
        auto_analyze_relationships: bool = False,
        inheritance_service: Optional[Any] = None,
    ):
        """
        Initialize the service.

        Args:
            store: The storage backend for skills
            matcher: Optional SkillMatcher for enhanced matching
            relationship_service: Optional SkillRelationshipService
            auto_analyze_relationships: If True, trigger relationship
                analysis automatically after each skill registration
            inheritance_service: Optional SkillInheritanceService for
                cross-scope inheritance resolution
        """
        self._store = store
        self._matcher = matcher
        self._relationship_service = relationship_service
        self._auto_analyze = auto_analyze_relationships
        self._inheritance_service = inheritance_service
        # Cache for skill metadata by (user_id, project_id, inherited) tuple
        self._metadata_cache: Dict[tuple, List[SkillMetadata]] = {}
        self._cache_enabled = True

    async def initialize(self) -> None:
        """Initialize the underlying store."""
        await self._store.initialize()

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable metadata caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._metadata_cache.clear()

    def invalidate_cache(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> None:
        """
        Invalidate the metadata cache.

        Args:
            user_id: If provided, only invalidate caches involving this user
            project_id: If provided, only invalidate caches involving this project
        """
        if user_id is None and project_id is None:
            self._metadata_cache.clear()
            return

        # Remove specific cache entries
        keys_to_remove = []
        for key in self._metadata_cache:
            cached_user = key[0]
            cached_project = key[1]
            if user_id and cached_user == user_id:
                keys_to_remove.append(key)
            elif project_id and cached_project == project_id:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._metadata_cache[key]

    # =========================================================================
    # Skill Registration
    # =========================================================================

    async def register_skill(self, skill: Skill) -> None:
        """
        Register a new skill.

        Saves the skill to the store, invalidates caches, and optionally
        triggers automatic relationship analysis.

        Args:
            skill: The skill to register
        """
        await self._store.save(skill)

        # Invalidate relevant caches
        if skill.scope == SkillScope.USER:
            self.invalidate_cache(user_id=skill.scope_id)
        elif skill.scope == SkillScope.PROJECT:
            self.invalidate_cache(project_id=skill.scope_id)
        else:
            # Base skill affects all caches
            self._metadata_cache.clear()

        logger.debug(f"Registered skill: {skill.name} ({skill.scope.value})")

        # Auto-analyze relationships if configured
        if self._auto_analyze and self._relationship_service:
            try:
                all_skills = await self.get_available_skills(
                    user_id=(
                        skill.scope_id
                        if skill.scope == SkillScope.USER else None
                    ),
                    project_id=(
                        skill.scope_id
                        if skill.scope == SkillScope.PROJECT else None
                    ),
                )
                if len(all_skills) >= 2:
                    await self._relationship_service.analyze_relationships(
                        all_skills,
                    )
            except Exception as exc:
                logger.warning(
                    "Auto relationship analysis failed: %s", exc,
                )

    async def register_base_skill(
        self,
        name: str,
        description: str,
        content: str,
        triggers: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Skill:
        """
        Register a base (system-wide) skill.

        Args:
            name: Skill name (lowercase, hyphens only)
            description: Short description (max 256 chars)
            content: Full markdown workflow content
            triggers: Keywords/patterns that activate this skill
            prerequisites: Other skills required
            allowed_tools: Tools this skill can use
            metadata: Additional metadata

        Returns:
            The registered skill
        """
        skill = Skill(
            name=name,
            description=description,
            scope=SkillScope.BASE,
            scope_id="system",
            content=content,
            triggers=triggers or [],
            prerequisites=prerequisites or [],
            allowed_tools=allowed_tools or [],
            metadata=metadata or {},
        )
        await self.register_skill(skill)
        return skill

    async def register_user_skill(
        self,
        user_id: str,
        name: str,
        description: str,
        content: str,
        triggers: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Skill:
        """
        Register a user-specific skill.

        Args:
            user_id: The user ID
            name: Skill name (lowercase, hyphens only)
            description: Short description (max 256 chars)
            content: Full markdown workflow content
            triggers: Keywords/patterns that activate this skill
            prerequisites: Other skills required
            allowed_tools: Tools this skill can use
            metadata: Additional metadata

        Returns:
            The registered skill
        """
        skill = Skill(
            name=name,
            description=description,
            scope=SkillScope.USER,
            scope_id=user_id,
            content=content,
            triggers=triggers or [],
            prerequisites=prerequisites or [],
            allowed_tools=allowed_tools or [],
            metadata=metadata or {},
        )
        await self.register_skill(skill)
        return skill

    async def register_project_skill(
        self,
        project_id: str,
        name: str,
        description: str,
        content: str,
        triggers: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Skill:
        """
        Register a project-specific skill.

        Args:
            project_id: The project ID
            name: Skill name (lowercase, hyphens only)
            description: Short description (max 256 chars)
            content: Full markdown workflow content
            triggers: Keywords/patterns that activate this skill
            prerequisites: Other skills required
            allowed_tools: Tools this skill can use
            metadata: Additional metadata

        Returns:
            The registered skill
        """
        skill = Skill(
            name=name,
            description=description,
            scope=SkillScope.PROJECT,
            scope_id=project_id,
            content=content,
            triggers=triggers or [],
            prerequisites=prerequisites or [],
            allowed_tools=allowed_tools or [],
            metadata=metadata or {},
        )
        await self.register_skill(skill)
        return skill

    # =========================================================================
    # Skill Retrieval
    # =========================================================================

    async def get_available_skills(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        include_inherited: bool = False,
    ) -> List[SkillMetadata]:
        """
        Get all available skills (metadata only) with layering.

        Returns skills from BASE, then USER (if user_id provided),
        then PROJECT (if project_id provided). Later scopes override
        earlier ones by name.

        When *include_inherited* is True and a ``SkillInheritanceService``
        is configured, returns the full additive inheritance set instead
        of the default override-only behavior.

        Args:
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills
            include_inherited: If True, use inheritance-aware resolution

        Returns:
            List of SkillMetadata with scope layering applied
        """
        cache_key = (user_id, project_id, include_inherited)

        if self._cache_enabled and cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        if include_inherited and self._inheritance_service is not None:
            if project_id:
                skills = await self._inheritance_service.get_inherited_skills(
                    scope=SkillScope.PROJECT,
                    scope_id=project_id,
                    user_id=user_id,
                )
            elif user_id:
                skills = await self._inheritance_service.get_inherited_skills(
                    scope=SkillScope.USER,
                    scope_id=user_id,
                )
            else:
                skills = await self._inheritance_service.get_inherited_skills(
                    scope=SkillScope.BASE,
                    scope_id="system",
                )
        else:
            skills = await self._store.list_all_metadata(user_id, project_id)

        if self._cache_enabled:
            self._metadata_cache[cache_key] = skills

        return skills

    async def get_skills_index(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> SkillsIndex:
        """
        Get a SkillsIndex for prompt injection.

        Args:
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills

        Returns:
            SkillsIndex with scope counts
        """
        skills = await self.get_available_skills(user_id, project_id)

        # Count by scope
        scope_counts: Dict[SkillScope, int] = {}
        for skill in skills:
            scope_counts[skill.scope] = scope_counts.get(skill.scope, 0) + 1

        return SkillsIndex(skills=skills, scope_counts=scope_counts)

    async def load_skill_content(
        self,
        name: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Optional[Skill]:
        """
        Load full skill content (on-demand).

        Searches in order: PROJECT -> USER -> BASE

        Args:
            name: The skill name
            user_id: Optional user ID
            project_id: Optional project ID

        Returns:
            Full Skill with content, or None if not found
        """
        # Try project first
        if project_id:
            skill = await self._store.get(name, SkillScope.PROJECT, project_id)
            if skill:
                return skill

        # Try user
        if user_id:
            skill = await self._store.get(name, SkillScope.USER, user_id)
            if skill:
                return skill

        # Try base
        skill = await self._store.get(name, SkillScope.BASE, "system")
        return skill

    # =========================================================================
    # Skill Matching
    # =========================================================================

    async def match_skills(
        self,
        query: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        threshold: float = 0.7
    ) -> List[SkillMatch]:
        """
        Find skills that match a query based on triggers.

        Args:
            query: The user query to match against
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills
            threshold: Minimum confidence threshold (0.0 - 1.0)

        Returns:
            List of SkillMatch objects sorted by confidence
        """
        if self._matcher:
            # Use enhanced matcher if available
            available = await self.get_available_skills(user_id, project_id)
            return await self._matcher.match(query, available, threshold)

        # Fall back to store's trigger search
        matches = await self._store.search_by_trigger(query, user_id, project_id)
        return [m for m in matches if m.confidence >= threshold]

    # =========================================================================
    # Category / Tag / Recommendation Queries
    # =========================================================================

    async def search_by_category(
        self,
        category: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """
        Search for skills matching a category.

        Args:
            category: The category to search for
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills

        Returns:
            List of SkillMetadata matching the category
        """
        return await self._store.search_by_category(
            category, user_id, project_id
        )

    async def search_by_tags(
        self,
        tags: List[str],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """
        Search for skills matching any of the provided tags.

        Args:
            tags: Tags to search for (matches any)
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills

        Returns:
            List of SkillMetadata matching at least one tag
        """
        all_meta = await self.get_available_skills(user_id, project_id)
        tag_set = set(t.lower() for t in tags)
        return [
            m for m in all_meta
            if tag_set & set(t.lower() for t in m.tags)
        ]

    async def get_recommended_skills(
        self,
        active_skills: List[str],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """
        Get recommended skills based on currently active skills.

        Returns skills that have COMPOSE_WITH or DEPEND_ON relationships
        with any of the active skills, excluding already-active ones.

        Args:
            active_skills: Names of currently active skills
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills

        Returns:
            List of recommended SkillMetadata
        """
        if not active_skills:
            return []

        active_set = set(active_skills)
        all_rels = await self._store.get_all_relationships()
        all_meta = await self.get_available_skills(user_id, project_id)
        meta_by_name = {m.name: m for m in all_meta}

        recommended_names: set = set()
        for rel in all_rels:
            if rel.relation_type not in (
                SkillRelationType.COMPOSE_WITH,
                SkillRelationType.DEPEND_ON,
            ):
                continue
            if rel.source in active_set and rel.target not in active_set:
                recommended_names.add(rel.target)
            if rel.target in active_set and rel.source not in active_set:
                recommended_names.add(rel.source)

        return [
            meta_by_name[name]
            for name in sorted(recommended_names)
            if name in meta_by_name
        ]

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_skills_index(self, skills: List[SkillMetadata]) -> str:
        """
        Format skill metadata for prompt injection.

        Args:
            skills: List of skill metadata

        Returns:
            Formatted string for prompt injection
        """
        index = SkillsIndex(skills=skills)
        return index.format_for_prompt()

    def format_skill_workflow(
        self,
        skill: Skill,
        detail_level: int = 1,
    ) -> str:
        """
        Format a skill's workflow for prompt injection.

        Args:
            skill: The full skill to format
            detail_level: 1=instructions only, 2=+scripts, 3=+references

        Returns:
            Formatted workflow string
        """
        from ctxforge.engine.services.skill_content_loader import (
            SkillContentLoader,
        )

        lines = [
            f"## Skill: {skill.name}",
            "",
            f"**Description:** {skill.description}",
            "",
        ]

        if skill.prerequisites:
            lines.append(f"**Prerequisites:** {', '.join(skill.prerequisites)}")
            lines.append("")

        if skill.allowed_tools:
            lines.append(f"**Allowed Tools:** {', '.join(skill.allowed_tools)}")
            lines.append("")

        lines.append("### Workflow")
        lines.append("")

        if skill.structured_content is not None:
            include_scripts = detail_level >= 2
            include_references = detail_level >= 3
            formatted = SkillContentLoader.format_for_prompt(
                skill.structured_content,
                include_scripts=include_scripts,
                include_references=include_references,
            )
            lines.append(formatted)
        else:
            lines.append(skill.content)

        return "\n".join(lines)

    # =========================================================================
    # Deletion
    # =========================================================================

    async def delete_skill(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> bool:
        """
        Delete a skill.

        Args:
            name: The skill name
            scope: The skill scope
            scope_id: The scope ID

        Returns:
            True if deleted, False if not found
        """
        deleted = await self._store.delete(name, scope, scope_id)

        if deleted:
            # Invalidate relevant caches
            if scope == SkillScope.USER:
                self.invalidate_cache(user_id=scope_id)
            elif scope == SkillScope.PROJECT:
                self.invalidate_cache(project_id=scope_id)
            else:
                self._metadata_cache.clear()

        return deleted

    # =========================================================================
    # Stats
    # =========================================================================

    async def count(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count skills, optionally filtered by scope."""
        return await self._store.count(scope, scope_id)
