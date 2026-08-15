"""
Skills Middleware.

Injects available skills into prompts with progressive disclosure,
and auto-activates matching skills based on user queries.

Supports companion skill loading via relationship graph,
sub-skill chain resolution for bundled skills, and
prerequisite enforcement via session state tracking.
"""
import logging
from typing import List, Optional, Set

from ctxforge.core.skill import Skill, SkillRelationType
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction
from ctxforge.protocols.skill import ISkillStore

logger = logging.getLogger(__name__)

# Session state key for tracking completed skills
COMPLETED_SKILLS_KEY = "completed_skills"


class SkillsMiddleware(BaseMiddleware):
    """
    Injects available skills into prompts with progressive disclosure.

    This middleware:
    1. Injects a skills index (lightweight metadata) into the prompt
    2. Optionally auto-activates matching skills based on the user query
    3. Loads full skill content on-demand when activated
    """

    def __init__(
        self,
        skill_service: SkillService,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        auto_activate: bool = True,
        max_auto_skills: int = 2,
        confidence_threshold: float = 0.7,
        enabled: bool = True,
        skill_store: Optional[ISkillStore] = None,
        load_companions: bool = True,
        resolve_composed: bool = True,
        enforce_prerequisites: bool = True,
        include_inherited: bool = False,
    ):
        """
        Initialize the middleware.

        Args:
            skill_service: The skill service
            user_id: Optional user ID for user-specific skills
            project_id: Optional project ID for project-specific skills
            auto_activate: Whether to auto-load matching skills
            max_auto_skills: Maximum number of skills to auto-activate
            confidence_threshold: Minimum confidence for auto-activation
            enabled: Whether the middleware is enabled
            skill_store: Optional skill store for relationship lookups
            load_companions: Whether to load companion skills via relationships
            resolve_composed: Whether to resolve composed_of sub-skill chains
            enforce_prerequisites: Whether to check prerequisite completion
                before activating skills
            include_inherited: Whether to include inherited skills in index
        """
        super().__init__(enabled=enabled)
        self._skill_service = skill_service
        self._user_id = user_id
        self._project_id = project_id
        self._auto_activate = auto_activate
        self._max_auto_skills = max_auto_skills
        self._confidence_threshold = confidence_threshold
        self._skill_store = skill_store
        self._load_companions = load_companions
        self._resolve_composed = resolve_composed
        self._enforce_prerequisites = enforce_prerequisites
        self._include_inherited = include_inherited

    @property
    def name(self) -> str:
        """Unique identifier for this middleware."""
        return "skills"

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID."""
        return self._user_id

    @user_id.setter
    def user_id(self, value: Optional[str]) -> None:
        """Set the user ID."""
        self._user_id = value

    @property
    def project_id(self) -> Optional[str]:
        """Get the project ID."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: Optional[str]) -> None:
        """Set the project ID."""
        self._project_id = value

    @property
    def auto_activate(self) -> bool:
        """Get auto-activate setting."""
        return self._auto_activate

    @auto_activate.setter
    def auto_activate(self, value: bool) -> None:
        """Set auto-activate setting."""
        self._auto_activate = value

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Inject skills into the context.

        Args:
            context: The current middleware context
            next: Function to call the next middleware

        Returns:
            The processed context with skills injected
        """
        skills_index_text = ""
        activated_skills_text = ""
        activated_skill_names: List[str] = []

        try:
            # 1. Get available skills (metadata only - progressive disclosure)
            skills = await self._skill_service.get_available_skills(
                user_id=self._user_id,
                project_id=self._project_id,
                include_inherited=self._include_inherited,
            )

            # 2. Format skills index for injection
            if skills:
                skills_index_text = self._skill_service.format_skills_index(skills)
                context.add_flag("skills_index_injected")

            # 3. Auto-activate matching skills
            if self._auto_activate and context.user_input:
                activated_skills = await self._auto_activate_skills(
                    context.user_input
                )

                # 3a. Enforce prerequisites: filter out skills whose
                # prerequisites have not been completed in this session.
                if self._enforce_prerequisites and activated_skills:
                    activated_skills, blocked = self._filter_by_prerequisites(
                        activated_skills, context
                    )
                    if blocked:
                        blocked_text = self._format_blocked_skills(blocked)
                        context.add_section(
                            name="blocked_skills",
                            content=blocked_text,
                            priority=66,
                            is_required=False,
                        )
                        context.add_flag("skills_blocked_by_prerequisites")

                if activated_skills:
                    activated_skills_text = self._format_activated_skills(
                        activated_skills
                    )
                    activated_skill_names = [s.name for s in activated_skills]
                    context.add_flag("skills_auto_activated")

            # 4. Inject as context sections (not into processed_input)
            if skills_index_text:
                context.add_section(
                    name="skills_index",
                    content=skills_index_text,
                    priority=60,
                    is_required=False,
                )
            if activated_skills_text:
                context.add_section(
                    name="activated_skills",
                    content=activated_skills_text,
                    priority=65,
                    is_required=False,
                )

            if skills_index_text or activated_skills_text:
                context.record_modification(self.name, {
                    "action": "injected_skills",
                    "available_count": len(skills),
                    "activated_skills": activated_skill_names,
                })

                logger.debug(
                    f"Injected {len(skills)} skills, "
                    f"activated: {activated_skill_names}"
                )

        except Exception as e:
            logger.warning(f"Failed to inject skills: {e}")
            context.set_metadata(f"{self.name}_error", str(e))

        return await next(context)

    def _get_completed_skills(self, context: MiddlewareContext) -> Set[str]:
        """Get the set of skill names completed in this session.

        Reads from session state if a session is available, otherwise
        falls back to middleware context metadata.
        """
        if context.session and hasattr(context.session, "state"):
            return set(
                context.session.state.get(COMPLETED_SKILLS_KEY, [])
            )
        return set(context.get_metadata(COMPLETED_SKILLS_KEY, []))

    def _filter_by_prerequisites(
        self,
        skills: List[Skill],
        context: MiddlewareContext,
    ) -> tuple:
        """Filter skills based on prerequisite completion.

        Args:
            skills: Candidate skills to activate.
            context: Current middleware context (provides session state).

        Returns:
            Tuple of (allowed_skills, blocked_skills) where each blocked
            entry is (skill, missing_prerequisites).
        """
        completed = self._get_completed_skills(context)
        allowed: List[Skill] = []
        blocked: List[tuple] = []

        for skill in skills:
            if not skill.prerequisites:
                allowed.append(skill)
                continue

            missing = [p for p in skill.prerequisites if p not in completed]
            if missing:
                blocked.append((skill, missing))
                logger.debug(
                    f"Skill '{skill.name}' blocked: prerequisites "
                    f"{missing} not completed"
                )
            else:
                allowed.append(skill)

        return allowed, blocked

    def _format_blocked_skills(
        self,
        blocked: List[tuple],
    ) -> str:
        """Format blocked skills info for context injection.

        Args:
            blocked: List of (skill, missing_prerequisites) tuples.

        Returns:
            Formatted text explaining which skills are blocked and why.
        """
        lines = [
            "## Blocked Skills (Prerequisites Not Met)",
            "",
            "The following skills matched your request but cannot be "
            "activated until their prerequisite skills are completed:",
            "",
        ]
        for skill, missing in blocked:
            missing_str = ", ".join(f"`{m}`" for m in missing)
            lines.append(
                f"- **{skill.name}**: requires {missing_str} first"
            )

        lines.append("")
        lines.append(
            "Complete the prerequisite skills, then mark them done with "
            "`mark_skill_completed` before proceeding."
        )
        return "\n".join(lines)

    @staticmethod
    def mark_skill_completed(
        context: MiddlewareContext,
        skill_name: str,
    ) -> None:
        """Mark a skill as completed in the session state.

        This should be called after a skill's workflow has been fully
        executed, allowing dependent skills to be activated.

        Args:
            context: The middleware context (with session attached).
            skill_name: Name of the completed skill.
        """
        if context.session and hasattr(context.session, "state"):
            completed = list(
                context.session.state.get(COMPLETED_SKILLS_KEY, [])
            )
            if skill_name not in completed:
                completed.append(skill_name)
                context.session.state.set(COMPLETED_SKILLS_KEY, completed)
        else:
            completed = list(
                context.get_metadata(COMPLETED_SKILLS_KEY, [])
            )
            if skill_name not in completed:
                completed.append(skill_name)
                context.set_metadata(COMPLETED_SKILLS_KEY, completed)

    async def _auto_activate_skills(
        self,
        user_input: str,
    ) -> List[Skill]:
        """
        Find and load matching skills based on user input.

        After loading the top matches, also loads companion skills
        (via COMPOSE_WITH relationships) and resolves composed_of
        sub-skill chains, up to max_auto_skills total.

        Args:
            user_input: The user's input text

        Returns:
            List of fully loaded skills
        """
        # Match skills to the query
        matches = await self._skill_service.match_skills(
            query=user_input,
            user_id=self._user_id,
            project_id=self._project_id,
            threshold=self._confidence_threshold,
        )

        # Take top N matches
        top_matches = matches[:self._max_auto_skills]

        # Load full skill content for each match
        activated: List[Skill] = []
        activated_names: Set[str] = set()
        for match in top_matches:
            skill = await self._skill_service.load_skill_content(
                name=match.skill.name,
                user_id=self._user_id,
                project_id=self._project_id,
            )
            if skill:
                activated.append(skill)
                activated_names.add(skill.name)
                logger.debug(
                    f"Auto-activated skill: {skill.name} "
                    f"(confidence: {match.confidence:.2f})"
                )

        # Load companion skills via COMPOSE_WITH relationships
        if self._load_companions and self._skill_store and activated_names:
            companions = await self._find_companion_skills(activated_names)
            for comp_name in companions:
                if len(activated) >= self._max_auto_skills:
                    break
                if comp_name in activated_names:
                    continue
                comp_skill = await self._skill_service.load_skill_content(
                    name=comp_name,
                    user_id=self._user_id,
                    project_id=self._project_id,
                )
                if comp_skill:
                    activated.append(comp_skill)
                    activated_names.add(comp_skill.name)
                    logger.debug(
                        f"Auto-activated companion skill: {comp_skill.name}"
                    )

        # Resolve composed_of sub-skill chains
        if self._resolve_composed:
            sub_skills = await self._resolve_composed_skills(
                activated, activated_names
            )
            for sub in sub_skills:
                if len(activated) >= self._max_auto_skills:
                    break
                activated.append(sub)
                activated_names.add(sub.name)
                logger.debug(
                    f"Auto-activated sub-skill: {sub.name}"
                )

        return activated

    async def _find_companion_skills(
        self,
        active_names: Set[str],
    ) -> List[str]:
        """Find companion skill names via COMPOSE_WITH relationships.

        Args:
            active_names: Names of currently active skills

        Returns:
            List of companion skill names not already active
        """
        companion_names: List[str] = []
        for name in active_names:
            rels = await self._skill_store.get_relationships(name)
            for rel in rels:
                if rel.relation_type != SkillRelationType.COMPOSE_WITH:
                    continue
                other = rel.target if rel.source == name else rel.source
                if other not in active_names and other not in companion_names:
                    companion_names.append(other)
        return companion_names

    async def _resolve_composed_skills(
        self,
        activated: List[Skill],
        activated_names: Set[str],
    ) -> List[Skill]:
        """Resolve composed_of sub-skill chains for activated skills.

        Args:
            activated: Currently activated skills
            activated_names: Names of activated skills

        Returns:
            List of additional sub-skills to load
        """
        sub_skills: List[Skill] = []
        for skill in list(activated):
            if not skill.composed_of:
                continue
            for sub_name in skill.composed_of:
                if sub_name in activated_names:
                    continue
                sub_skill = await self._skill_service.load_skill_content(
                    name=sub_name,
                    user_id=self._user_id,
                    project_id=self._project_id,
                )
                if sub_skill:
                    sub_skills.append(sub_skill)
                    activated_names.add(sub_name)
        return sub_skills

    def _format_activated_skills(self, skills: List[Skill]) -> str:
        """
        Format activated skills for prompt injection.

        Args:
            skills: List of activated skills

        Returns:
            Formatted skills text
        """
        if not skills:
            return ""

        lines = ["## Activated Skills", ""]
        lines.append(
            "The following skills have been activated based on your request:"
        )
        lines.append("")

        for skill in skills:
            lines.append(self._skill_service.format_skill_workflow(skill))
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

class SkillRequestMiddleware(BaseMiddleware):
    """
    Middleware that handles explicit skill requests.

    Detects when a user explicitly requests a skill by name
    and loads the full skill content.
    """

    def __init__(
        self,
        skill_service: SkillService,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        request_patterns: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.

        Args:
            skill_service: The skill service
            user_id: Optional user ID
            project_id: Optional project ID
            request_patterns: Patterns that indicate a skill request
                             (default: ["use skill", "activate skill", "run skill"])
            enabled: Whether the middleware is enabled
        """
        super().__init__(enabled=enabled)
        self._skill_service = skill_service
        self._user_id = user_id
        self._project_id = project_id
        self._request_patterns = request_patterns or [
            "use skill",
            "activate skill",
            "run skill",
            "execute skill",
            "apply skill",
        ]

    @property
    def name(self) -> str:
        """Unique identifier for this middleware."""
        return "skill_request"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Check for explicit skill requests and load the skill.

        Args:
            context: The current middleware context
            next: Function to call the next middleware

        Returns:
            The processed context
        """
        if not context.user_input:
            return await next(context)

        input_lower = context.user_input.lower()

        # Check for skill request patterns
        for pattern in self._request_patterns:
            if pattern in input_lower:
                # Extract skill name (assume it follows the pattern)
                skill_name = self._extract_skill_name(
                    context.user_input, pattern
                )
                if skill_name:
                    await self._load_and_inject_skill(context, skill_name)
                break

        return await next(context)

    def _extract_skill_name(
        self,
        input_text: str,
        pattern: str,
    ) -> Optional[str]:
        """
        Extract skill name from input following a pattern.

        Args:
            input_text: The user input
            pattern: The pattern that was matched

        Returns:
            The skill name or None
        """
        input_lower = input_text.lower()
        pattern_idx = input_lower.find(pattern)
        if pattern_idx == -1:
            return None

        # Get text after the pattern
        after_pattern = input_text[pattern_idx + len(pattern):].strip()

        # Extract the first word (skill name)
        words = after_pattern.split()
        if words:
            # Clean up the skill name (remove quotes, punctuation)
            skill_name = words[0].strip("'\".,!?")
            # Validate format (lowercase with hyphens)
            if skill_name and all(c.isalnum() or c == '-' for c in skill_name):
                return skill_name.lower()

        return None

    async def _load_and_inject_skill(
        self,
        context: MiddlewareContext,
        skill_name: str,
    ) -> None:
        """
        Load a skill and inject it into the context.

        Args:
            context: The middleware context
            skill_name: Name of the skill to load
        """
        try:
            skill = await self._skill_service.load_skill_content(
                name=skill_name,
                user_id=self._user_id,
                project_id=self._project_id,
            )

            if skill:
                workflow = self._skill_service.format_skill_workflow(skill)
                context.add_section(
                    name="requested_skill",
                    content=(
                        f"## Requested Skill: {skill_name}\n\n"
                        f"{workflow}"
                    ),
                    priority=70,
                    is_required=True,
                )
                context.add_flag("skill_requested")
                context.set_metadata("requested_skill", skill_name)
                context.record_modification(self.name, {
                    "action": "loaded_skill",
                    "skill_name": skill_name,
                })
                logger.debug(f"Loaded requested skill: {skill_name}")
            else:
                context.set_metadata("skill_not_found", skill_name)
                logger.debug(f"Requested skill not found: {skill_name}")

        except Exception as e:
            logger.warning(f"Failed to load skill {skill_name}: {e}")
            context.set_metadata(f"{self.name}_error", str(e))
