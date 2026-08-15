"""
Skill Generator Service.

Generates reusable skills from completed sessions, observations,
scoped memories, or user prompts using a two-phase LLM approach.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from ctxforge.config.base import SkillGenerationConfig
from ctxforge.core.events import Event
from ctxforge.core.observation import Observation
from ctxforge.core.scoped_memory import ScopedMemory
from ctxforge.core.skill import Skill, SkillContent, SkillScope
from ctxforge.engine.prompts.skill_generation import (
    CANDIDATE_METADATA_SYSTEM_PROMPT,
    PROMPT_SKILL_SYSTEM_PROMPT,
    SKILL_CONTENT_SYSTEM_PROMPT,
    build_candidate_prompt,
    build_content_prompt,
    build_prompt_skill_prompt,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.engine.services.skill_validator import SkillValidator
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

logger = logging.getLogger(__name__)

_KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class SkillGeneratorService:
    """Generate skills from sessions, observations, memories, or prompts."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        skill_service: SkillService,
        config: Optional[SkillGenerationConfig] = None,
        validator: Optional[SkillValidator] = None,
    ):
        self._llm = llm_provider
        self._skill_service = skill_service
        self._config = config or SkillGenerationConfig()
        self._validator = validator or SkillValidator()

    async def generate_from_session(
        self,
        events: List[Event],
        project_id: str,
        max_retries: int = 2,
        deduplicate: bool = True,
    ) -> List[Skill]:
        """Generate skills from session events with dedup and retry.

        Phase 1: Extract candidate metadata.
        Phase 1.5: Deduplicate against existing skills in the database.
        Phase 2: Generate content with validation + retry.
        Phase 3: Persist each valid skill via register_skill().

        Args:
            events: List of session events.
            project_id: Project to register generated skills under.
            max_retries: Number of retries on validation failure.
            deduplicate: If True, skip candidates whose names already
                exist in the store.

        Returns:
            List of generated Skill objects.
        """
        if len(events) < self._config.min_session_events:
            return []

        events_text = self._format_events(events)
        candidates = await self._extract_candidates(events_text)

        if deduplicate:
            existing = await self._skill_service.get_available_skills(
                project_id=project_id,
            )
            existing_names = {s.name for s in existing}
            candidates = [
                c for c in candidates if c["name"] not in existing_names
            ]

        return await self._generate_skills_with_retry(
            candidates, project_id, events_text, max_retries,
        )

    async def generate_from_observations(
        self,
        observations: List[Observation],
        project_id: str,
    ) -> List[Skill]:
        """Generate skills from extracted observations.

        Args:
            observations: List of observations.
            project_id: Project to register generated skills under.

        Returns:
            List of generated Skill objects.
        """
        if not observations:
            return []

        obs_text = self._format_observations(observations)
        candidates = await self._extract_candidates(obs_text)
        return await self._generate_skills(candidates, project_id, obs_text)

    async def generate_from_memories(
        self,
        memories: List[ScopedMemory],
        project_id: str,
    ) -> List[Skill]:
        """Generate skills from scoped memory items.

        Args:
            memories: List of scoped memories.
            project_id: Project to register generated skills under.

        Returns:
            List of generated Skill objects.
        """
        if not memories:
            return []

        mem_text = self._format_memories(memories)
        candidates = await self._extract_candidates(mem_text)
        return await self._generate_skills(candidates, project_id, mem_text)

    async def generate_from_prompt(
        self,
        description: str,
        project_id: str,
    ) -> Optional[Skill]:
        """Generate a single skill from a text description.

        Args:
            description: Natural language description of the desired skill.
            project_id: Project to register the skill under.

        Returns:
            A generated Skill, or None if generation failed.
        """
        user_prompt = build_prompt_skill_prompt(description)
        messages = [
            ChatMessage(role="system", content=PROMPT_SKILL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.chat(
            messages=messages,
            model=self._config.model,
            temperature=0.3,
            max_tokens=2048,
        )

        data = self._parse_json(response.content)
        if data is None:
            return None

        name = data.get("name", "")
        if not _KEBAB_RE.match(name):
            logger.debug("Generated skill name '%s' is not valid kebab-case", name)
            return None

        skill_description = data.get("description", "")[:256]
        instructions = data.get("instructions", "")
        triggers = data.get("triggers", [])
        category = data.get("category", "other")
        when_to_use = data.get("when_to_use", "")

        skill = Skill(
            name=name,
            description=skill_description,
            scope=SkillScope.PROJECT,
            scope_id=project_id,
            content=instructions,
            triggers=triggers,
            category=category,
            when_to_use=when_to_use,
            structured_content=SkillContent(instructions=instructions),
        )

        await self._skill_service.register_skill(skill)
        return skill

    async def generate_from_github(
        self,
        github_url: str,
        project_id: str = "default",
        github_token: Optional[str] = None,
    ) -> Optional[Skill]:
        """Generate a skill from a GitHub repository URL.

        Validates and persists the generated skill to the database.

        Args:
            github_url: GitHub repository URL.
            project_id: Project to register the skill under.
            github_token: Optional GitHub API token.

        Returns:
            A generated Skill, or None if generation/validation failed.
        """
        from ctxforge.engine.services.github_skill_creator import (
            GitHubSkillCreator,
        )

        creator = GitHubSkillCreator(
            llm_provider=self._llm,
            github_token=github_token,
        )
        skill = await creator.create_from_url(github_url, project_id)

        if skill is None:
            return None

        validation = self._validator.validate(skill)
        if not validation.is_valid:
            logger.warning(
                "GitHub skill failed validation: %s", validation.errors,
            )
            return None

        await self._skill_service.register_skill(skill)
        return skill

    async def generate_from_document(
        self,
        file_path: str,
        project_id: str = "default",
    ) -> Optional[Skill]:
        """Generate a skill from a PDF, DOCX, or PPTX file.

        Validates and persists the generated skill to the database.

        Args:
            file_path: Path to the office document.
            project_id: Project to register the skill under.

        Returns:
            A generated Skill, or None if generation/validation failed.
        """
        from ctxforge.engine.services.document_skill_creator import (
            DocumentSkillCreator,
        )

        creator = DocumentSkillCreator(llm_provider=self._llm)
        skill = await creator.create_from_file(file_path, project_id)

        if skill is None:
            return None

        validation = self._validator.validate(skill)
        if not validation.is_valid:
            logger.warning(
                "Document skill failed validation: %s", validation.errors,
            )
            return None

        await self._skill_service.register_skill(skill)
        return skill

    async def _extract_candidates(
        self, context_text: str
    ) -> List[Dict[str, Any]]:
        """Phase 1: Extract candidate skill metadata from context text."""
        user_prompt = build_candidate_prompt(context_text)
        messages = [
            ChatMessage(
                role="system", content=CANDIDATE_METADATA_SYSTEM_PROMPT
            ),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.chat(
            messages=messages,
            model=self._config.model,
            temperature=0.3,
            max_tokens=2048,
        )

        data = self._parse_json(response.content)
        if not isinstance(data, list):
            return []

        valid: List[Dict[str, Any]] = []
        for item in data:
            name = item.get("name", "")
            if _KEBAB_RE.match(name):
                valid.append(item)
            else:
                logger.debug("Skipping candidate with invalid name: '%s'", name)
        return valid

    async def _generate_skills(
        self,
        candidates: List[Dict[str, Any]],
        project_id: str,
        context_text: str,
    ) -> List[Skill]:
        """Phase 2: Generate full skill content for each candidate."""
        skills: List[Skill] = []
        for candidate in candidates:
            name = candidate.get("name", "")
            description = candidate.get("description", "")[:256]
            when_to_use = candidate.get("when_to_use", "")
            category = candidate.get("category", "other")

            user_prompt = build_content_prompt(
                name=name,
                description=description,
                when_to_use=when_to_use,
                category=category,
                context=context_text[:2000],
            )
            messages = [
                ChatMessage(
                    role="system", content=SKILL_CONTENT_SYSTEM_PROMPT
                ),
                ChatMessage(role="user", content=user_prompt),
            ]

            response = await self._llm.chat(
                messages=messages,
                model=self._config.model,
                temperature=0.3,
                max_tokens=2048,
            )

            content_data = self._parse_json(response.content)
            if content_data is None:
                continue

            instructions = content_data.get("instructions", "")
            scripts = content_data.get("scripts", {})
            references = content_data.get("references", {})

            skill = Skill(
                name=name,
                description=description,
                scope=SkillScope.PROJECT,
                scope_id=project_id,
                content=instructions,
                category=category,
                when_to_use=when_to_use,
                structured_content=SkillContent(
                    instructions=instructions,
                    scripts=scripts,
                    references=references,
                ),
            )

            await self._skill_service.register_skill(skill)
            skills.append(skill)

        return skills

    async def _generate_skills_with_retry(
        self,
        candidates: List[Dict[str, Any]],
        project_id: str,
        context_text: str,
        max_retries: int = 2,
    ) -> List[Skill]:
        """Generate skills with validation and retry on failure.

        For each candidate, attempts generation up to max_retries + 1 times.
        Only persists skills that pass validation.
        """
        skills: List[Skill] = []
        for candidate in candidates:
            for attempt in range(max_retries + 1):
                skill = await self._generate_single(
                    candidate, project_id, context_text,
                )
                if skill is None:
                    break  # LLM parse failure, no point retrying

                validation = self._validator.validate(skill)
                if validation.is_valid:
                    await self._skill_service.register_skill(skill)
                    skills.append(skill)
                    break
                elif attempt < max_retries:
                    logger.warning(
                        "Validation failed for '%s', retry %d/%d",
                        candidate.get("name", ""),
                        attempt + 1,
                        max_retries,
                    )
        return skills

    async def _generate_single(
        self,
        candidate: Dict[str, Any],
        project_id: str,
        context_text: str,
    ) -> Optional[Skill]:
        """Generate a single skill from a candidate metadata dict."""
        name = candidate.get("name", "")
        description = candidate.get("description", "")[:256]
        when_to_use = candidate.get("when_to_use", "")
        category = candidate.get("category", "other")
        triggers = candidate.get("triggers", [])

        user_prompt = build_content_prompt(
            name=name,
            description=description,
            when_to_use=when_to_use,
            category=category,
            context=context_text[:2000],
        )
        messages = [
            ChatMessage(
                role="system", content=SKILL_CONTENT_SYSTEM_PROMPT
            ),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.chat(
            messages=messages,
            model=self._config.model,
            temperature=0.3,
            max_tokens=2048,
        )

        content_data = self._parse_json(response.content)
        if content_data is None or not isinstance(content_data, dict):
            return None

        instructions = content_data.get("instructions", "")
        scripts = content_data.get("scripts", {})
        references = content_data.get("references", {})
        content_triggers = content_data.get("triggers", [])

        return Skill(
            name=name,
            description=description,
            scope=SkillScope.PROJECT,
            scope_id=project_id,
            content=instructions,
            category=category,
            when_to_use=when_to_use,
            triggers=content_triggers or triggers,
            structured_content=SkillContent(
                instructions=instructions,
                scripts=scripts,
                references=references,
            ),
        )

    def _format_events(self, events: List[Event]) -> str:
        """Format session events for LLM consumption."""
        parts: List[str] = []
        for ev in events:
            parts.append(f"[{ev.type.value}] {ev.content[:500]}")
        return "\n".join(parts)

    def _format_observations(self, observations: List[Observation]) -> str:
        """Format observations for LLM consumption."""
        parts: List[str] = []
        for obs in observations:
            detail = f" - {obs.detail}" if obs.detail else ""
            parts.append(f"[{obs.type.value}] {obs.summary}{detail}")
        return "\n".join(parts)

    def _format_memories(self, memories: List[ScopedMemory]) -> str:
        """Format scoped memories for LLM consumption."""
        parts: List[str] = []
        for mem in memories:
            parts.append(f"[{mem.category.value}] {mem.content[:500]}")
        return "\n".join(parts)

    def _parse_json(self, raw: str) -> Any:
        """Parse JSON from LLM response, handling markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON: %s", exc)
            return None
