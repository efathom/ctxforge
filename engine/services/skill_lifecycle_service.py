"""
Skill Lifecycle Service.

Orchestrates the full skill lifecycle from creation to retirement:
Generate -> Validate -> Evaluate -> Persist -> Analyze Relationships.

Every step persists results to the database via ISkillStore.
"""
import logging
from typing import Dict, List, Optional

from ctxforge.core.events import Event
from ctxforge.core.skill import Skill, SkillEvaluation, SkillScope
from ctxforge.engine.services.skill_effectiveness_service import (
    SkillEffectivenessService,
)
from ctxforge.engine.services.skill_evaluation_service import (
    SkillEvaluationService,
)
from ctxforge.engine.services.skill_generator_service import (
    SkillGeneratorService,
)
from ctxforge.engine.services.skill_relationship_service import (
    SkillRelationshipService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.engine.services.skill_validator import SkillValidator

logger = logging.getLogger(__name__)


class SkillLifecycleService:
    """Orchestrates the full skill lifecycle from creation to retirement.

    Every step persists results to the database.
    """

    def __init__(
        self,
        generator: SkillGeneratorService,
        validator: SkillValidator,
        evaluator: SkillEvaluationService,
        skill_service: SkillService,
        relationship_service: Optional[SkillRelationshipService] = None,
        effectiveness_service: Optional[SkillEffectivenessService] = None,
        min_evaluation_score: float = 0.4,
    ):
        self._generator = generator
        self._validator = validator
        self._evaluator = evaluator
        self._skill_service = skill_service
        self._relationship_service = relationship_service
        self._effectiveness_service = effectiveness_service
        self._min_evaluation_score = min_evaluation_score

    async def create_from_session(
        self,
        events: List[Event],
        project_id: str,
    ) -> List[Skill]:
        """Full pipeline: generate, validate, evaluate, persist, analyze.

        1. Generate candidate skills from events
        2. Validate each candidate -> reject invalid
        3. Evaluate each valid candidate via LLM
        4. Reject candidates scoring < min_evaluation_score
        5. Persist each passing skill with evaluation attached
        6. Analyze relationships among all skills
        7. Return list of registered skills

        Args:
            events: List of session events.
            project_id: Project to register generated skills under.

        Returns:
            List of persisted Skill objects that passed all gates.
        """
        raw_skills = await self._generator.generate_from_session(
            events, project_id, deduplicate=True,
        )
        if not raw_skills:
            return []

        accepted: List[Skill] = []
        for skill in raw_skills:
            result = await self._validate_evaluate_persist(skill)
            if result is not None:
                accepted.append(result)

        if accepted and self._relationship_service:
            await self._analyze_relationships(project_id)

        return accepted

    async def create_from_github(
        self,
        github_url: str,
        project_id: str,
        github_token: Optional[str] = None,
    ) -> Optional[Skill]:
        """Pipeline for GitHub-sourced skills.

        1. Generate from repo
        2. Validate -> Evaluate -> Persist (with evaluation)
        3. Analyze relationships

        Args:
            github_url: GitHub repository URL.
            project_id: Project to register the skill under.
            github_token: Optional GitHub API token.

        Returns:
            A persisted Skill, or None if any gate rejected it.
        """
        skill = await self._generator.generate_from_github(
            github_url, project_id, github_token,
        )
        if skill is None:
            return None

        result = await self._validate_evaluate_persist(skill)
        if result is not None and self._relationship_service:
            await self._analyze_relationships(project_id)

        return result

    async def create_from_document(
        self,
        file_path: str,
        project_id: str,
    ) -> Optional[Skill]:
        """Pipeline for document-sourced skills.

        1. Generate from file
        2. Validate -> Evaluate -> Persist (with evaluation)
        3. Analyze relationships

        Args:
            file_path: Path to the office document.
            project_id: Project to register the skill under.

        Returns:
            A persisted Skill, or None if any gate rejected it.
        """
        skill = await self._generator.generate_from_document(
            file_path, project_id,
        )
        if skill is None:
            return None

        result = await self._validate_evaluate_persist(skill)
        if result is not None and self._relationship_service:
            await self._analyze_relationships(project_id)

        return result

    async def retire_underperforming(
        self,
        scope: SkillScope,
        scope_id: str,
        min_success_rate: float = 0.3,
        min_usage_count: int = 5,
    ) -> List[str]:
        """Find and delete skills with poor effectiveness.

        Only retires skills that have been used at least min_usage_count
        times AND have a success_rate below min_success_rate.

        Args:
            scope: Skill scope to scan.
            scope_id: Scope identifier.
            min_success_rate: Minimum acceptable success rate.
            min_usage_count: Minimum usage count before considering retirement.

        Returns:
            List of retired skill names.
        """
        all_meta = await self._skill_service._store.list_metadata(
            scope, scope_id,
        )

        retired: List[str] = []
        for meta in all_meta:
            skill = await self._skill_service._store.get(
                meta.name, scope, scope_id,
            )
            if skill is None:
                continue

            eff = skill.effectiveness or {}
            usage_count = eff.get("usage_count", 0)
            success_rate = eff.get("success_rate", 0.0)

            if usage_count >= min_usage_count and success_rate < min_success_rate:
                deleted = await self._skill_service.delete_skill(
                    meta.name, scope, scope_id,
                )
                if deleted:
                    retired.append(meta.name)
                    logger.info(
                        "Retired skill '%s' (usage=%d, success_rate=%.2f)",
                        meta.name, usage_count, success_rate,
                    )

        return retired

    async def refresh_evaluation(
        self,
        scope: SkillScope,
        scope_id: str,
    ) -> Dict[str, SkillEvaluation]:
        """Re-evaluate all skills in a scope and persist updated evaluations.

        Args:
            scope: Skill scope to re-evaluate.
            scope_id: Scope identifier.

        Returns:
            Dict mapping skill name to its new evaluation.
        """
        all_meta = await self._skill_service._store.list_metadata(
            scope, scope_id,
        )

        results: Dict[str, SkillEvaluation] = {}
        for meta in all_meta:
            skill = await self._skill_service._store.get(
                meta.name, scope, scope_id,
            )
            if skill is None:
                continue

            try:
                evaluation = await self._evaluator.evaluate(skill)
                skill.evaluation = evaluation
                await self._skill_service._store.save(skill)
                results[skill.name] = evaluation
            except Exception as exc:
                logger.warning(
                    "Failed to re-evaluate skill '%s': %s",
                    skill.name, exc,
                )

        return results

    async def _validate_evaluate_persist(
        self, skill: Skill,
    ) -> Optional[Skill]:
        """Validate, evaluate, and persist a single skill.

        Returns the skill with evaluation attached if it passes all gates,
        or None if rejected.
        """
        validation = self._validator.validate(skill)
        if not validation.is_valid:
            logger.info(
                "Skill '%s' rejected by validator: %s",
                skill.name, validation.errors,
            )
            return None

        try:
            evaluation = await self._evaluator.evaluate(skill)
        except Exception as exc:
            logger.warning(
                "Evaluation failed for skill '%s': %s", skill.name, exc,
            )
            return None

        if evaluation.overall_score < self._min_evaluation_score:
            logger.info(
                "Skill '%s' rejected: score %.2f < threshold %.2f",
                skill.name, evaluation.overall_score,
                self._min_evaluation_score,
            )
            return None

        skill.evaluation = evaluation
        await self._skill_service.register_skill(skill)
        return skill

    async def _analyze_relationships(self, project_id: str) -> None:
        """Analyze relationships among all available skills."""
        if not self._relationship_service:
            return

        try:
            all_skills = await self._skill_service.get_available_skills(
                project_id=project_id,
            )
            if len(all_skills) >= 2:
                await self._relationship_service.analyze_relationships(
                    all_skills,
                )
        except Exception as exc:
            logger.warning("Relationship analysis failed: %s", exc)
