"""
Skill Evaluation Service.

Provides LLM-based quality evaluation of skills across five dimensions:
Safety, Completeness, Executability, Maintainability, and Cost-Awareness.

Optionally runs Python scripts found in a skill's structured content and
feeds execution results into the evaluation prompt for grounded assessment.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple

from ctxforge.config.base import SkillEvaluationConfig
from ctxforge.core.skill import (
    EvaluationLevel,
    Skill,
    SkillEvaluation,
)
from ctxforge.engine.prompts.skill_evaluation import (
    SKILL_EVALUATION_SYSTEM_PROMPT,
    build_evaluation_prompt,
)
from ctxforge.engine.services.script_runner import (
    ScriptExecutionResult,
    ScriptRunner,
)
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

logger = logging.getLogger(__name__)

_VALID_LEVELS = {"good", "average", "poor"}


class SkillEvaluationService:
    """Evaluate skills using an LLM across five quality dimensions.

    Optionally integrates a ScriptRunner to execute Python scripts from
    a skill's structured content and include execution results in the
    evaluation prompt.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        config: Optional[SkillEvaluationConfig] = None,
        script_runner: Optional[ScriptRunner] = None,
    ):
        self._llm = llm_provider
        self._config = config or SkillEvaluationConfig()
        self._script_runner = script_runner

    async def evaluate(self, skill: Skill) -> SkillEvaluation:
        """Evaluate a single skill.

        If a ScriptRunner is configured and the skill has scripts in its
        structured content, the scripts are executed and the results are
        appended to the evaluation prompt.

        Args:
            skill: The skill to evaluate.

        Returns:
            A SkillEvaluation with ratings for all five dimensions.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        scripts_text = ""
        references_text = ""
        scripts_dict = {}
        if skill.structured_content:
            if skill.structured_content.scripts:
                parts = []
                for name, code in skill.structured_content.scripts.items():
                    parts.append(f"### {name}\n```\n{code}\n```")
                scripts_text = "\n\n".join(parts)
                scripts_dict = dict(skill.structured_content.scripts)
            if skill.structured_content.references:
                parts = []
                for name, ref in skill.structured_content.references.items():
                    parts.append(f"### {name}\n{ref}")
                references_text = "\n\n".join(parts)

        # Optional script execution
        script_results_text = ""
        if self._script_runner and scripts_dict:
            try:
                exec_results = await self._script_runner.run_scripts(scripts_dict)
                if exec_results:
                    script_results_text = self._format_script_results(exec_results)
            except Exception as exc:
                logger.warning("Script execution failed: %s", exc)
                script_results_text = f"[Script execution error: {exc}]"

        tools_text = ", ".join(skill.allowed_tools) if skill.allowed_tools else ""

        user_prompt = build_evaluation_prompt(
            name=skill.name,
            description=skill.description,
            content=skill.content,
            scripts=scripts_text,
            references=references_text,
            allowed_tools=tools_text,
            script_execution_results=script_results_text,
        )

        messages = [
            ChatMessage(role="system", content=SKILL_EVALUATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.chat(
            messages=messages,
            model=self._config.model,
            temperature=0.2,
            max_tokens=1024,
        )

        return self._parse_response(response.content)

    async def evaluate_batch(
        self,
        skills: List[Skill],
        max_concurrency: int = 5,
        on_result: Optional[Callable[[Skill, SkillEvaluation], Any]] = None,
    ) -> List[SkillEvaluation]:
        """Evaluate multiple skills in parallel with bounded concurrency.

        Args:
            skills: The skills to evaluate.
            max_concurrency: Maximum number of concurrent evaluations.
            on_result: Optional callback invoked as each result completes.

        Returns:
            A list of SkillEvaluation objects in the same order as input.
        """
        if not skills:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _eval_one(
            idx: int, skill: Skill,
        ) -> Tuple[int, SkillEvaluation]:
            async with semaphore:
                result = await self.evaluate(skill)
                if on_result:
                    on_result(skill, result)
                return (idx, result)

        tasks = [_eval_one(i, s) for i, s in enumerate(skills)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Optional[SkillEvaluation]] = [None] * len(skills)
        for i, item in enumerate(completed):
            if isinstance(item, BaseException):
                logger.error("Batch evaluation error: %s", item)
                results[i] = self._create_error_evaluation(str(item))
            else:
                idx, evaluation = item
                results[idx] = evaluation

        # Fill any remaining None slots with error evaluations
        for i, r in enumerate(results):
            if r is None:
                results[i] = self._create_error_evaluation("Missing result")

        return results  # type: ignore[return-value]

    @staticmethod
    def _format_script_results(
        exec_results: List[ScriptExecutionResult],
    ) -> str:
        """Format script execution results for inclusion in the LLM prompt."""
        lines: List[str] = []
        for r in exec_results:
            line = f"- {r.script_name}: {r.status}"
            if r.exit_code is not None:
                line += f" (exit={r.exit_code})"
            line += f" | cmd: {r.command}"
            if r.note:
                line += f" | note: {r.note}"
            if r.stderr:
                clean_err = " ".join(r.stderr.splitlines())
                line += f" | error: {clean_err}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _create_error_evaluation(error_msg: str) -> SkillEvaluation:
        """Create a default error evaluation with Poor ratings."""
        return SkillEvaluation(
            safety=EvaluationLevel.POOR,
            safety_reason=f"Evaluation failed: {error_msg}",
            completeness=EvaluationLevel.POOR,
            completeness_reason=f"Evaluation failed: {error_msg}",
            executability=EvaluationLevel.POOR,
            executability_reason=f"Evaluation failed: {error_msg}",
            maintainability=EvaluationLevel.POOR,
            maintainability_reason=f"Evaluation failed: {error_msg}",
            cost_awareness=EvaluationLevel.POOR,
            cost_awareness_reason=f"Evaluation failed: {error_msg}",
            evaluated_at=datetime.now(),
            overall_score=0.0,
        )

    def _parse_response(self, raw: str) -> SkillEvaluation:
        """Parse the LLM JSON response into a SkillEvaluation.

        Args:
            raw: Raw LLM response text (expected to be JSON).

        Returns:
            A SkillEvaluation instance.

        Raises:
            ValueError: If the response is not valid JSON or has invalid levels.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        for dim in ("safety", "completeness", "executability",
                     "maintainability", "cost_awareness"):
            val = data.get(dim, "").lower().strip()
            if val not in _VALID_LEVELS:
                raise ValueError(
                    f"Invalid evaluation level for {dim}: '{val}'. "
                    f"Expected one of {_VALID_LEVELS}."
                )
            data[dim] = val

        safety = EvaluationLevel(data["safety"])
        completeness = EvaluationLevel(data["completeness"])
        executability = EvaluationLevel(data["executability"])
        maintainability = EvaluationLevel(data["maintainability"])
        cost_awareness = EvaluationLevel(data["cost_awareness"])

        overall = SkillEvaluation.compute_overall_score(
            safety, completeness, executability,
            maintainability, cost_awareness,
        )

        return SkillEvaluation(
            safety=safety,
            safety_reason=data.get("safety_reason", ""),
            completeness=completeness,
            completeness_reason=data.get("completeness_reason", ""),
            executability=executability,
            executability_reason=data.get("executability_reason", ""),
            maintainability=maintainability,
            maintainability_reason=data.get("maintainability_reason", ""),
            cost_awareness=cost_awareness,
            cost_awareness_reason=data.get("cost_awareness_reason", ""),
            evaluated_at=datetime.now(),
            overall_score=overall,
        )
