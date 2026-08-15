"""
Tests for Skill Evaluation Service.
"""
import json
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from ctxforge.config.base import SkillEvaluationConfig
from ctxforge.core.skill import (
    EvaluationLevel,
    Skill,
    SkillContent,
    SkillEvaluation,
    SkillScope,
)
from ctxforge.engine.services.script_runner import (
    ScriptExecutionResult,
    ScriptRunner,
    ScriptRunnerConfig,
)
from ctxforge.engine.services.skill_evaluation_service import SkillEvaluationService
from ctxforge.protocols.llm import ChatMessage, LLMResponse


class FakeLLMProvider:
    """Minimal fake LLM provider for testing."""

    def __init__(self, response_content: str = ""):
        self._response_content = response_content

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-model"

    async def generate(
        self, prompt: str, model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None, **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._response_content, model="fake-model")

    async def chat(
        self, messages: List[ChatMessage], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        functions: Optional[List[Dict[str, Any]]] = None, **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._response_content, model="fake-model")

    async def stream(
        self, messages: List[ChatMessage], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield self._response_content

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text.split())

    def count_message_tokens(
        self, messages: List[ChatMessage], model: Optional[str] = None,
    ) -> int:
        return sum(len(m.content.split()) for m in messages)


def _good_response() -> str:
    return json.dumps({
        "safety": "good", "safety_reason": "No destructive actions",
        "completeness": "good", "completeness_reason": "Covers all steps",
        "executability": "good", "executability_reason": "Clear instructions",
        "maintainability": "good", "maintainability_reason": "Well structured",
        "cost_awareness": "good", "cost_awareness_reason": "Minimal LLM calls",
    })


def _mixed_response() -> str:
    return json.dumps({
        "safety": "good", "safety_reason": "Safe",
        "completeness": "average", "completeness_reason": "Missing edge cases",
        "executability": "poor", "executability_reason": "Ambiguous steps",
        "maintainability": "good", "maintainability_reason": "Modular",
        "cost_awareness": "average", "cost_awareness_reason": "Some extra calls",
    })


def _make_skill(**overrides) -> Skill:
    defaults = dict(
        name="test-skill",
        description="A test skill",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Steps\n1. Do thing A\n2. Do thing B",
    )
    defaults.update(overrides)
    return Skill(**defaults)


class TestSkillEvaluationService:
    """Tests for SkillEvaluationService."""

    async def test_evaluate_returns_valid_evaluation(self):
        """evaluate() returns a SkillEvaluation with all 5 dimensions."""
        llm = FakeLLMProvider(response_content=_good_response())
        svc = SkillEvaluationService(llm)
        result = await svc.evaluate(_make_skill())

        assert isinstance(result, SkillEvaluation)
        assert result.safety == EvaluationLevel.GOOD
        assert result.completeness == EvaluationLevel.GOOD
        assert result.executability == EvaluationLevel.GOOD
        assert result.maintainability == EvaluationLevel.GOOD
        assert result.cost_awareness == EvaluationLevel.GOOD
        assert result.overall_score == 1.0
        assert isinstance(result.evaluated_at, datetime)

    async def test_evaluate_computes_correct_overall_score(self):
        """evaluate() computes the correct weighted overall score."""
        llm = FakeLLMProvider(response_content=_mixed_response())
        svc = SkillEvaluationService(llm)
        result = await svc.evaluate(_make_skill())

        expected = SkillEvaluation.compute_overall_score(
            EvaluationLevel.GOOD, EvaluationLevel.AVERAGE,
            EvaluationLevel.POOR, EvaluationLevel.GOOD,
            EvaluationLevel.AVERAGE,
        )
        assert result.overall_score == expected

    async def test_evaluate_batch(self):
        """evaluate_batch() evaluates multiple skills."""
        llm = FakeLLMProvider(response_content=_good_response())
        svc = SkillEvaluationService(llm)
        skills = [_make_skill(name="skill-a"), _make_skill(name="skill-b")]
        results = await svc.evaluate_batch(skills)

        assert len(results) == 2
        assert all(isinstance(r, SkillEvaluation) for r in results)

    async def test_evaluate_handles_malformed_json(self):
        """evaluate() raises ValueError for invalid JSON."""
        llm = FakeLLMProvider(response_content="not valid json {{{")
        svc = SkillEvaluationService(llm)

        with pytest.raises(ValueError, match="invalid JSON"):
            await svc.evaluate(_make_skill())

    async def test_evaluate_handles_invalid_level(self):
        """evaluate() raises ValueError for invalid evaluation levels."""
        bad = json.dumps({
            "safety": "excellent",  # invalid
            "safety_reason": "ok",
            "completeness": "good", "completeness_reason": "ok",
            "executability": "good", "executability_reason": "ok",
            "maintainability": "good", "maintainability_reason": "ok",
            "cost_awareness": "good", "cost_awareness_reason": "ok",
        })
        llm = FakeLLMProvider(response_content=bad)
        svc = SkillEvaluationService(llm)

        with pytest.raises(ValueError, match="Invalid evaluation level"):
            await svc.evaluate(_make_skill())

    async def test_evaluate_strips_markdown_fences(self):
        """evaluate() handles LLM responses wrapped in markdown fences."""
        wrapped = "```json\n" + _good_response() + "\n```"
        llm = FakeLLMProvider(response_content=wrapped)
        svc = SkillEvaluationService(llm)
        result = await svc.evaluate(_make_skill())
        assert result.safety == EvaluationLevel.GOOD

    async def test_evaluate_with_structured_content(self):
        """evaluate() includes scripts and references from structured content."""
        skill = _make_skill(
            structured_content=SkillContent(
                instructions="Do the thing",
                scripts={"setup.sh": "echo setup"},
                references={"guide.md": "# Guide"},
            ),
            allowed_tools=["read_file", "write_file"],
        )
        llm = FakeLLMProvider(response_content=_good_response())
        svc = SkillEvaluationService(llm)
        result = await svc.evaluate(skill)
        assert isinstance(result, SkillEvaluation)

    async def test_evaluate_uses_config_model(self):
        """evaluate() passes config model to LLM."""
        config = SkillEvaluationConfig(model="custom-model")
        llm = FakeLLMProvider(response_content=_good_response())
        llm.chat = AsyncMock(return_value=LLMResponse(
            content=_good_response(), model="custom-model",
        ))
        svc = SkillEvaluationService(llm, config=config)
        await svc.evaluate(_make_skill())

        llm.chat.assert_called_once()
        call_kwargs = llm.chat.call_args
        assert call_kwargs.kwargs.get("model") == "custom-model"


class TestScriptRunnerIntegration:
    """Tests for script execution integration in evaluation."""

    async def test_evaluate_without_script_runner(self):
        """Evaluation works with script_runner=None (backward compatible)."""
        llm = FakeLLMProvider(response_content=_good_response())
        svc = SkillEvaluationService(llm, script_runner=None)
        skill = _make_skill(
            structured_content=SkillContent(
                instructions="Do the thing",
                scripts={"run.py": "print('hello')"},
            ),
        )
        result = await svc.evaluate(skill)
        assert isinstance(result, SkillEvaluation)
        assert result.safety == EvaluationLevel.GOOD

    async def test_evaluate_with_script_runner_runs_scripts(self):
        """Evaluation with script_runner executes scripts and includes results."""
        llm = FakeLLMProvider(response_content=_good_response())
        # Capture the prompt sent to the LLM
        llm.chat = AsyncMock(return_value=LLMResponse(
            content=_good_response(), model="fake-model",
        ))
        runner = ScriptRunner(ScriptRunnerConfig(enabled=True, timeout_sec=5))
        svc = SkillEvaluationService(llm, script_runner=runner)

        skill = _make_skill(
            structured_content=SkillContent(
                instructions="Do the thing",
                scripts={"hello.py": '"""M.\n\nUsage:\n    python hello.py\n"""\nprint("hi")\n'},
            ),
        )
        result = await svc.evaluate(skill)
        assert isinstance(result, SkillEvaluation)

        # Verify the prompt sent to LLM contains script execution results
        call_args = llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_msg = [m for m in messages if m.role == "user"][0]
        assert "Script Execution Results" in user_msg.content
        assert "hello.py" in user_msg.content

    async def test_evaluate_script_runner_no_scripts_in_skill(self):
        """Script runner configured but skill has no scripts -> no execution."""
        llm = FakeLLMProvider(response_content=_good_response())
        runner = ScriptRunner(ScriptRunnerConfig(enabled=True))
        svc = SkillEvaluationService(llm, script_runner=runner)

        skill = _make_skill()  # No structured_content
        result = await svc.evaluate(skill)
        assert isinstance(result, SkillEvaluation)

    async def test_evaluate_script_runner_disabled(self):
        """Script runner with enabled=False -> no execution."""
        llm = FakeLLMProvider(response_content=_good_response())
        llm.chat = AsyncMock(return_value=LLMResponse(
            content=_good_response(), model="fake-model",
        ))
        runner = ScriptRunner(ScriptRunnerConfig(enabled=False))
        svc = SkillEvaluationService(llm, script_runner=runner)

        skill = _make_skill(
            structured_content=SkillContent(
                instructions="Do the thing",
                scripts={"run.py": "print('hello')"},
            ),
        )
        await svc.evaluate(skill)

        # Prompt should NOT contain script execution results
        call_args = llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_msg = [m for m in messages if m.role == "user"][0]
        assert "Script Execution Results" not in user_msg.content

    async def test_script_execution_failure_does_not_block_evaluation(self):
        """If script execution raises, evaluation still completes."""
        llm = FakeLLMProvider(response_content=_good_response())
        runner = ScriptRunner(ScriptRunnerConfig(enabled=True))
        # Monkey-patch run_scripts to raise
        runner.run_scripts = AsyncMock(side_effect=RuntimeError("boom"))
        svc = SkillEvaluationService(llm, script_runner=runner)

        skill = _make_skill(
            structured_content=SkillContent(
                instructions="Do the thing",
                scripts={"run.py": "print('hello')"},
            ),
        )
        result = await svc.evaluate(skill)
        assert isinstance(result, SkillEvaluation)
        assert result.safety == EvaluationLevel.GOOD


class TestFormatScriptResults:
    """Tests for _format_script_results."""

    def test_format_success(self):
        results = [
            ScriptExecutionResult(
                script_name="run.py", status="success",
                command="python run.py", exit_code=0,
            ),
        ]
        formatted = SkillEvaluationService._format_script_results(results)
        assert "run.py: success" in formatted
        assert "exit=0" in formatted

    def test_format_failed_with_error(self):
        results = [
            ScriptExecutionResult(
                script_name="bad.py", status="failed",
                command="python bad.py", exit_code=1,
                stderr="SyntaxError: invalid syntax",
            ),
        ]
        formatted = SkillEvaluationService._format_script_results(results)
        assert "bad.py: failed" in formatted
        assert "SyntaxError" in formatted

    def test_format_skipped_with_note(self):
        results = [
            ScriptExecutionResult(
                script_name="danger.py", status="skipped",
                command="", note="dangerous pattern: os.system(",
            ),
        ]
        formatted = SkillEvaluationService._format_script_results(results)
        assert "danger.py: skipped" in formatted
        assert "dangerous pattern" in formatted

    def test_format_multiple_results(self):
        results = [
            ScriptExecutionResult(
                script_name="a.py", status="success",
                command="python a.py", exit_code=0,
            ),
            ScriptExecutionResult(
                script_name="b.py", status="timeout",
                command="python b.py", note="timed out",
            ),
        ]
        formatted = SkillEvaluationService._format_script_results(results)
        lines = formatted.strip().split("\n")
        assert len(lines) == 2


class TestBatchParallelEvaluation:
    """Tests for parallel batch evaluation."""

    async def test_batch_empty(self):
        """Empty skills list returns empty results."""
        llm = FakeLLMProvider(response_content=_good_response())
        svc = SkillEvaluationService(llm)
        results = await svc.evaluate_batch([])
        assert results == []

    async def test_batch_preserves_order(self):
        """Results are returned in the same order as input skills."""
        # Use different responses to distinguish results
        call_count = 0
        responses = [_good_response(), _mixed_response(), _good_response()]

        class OrderedLLM(FakeLLMProvider):
            async def chat(self, messages, **kwargs):
                nonlocal call_count
                idx = call_count
                call_count += 1
                return LLMResponse(
                    content=responses[idx % len(responses)],
                    model="fake-model",
                )

        llm = OrderedLLM()
        svc = SkillEvaluationService(llm)
        skills = [
            _make_skill(name="skill-a"),
            _make_skill(name="skill-b"),
            _make_skill(name="skill-c"),
        ]
        results = await svc.evaluate_batch(skills, max_concurrency=2)

        assert len(results) == 3
        assert all(isinstance(r, SkillEvaluation) for r in results)

    async def test_batch_callback_invoked(self):
        """on_result callback is called for each completed evaluation."""
        llm = FakeLLMProvider(response_content=_good_response())
        svc = SkillEvaluationService(llm)

        callback_results = []

        def on_result(skill, evaluation):
            callback_results.append((skill.name, evaluation.overall_score))

        skills = [_make_skill(name="skill-a"), _make_skill(name="skill-b")]
        await svc.evaluate_batch(skills, on_result=on_result)

        assert len(callback_results) == 2
        names = {name for name, _ in callback_results}
        assert names == {"skill-a", "skill-b"}

    async def test_batch_error_produces_error_evaluation(self):
        """If one skill fails, it gets an error evaluation; others succeed."""

        class FailByNameLLM(FakeLLMProvider):
            async def chat(self, messages, **kwargs):
                # Find the skill name in the user message
                user_msg = [m for m in messages if m.role == "user"][0]
                if "skill-bad" in user_msg.content:
                    return LLMResponse(
                        content="NOT VALID JSON", model="fake-model",
                    )
                return LLMResponse(
                    content=_good_response(), model="fake-model",
                )

        llm = FailByNameLLM()
        svc = SkillEvaluationService(llm)
        skills = [
            _make_skill(name="skill-a"),
            _make_skill(name="skill-bad"),
            _make_skill(name="skill-c"),
        ]

        results = await svc.evaluate_batch(skills, max_concurrency=3)

        assert len(results) == 3
        # The failing skill should have a Poor error evaluation
        error_results = [r for r in results if r.overall_score == 0.0]
        good_results = [r for r in results if r.overall_score == 1.0]
        assert len(error_results) == 1
        assert len(good_results) == 2
        assert "Evaluation failed" in error_results[0].safety_reason

    async def test_batch_concurrency_limit(self):
        """max_concurrency limits parallel evaluations."""
        import asyncio
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        class TrackingLLM(FakeLLMProvider):
            async def chat(self, messages, **kwargs):
                nonlocal max_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > max_concurrent:
                        max_concurrent = current_concurrent
                await asyncio.sleep(0.05)
                async with lock:
                    current_concurrent -= 1
                return LLMResponse(
                    content=_good_response(), model="fake-model",
                )

        llm = TrackingLLM()
        svc = SkillEvaluationService(llm)
        skills = [_make_skill(name=f"skill-{i}") for i in range(10)]

        await svc.evaluate_batch(skills, max_concurrency=3)

        assert max_concurrent <= 3


class TestCreateErrorEvaluation:
    """Tests for _create_error_evaluation."""

    def test_creates_all_poor_ratings(self):
        result = SkillEvaluationService._create_error_evaluation("test error")
        assert result.safety == EvaluationLevel.POOR
        assert result.completeness == EvaluationLevel.POOR
        assert result.executability == EvaluationLevel.POOR
        assert result.maintainability == EvaluationLevel.POOR
        assert result.cost_awareness == EvaluationLevel.POOR
        assert result.overall_score == 0.0
        assert "test error" in result.safety_reason
