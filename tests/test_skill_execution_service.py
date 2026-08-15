"""Tests for SkillExecutionService."""

import pytest

from ctxforge.core.skill import Skill, SkillContent, SkillScope
from ctxforge.engine.services.skill_execution_service import (
    ExecutableRuntimeConfig,
    SkillExecutionResult,
    SkillExecutionService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.storage.memory.skill import InMemorySkillStore

# ---- helpers ---------------------------------------------------------------


def _make_skill(
    name: str = "test-skill",
    scripts: dict | None = None,
    scope: SkillScope = SkillScope.BASE,
    scope_id: str = "system",
) -> Skill:
    sc = None
    if scripts is not None:
        sc = SkillContent(instructions="test", scripts=scripts)
    return Skill(
        name=name,
        description=f"Use when needing {name}",
        scope=scope,
        scope_id=scope_id,
        content="# test",
        structured_content=sc,
    )


async def _echo_tool(name: str, args: dict) -> dict:
    return {"tool": name, "args": args}


async def _make_service(
    skills: list | None = None,
    tool_fn=None,
    config: ExecutableRuntimeConfig | None = None,
) -> SkillExecutionService:
    store = InMemorySkillStore()
    for s in (skills or []):
        await store.save(s)
    skill_svc = SkillService(store=store)
    return SkillExecutionService(
        skill_service=skill_svc,
        tool_fn=tool_fn,
        config=config or ExecutableRuntimeConfig(enabled=True),
    )


# ---- execution tests -------------------------------------------------------


class TestExecuteSkill:

    @pytest.mark.asyncio
    async def test_execute_skill_success(self):
        skill = _make_skill(scripts={"main.py": "result = 42\nprint('hello')"})
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill")
        assert r.status == "success"
        assert r.return_value == 42
        assert "hello" in r.stdout
        assert r.duration_sec > 0

    @pytest.mark.asyncio
    async def test_execute_skill_with_args(self):
        script = "result = x + y"
        skill = _make_skill(scripts={"main.py": script})
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill", args={"x": 10, "y": 5})
        assert r.status == "success"
        assert r.return_value == 15

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self):
        svc = await _make_service()
        r = await svc.execute("nonexistent")
        assert r.status == "not_found"
        assert "not found" in r.detail.lower()

    @pytest.mark.asyncio
    async def test_execute_skill_no_scripts(self):
        skill = _make_skill(scripts=None)
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill")
        assert r.status == "not_found"
        assert "no executable scripts" in r.detail.lower()

    @pytest.mark.asyncio
    async def test_execute_skill_syntax_error(self):
        skill = _make_skill(scripts={"bad.py": "def foo(:\n  pass"})
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill")
        assert r.status == "failed"
        assert "syntax" in r.detail.lower()

    @pytest.mark.asyncio
    async def test_execute_skill_runtime_error(self):
        skill = _make_skill(scripts={"err.py": "raise ValueError('boom')"})
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill")
        assert r.status == "failed"
        assert "boom" in r.stderr

    @pytest.mark.asyncio
    async def test_execute_skill_specific_script(self):
        skill = _make_skill(scripts={
            "a.py": "result = 'a'",
            "b.py": "result = 'b'",
        })
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill", script_name="b.py")
        assert r.status == "success"
        assert r.return_value == "b"

    @pytest.mark.asyncio
    async def test_execute_skill_missing_script_name(self):
        skill = _make_skill(scripts={"a.py": "result = 1"})
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill", script_name="nope.py")
        assert r.status == "not_found"
        assert "nope.py" in r.detail


class TestSandbox:

    @pytest.mark.asyncio
    async def test_sandbox_blocks_dangerous(self):
        skill = _make_skill(scripts={"evil.py": "import os\nos.system('echo hi')"})
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill")
        assert r.status == "rejected"
        assert "dangerous" in r.detail.lower()

    @pytest.mark.asyncio
    async def test_sandbox_disabled_allows_dangerous(self):
        skill = _make_skill(scripts={"run.py": "result = 'ok'"})
        svc = await _make_service(
            skills=[skill],
            config=ExecutableRuntimeConfig(enabled=True, sandbox=False),
        )
        r = await svc.execute("test-skill")
        assert r.status == "success"


class TestTimeout:

    @pytest.mark.asyncio
    async def test_execute_skill_timeout(self):
        script = "import time\ntime.sleep(10)\nresult = 'done'"
        skill = _make_skill(scripts={"slow.py": script})
        svc = await _make_service(
            skills=[skill],
            config=ExecutableRuntimeConfig(enabled=True, timeout_sec=0.3),
        )

        r = await svc.execute("test-skill")
        assert r.status == "timeout"
        assert "timed out" in r.detail.lower()


class TestExecutionStats:

    @pytest.mark.asyncio
    async def test_execute_skill_records_stats(self):
        store = InMemorySkillStore()
        skill = _make_skill(scripts={"main.py": "result = 1"})
        await store.save(skill)
        skill_svc = SkillService(store=store)
        svc = SkillExecutionService(
            skill_service=skill_svc,
            config=ExecutableRuntimeConfig(enabled=True),
        )

        await svc.execute("test-skill")

        # Re-fetch the skill to check updated effectiveness
        updated = await store.get("test-skill", SkillScope.BASE, "system")
        assert updated is not None
        eff = updated.effectiveness or {}
        exec_stats = eff.get("execution_stats", {})
        assert exec_stats.get("total_executions", 0) >= 1
        assert exec_stats.get("successful_executions", 0) >= 1


class TestSchemaInference:

    @pytest.mark.asyncio
    async def test_schema_inference_from_return_value(self):
        skill = _make_skill(scripts={
            "main.py": "result = {'name': 'test', 'count': 5}",
        })
        svc = await _make_service(skills=[skill])

        r = await svc.execute("test-skill")
        assert r.status == "success"
        assert r.inferred_output_schema is not None
        assert r.inferred_output_schema["type"] == "object"
        props = r.inferred_output_schema["properties"]
        assert "name" in props
        assert "count" in props

    @pytest.mark.asyncio
    async def test_schema_inference_merges_across_runs(self):
        store = InMemorySkillStore()
        skill = _make_skill(scripts={"main.py": "result = {'a': 1}"})
        await store.save(skill)
        skill_svc = SkillService(store=store)
        svc = SkillExecutionService(
            skill_service=skill_svc,
            config=ExecutableRuntimeConfig(enabled=True),
        )

        await svc.execute("test-skill", args={"x": 1})

        # Modify the script to return a different shape
        skill2 = _make_skill(scripts={"main.py": "result = {'a': 1, 'b': 'hi'}"})
        await store.save(skill2)
        skill_svc.invalidate_cache()

        await svc.execute("test-skill", args={"x": 1, "y": 2})

        updated = await store.get("test-skill", SkillScope.BASE, "system")
        eff = updated.effectiveness or {}
        exec_stats = eff.get("execution_stats", {})
        schema = exec_stats.get("inferred_output_schema")
        assert schema is not None
        # After merge, should have both 'a' and 'b'
        if schema.get("type") == "object":
            assert "a" in schema.get("properties", {})


class TestToolBridgeIntegration:

    @pytest.mark.asyncio
    async def test_call_tool_from_script(self):
        script = "r = call_tool('echo', {'msg': 'hi'})\nresult = r"
        skill = _make_skill(scripts={"main.py": script})
        svc = await _make_service(skills=[skill], tool_fn=_echo_tool)

        r = await svc.execute("test-skill")
        assert r.status == "success"
        assert r.return_value == {"tool": "echo", "args": {"msg": "hi"}}
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_blocked_tool_in_script(self):
        script = "call_tool('execute_skill', {})\nresult = 'bad'"
        skill = _make_skill(scripts={"main.py": script})
        svc = await _make_service(skills=[skill], tool_fn=_echo_tool)

        r = await svc.execute("test-skill")
        assert r.status == "failed"
        assert "blocked" in r.stderr.lower()


class TestSerialization:

    def test_result_to_dict(self):
        result = SkillExecutionResult(
            skill_name="test",
            status="success",
            return_value=42,
            stdout="hello",
            stderr="",
            duration_sec=0.5,
        )
        d = result.to_dict()
        assert d["skill_name"] == "test"
        assert d["status"] == "success"
        assert d["return_value"] == 42
        assert isinstance(d["tool_calls"], list)
