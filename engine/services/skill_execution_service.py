"""
Skill Execution Service.

Executes skill scripts at runtime with parameter injection,
tool bridge integration, and execution statistics tracking.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ctxforge.core.skill import SkillScope
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.engine.services.tool_bridge import (
    ToolBridge,
    ToolBridgeConfig,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)

# Patterns that indicate a script should NOT be executed.
_DANGEROUS_PATTERNS = [
    "os.system(",
    "subprocess.call(",
    "subprocess.run(",
    "subprocess.Popen(",
    "shutil.rmtree(",
    "rm -rf",
    "rm -r ",
    "os.remove(",
    "os.unlink(",
    "os.rmdir(",
    "eval(",
    "exec(",
]

# Bootstrap script run inside the isolated subprocess. It reads the skill source
# and serialized args, executes the skill, and writes a JSON result payload. Any
# resource limits are applied by this process itself (before user code runs),
# which is safer and more portable than `preexec_fn`.
_SUBPROCESS_RUNNER = r'''
import json
import os
import traceback
from pathlib import Path

_HERE = Path(".")

try:
    import resource as _resource

    _cpu = int(os.environ.get("CTXFORGE_CPU_LIMIT", "0") or "0")
    _mem_mb = int(os.environ.get("CTXFORGE_MEM_LIMIT", "0") or "0")
    if _cpu > 0:
        _resource.setrlimit(_resource.RLIMIT_CPU, (_cpu, _cpu))
    if _mem_mb > 0:
        _bytes = _mem_mb * 1024 * 1024
        _resource.setrlimit(_resource.RLIMIT_AS, (_bytes, _bytes))
except Exception:
    pass

args = json.loads((_HERE / "args.json").read_text())

namespace = {"__builtins__": __builtins__, "args": args}
namespace.update({str(k): v for k, v in args.items()})

try:
    source = (_HERE / "__skill__.py").read_text()
    code = compile(source, "<skill>", "exec")
    exec(code, namespace)
    result = {"status": "success", "return_value": namespace.get("result")}
except Exception:
    result = {"status": "failed", "error": traceback.format_exc()}

(_HERE / "result.json").write_text(json.dumps(result, default=str))
'''


@dataclass
class SkillExecutionResult:
    """Result of executing a skill script."""

    skill_name: str
    status: str  # "success" | "failed" | "timeout" | "not_found" | "rejected"
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    duration_sec: float = 0.0
    inferred_output_schema: Optional[Dict] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "status": self.status,
            "return_value": self.return_value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "duration_sec": self.duration_sec,
            "inferred_output_schema": self.inferred_output_schema,
            "detail": self.detail,
        }


@dataclass
class ExecutableRuntimeConfig:
    """Configuration for the executable skill runtime."""

    enabled: bool = False
    timeout_sec: float = 10.0
    max_concurrent: int = 3
    sandbox: bool = True
    max_output_chars: int = 4000
    # Execution boundary: "subprocess" (isolated process + resource limits) or
    # "inprocess" (worker thread; required for the call_tool bridge).
    isolation: str = "subprocess"
    # Resource limits for subprocess isolation (0 = unlimited).
    cpu_time_limit_sec: int = 0
    memory_limit_mb: int = 0
    blocked_tools: List[str] = field(default_factory=lambda: [
        "save_skill", "execute_skill", "list_skills", "get_skill",
    ])


class SkillExecutionService:
    """Execute skill scripts at runtime with tool bridge support.

    Resolves skills via ``SkillService``, validates scripts for safety,
    runs them in a worker thread with ``call_tool`` injected, and
    captures results including stdout, return value, and tool call records.
    """

    def __init__(
        self,
        *,
        skill_service: SkillService,
        tool_fn: Optional[Callable[..., Coroutine]] = None,
        config: Optional[ExecutableRuntimeConfig] = None,
    ):
        self._skill_service = skill_service
        self._tool_fn = tool_fn
        self._config = config or ExecutableRuntimeConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)

    @property
    def config(self) -> ExecutableRuntimeConfig:
        return self._config

    async def execute(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        script_name: Optional[str] = None,
    ) -> SkillExecutionResult:
        """Execute a skill's script.

        Args:
            name: Skill name.
            args: Arguments to inject into the script namespace.
            user_id: Optional user ID for skill resolution.
            project_id: Optional project ID for skill resolution.
            script_name: Specific script to run within the skill's
                ``structured_content.scripts``. If None, runs the first
                available script.

        Returns:
            Execution result with stdout, return value, and tool calls.
        """
        async with self._semaphore:
            return await self._execute_impl(
                name, args or {}, user_id, project_id, script_name,
            )

    async def _execute_impl(
        self,
        name: str,
        args: Dict[str, Any],
        user_id: Optional[str],
        project_id: Optional[str],
        script_name: Optional[str],
    ) -> SkillExecutionResult:
        # 1. Resolve the skill
        skill = await self._skill_service.load_skill_content(
            name=name, user_id=user_id, project_id=project_id,
        )
        if skill is None:
            return SkillExecutionResult(
                skill_name=name,
                status="not_found",
                detail=f"Skill '{name}' not found.",
            )

        # 2. Find the script to execute
        if skill.structured_content is None or not skill.structured_content.scripts:
            return SkillExecutionResult(
                skill_name=name,
                status="not_found",
                detail=f"Skill '{name}' has no executable scripts.",
            )

        scripts = skill.structured_content.scripts
        if script_name:
            if script_name not in scripts:
                return SkillExecutionResult(
                    skill_name=name,
                    status="not_found",
                    detail=(
                        f"Script '{script_name}' not found in skill '{name}'. "
                        f"Available: {list(scripts.keys())}"
                    ),
                )
            source = scripts[script_name]
        else:
            script_name = next(iter(scripts))
            source = scripts[script_name]

        # 3. Safety check
        if self._config.sandbox:
            danger = self._detect_dangerous(source)
            if danger:
                return SkillExecutionResult(
                    skill_name=name,
                    status="rejected",
                    detail=f"Script rejected: dangerous pattern '{danger}'.",
                )

        # 4. Compile check
        try:
            code = compile(source, f"<skill:{name}/{script_name}>", "exec")
        except SyntaxError as exc:
            return SkillExecutionResult(
                skill_name=name,
                status="failed",
                stderr=str(exc),
                detail="Script has syntax errors.",
            )

        # 5. Build namespace with args and optional tool bridge
        namespace: Dict[str, Any] = {"__builtins__": __builtins__}
        namespace["args"] = args
        for k, v in args.items():
            namespace[k] = v

        bridge: Optional[ToolBridge] = None
        if self._tool_fn is not None:
            bridge_config = ToolBridgeConfig(
                timeout_sec=self._config.timeout_sec,
                blocked_tools=set(self._config.blocked_tools),
            )
            bridge = ToolBridge(
                tool_fn=self._tool_fn, config=bridge_config,
            )
            loop = asyncio.get_running_loop()
            namespace["call_tool"] = bridge.make_call_tool(loop)

        # 6. Execute in an isolated subprocess (default) or a worker thread.
        start = time.time()
        use_subprocess = self._config.isolation == "subprocess" and bridge is None
        if use_subprocess:
            result = await self._run_subprocess(source, args, name, script_name)
        else:
            if self._config.isolation == "subprocess" and bridge is not None:
                logger.warning(
                    "Skill '%s' uses call_tool; falling back to in-process "
                    "execution (subprocess isolation unavailable with tool_fn).",
                    name,
                )
            result = await self._run_in_thread(code, namespace, name, script_name)

        duration = round(time.time() - start, 4)
        result.duration_sec = duration

        if bridge is not None:
            result.tool_calls = bridge.records

        # 7. Infer output schema from return value
        if result.status == "success" and result.return_value is not None:
            result.inferred_output_schema = self._infer_schema(
                result.return_value,
            )

        # 8. Update execution stats on the skill
        await self._update_execution_stats(
            name, skill.scope, skill.scope_id, result, args,
        )

        return result

    async def _run_in_thread(
        self,
        code: Any,
        namespace: Dict[str, Any],
        skill_name: str,
        script_name: str,
    ) -> SkillExecutionResult:
        """Run compiled code in a worker thread with captured output."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        return_holder: Dict[str, Any] = {"value": None, "error": None}
        done_event = threading.Event()

        def _worker() -> None:
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, namespace)  # noqa: S102
                # Convention: if the script sets `result` in its namespace,
                # that becomes the return value.
                return_holder["value"] = namespace.get("result")
            except Exception as exc:
                return_holder["error"] = str(exc)
            finally:
                done_event.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        # Wait for the thread with timeout
        finished = await asyncio.get_running_loop().run_in_executor(
            None, done_event.wait, self._config.timeout_sec,
        )

        if not finished:
            stdout_text = self._truncate(stdout_buf.getvalue())
            stderr_text = self._truncate(stderr_buf.getvalue())
            return SkillExecutionResult(
                skill_name=skill_name,
                status="timeout",
                stdout=stdout_text,
                stderr=stderr_text,
                detail=(
                    f"Script '{script_name}' timed out after "
                    f"{self._config.timeout_sec}s."
                ),
            )

        stdout_text = self._truncate(stdout_buf.getvalue())
        stderr_text = self._truncate(stderr_buf.getvalue())

        if return_holder["error"] is not None:
            return SkillExecutionResult(
                skill_name=skill_name,
                status="failed",
                stdout=stdout_text,
                stderr=return_holder["error"],
                detail=f"Script '{script_name}' raised an exception.",
            )

        return SkillExecutionResult(
            skill_name=skill_name,
            status="success",
            return_value=return_holder["value"],
            stdout=stdout_text,
            stderr=stderr_text,
            detail=f"Script '{script_name}' executed successfully.",
        )

    async def _run_subprocess(
        self,
        source: str,
        args: Dict[str, Any],
        skill_name: str,
        script_name: str,
    ) -> SkillExecutionResult:
        """Run the script in an isolated subprocess with resource limits.

        The user source and serialized args are written to a temporary directory,
        and a bootstrap runner executes them in a fresh Python interpreter. A JSON
        payload (result or error) is written back and parsed by this method.
        """
        tmp_dir = tempfile.TemporaryDirectory(prefix="ctxforge-skill-")
        try:
            tmp = Path(tmp_dir.name)
            (tmp / "args.json").write_text(json.dumps(args))
            (tmp / "__skill__.py").write_text(source)

            env = dict(os.environ)
            env["CTXFORGE_CPU_LIMIT"] = str(int(self._config.cpu_time_limit_sec))
            env["CTXFORGE_MEM_LIMIT"] = str(int(self._config.memory_limit_mb))

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    _SUBPROCESS_RUNNER,
                    cwd=str(tmp),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._config.timeout_sec,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
                return SkillExecutionResult(
                    skill_name=skill_name,
                    status="timeout",
                    detail=(
                        f"Script '{script_name}' timed out after "
                        f"{self._config.timeout_sec}s."
                    ),
                )

            stdout_text = self._truncate(stdout_bytes.decode(errors="replace"))
            stderr_text = self._truncate(stderr_bytes.decode(errors="replace"))

            result_file = tmp / "result.json"
            if not result_file.exists():
                return SkillExecutionResult(
                    skill_name=skill_name,
                    status="failed",
                    stdout=stdout_text,
                    stderr=stderr_text,
                    detail=f"Script '{script_name}' did not produce a result.",
                )

            try:
                payload = json.loads(result_file.read_text())
            except Exception as exc:
                return SkillExecutionResult(
                    skill_name=skill_name,
                    status="failed",
                    stdout=stdout_text,
                    stderr=stderr_text or f"Failed to parse result: {exc}",
                    detail=f"Script '{script_name}' produced invalid output.",
                )

            if payload.get("status") == "success":
                return SkillExecutionResult(
                    skill_name=skill_name,
                    status="success",
                    return_value=payload.get("return_value"),
                    stdout=stdout_text,
                    stderr=stderr_text,
                    detail=f"Script '{script_name}' executed successfully.",
                )

            return SkillExecutionResult(
                skill_name=skill_name,
                status="failed",
                stdout=stdout_text,
                stderr=self._truncate(payload.get("error") or stderr_text),
                detail=f"Script '{script_name}' raised an exception.",
            )
        finally:
            tmp_dir.cleanup()

    async def _update_execution_stats(
        self,
        skill_name: str,
        scope: SkillScope,
        scope_id: str,
        result: SkillExecutionResult,
        args: Dict[str, Any],
    ) -> None:
        """Update execution statistics on the skill's effectiveness dict."""
        try:
            skill = await self._skill_service.load_skill_content(
                name=skill_name,
            )
            if skill is None:
                return

            eff = dict(skill.effectiveness or {})
            exec_stats = dict(eff.get("execution_stats", {}))

            total = exec_stats.get("total_executions", 0) + 1
            exec_stats["total_executions"] = total

            if result.status == "success":
                exec_stats["successful_executions"] = (
                    exec_stats.get("successful_executions", 0) + 1
                )
            else:
                exec_stats["failed_executions"] = (
                    exec_stats.get("failed_executions", 0) + 1
                )

            exec_stats["last_execution_status"] = result.status

            # Running average of duration
            old_avg = exec_stats.get("avg_duration_sec", 0.0)
            old_count = total - 1
            if old_count > 0:
                exec_stats["avg_duration_sec"] = round(
                    (old_avg * old_count + result.duration_sec) / total, 4,
                )
            else:
                exec_stats["avg_duration_sec"] = result.duration_sec

            # Schema inference: merge observed arg types
            if args and result.status == "success":
                exec_stats["inferred_input_schema"] = self._merge_schema(
                    exec_stats.get("inferred_input_schema"),
                    self._infer_schema(args),
                )
            if result.inferred_output_schema and result.status == "success":
                exec_stats["inferred_output_schema"] = self._merge_schema(
                    exec_stats.get("inferred_output_schema"),
                    result.inferred_output_schema,
                )

            eff["execution_stats"] = exec_stats

            store = self._skill_service._store
            await store.update_effectiveness(
                skill_name, scope, scope_id, eff,
            )

        except Exception as exc:
            logger.debug(
                "Failed to update execution stats for '%s': %s",
                skill_name, exc,
            )

    @staticmethod
    def _infer_schema(value: Any) -> Optional[Dict]:
        """Infer a simple JSON-schema-like dict from a Python value."""
        if value is None:
            return None
        if isinstance(value, dict):
            properties = {}
            for k, v in value.items():
                properties[str(k)] = {"type": type(v).__name__}
            return {"type": "object", "properties": properties}
        if isinstance(value, list):
            if value:
                item_type = type(value[0]).__name__
            else:
                item_type = "any"
            return {"type": "array", "items": {"type": item_type}}
        return {"type": type(value).__name__}

    @staticmethod
    def _merge_schema(
        existing: Optional[Dict],
        incoming: Optional[Dict],
    ) -> Optional[Dict]:
        """Merge two schemas using a simple union strategy."""
        if existing is None:
            return incoming
        if incoming is None:
            return existing

        # For object schemas, merge properties
        if (
            existing.get("type") == "object"
            and incoming.get("type") == "object"
        ):
            merged_props = dict(existing.get("properties", {}))
            for k, v in incoming.get("properties", {}).items():
                if k not in merged_props:
                    merged_props[k] = v
                elif merged_props[k].get("type") != v.get("type"):
                    merged_props[k] = {"type": "any"}
            return {"type": "object", "properties": merged_props}

        # Different top-level types
        if existing.get("type") != incoming.get("type"):
            return {"type": "any"}

        return incoming

    @staticmethod
    def _detect_dangerous(source: str) -> Optional[str]:
        """Return the dangerous pattern found, or None if safe."""
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in source:
                return pattern
        return None

    def _truncate(self, text: str) -> str:
        if not text:
            return ""
        limit = self._config.max_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"
