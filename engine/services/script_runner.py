"""
Script Runner for Skill Evaluation.

Intelligently executes Python scripts from a skill's structured content
with usage extraction, placeholder detection, safety checks, and fallback
to compile-only verification.
"""
import ast
import asyncio
import logging
import os
import shlex
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


PATH_LIKE_EXTS = {
    ".xml", ".json", ".yaml", ".yml", ".csv", ".tsv", ".txt", ".md",
    ".ini", ".toml", ".coverage", ".db", ".sqlite", ".sql", ".parquet",
}

# Patterns that indicate a script should NOT be auto-executed.
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

_DANGEROUS_SHELL_TRUE = "shell=True"


@dataclass
class ScriptExecutionResult:
    """Result of executing a single script."""
    script_name: str
    status: str  # "success" | "compiled_only" | "failed" | "timeout" | "skipped"
    command: str
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration_sec: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script_name": self.script_name,
            "status": self.status,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_sec": self.duration_sec,
            "note": self.note,
        }


@dataclass
class ScriptRunnerConfig:
    """Configuration for script execution."""
    enabled: bool = False
    timeout_sec: int = 8
    max_scripts: int = 5
    max_output_chars: int = 400
    python_bin: str = "python3"
    sandbox: bool = True


class ScriptRunner:
    """Execute Python scripts from a skill's structured content.

    Behaviours:
    - Extracts usage examples from docstrings.
    - Detects placeholder tokens (<file>, [options]) and skips unrunnable scripts.
    - Detects dangerous patterns (os.system, rm -rf, etc.) when sandbox=True.
    - Falls back: runnable command -> compile-only verification.
    - Captures stdout/stderr with timeout enforcement.
    - Truncates output to max_output_chars.
    """

    def __init__(self, config: Optional[ScriptRunnerConfig] = None):
        self._config = config or ScriptRunnerConfig()

    async def run_scripts(
        self,
        scripts: Dict[str, str],
        working_dir: Optional[str] = None,
    ) -> List[ScriptExecutionResult]:
        """Execute scripts intelligently.

        Args:
            scripts: Mapping of script name to source code.
            working_dir: Working directory for execution (uses a temp dir if None).

        Returns:
            List of execution results, one per script (up to max_scripts).
        """
        if not self._config.enabled or not scripts:
            return []

        results: List[ScriptExecutionResult] = []
        names = sorted(scripts.keys())[: self._config.max_scripts]

        use_temp = working_dir is None
        tmp_dir = tempfile.mkdtemp(prefix="ctxforge_scripts_") if use_temp else None
        cwd = tmp_dir if use_temp else working_dir

        try:
            for name in names:
                content = scripts[name]
                result = await self._run_single(name, content, cwd)
                results.append(result)
        finally:
            if use_temp and tmp_dir:
                # Best-effort cleanup
                try:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        if len(scripts) > self._config.max_scripts:
            logger.info(
                "Found %d scripts, truncated to %d for execution",
                len(scripts), self._config.max_scripts,
            )

        return results

    # ------------------------------------------------------------------
    # Single-script execution
    # ------------------------------------------------------------------

    async def _run_single(
        self, name: str, content: str, cwd: str,
    ) -> ScriptExecutionResult:
        """Run a single script through the analysis and execution pipeline."""

        # Safety check
        if self._config.sandbox:
            reason = self._detect_dangerous_patterns(content)
            if reason:
                return ScriptExecutionResult(
                    script_name=name,
                    status="skipped",
                    command="",
                    note=f"dangerous pattern: {reason}",
                )

        # Write script to disk
        script_path = os.path.join(cwd, name)
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)

        rel_path = name
        script_name_base = os.path.basename(name)

        # Try to build a usage-derived command
        usage_cmd = self._build_usage_command(
            content, script_path, rel_path, script_name_base,
        )

        if usage_cmd:
            missing = self._detect_missing_inputs(usage_cmd, cwd)
            if missing:
                # Compile-only fallback
                compile_result = await self._execute(
                    cwd,
                    [self._config.python_bin, "-m", "py_compile", rel_path],
                )
                note = f"missing inputs: {', '.join(missing)}"
                return self._build_result(
                    name, compile_result, note,
                    success_status="compiled_only",
                )

            # Run the usage command
            run_result = await self._execute(cwd, usage_cmd)
            return self._build_result(
                name, run_result, "usage-derived command",
            )

        # No usage found — compile-only
        compile_result = await self._execute(
            cwd,
            [self._config.python_bin, "-m", "py_compile", rel_path],
        )
        return self._build_result(
            name, compile_result,
            "no usage examples found; py_compile check",
            success_status="compiled_only",
        )

    # ------------------------------------------------------------------
    # Usage extraction (from docstrings)
    # ------------------------------------------------------------------

    def _extract_usage_lines(
        self, content: str, script_name: str,
    ) -> List[str]:
        """Parse the module docstring for usage examples."""
        try:
            tree = ast.parse(content)
            doc = ast.get_docstring(tree) or ""
        except Exception:
            doc = ""

        if not doc:
            return []

        lines = doc.splitlines()
        usage_lines: List[str] = []

        # Look for explicit "Usage:" section
        for idx, line in enumerate(lines):
            if line.strip().lower().startswith("usage:"):
                for follow in lines[idx + 1:]:
                    if not follow.strip():
                        break
                    usage_lines.append(follow.strip())

        if usage_lines:
            return usage_lines

        # Fallback: lines that look like invocation commands
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("./", "python", "python3")):
                usage_lines.append(stripped)
                continue
            if stripped.startswith(script_name):
                usage_lines.append(stripped)

        return usage_lines

    # ------------------------------------------------------------------
    # Command building and analysis
    # ------------------------------------------------------------------

    def _build_usage_command(
        self,
        content: str,
        script_path: str,
        rel_path: str,
        script_name: str,
    ) -> Optional[List[str]]:
        """Build a runnable command from usage examples in the docstring."""
        usage_lines = self._extract_usage_lines(content, script_name)
        if not usage_lines:
            return None

        candidates: List[List[str]] = []
        for line in usage_lines:
            cmd = self._parse_usage_line(line, rel_path, script_name)
            if cmd:
                candidates.append(cmd)

        if not candidates:
            return None

        # Prefer non-placeholder, non-help commands
        runnable = [c for c in candidates if not self._has_placeholder_tokens(c)]
        for cmd in runnable:
            if not self._is_help_command(cmd):
                return cmd

        # Fallback to help commands
        for cmd in candidates:
            if self._is_help_command(cmd):
                return cmd

        return None

    def _parse_usage_line(
        self, line: str, rel_path: str, script_name: str,
    ) -> Optional[List[str]]:
        """Parse a usage line into a command list."""
        try:
            tokens = shlex.split(line)
        except ValueError:
            return None

        if not tokens:
            return None

        python_prefix = tokens[0].startswith("python")
        if python_prefix:
            tokens = [self._config.python_bin] + tokens[1:]

        script_idx = None
        for idx, token in enumerate(tokens):
            if os.path.basename(token) == script_name:
                script_idx = idx
                break

        if script_idx is None:
            return None

        tokens[script_idx] = rel_path

        if not python_prefix:
            tokens = [self._config.python_bin] + tokens[script_idx:]

        return tokens

    # ------------------------------------------------------------------
    # Token analysis helpers
    # ------------------------------------------------------------------

    def _iter_non_flag_tokens(self, cmd: List[str]) -> Iterator[str]:
        """Iterate over non-flag, non-python-bin tokens."""
        for token in cmd:
            if not token or token.startswith("-"):
                continue
            if token == self._config.python_bin:
                continue
            yield token

    @staticmethod
    def _is_help_command(cmd: List[str]) -> bool:
        for token in cmd:
            if token.lower() in {"--help", "-h", "help"}:
                return True
        return False

    def _has_placeholder_tokens(self, cmd: List[str]) -> bool:
        for token in self._iter_non_flag_tokens(cmd):
            if self._is_placeholder_token(token):
                return True
        return False

    @staticmethod
    def _is_placeholder_token(token: str) -> bool:
        if any(ch in token for ch in ("<", ">", "[", "]", "{", "}")):
            return True
        return token.strip().lower() in {
            "options", "[options]", "<options>", "{options}",
        }

    def _detect_missing_inputs(
        self, cmd: List[str], cwd: str,
    ) -> List[str]:
        """Check if path-like tokens reference existing files."""
        missing: List[str] = []
        for token in self._iter_non_flag_tokens(cmd):
            if self._is_placeholder_token(token):
                continue
            if not self._looks_like_path(token):
                continue
            path = token if os.path.isabs(token) else os.path.join(cwd, token)
            if not os.path.exists(path):
                missing.append(token)
        return missing

    @staticmethod
    def _looks_like_path(token: str) -> bool:
        if "/" in token or token.startswith("."):
            return True
        _, ext = os.path.splitext(token)
        return ext.lower() in PATH_LIKE_EXTS

    @staticmethod
    def _detect_dangerous_patterns(content: str) -> Optional[str]:
        """Return reason string if dangerous, None if safe."""
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in content:
                return pattern
        if _DANGEROUS_SHELL_TRUE in content:
            return "subprocess with shell=True"
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute(
        self, cwd: str, command: List[str],
    ) -> Dict[str, Any]:
        """Run a command via asyncio subprocess with timeout."""
        command_str = shlex.join(command)
        start = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._config.timeout_sec,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.time() - start
                return {
                    "command": command_str,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "duration_sec": round(duration, 3),
                    "timed_out": True,
                    "error": f"Timeout after {self._config.timeout_sec}s",
                }

            duration = time.time() - start
            return {
                "command": command_str,
                "exit_code": proc.returncode,
                "stdout": self._truncate(
                    stdout_bytes.decode("utf-8", errors="replace"),
                ),
                "stderr": self._truncate(
                    stderr_bytes.decode("utf-8", errors="replace"),
                ),
                "duration_sec": round(duration, 3),
                "timed_out": False,
            }
        except FileNotFoundError as exc:
            duration = time.time() - start
            return {
                "command": command_str,
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
                "duration_sec": round(duration, 3),
                "timed_out": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_result(
        self,
        name: str,
        run_result: Dict[str, Any],
        note: Optional[str] = None,
        success_status: str = "success",
    ) -> ScriptExecutionResult:
        """Build a ScriptExecutionResult from a raw run_result dict."""
        if run_result.get("timed_out"):
            return ScriptExecutionResult(
                script_name=name,
                status="timeout",
                command=run_result["command"],
                exit_code=None,
                stderr=run_result.get("error"),
                duration_sec=run_result.get("duration_sec"),
                note=note,
            )
        if run_result.get("exit_code") == 0:
            return ScriptExecutionResult(
                script_name=name,
                status=success_status,
                command=run_result["command"],
                exit_code=0,
                stdout=run_result.get("stdout"),
                stderr=run_result.get("stderr"),
                duration_sec=run_result.get("duration_sec"),
                note=note,
            )
        return ScriptExecutionResult(
            script_name=name,
            status="failed",
            command=run_result["command"],
            exit_code=run_result.get("exit_code"),
            stdout=run_result.get("stdout"),
            stderr=self._pick_error(run_result),
            duration_sec=run_result.get("duration_sec"),
            note=note,
        )

    @staticmethod
    def _pick_error(result: Dict[str, Any]) -> Optional[str]:
        stderr = (result.get("stderr") or "").strip()
        if stderr:
            return stderr
        stdout = (result.get("stdout") or "").strip()
        if stdout:
            return stdout
        if result.get("error"):
            return str(result["error"])
        return None

    def _truncate(self, text: str) -> str:
        if not text:
            return ""
        limit = self._config.max_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"
