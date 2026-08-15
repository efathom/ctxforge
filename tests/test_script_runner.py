"""
Tests for Script Runner.
"""
from ctxforge.engine.services.script_runner import (
    ScriptExecutionResult,
    ScriptRunner,
    ScriptRunnerConfig,
)


def _make_runner(**overrides) -> ScriptRunner:
    defaults = dict(enabled=True, timeout_sec=5, max_scripts=5,
                    max_output_chars=400, sandbox=True)
    defaults.update(overrides)
    return ScriptRunner(ScriptRunnerConfig(**defaults))


class TestScriptRunnerDisabled:
    """Tests for disabled runner."""

    async def test_disabled_returns_empty(self):
        runner = ScriptRunner(ScriptRunnerConfig(enabled=False))
        results = await runner.run_scripts({"hello.py": "print('hi')"})
        assert results == []

    async def test_empty_scripts_returns_empty(self):
        runner = _make_runner()
        results = await runner.run_scripts({})
        assert results == []


class TestScriptExecution:
    """Tests for actual script execution."""

    async def test_successful_script(self):
        """Script that prints and exits 0 -> status='success'."""
        runner = _make_runner()
        scripts = {
            "hello.py": '"""Module.\n\nUsage:\n    python hello.py\n"""\nprint("hello world")\n'
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        r = results[0]
        assert r.script_name == "hello.py"
        assert r.status == "success"
        assert r.exit_code == 0
        assert "hello world" in (r.stdout or "")

    async def test_syntax_error_script(self):
        """Script with syntax error -> status='failed'."""
        runner = _make_runner()
        scripts = {"bad.py": "def foo(\n"}
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "failed"

    async def test_compile_only_no_usage(self):
        """Script without usage docstring -> compile-only."""
        runner = _make_runner()
        scripts = {"lib.py": "def helper():\n    return 42\n"}
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "compiled_only"
        assert "no usage" in (results[0].note or "")

    async def test_timeout_script(self):
        """Script that runs forever -> status='timeout'."""
        runner = _make_runner(timeout_sec=1)
        scripts = {
            "slow.py": '"""Module.\n\nUsage:\n    python slow.py\n"""\nimport time\ntime.sleep(60)\n'
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "timeout"


class TestDangerousPatternDetection:
    """Tests for sandbox safety checks."""

    async def test_os_system_skipped(self):
        runner = _make_runner(sandbox=True)
        scripts = {"danger.py": 'import os\nos.system("ls")\n'}
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert "os.system(" in (results[0].note or "")

    async def test_subprocess_shell_true_skipped(self):
        runner = _make_runner(sandbox=True)
        scripts = {
            "danger2.py": 'import subprocess\nsubprocess.run("ls", shell=True)\n'
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "skipped"

    async def test_rm_rf_skipped(self):
        runner = _make_runner(sandbox=True)
        scripts = {"danger3.py": 'import os\nos.system("rm -rf /")\n'}
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "skipped"

    async def test_sandbox_disabled_allows_patterns(self):
        """When sandbox=False, dangerous patterns are NOT blocked."""
        runner = _make_runner(sandbox=False)
        # Script uses eval() but is otherwise harmless
        scripts = {"evaltest.py": "x = eval('1 + 1')\nprint(x)\n"}
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        # Should compile (no usage docstring -> compile-only)
        assert results[0].status == "compiled_only"

    async def test_shutil_rmtree_skipped(self):
        runner = _make_runner(sandbox=True)
        scripts = {"danger4.py": 'import shutil\nshutil.rmtree("/tmp/foo")\n'}
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "skipped"


class TestPlaceholderDetection:
    """Tests for placeholder token detection."""

    async def test_placeholder_angle_brackets_skipped(self):
        """Script with <file> placeholder -> compile-only (missing inputs)."""
        runner = _make_runner()
        scripts = {
            "tool.py": '"""Tool.\n\nUsage:\n    python tool.py <input_file>\n"""\nimport sys\nprint(sys.argv)\n'
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        # Placeholder detected -> no runnable command -> compile-only
        assert results[0].status == "compiled_only"

    async def test_placeholder_square_brackets(self):
        runner = _make_runner()
        scripts = {
            "tool2.py": '"""Tool.\n\nUsage:\n    python tool2.py [options]\n"""\nprint("ok")\n'
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        assert results[0].status == "compiled_only"


class TestUsageExtraction:
    """Tests for docstring usage line extraction."""

    def test_extract_usage_section(self):
        runner = _make_runner()
        content = '"""My script.\n\nUsage:\n    python myscript.py --verbose\n    python myscript.py --help\n\nMore info here.\n"""\nprint("hi")\n'
        lines = runner._extract_usage_lines(content, "myscript.py")
        assert len(lines) == 2
        assert "python myscript.py --verbose" in lines[0]

    def test_extract_fallback_python_lines(self):
        runner = _make_runner()
        content = '"""My script.\n\npython3 myscript.py input.txt\n"""\n'
        lines = runner._extract_usage_lines(content, "myscript.py")
        assert len(lines) >= 1
        assert "python3" in lines[0]

    def test_no_docstring(self):
        runner = _make_runner()
        content = "print('hello')\n"
        lines = runner._extract_usage_lines(content, "hello.py")
        assert lines == []

    def test_empty_docstring(self):
        runner = _make_runner()
        content = '"""  """\nprint("hi")\n'
        lines = runner._extract_usage_lines(content, "test.py")
        assert lines == []


class TestMaxScriptsLimit:
    """Tests for max_scripts enforcement."""

    async def test_max_scripts_honored(self):
        runner = _make_runner(max_scripts=2)
        scripts = {
            "a.py": "x = 1\n",
            "b.py": "x = 2\n",
            "c.py": "x = 3\n",
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 2


class TestTruncation:
    """Tests for output truncation."""

    async def test_long_output_truncated(self):
        runner = _make_runner(max_output_chars=20)
        scripts = {
            "verbose.py": '"""M.\n\nUsage:\n    python verbose.py\n"""\nprint("A" * 100)\n'
        }
        results = await runner.run_scripts(scripts)
        assert len(results) == 1
        if results[0].status == "success" and results[0].stdout:
            assert results[0].stdout.endswith("...[truncated]")
            assert len(results[0].stdout) <= 20 + len("...[truncated]")


class TestToDict:
    """Tests for serialization."""

    def test_to_dict(self):
        r = ScriptExecutionResult(
            script_name="test.py",
            status="success",
            command="python test.py",
            exit_code=0,
            stdout="ok",
            duration_sec=0.1,
        )
        d = r.to_dict()
        assert d["script_name"] == "test.py"
        assert d["status"] == "success"
        assert d["exit_code"] == 0
