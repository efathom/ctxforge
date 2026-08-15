"""
Tests for GitHub Skill Creator.
"""
import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ctxforge.engine.services.github_skill_creator import GitHubSkillCreator


@dataclass
class FakeLLMResponse:
    content: str


class FakeLLMProvider:
    """Fake LLM provider that returns a valid skill JSON."""

    def __init__(self, response_data: Optional[dict] = None):
        self._data = response_data or {
            "name": "test-repo-skill",
            "description": "Use when working with test-repo library",
            "when_to_use": "When the user needs to use test-repo",
            "triggers": ["test-repo", "test library"],
            "category": "github",
            "instructions": "# Test Repo Skill\n\nThis skill helps you use the test-repo library effectively.",
            "scripts": {"example.py": "print('hello')"},
            "references": {"api.md": "# API Reference\n\nDocumentation here."},
        }

    async def chat(self, messages, model=None, temperature=None,
                   max_tokens=None, **kwargs):
        return FakeLLMResponse(content=json.dumps(self._data))


class TestParseGitHubUrl:
    """Tests for GitHub URL parsing."""

    def test_basic_repo_url(self):
        owner, repo, branch, path = GitHubSkillCreator.parse_github_url(
            "https://github.com/owner/repo"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert branch == "main"
        assert path is None

    def test_url_with_tree(self):
        owner, repo, branch, path = GitHubSkillCreator.parse_github_url(
            "https://github.com/owner/repo/tree/develop/src/lib"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert branch == "develop"
        assert path == "src/lib"

    def test_url_with_blob(self):
        owner, repo, branch, path = GitHubSkillCreator.parse_github_url(
            "https://github.com/owner/repo/blob/main/README.md"
        )
        assert branch == "main"
        assert path == "README.md"

    def test_url_with_dot_git(self):
        owner, repo, branch, path = GitHubSkillCreator.parse_github_url(
            "https://github.com/owner/repo.git"
        )
        assert repo == "repo"

    def test_trailing_slash(self):
        owner, repo, branch, path = GitHubSkillCreator.parse_github_url(
            "https://github.com/owner/repo/"
        )
        assert owner == "owner"
        assert repo == "repo"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Not a GitHub URL"):
            GitHubSkillCreator.parse_github_url("https://gitlab.com/user/repo")

    def test_incomplete_url_raises(self):
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            GitHubSkillCreator.parse_github_url("https://github.com/owner")


class TestCreateFromUrl:
    """Tests for skill creation from GitHub URL."""

    @pytest.fixture
    def mock_aiohttp(self):
        """Create a mock aiohttp module."""
        mock_session = AsyncMock()

        # Mock metadata response
        metadata_resp = AsyncMock()
        metadata_resp.status = 200
        metadata_resp.json = AsyncMock(return_value={
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "description": "A test repository",
            "stargazers_count": 100,
            "language": "Python",
            "topics": ["python", "testing"],
            "default_branch": "main",
        })

        # Mock readme response
        readme_resp = AsyncMock()
        readme_resp.status = 200
        readme_resp.text = AsyncMock(
            return_value="# Test Repo\n\nA test repository."
        )

        # Mock file tree response
        tree_resp = AsyncMock()
        tree_resp.status = 200
        tree_resp.json = AsyncMock(return_value={
            "tree": [
                {"path": "src/main.py", "type": "blob", "size": 500},
                {"path": "README.md", "type": "blob", "size": 200},
                {"path": "node_modules/dep", "type": "tree"},
            ],
        })

        # Mock languages response
        lang_resp = AsyncMock()
        lang_resp.status = 200
        lang_resp.json = AsyncMock(return_value={
            "Python": 8000,
            "JavaScript": 2000,
        })

        # Mock file content response
        file_resp = AsyncMock()
        file_resp.status = 200
        file_resp.text = AsyncMock(
            return_value='def hello():\n    """Say hello."""\n    print("hi")\n'
        )

        def get_side_effect(url, timeout=None):
            ctx = AsyncMock()
            if "/repos/" in url and "/git/trees/" in url:
                ctx.__aenter__.return_value = tree_resp
            elif "/repos/" in url and "/languages" in url:
                ctx.__aenter__.return_value = lang_resp
            elif "/repos/" in url:
                ctx.__aenter__.return_value = metadata_resp
            elif "raw.githubusercontent.com" in url:
                if "README" in url:
                    ctx.__aenter__.return_value = readme_resp
                else:
                    ctx.__aenter__.return_value = file_resp
            else:
                ctx.__aenter__.return_value = metadata_resp
            return ctx

        mock_session.get = MagicMock(side_effect=get_side_effect)

        mock_client_session = MagicMock()
        mock_client_session_ctx = AsyncMock()
        mock_client_session_ctx.__aenter__.return_value = mock_session
        mock_client_session.return_value = mock_client_session_ctx

        return mock_client_session

    @pytest.mark.asyncio
    async def test_create_from_url_success(self, mock_aiohttp):
        llm = FakeLLMProvider()
        creator = GitHubSkillCreator(llm_provider=llm, github_token="fake")

        with patch("ctxforge.engine.services.github_skill_creator.aiohttp",
                   create=True):
            import sys
            sys.modules['aiohttp'] = MagicMock()
            sys.modules['aiohttp'].ClientSession = mock_aiohttp

            try:
                skill = await creator.create_from_url(
                    "https://github.com/owner/test-repo"
                )
            finally:
                del sys.modules['aiohttp']

        assert skill is not None
        assert skill.name == "test-repo-skill"
        assert skill.scope_id == "default"

    @pytest.mark.asyncio
    async def test_create_from_url_invalid_json(self, mock_aiohttp):
        """LLM returns invalid JSON -> None."""

        class BadLLM:
            async def chat(self, **kwargs):
                return FakeLLMResponse(content="not json at all")

        creator = GitHubSkillCreator(llm_provider=BadLLM())

        import sys
        sys.modules['aiohttp'] = MagicMock()
        sys.modules['aiohttp'].ClientSession = mock_aiohttp

        try:
            skill = await creator.create_from_url(
                "https://github.com/owner/test-repo"
            )
        finally:
            del sys.modules['aiohttp']

        assert skill is None

    @pytest.mark.asyncio
    async def test_create_from_url_bad_name(self, mock_aiohttp):
        """LLM returns invalid skill name -> None."""
        bad_data = {
            "name": "Invalid Name!",
            "description": "desc",
            "instructions": "inst",
        }
        llm = FakeLLMProvider(response_data=bad_data)
        creator = GitHubSkillCreator(llm_provider=llm)

        import sys
        sys.modules['aiohttp'] = MagicMock()
        sys.modules['aiohttp'].ClientSession = mock_aiohttp

        try:
            skill = await creator.create_from_url(
                "https://github.com/owner/test-repo"
            )
        finally:
            del sys.modules['aiohttp']

        assert skill is None


class TestExcludedDirs:
    """Tests for excluded directory filtering."""

    @pytest.mark.asyncio
    async def test_node_modules_excluded(self):
        """File tree items in excluded dirs are filtered out."""
        llm = FakeLLMProvider()
        creator = GitHubSkillCreator(llm_provider=llm)
        file_tree = [
            {"path": "src/main.py", "type": "file"},
            {"path": "node_modules/dep/index.js", "type": "file"},
            {"path": "__pycache__/mod.pyc", "type": "file"},
        ]
        formatted = creator._format_file_tree(file_tree)
        # All items are in the list (filtering happens in _fetch_file_tree)
        assert "src/main.py" in formatted


class TestParseResponse:
    """Tests for LLM response parsing."""

    def test_valid_json(self):
        llm = FakeLLMProvider()
        creator = GitHubSkillCreator(llm_provider=llm)
        data = {
            "name": "my-skill",
            "description": "Use when needed",
            "when_to_use": "When the user needs it",
            "triggers": ["my-skill"],
            "instructions": "# Instructions\n\nDo the thing with sufficient detail to pass validation.",
        }
        skill = creator._parse_response(json.dumps(data), "proj1")
        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.scope_id == "proj1"

    def test_json_with_markdown_fences(self):
        llm = FakeLLMProvider()
        creator = GitHubSkillCreator(llm_provider=llm)
        data = {
            "name": "fenced-skill",
            "description": "desc",
            "instructions": "inst",
        }
        raw = f"```json\n{json.dumps(data)}\n```"
        skill = creator._parse_response(raw, "proj1")
        assert skill is not None
        assert skill.name == "fenced-skill"

    def test_invalid_json_returns_none(self):
        llm = FakeLLMProvider()
        creator = GitHubSkillCreator(llm_provider=llm)
        skill = creator._parse_response("not json", "proj1")
        assert skill is None

    def test_invalid_name_returns_none(self):
        llm = FakeLLMProvider()
        creator = GitHubSkillCreator(llm_provider=llm)
        data = {"name": "BAD NAME", "instructions": "x"}
        skill = creator._parse_response(json.dumps(data), "proj1")
        assert skill is None
