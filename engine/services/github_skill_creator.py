"""
GitHub Repository Skill Creator.

Fetches GitHub repository data (metadata, README, file tree, code analysis)
and generates skills via LLM synthesis. Uses async HTTP (aiohttp) for all
GitHub API requests.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ctxforge.core.skill import Skill, SkillContent, SkillScope
from ctxforge.engine.prompts.github_skill import (
    GITHUB_SKILL_SYSTEM_PROMPT,
    build_github_skill_prompt,
)
from ctxforge.engine.services.code_analyzer import CodeAnalyzer
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

logger = logging.getLogger(__name__)

_KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]*$")

_EXCLUDED_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "env", ".env", "build", "dist", ".pytest_cache",
    ".mypy_cache", "htmlcov", ".tox", ".eggs", "vendor", "target",
})


class GitHubSkillCreator:
    """Fetch GitHub repository data and generate skills from code analysis.

    Uses aiohttp for async HTTP requests and CodeAnalyzer for multi-language
    code analysis.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        github_token: Optional[str] = None,
        max_files: int = 50,
        model: Optional[str] = None,
    ):
        self._llm = llm_provider
        self._token = github_token or os.getenv("GITHUB_TOKEN")
        self._max_files = max_files
        self._model = model
        self._analyzer = CodeAnalyzer()

    async def create_from_url(
        self,
        github_url: str,
        project_id: str = "default",
    ) -> Optional[Skill]:
        """Generate a skill from a GitHub repository URL.

        Args:
            github_url: GitHub repository URL.
            project_id: Project scope ID for the generated skill.

        Returns:
            A Skill object, or None if generation failed.
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for GitHub skill creation. "
                "Install with: pip install aiohttp"
            ) from None

        owner, repo, branch, _ = self.parse_github_url(github_url)
        logger.info("Fetching data for %s/%s @ %s", owner, repo, branch)

        headers = self._build_headers()

        async with aiohttp.ClientSession(headers=headers) as session:
            metadata = await self._fetch_repo_metadata(session, owner, repo)
            branch = metadata.get("default_branch", branch)

            readme = await self._fetch_readme(session, owner, repo, branch)
            file_tree = await self._fetch_file_tree(
                session, owner, repo, branch,
            )
            languages = await self._fetch_languages(session, owner, repo)

            code_results = await self._analyze_code_files(
                session, owner, repo, branch, file_tree,
            )

        code_summary = self._analyzer.format_summary(code_results)
        file_tree_str = self._format_file_tree(file_tree)

        lang_str = ", ".join(
            f"{lang}: {pct}%"
            for lang, pct in list(languages.items())[:5]
        ) or "N/A"

        readme_content = (readme or "No README available")[:15000]

        user_prompt = build_github_skill_prompt(
            repo_name=metadata.get(
                "full_name", f"{owner}/{repo}",
            ),
            repo_url=github_url,
            repo_description=(
                metadata.get("description") or "No description available"
            ),
            language=metadata.get("language") or "Unknown",
            languages_breakdown=lang_str,
            stars=metadata.get("stars", 0),
            topics=(
                ", ".join(metadata.get("topics", []))
                if metadata.get("topics") else "None"
            ),
            readme_content=readme_content,
            file_tree=file_tree_str,
            code_summary=code_summary,
        )

        messages = [
            ChatMessage(role="system", content=GITHUB_SKILL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.chat(
            messages=messages,
            model=self._model,
            temperature=0.3,
            max_tokens=4096,
        )

        return self._parse_response(response.content, project_id)

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ctxforge/1.0",
        }
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        return headers

    async def _fetch_repo_metadata(
        self, session: Any, owner: str, repo: str,
    ) -> Dict[str, Any]:
        """Fetch repository metadata from GitHub API."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        data = await self._get_json(session, url)
        if data is None:
            return {"name": repo, "full_name": f"{owner}/{repo}"}

        return {
            "name": data.get("name", repo),
            "full_name": data.get("full_name", f"{owner}/{repo}"),
            "description": data.get("description"),
            "stars": data.get("stargazers_count", 0),
            "language": data.get("language"),
            "topics": data.get("topics", []),
            "default_branch": data.get("default_branch", "main"),
        }

    async def _fetch_readme(
        self, session: Any, owner: str, repo: str, branch: str,
    ) -> Optional[str]:
        """Fetch README content from repository."""
        for name in ("README.md", "README.rst", "README.txt", "README"):
            url = (
                f"https://raw.githubusercontent.com/"
                f"{owner}/{repo}/{branch}/{name}"
            )
            text = await self._get_text(session, url)
            if text is not None:
                return text
        return None

    async def _fetch_file_tree(
        self, session: Any, owner: str, repo: str, branch: str,
    ) -> List[Dict[str, Any]]:
        """Fetch repository file tree, filtering excluded directories."""
        url = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/git/trees/{branch}?recursive=1"
        )
        data = await self._get_json(session, url)
        if data is None:
            return []

        file_tree: List[Dict[str, Any]] = []
        for item in data.get("tree", []):
            path = item.get("path", "")
            if any(excl in path.split("/") for excl in _EXCLUDED_DIRS):
                continue
            file_tree.append({
                "path": path,
                "type": "dir" if item.get("type") == "tree" else "file",
                "size": item.get("size"),
            })
        return file_tree

    async def _fetch_languages(
        self, session: Any, owner: str, repo: str,
    ) -> Dict[str, float]:
        """Fetch language breakdown from GitHub API."""
        url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        data = await self._get_json(session, url)
        if not data:
            return {}

        total = sum(data.values())
        if total == 0:
            return {}
        return {
            lang: round((count / total) * 100, 2)
            for lang, count in data.items()
        }

    async def _fetch_file_content(
        self, session: Any, owner: str, repo: str,
        path: str, branch: str,
    ) -> Optional[str]:
        """Fetch content of a specific file."""
        url = (
            f"https://raw.githubusercontent.com/"
            f"{owner}/{repo}/{branch}/{path}"
        )
        return await self._get_text(session, url)

    async def _analyze_code_files(
        self,
        session: Any,
        owner: str,
        repo: str,
        branch: str,
        file_tree: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze code files to extract class/function signatures."""
        code_files = [
            f for f in file_tree
            if f.get("type") == "file"
            and CodeAnalyzer.is_supported(f.get("path", ""))
        ][:self._max_files]

        results: List[Dict[str, Any]] = []
        for file_info in code_files:
            path = file_info.get("path", "")
            content = await self._fetch_file_content(
                session, owner, repo, path, branch,
            )
            if not content:
                continue

            analysis = self._analyzer.analyze(content, path)
            if analysis.get("classes") or analysis.get("functions"):
                results.append({"file": path, **analysis})

        return results

    async def _get_json(
        self, session: Any, url: str,
    ) -> Optional[Dict[str, Any]]:
        """HTTP GET returning parsed JSON, or None on failure."""
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    logger.debug("HTTP %d for %s", resp.status, url)
                    return None
                return await resp.json()
        except Exception as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    async def _get_text(
        self, session: Any, url: str,
    ) -> Optional[str]:
        """HTTP GET returning text, or None on failure."""
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    return None
                return await resp.text()
        except Exception as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    @staticmethod
    def parse_github_url(url: str) -> Tuple[str, str, str, Optional[str]]:
        """Parse a GitHub URL into (owner, repo, branch, subdir_path).

        Handles:
        - https://github.com/owner/repo
        - https://github.com/owner/repo/tree/main/subdir
        - https://github.com/owner/repo/blob/main/file.py
        - https://github.com/owner/repo.git

        Returns:
            Tuple of (owner, repo, branch, path).

        Raises:
            ValueError: If the URL cannot be parsed.
        """
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]

        if "github.com/" not in url:
            raise ValueError(f"Not a GitHub URL: {url}")

        parts = url.split("github.com/")[-1].split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub URL format: {url}")

        owner, repo = parts[0], parts[1]
        branch = "main"
        path = None

        if len(parts) > 3 and parts[2] in ("tree", "blob"):
            branch = parts[3]
            if len(parts) > 4:
                path = "/".join(parts[4:])

        return owner, repo, branch, path

    def _format_file_tree(self, file_tree: List[Dict[str, Any]]) -> str:
        """Format file tree for LLM prompt."""
        if not file_tree:
            return "No file tree available."

        lines: List[str] = []
        for item in file_tree[:50]:
            path = item.get("path", "")
            icon = "dir" if item.get("type") == "dir" else "file"
            lines.append(f"[{icon}] {path}")

        if len(file_tree) > 50:
            lines.append(f"... and {len(file_tree) - 50} more files")

        return "\n".join(lines)

    def _parse_response(
        self, raw: str, project_id: str,
    ) -> Optional[Skill]:
        """Parse LLM JSON response into a Skill object."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse GitHub skill JSON: %s", exc)
            return None

        name = data.get("name", "")
        if not _KEBAB_RE.match(name):
            logger.warning("Invalid skill name from LLM: '%s'", name)
            return None

        description = data.get("description", "")[:256]
        instructions = data.get("instructions", "")
        triggers = data.get("triggers", [])
        category = data.get("category", "github")
        when_to_use = data.get("when_to_use", "")
        scripts = data.get("scripts", {})
        references = data.get("references", {})

        return Skill(
            name=name,
            description=description,
            scope=SkillScope.PROJECT,
            scope_id=project_id,
            content=instructions,
            triggers=triggers,
            category=category,
            when_to_use=when_to_use,
            structured_content=SkillContent(
                instructions=instructions,
                scripts=scripts if isinstance(scripts, dict) else {},
                references=references if isinstance(references, dict) else {},
            ),
        )
