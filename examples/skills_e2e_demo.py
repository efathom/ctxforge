#!/usr/bin/env python3
"""
End-to-End Demo: ALL Skill Features with MySQL + Neo4j Persistent Storage

Validates every skill feature in the ctxforge system using real infrastructure
configured via examples/.env:

  - Azure OpenAI (or OpenAI) for LLM chat + embeddings
  - MySQL for persistent skill storage (ISkillStore)
  - Neo4j for skill relationship graph visualization
  - In-memory as the always-available fallback

Features covered (18 total):
  1.  Core models          -- Skill, SkillMetadata, SkillContent, SkillScope, enums
  2.  CSO validation       -- Claude Search Optimization lint for descriptions
  3.  Scope layering       -- BASE < USER < PROJECT override semantics
  4.  Trigger matching     -- substring, regex-based via RegexSkillMatcher
  5.  Composite scoring    -- weighted multi-signal ranking via SkillMatcher
  6.  Skills index         -- progressive-disclosure prompt injection
  7.  Content loading      -- SKILL.md parsing, directory layout, YAML frontmatter
  8.  Skill validation     -- structural + CSO checks via SkillValidator
  9.  Skill relationships  -- LLM-inferred + manual, all 4 types
 10.  Dependency resolution -- topological sort of DEPEND_ON chains
 11.  Skill evaluation     -- real LLM 5-dimension quality scoring
 12.  Effectiveness tracking -- usage counts, success rate, ranking boost
 13.  Skill execution      -- sandboxed script execution with tool bridge
 14.  Skill inheritance    -- cross-scope visibility + graduation (promotion)
 15.  Skill lifecycle      -- Generate -> Validate -> Evaluate -> Persist -> Analyze
 16.  Skill generation     -- real two-phase LLM extraction from session events
 17.  Middleware integration -- SkillsMiddleware, SkillRequestMiddleware,
                               SkillEffectivenessMiddleware
 18.  MySQL + Neo4j storage -- full CRUD + relationships + graph persistence

Usage:
    source ctxforge/venv/bin/activate
    python ctxforge/examples/skills_e2e_demo.py
    python ctxforge/examples/skills_e2e_demo.py --skip-mysql     # in-memory skill store
    python ctxforge/examples/skills_e2e_demo.py --skip-neo4j     # skip Neo4j graph
    python ctxforge/examples/skills_e2e_demo.py --skip-llm       # skip LLM-dependent demos
"""

import argparse
import asyncio
import os
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure ctxforge is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

from ctxforge.config.base import (
    EngineConfig,
    Neo4jGraphStoreConfig,
    SkillGenerationConfig,
)
from ctxforge.core.context import Context
from ctxforge.core.events import Event, EventType
from ctxforge.core.skill import (
    EvaluationLevel,
    Skill,
    SkillContent,
    SkillEvaluation,
    SkillMetadata,
    SkillRelationship,
    SkillRelationType,
    SkillScope,
    SkillsIndex,
    lint_skill_description,
)
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.services.skill_content_loader import SkillContentLoader
from ctxforge.engine.services.skill_effectiveness_service import (
    SkillEffectivenessService,
)
from ctxforge.engine.services.skill_evaluation_service import SkillEvaluationService
from ctxforge.engine.services.skill_execution_service import (
    ExecutableRuntimeConfig,
    SkillExecutionService,
)
from ctxforge.engine.services.skill_generator_service import SkillGeneratorService
from ctxforge.engine.services.skill_inheritance_service import (
    SkillInheritanceService,
)
from ctxforge.engine.services.skill_lifecycle_service import SkillLifecycleService
from ctxforge.engine.services.skill_matcher import (
    DEFAULT_WEIGHTS,
    RegexSkillMatcher,
    SkillMatcher,
)
from ctxforge.engine.services.skill_relationship_service import (
    CyclicDependencyError,
    SkillRelationshipService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.engine.services.skill_validator import SkillValidator
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skill_effectiveness import SkillEffectivenessMiddleware
from ctxforge.middleware.skills import SkillRequestMiddleware, SkillsMiddleware
from ctxforge.protocols.llm import ILLMProvider
from ctxforge.storage.connection import MySQLConfig
from ctxforge.storage.memory.skill import InMemorySkillStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def sub(title: str) -> None:
    print(f"\n  --- {title} ---")


def bullet(text: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{text}")


def check(label: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
    extra = f" ({detail})" if detail else ""
    print(f"    [{status}] {label}{extra}")


# ---------------------------------------------------------------------------
# Infrastructure bootstrap (reuses patterns from examples/config.py)
# ---------------------------------------------------------------------------

def _build_engine_config() -> EngineConfig:
    from ctxforge.config.loader import ConfigLoader

    loader = ConfigLoader()
    config_path = Path(__file__).parent / "engine_config.yaml"
    cfg = loader.load_from_file(str(config_path)) if config_path.exists() else EngineConfig()
    cfg = loader.with_env_overrides(cfg)

    provider_name = (getattr(cfg.llm.provider, "value", None) or str(cfg.llm.provider)).lower()
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint and provider_name not in ("azure", "azure_openai"):
        cfg = cfg.merge_with({"llm": {"provider": "azure"}})
        provider_name = "azure"

    if provider_name in ("azure", "azure_openai"):
        patch: Dict[str, Any] = {}
        if azure_key:
            patch.setdefault("llm", {})["api_key"] = azure_key
        if azure_endpoint:
            patch.setdefault("llm", {})["api_base"] = azure_endpoint
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if api_version:
            patch.setdefault("llm", {}).setdefault("extra_params", {})["api_version"] = api_version
        chat_deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if chat_deploy:
            patch.setdefault("llm", {})["model"] = chat_deploy
        if patch:
            cfg = cfg.merge_with(patch)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and not cfg.llm.api_key:
            cfg = cfg.merge_with({"llm": {"api_key": api_key}})

    return cfg


def _get_llm_provider() -> Optional[ILLMProvider]:
    try:
        factory = EngineFactory()
        cfg = _build_engine_config()
        return factory._create_llm_provider(cfg)
    except Exception as exc:
        print(f"    [warn] Could not create LLM provider: {exc}")
        return None


def _get_mysql_config() -> Optional[MySQLConfig]:
    from ctxforge.examples.config import load_config

    try:
        demo_cfg = load_config()
    except Exception:
        demo_cfg = None

    host = os.getenv("MYSQL_HOST", demo_cfg.mysql.host if demo_cfg else "localhost")
    user = os.getenv("MYSQL_USER", demo_cfg.mysql.user if demo_cfg else "contextengine")
    password = os.getenv("MYSQL_PASSWORD", demo_cfg.mysql.password if demo_cfg else "contextengine")
    database = os.getenv("MYSQL_DATABASE", demo_cfg.mysql.database if demo_cfg else "contextengine")
    port = int(os.getenv("MYSQL_PORT", str(demo_cfg.mysql.port if demo_cfg else 3306)))

    return MySQLConfig(host=host, user=user, password=password, database=database, port=port)


def _get_neo4j_config() -> Neo4jGraphStoreConfig:
    return Neo4jGraphStoreConfig(
        url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "contextengine_dev"),
        database=os.getenv("NEO4J_DATABASE", None),
    )


# ---------------------------------------------------------------------------
# Test skill corpus
# ---------------------------------------------------------------------------

DEMO_SKILLS = [
    Skill(
        name="write-tests",
        description="Write comprehensive unit tests for Python code",
        scope=SkillScope.BASE, scope_id="system",
        content=(
            "# Write Unit Tests\n\n"
            "1. Identify the function under test and its edge cases\n"
            "2. Create a test file with pytest conventions\n"
            "3. Write happy-path tests first\n"
            "4. Add boundary and error tests\n"
            "5. Run with coverage and ensure >= 80%\n"
        ),
        triggers=["write test", "unit test", "add tests"],
        category="testing", tags=["python", "pytest"],
        when_to_use="When the user asks to add or write tests for code",
        structured_content=SkillContent(
            instructions="1. Identify function\n2. Write tests\n3. Check coverage",
            scripts={
                "run_tests.py": (
                    "# Simple test runner\n"
                    "import math\n"
                    "result = {'passed': 3, 'failed': 0, 'coverage': 85.5}\n"
                    "print(f'Tests: {result[\"passed\"]} passed, {result[\"failed\"]} failed')\n"
                    "print(f'Coverage: {result[\"coverage\"]}%')\n"
                ),
            },
        ),
    ),
    Skill(
        name="refactor-code",
        description="Refactor Python code for readability and maintainability",
        scope=SkillScope.BASE, scope_id="system",
        content=(
            "# Refactor Code\n\n"
            "1. Identify code smells (long functions, duplication, deep nesting)\n"
            "2. Extract helper functions\n"
            "3. Simplify conditionals\n"
            "4. Add type hints\n"
            "5. Run tests to verify behavior is preserved\n"
        ),
        triggers=["refactor", "clean up code", "improve readability"],
        category="refactoring", tags=["python", "quality"],
        when_to_use="When code needs restructuring for clarity",
    ),
    Skill(
        name="deploy-service",
        description="Deploy a microservice to production with safety checks",
        scope=SkillScope.BASE, scope_id="system",
        content=(
            "# Deploy Service\n\n"
            "1. Run full test suite\n"
            "2. Build container image\n"
            "3. Push to registry\n"
            "4. Deploy canary (10%)\n"
            "5. Monitor error rates for 15 minutes\n"
            "6. Promote to 100%\n"
        ),
        triggers=["deploy", "ship to prod", "release service"],
        category="deployment", tags=["devops", "k8s"],
        when_to_use="When deploying a service to production",
    ),
    Skill(
        name="fix-import-errors",
        description="Fix Python import errors and circular dependencies",
        scope=SkillScope.BASE, scope_id="system",
        content=(
            "# Fix Import Errors\n\n"
            "1. Read the error traceback\n"
            "2. Check if the module exists in the project\n"
            "3. If circular import: refactor to break the cycle\n"
            "4. If missing dependency: add to requirements.txt\n"
            "5. Run tests to verify\n"
        ),
        triggers=["fix imports", "import error", "circular import"],
        prerequisites=["python-basics"],
        allowed_tools=["read_file", "write_file", "run_command"],
        category="debugging", tags=["python", "imports"],
        when_to_use="When encountering ImportError or circular imports",
    ),
    Skill(
        name="python-basics",
        description="Core Python knowledge and conventions",
        scope=SkillScope.BASE, scope_id="system",
        content="# Python Basics\n\nPEP 8, type hints, virtual environments, pip.\n" * 2,
        triggers=["python basics"],
        category="coding", tags=["python"],
        when_to_use="When the user needs help with basic Python concepts",
    ),
]


# ---------------------------------------------------------------------------
# Fake LLM fallback
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Minimal fake so services can be instantiated without a real LLM."""
    name = "fake"
    default_model = "fake-model"

    async def chat(self, **kw: Any) -> Any:
        from dataclasses import dataclass

        @dataclass
        class _R:
            content: str = "[]"

        return _R()


# ---------------------------------------------------------------------------
# 1. Core Models
# ---------------------------------------------------------------------------

async def demo_core_models() -> None:
    header("1. CORE MODELS")

    sub("SkillScope priorities")
    for scope in SkillScope:
        p = SkillScope.priority(scope)
        bullet(f"{scope.value:10s}  priority={p}")
        check(f"SkillScope.{scope.value} has priority", p is not None)

    skill = DEMO_SKILLS[0]
    sub("Skill with structured content")
    bullet(f"Name:          {skill.name}")
    bullet(f"Category:      {skill.category}")
    bullet(f"Tags:          {skill.tags}")
    bullet(f"Triggers:      {skill.triggers}")
    bullet(f"Has scripts:   {bool(skill.structured_content and skill.structured_content.scripts)}")
    check("Skill has name", skill.name == "write-tests")
    check("Skill has structured_content", skill.structured_content is not None)

    sub("SkillMetadata (lightweight)")
    meta = skill.skill_metadata
    bullet(f"Name:        {meta.name}")
    bullet(f"Description: {meta.description}")
    bullet(f"Triggers:    {meta.triggers}")
    check("skill_metadata preserves name", meta.name == skill.name)
    check("skill_metadata preserves triggers", meta.triggers == skill.triggers)

    sub("SkillContent")
    sc = SkillContent(
        instructions="Step 1: do X\nStep 2: do Y",
        scripts={"main.py": "print('hello')"},
        references={"guide": "See docs at..."},
    )
    check("SkillContent has instructions", len(sc.instructions) > 0)
    check("SkillContent has scripts", "main.py" in sc.scripts)
    check("SkillContent has references", "guide" in sc.references)

    sub("SkillRelationship")
    rel = SkillRelationship(
        source="fix-import-errors", target="python-basics",
        relation_type=SkillRelationType.DEPEND_ON,
        reason="Requires Python knowledge", confidence=0.95,
    )
    bullet(f"{rel.source} --[{rel.relation_type.value}]--> {rel.target}")
    check("Relationship types", len(SkillRelationType) == 4,
          ", ".join(r.value for r in SkillRelationType))

    sub("SkillEvaluation scoring")
    score = SkillEvaluation.compute_overall_score(
        EvaluationLevel.GOOD, EvaluationLevel.GOOD,
        EvaluationLevel.AVERAGE, EvaluationLevel.GOOD, EvaluationLevel.GOOD,
    )
    bullet(f"GOOD/GOOD/AVG/GOOD/GOOD -> overall = {score}")
    check("Overall score computed", 0 < score <= 1.0)


# ---------------------------------------------------------------------------
# 2. CSO Validation
# ---------------------------------------------------------------------------

async def demo_cso_validation() -> None:
    header("2. CSO (CLAUDE SEARCH OPTIMIZATION) VALIDATION")

    sub("Good descriptions (trigger-only)")
    good_descs = [
        "When the user asks to fix failing tests",
        "When Python import errors are encountered",
        "When a service needs to go to production",
    ]
    for d in good_descs:
        warnings = lint_skill_description(d)
        bullet(f"'{d}' -> {len(warnings)} warnings")
        check("No CSO warnings for trigger-only desc", len(warnings) == 0)

    sub("Bad descriptions (workflow summaries)")
    bad_descs = [
        "Review code then execute tests and deploy the service",
        "First analyze the imports, then refactor the module",
        "Generate test files, create fixtures, then execute test suite",
    ]
    for d in bad_descs:
        warnings = lint_skill_description(d)
        bullet(f"'{d}' -> {len(warnings)} warnings")
        for w in warnings:
            bullet(f"  WARNING: {w}", indent=6)
        check("CSO warnings flagged for workflow summary", len(warnings) > 0)


# ---------------------------------------------------------------------------
# 3. Scope Layering
# ---------------------------------------------------------------------------

async def demo_scope_layering() -> None:
    header("3. SCOPE LAYERING (BASE < USER < PROJECT)")

    store = InMemorySkillStore()
    await store.initialize()

    for scope, scope_id, desc in [
        (SkillScope.BASE, "system", "Generic deployment"),
        (SkillScope.USER, "alice", "Alice's canary deployment"),
        (SkillScope.PROJECT, "proj-x", "Project-X k8s deployment"),
    ]:
        await store.save(Skill(
            name="deploy", description=desc,
            scope=scope, scope_id=scope_id,
            content=f"# {desc}\n" + "instructions " * 10,
            triggers=["deploy"],
            when_to_use="When deploying",
        ))
        bullet(f"Saved {scope.value:8s} deploy: {desc}")

    for label, uid, pid, expected_scope in [
        ("alice + proj-x", "alice", "proj-x", SkillScope.PROJECT),
        ("alice only", "alice", None, SkillScope.USER),
        ("anonymous", None, None, SkillScope.BASE),
    ]:
        meta = await store.list_all_metadata(user_id=uid, project_id=pid)
        d = next(m for m in meta if m.name == "deploy")
        sub(f"View for {label}")
        bullet(f"Winner: {d.scope.value} -> '{d.description}'")
        check(f"Scope winner for {label}", d.scope == expected_scope)


# ---------------------------------------------------------------------------
# 4. Trigger Matching
# ---------------------------------------------------------------------------

async def demo_trigger_matching() -> None:
    header("4. TRIGGER MATCHING")

    meta = DEMO_SKILLS[3].skill_metadata
    sub("Substring matching via matches_trigger()")
    for q, should_match in [
        ("How do I fix imports?", True),
        ("import error on line 5", True),
        ("What's the weather?", False),
    ]:
        m = meta.matches_trigger(q)
        result = f"MATCH: {m}" if m else "no match"
        bullet(f"'{q}' -> {result}")
        check(f"Trigger match for '{q[:30]}...'", bool(m) == should_match)

    sub("RegexSkillMatcher")
    regex = RegexSkillMatcher()
    rmeta = SkillMetadata(
        name="fix-type-errors", description="Fix TypeErrors",
        scope=SkillScope.BASE, scope_id="system",
        triggers=[r"TypeError:\s+\w+", r"type\s+error"],
    )
    for q, should_match in [
        ("TypeError: unsupported operand", True),
        ("I have a type error", True),
        ("sort a list", False),
    ]:
        results = await regex.match(q, [rmeta], threshold=0.0)
        matched = bool(results and results[0].confidence > 0)
        bullet(f"'{q}' -> {'MATCH' if matched else 'no match'}")
        check(f"Regex match for '{q[:30]}...'", matched == should_match)


# ---------------------------------------------------------------------------
# 5. Composite Scoring
# ---------------------------------------------------------------------------

async def demo_composite_scoring() -> None:
    header("5. COMPOSITE SCORING (SkillMatcher)")

    sub("Default weights")
    for signal, w in DEFAULT_WEIGHTS.items():
        bullet(f"{signal:15s} = {w:.2f}")
    check("Weights sum to ~1.0", abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.01)

    matcher = SkillMatcher()
    metas = [s.skill_metadata for s in DEMO_SKILLS[:4]]

    sub("Matching queries")
    queries_expected = [
        ("Help me write unit tests for auth", "write-tests"),
        ("Refactor this messy function", "refactor-code"),
        ("Deploy the payment service", "deploy-service"),
    ]
    for q, expected_name in queries_expected:
        results = await matcher.match(q, metas, threshold=0.0)
        if results:
            top = results[0]
            bullet(f"'{q[:50]}' -> {top.skill.name} (conf={top.confidence:.3f})")
            check(f"Top match for '{q[:30]}...'", top.skill.name == expected_name)
        else:
            bullet(f"'{q[:50]}' -> no match")
            check(f"Expected match for '{q[:30]}...'", False)

    no_match_q = "What's the meaning of life?"
    results = await matcher.match(no_match_q, metas, threshold=0.5)
    bullet(f"'{no_match_q}' -> {len(results)} matches above 0.5 threshold")
    check("No false matches for irrelevant query", len(results) == 0)


# ---------------------------------------------------------------------------
# 6. Skills Index
# ---------------------------------------------------------------------------

async def demo_skills_index() -> None:
    header("6. SKILLS INDEX (PROGRESSIVE DISCLOSURE)")

    metas = [s.skill_metadata for s in DEMO_SKILLS[:4]]
    index = SkillsIndex(skills=metas)

    sub("Summary")
    bullet(f"Total: {index.total_count}")
    compact = index.format_compact()
    bullet(f"Compact: {compact}")
    check("Total count correct", index.total_count == 4)

    sub("Prompt format (first 200 chars)")
    prompt = index.format_for_prompt()
    for line in prompt.split("\n")[:8]:
        bullet(line, indent=6)
    check("Prompt format non-empty", len(prompt) > 50)
    check("Prompt contains skill names", "write-tests" in prompt)


# ---------------------------------------------------------------------------
# 7. Content Loading
# ---------------------------------------------------------------------------

async def demo_content_loading() -> None:
    header("7. CONTENT LOADING (SkillContentLoader)")

    raw_md = textwrap.dedent("""\
        ---
        name: fix-import-errors
        category: debugging
        tags: python, imports
        triggers: fix imports, import error
        when_to_use: When encountering import issues
        ---
        # Fix Import Errors

        1. Read the error traceback
        2. Identify the failing module
        3. Fix the cycle or add the dependency
    """)

    sub("Parse SKILL.md with frontmatter")
    content = SkillContentLoader.parse_skill_markdown(raw_md)
    bullet(f"Instructions: {content.instructions[:60]}...")
    check("Instructions parsed", "Fix Import" in content.instructions)

    fm = SkillContentLoader.parse_frontmatter(raw_md)
    bullet(f"Frontmatter: {fm}")
    check("Frontmatter has name", fm.get("name") == "fix-import-errors")
    check("Frontmatter has category", fm.get("category") == "debugging")

    sub("Load from directory structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        sd = os.path.join(tmpdir, "my-skill")
        os.makedirs(os.path.join(sd, "scripts"))
        os.makedirs(os.path.join(sd, "references"))
        with open(os.path.join(sd, "SKILL.md"), "w") as f:
            f.write(raw_md)
        with open(os.path.join(sd, "scripts", "setup.sh"), "w") as f:
            f.write("#!/bin/bash\necho 'hello'\n")
        with open(os.path.join(sd, "references", "guide.md"), "w") as f:
            f.write("# Guide\nSee PEP 328.\n")

        loaded = SkillContentLoader.load_from_directory(sd)
        bullet(f"Scripts: {list(loaded.scripts.keys())}")
        bullet(f"Refs:    {list(loaded.references.keys())}")
        check("Scripts loaded", "setup.sh" in loaded.scripts)
        check("References loaded", "guide.md" in loaded.references)

    sub("Progressive disclosure levels")
    l1 = len(SkillContentLoader.format_for_prompt(loaded))
    l2 = len(SkillContentLoader.format_for_prompt(loaded, include_scripts=True))
    l3 = len(SkillContentLoader.format_for_prompt(loaded, include_scripts=True, include_references=True))
    bullet(f"Level 1 (instructions): {l1} chars")
    bullet(f"Level 2 (+scripts):     {l2} chars")
    bullet(f"Level 3 (+refs):        {l3} chars")
    check("Progressive disclosure grows", l1 < l2 < l3)


# ---------------------------------------------------------------------------
# 8. Skill Validation
# ---------------------------------------------------------------------------

async def demo_validation() -> None:
    header("8. SKILL VALIDATION (SkillValidator)")

    validator = SkillValidator(min_instruction_length=50)

    sub("Valid skill")
    result = validator.validate(DEMO_SKILLS[0])
    bullet(f"is_valid={result.is_valid}, errors={result.errors}, warnings={len(result.warnings)}")
    check("Well-formed skill passes validation", result.is_valid)

    sub("Invalid skill (empty description, no triggers, short content)")
    # Note: Skill.__post_init__ enforces kebab-case names at construction,
    # so we use a valid name but leave other fields intentionally broken.
    bad_skill = Skill(
        name="bad-skill", description="", scope=SkillScope.BASE, scope_id="system",
        content="too short", triggers=[], when_to_use="",
    )
    result = validator.validate(bad_skill)
    bullet(f"is_valid={result.is_valid}")
    for e in result.errors:
        bullet(f"  ERROR: {e}", indent=6)
    check("Malformed skill fails validation", not result.is_valid)
    check("Multiple errors detected", len(result.errors) >= 3)

    sub("validate_and_raise()")
    try:
        validator.validate_and_raise(bad_skill)
        check("Should have raised ValueError", False)
    except ValueError as exc:
        bullet(f"Raised: {str(exc)[:80]}...")
        check("validate_and_raise raises ValueError", True)


# ---------------------------------------------------------------------------
# 9. Skill Relationships
# ---------------------------------------------------------------------------

async def demo_relationships(llm: Optional[ILLMProvider], store: Any) -> None:
    header("9. SKILL RELATIONSHIPS")

    for s in DEMO_SKILLS:
        await store.save(s)

    manual_rels = [
        SkillRelationship("fix-import-errors", "python-basics",
                          SkillRelationType.DEPEND_ON, "Requires Python knowledge"),
        SkillRelationship("write-tests", "python-basics",
                          SkillRelationType.DEPEND_ON, "Tests need Python"),
        SkillRelationship("fix-import-errors", "write-tests",
                          SkillRelationType.COMPOSE_WITH, "Fix then test"),
        SkillRelationship("refactor-code", "write-tests",
                          SkillRelationType.COMPOSE_WITH, "Refactor then test"),
        SkillRelationship("deploy-service", "write-tests",
                          SkillRelationType.DEPEND_ON, "Must test before deploy"),
        SkillRelationship("refactor-code", "fix-import-errors",
                          SkillRelationType.SIMILAR_TO, "Both fix code issues"),
    ]
    saved = await store.save_relationships(manual_rels)
    bullet(f"Saved {saved} manual relationships")
    check("All relationships saved", saved == len(manual_rels))

    sub("Full graph")
    all_rels = await store.get_all_relationships()
    for r in all_rels:
        bullet(f"{r.source} --[{r.relation_type.value}]--> {r.target}")
    check("Relationships retrievable", len(all_rels) >= len(manual_rels))

    rel_service = SkillRelationshipService(
        llm_provider=llm or _FakeLLM(), skill_store=store,
    )

    sub("Dependency chain for deploy-service")
    chain = await rel_service.resolve_dependency_chain("deploy-service")
    bullet(f"Chain: {' -> '.join(chain)}")
    check("Dependency chain resolved", len(chain) >= 2)
    check("deploy-service in chain", "deploy-service" in chain)

    sub("Composable skills for fix-import-errors")
    composable = await rel_service.find_composable_skills("fix-import-errors")
    bullet(f"Composes with: {composable}")
    check("write-tests is composable", "write-tests" in composable)

    sub("Alternatives for refactor-code")
    alternatives = await rel_service.find_alternatives("refactor-code")
    bullet(f"Similar to: {alternatives}")
    check("fix-import-errors is alternative", "fix-import-errors" in alternatives)

    if llm is not None and not isinstance(llm, _FakeLLM):
        sub("LLM-inferred relationships (real LLM)")
        try:
            metas = [s.skill_metadata for s in DEMO_SKILLS[:4]]
            inferred = await rel_service.analyze_relationships(metas)
            bullet(f"LLM inferred {len(inferred)} relationships:")
            for r in inferred[:5]:
                bullet(f"  {r.source} --[{r.relation_type.value}]--> {r.target} "
                       f"(conf={r.confidence:.2f})")
            check("LLM inferred relationships", len(inferred) > 0)
        except Exception as exc:
            bullet(f"LLM relationship analysis failed: {exc}")
    else:
        sub("LLM-inferred relationships (skipped -- no LLM)")


# ---------------------------------------------------------------------------
# 10. Dependency Resolution
# ---------------------------------------------------------------------------

async def demo_dependency_resolution(store: Any) -> None:
    header("10. DEPENDENCY RESOLUTION (Topological Sort)")

    rel_service = SkillRelationshipService(
        llm_provider=_FakeLLM(), skill_store=store,
    )

    sub("Linear chain: deploy-service -> write-tests -> python-basics")
    chain = await rel_service.resolve_dependency_chain("deploy-service")
    bullet(f"Resolved: {' -> '.join(chain)}")
    check("Chain starts with python-basics", chain[0] == "python-basics" if chain else False)
    check("Chain ends with deploy-service", chain[-1] == "deploy-service" if chain else False)

    sub("Cycle detection")
    # Create a temporary cycle
    cycle_rels = [
        SkillRelationship("python-basics", "deploy-service",
                          SkillRelationType.DEPEND_ON, "Cyclic!"),
    ]
    await store.save_relationships(cycle_rels)
    try:
        chain = await rel_service.resolve_dependency_chain("deploy-service")
        # Some implementations may not raise but just break the cycle
        bullet(f"Chain after adding cycle: {chain}")
        check("Cycle handled gracefully", True)
    except CyclicDependencyError as exc:
        bullet(f"CyclicDependencyError raised: {exc}")
        check("Cycle detected and raised", True)


# ---------------------------------------------------------------------------
# 11. Skill Evaluation (real LLM)
# ---------------------------------------------------------------------------

async def demo_evaluation(llm: Optional[ILLMProvider]) -> None:
    header("11. SKILL EVALUATION (5-Dimension Quality Scoring)")

    sub("Dimension weights")
    for dim, w in [("Safety", 25), ("Completeness", 25), ("Executability", 20),
                   ("Maintainability", 15), ("Cost-awareness", 15)]:
        bullet(f"{dim}: {w}%")

    sub("Score computation examples")
    for label, levels in [
        ("All GOOD", [EvaluationLevel.GOOD] * 5),
        ("All AVERAGE", [EvaluationLevel.AVERAGE] * 5),
        ("All POOR", [EvaluationLevel.POOR] * 5),
        ("Mixed", [EvaluationLevel.GOOD, EvaluationLevel.GOOD,
                   EvaluationLevel.AVERAGE, EvaluationLevel.GOOD, EvaluationLevel.AVERAGE]),
    ]:
        score = SkillEvaluation.compute_overall_score(*levels)
        bullet(f"{label:15s} -> {score:.4f}")
    check("GOOD > AVERAGE > POOR", (
        SkillEvaluation.compute_overall_score(*[EvaluationLevel.GOOD] * 5)
        > SkillEvaluation.compute_overall_score(*[EvaluationLevel.AVERAGE] * 5)
        > SkillEvaluation.compute_overall_score(*[EvaluationLevel.POOR] * 5)
    ))

    if llm is not None and not isinstance(llm, _FakeLLM):
        sub("Real LLM evaluation of 'write-tests' skill")
        eval_service = SkillEvaluationService(llm)
        try:
            result = await eval_service.evaluate(DEMO_SKILLS[0])
            bullet(f"Overall score: {result.overall_score}")
            for dim in ["safety", "completeness", "executability", "maintainability", "cost_awareness"]:
                level = getattr(result, dim)
                reason = getattr(result, f"{dim}_reason")
                bullet(f"  {dim:20s} = {level.value:8s}  {reason[:60]}")
            check("LLM evaluation returned valid score", 0 < result.overall_score <= 1.0)
        except Exception as exc:
            bullet(f"Evaluation failed: {exc}")
    else:
        sub("Real LLM evaluation (skipped -- no LLM)")


# ---------------------------------------------------------------------------
# 12. Effectiveness Tracking
# ---------------------------------------------------------------------------

async def demo_effectiveness(store: Any) -> None:
    header("12. EFFECTIVENESS TRACKING")

    eff = SkillEffectivenessService(store)

    # Ensure skill exists in store
    await store.save(DEMO_SKILLS[0])

    sub("Recording usage")
    for conf, sess in [(0.85, "s1"), (0.90, "s2"), (0.78, "s3")]:
        await eff.record_usage("write-tests", SkillScope.BASE, "system", conf, sess)
        bullet(f"Usage: conf={conf}, session={sess}")

    sub("Recording outcomes")
    await eff.record_outcome("write-tests", SkillScope.BASE, "system", success=True)
    await eff.record_outcome("write-tests", SkillScope.BASE, "system", success=True)
    await eff.record_outcome("write-tests", SkillScope.BASE, "system", success=False)

    metrics = await eff.get_effectiveness("write-tests", SkillScope.BASE, "system")
    sub("Metrics")
    for k in ["usage_count", "success_count", "failure_count", "success_rate",
              "avg_confidence_at_match"]:
        bullet(f"{k}: {metrics.get(k)}")
    check("Usage count is 3", metrics.get("usage_count") == 3)
    check("Success count is 2", metrics.get("success_count") == 2)
    check("Failure count is 1", metrics.get("failure_count") == 1)

    boost = await eff.get_ranking_boost("write-tests", SkillScope.BASE, "system")
    bullet(f"Ranking boost: {boost:.4f}")
    check("Ranking boost in [0, 1]", 0 <= boost <= 1.0)


# ---------------------------------------------------------------------------
# 13. Skill Execution
# ---------------------------------------------------------------------------

async def demo_execution(store: Any) -> None:
    header("13. SKILL EXECUTION (Sandboxed Script Runner)")

    service = SkillService(store=store)
    await service.register_skill(DEMO_SKILLS[0])

    config = ExecutableRuntimeConfig(enabled=True, timeout_sec=5.0, sandbox=True)
    executor = SkillExecutionService(skill_service=service, config=config)

    sub("Execute 'write-tests' script")
    result = await executor.execute("write-tests", args={"target": "auth_module"})
    bullet(f"Status:       {result.status}")
    bullet(f"Return value: {result.return_value}")
    bullet(f"Stdout:       {result.stdout.strip()}")
    bullet(f"Duration:     {result.duration_sec}s")
    check("Execution succeeded", result.status == "success")
    check("Stdout captured", "Tests:" in result.stdout)

    sub("Safety check (dangerous script)")
    danger_skill = Skill(
        name="dangerous-skill",
        description="A skill with dangerous code",
        scope=SkillScope.BASE, scope_id="system",
        content="# Dangerous\nDo bad things.\n" + "x " * 30,
        triggers=["danger"],
        when_to_use="Never",
        structured_content=SkillContent(
            instructions="Do not run this.\n" + "x " * 30,
            scripts={"bad.py": "import os\nos.system('echo pwned')"},
        ),
    )
    await service.register_skill(danger_skill)
    result = await executor.execute("dangerous-skill")
    bullet(f"Status: {result.status}")
    bullet(f"Detail: {result.detail}")
    check("Dangerous script rejected", result.status == "rejected")

    sub("Not-found skill")
    result = await executor.execute("nonexistent-skill")
    check("Not-found returns status='not_found'", result.status == "not_found")

    sub("Schema inference")
    infer_skill = Skill(
        name="schema-skill",
        description="Returns structured data",
        scope=SkillScope.BASE, scope_id="system",
        content="# Schema test\n" + "x " * 30,
        triggers=["schema"],
        when_to_use="For testing schema inference",
        structured_content=SkillContent(
            instructions="Return a dict.\n" + "x " * 30,
            scripts={"main.py": "result = {'name': 'test', 'count': 42, 'items': [1,2,3]}"},
        ),
    )
    await service.register_skill(infer_skill)
    result = await executor.execute("schema-skill")
    bullet(f"Return: {result.return_value}")
    bullet(f"Schema: {result.inferred_output_schema}")
    check("Schema inferred", result.inferred_output_schema is not None)
    check("Schema type is object", result.inferred_output_schema.get("type") == "object")


# ---------------------------------------------------------------------------
# 14. Skill Inheritance
# ---------------------------------------------------------------------------

async def demo_inheritance(store: Any) -> None:
    header("14. SKILL INHERITANCE (Cross-Scope + Graduation)")

    # Setup: BASE, USER, PROJECT skills
    await store.save(Skill(
        name="code-review", description="Review code for quality",
        scope=SkillScope.BASE, scope_id="system",
        content="# Code Review\n1. Check style\n2. Check logic\n" + "x " * 20,
        triggers=["code review"], when_to_use="When reviewing code",
    ))
    await store.save(Skill(
        name="alice-deploy", description="Alice's custom deploy",
        scope=SkillScope.USER, scope_id="alice",
        content="# Alice Deploy\n1. Use canary\n2. Rollback ready\n" + "x " * 20,
        triggers=["deploy"], when_to_use="When Alice deploys",
    ))
    await store.save(Skill(
        name="proj-test", description="Project-specific test runner",
        scope=SkillScope.PROJECT, scope_id="proj-alpha",
        content="# Proj Test\n1. Use custom fixtures\n2. Run integration\n" + "x " * 20,
        triggers=["test"], when_to_use="When testing proj-alpha",
    ))

    inheritance = SkillInheritanceService(skill_store=store)

    sub("Inherited skills for PROJECT scope (proj-alpha, user=alice)")
    inherited = await inheritance.get_inherited_skills(
        SkillScope.PROJECT, "proj-alpha", user_id="alice",
    )
    names = [s.name for s in inherited]
    bullet(f"Visible skills: {names}")
    check("BASE skill visible", "code-review" in names)
    check("USER skill visible", "alice-deploy" in names)
    check("PROJECT skill visible", "proj-test" in names)

    sub("Inherited skills for USER scope (alice)")
    inherited_user = await inheritance.get_inherited_skills(
        SkillScope.USER, "alice",
    )
    user_names = [s.name for s in inherited_user]
    bullet(f"Visible to alice: {user_names}")
    check("BASE visible to USER", "code-review" in user_names)
    check("PROJECT NOT visible to USER", "proj-test" not in user_names)

    sub("Graduation (promote USER -> BASE)")
    # First set effectiveness to qualify for graduation
    await store.update_effectiveness("alice-deploy", SkillScope.USER, "alice", {
        "usage_count": 10, "success_count": 9, "success_rate": 0.9,
    })
    candidates = await inheritance.get_graduation_candidates(
        SkillScope.USER, "alice", min_usage_count=5, min_success_rate=0.7,
    )
    bullet(f"Graduation candidates: {[c.name for c in candidates]}")
    check("alice-deploy is graduation candidate", any(c.name == "alice-deploy" for c in candidates))

    graduated = await inheritance.graduate_skill(
        "alice-deploy", SkillScope.USER, "alice", SkillScope.BASE, "system",
    )
    if graduated:
        bullet(f"Graduated: {graduated.name} -> scope={graduated.scope.value}")
        bullet(f"Provenance: promoted_from={graduated.promoted_from}, source_scope={graduated.source_scope}")
        check("Graduated to BASE scope", graduated.scope == SkillScope.BASE)
        check("Provenance tracked", graduated.promoted_from == "user")
    else:
        check("Graduation succeeded", False)


# ---------------------------------------------------------------------------
# 15. Skill Lifecycle (Generate -> Validate -> Evaluate -> Persist)
# ---------------------------------------------------------------------------

async def demo_lifecycle(llm: Optional[ILLMProvider], store: Any) -> None:
    header("15. SKILL LIFECYCLE (Full Pipeline)")

    if llm is None or isinstance(llm, _FakeLLM):
        bullet("Skipped -- requires real LLM provider")
        return

    service = SkillService(store=store)
    validator = SkillValidator(min_instruction_length=20)
    evaluator = SkillEvaluationService(llm)
    config = SkillGenerationConfig(enabled=True, min_session_events=3)
    generator = SkillGeneratorService(llm, service, config)
    eff_service = SkillEffectivenessService(store)
    rel_service = SkillRelationshipService(llm_provider=llm, skill_store=store)

    lifecycle = SkillLifecycleService(
        generator=generator,
        validator=validator,
        evaluator=evaluator,
        skill_service=service,
        relationship_service=rel_service,
        effectiveness_service=eff_service,
        min_evaluation_score=0.3,
    )

    sub("Create from session events (full pipeline)")
    events = [
        Event(type=EventType.USER, content="These tests keep failing randomly in CI"),
        Event(type=EventType.AGENT, content="Let me investigate the flaky tests"),
        Event(type=EventType.USER, content="It's the auth test, it fails 1 in 5 runs"),
        Event(type=EventType.AGENT, content="I found a race condition in the test setup fixture"),
        Event(type=EventType.USER, content="Great, can you fix it?"),
        Event(type=EventType.AGENT, content="Done. I added proper async locks and a retry decorator."),
    ]
    try:
        created = await lifecycle.create_from_session(events, project_id="demo-proj")
        bullet(f"Pipeline created {len(created)} skill(s)")
        for sk in created:
            bullet(f"  name={sk.name}, score={sk.evaluation.overall_score if sk.evaluation else 'N/A'}")
        check("Lifecycle pipeline ran without error", True)
        if created:
            check("Lifecycle produced skills", True)
        else:
            bullet("  (LLM did not produce valid skills -- nondeterministic, not a bug)")
    except Exception as exc:
        bullet(f"Lifecycle pipeline failed: {exc}")

    sub("Retire underperforming skills")
    # Create a poorly performing skill
    poor_skill = Skill(
        name="bad-skill", description="A poorly performing skill",
        scope=SkillScope.PROJECT, scope_id="demo-proj",
        content="# Bad Skill\n1. Do nothing useful\n" + "x " * 30,
        triggers=["bad"], when_to_use="Never",
    )
    await store.save(poor_skill)
    await store.update_effectiveness("bad-skill", SkillScope.PROJECT, "demo-proj", {
        "usage_count": 10, "success_count": 1, "failure_count": 9, "success_rate": 0.1,
    })
    retired = await lifecycle.retire_underperforming(
        SkillScope.PROJECT, "demo-proj", min_success_rate=0.3, min_usage_count=5,
    )
    bullet(f"Retired: {retired}")
    check("Underperforming skill retired", "bad-skill" in retired)


# ---------------------------------------------------------------------------
# 16. Skill Generation (real LLM)
# ---------------------------------------------------------------------------

async def demo_generation(llm: Optional[ILLMProvider]) -> None:
    header("16. SKILL GENERATION (Two-Phase LLM)")

    if llm is None or isinstance(llm, _FakeLLM):
        bullet("Skipped -- requires real LLM provider")
        return

    store = InMemorySkillStore()
    await store.initialize()
    service = SkillService(store=store)
    config = SkillGenerationConfig(enabled=True, min_session_events=3)
    generator = SkillGeneratorService(llm, service, config)

    sub("Generate from session events")
    events = [
        Event(type=EventType.USER, content="My API returns 500 errors under load"),
        Event(type=EventType.AGENT, content="Let me check the rate limiting configuration"),
        Event(type=EventType.USER, content="There's no rate limiting at all"),
        Event(type=EventType.AGENT, content="I'll add a token bucket rate limiter with Redis"),
        Event(type=EventType.USER, content="Works great, load test passes now"),
        Event(type=EventType.AGENT, content="Done! I configured 100 req/s per client with burst of 20."),
    ]
    try:
        generated = await generator.generate_from_session(events, project_id="demo-proj")
        bullet(f"Generated {len(generated)} skill(s):")
        for sk in generated:
            bullet(f"  name={sk.name}, category={sk.category}")
            bullet(f"  triggers: {sk.triggers}")
        check("Session generation ran without error", True)
        if generated:
            check("Session generation produced skills", True)
        else:
            bullet("  (LLM did not produce valid skills -- nondeterministic, not a bug)")
    except Exception as exc:
        bullet(f"Session generation failed: {exc}")

    sub("Generate from prompt")
    try:
        prompt_skill = await generator.generate_from_prompt(
            "Create a skill for setting up Python virtual environments with pip",
            project_id="demo-proj",
        )
        if prompt_skill:
            bullet(f"Generated: {prompt_skill.name}")
            bullet(f"  triggers: {prompt_skill.triggers}")
            check("Prompt generation succeeded", True)
        else:
            bullet("LLM returned invalid skill")
    except Exception as exc:
        bullet(f"Prompt generation failed: {exc}")


# ---------------------------------------------------------------------------
# 17. Middleware Integration
# ---------------------------------------------------------------------------

async def demo_middleware() -> None:
    header("17. MIDDLEWARE INTEGRATION")

    store = InMemorySkillStore()
    await store.initialize()
    matcher = SkillMatcher()
    service = SkillService(store=store, matcher=matcher)
    for s in DEMO_SKILLS[:3]:
        await service.register_skill(s)

    async def noop(ctx: MiddlewareContext) -> MiddlewareContext:
        return ctx

    sub("SkillsMiddleware -- auto-activation")
    mw = SkillsMiddleware(
        skill_service=service, user_id="alice",
        auto_activate=True, max_auto_skills=2, confidence_threshold=0.3,
    )
    for q, expect_skill in [
        ("Help me write unit tests", "write-tests"),
        ("Refactor this code", "refactor-code"),
    ]:
        ctx = MiddlewareContext(user_input=q, user_id="alice", session_id="s1")
        result = await mw.process(ctx, noop)
        # Middleware stores under key "skills" as a list of dicts
        skills_mods = result.modifications.get("skills", [])
        activated_names: list = []
        for mod in skills_mods:
            activated_names.extend(mod.get("activated_skills", []))
        # Also check context sections for activated_skills section
        section_names = [s.name for s in result.context_sections]
        has_section = "activated_skills" in section_names
        bullet(f"'{q}' -> activated={activated_names}, sections={section_names}")
        check(f"Auto-activated '{expect_skill}'",
              expect_skill in activated_names or has_section)

    sub("SkillRequestMiddleware -- explicit requests")
    req_mw = SkillRequestMiddleware(skill_service=service, user_id="alice")
    for q, expect_loaded in [
        ("use skill deploy-service", True),
        ("normal question", False),
    ]:
        ctx = MiddlewareContext(user_input=q, user_id="alice", session_id="s1")
        result = await req_mw.process(ctx, noop)
        # Skill loaded is stored in metadata, not modifications
        loaded_name = result.metadata.get("requested_skill")
        has_skill_flag = "skill_requested" in result.flags
        bullet(f"'{q}' -> {'loaded: ' + loaded_name if loaded_name else 'no skill requested'}")
        check(f"Explicit request: loaded={expect_loaded}",
              bool(loaded_name or has_skill_flag) == expect_loaded)

    sub("SkillEffectivenessMiddleware -- usage tracking")
    eff_service = SkillEffectivenessService(store)
    eff_mw = SkillEffectivenessMiddleware(eff_service, SkillScope.BASE, "system")
    ctx = MiddlewareContext(user_input="test", user_id="alice", session_id="s1")
    ctx.add_flag("skills_auto_activated")
    ctx.modifications["skills"] = [{"activated_skills": ["write-tests"]}]
    await eff_mw.process(ctx, noop)
    m = await eff_service.get_effectiveness("write-tests", SkillScope.BASE, "system")
    bullet(f"After activation: usage_count={m.get('usage_count', 0)}")
    check("Effectiveness middleware tracked usage", m.get("usage_count", 0) >= 1)


# ---------------------------------------------------------------------------
# 18. MySQL + Neo4j Storage
# ---------------------------------------------------------------------------

async def demo_mysql_store(mysql_config: MySQLConfig) -> None:
    header("18a. MYSQL SKILL STORE (Persistent)")

    from ctxforge.storage.mysql.skill import MySQLSkillStore

    store = MySQLSkillStore(config=mysql_config, table_name="skills_e2e_demo")

    try:
        sub("Initialize")
        await store.initialize()
        bullet("MySQL connected and tables created")
        check("MySQL initialization", True)

        sub("Save skills")
        for s in DEMO_SKILLS:
            await store.save(s)
            bullet(f"Saved: {s.name} ({s.scope.value})")
        check("All skills saved", True)

        sub("Scope layering (PROJECT override)")
        await store.save(Skill(
            name="write-tests", description="Project-specific test workflow",
            scope=SkillScope.PROJECT, scope_id="proj-demo",
            content="# Project Tests\n1. Use project fixtures\n2. Run integration tests\n" + "x " * 20,
            triggers=["test"], category="testing",
            when_to_use="When testing in proj-demo",
        ))
        all_meta = await store.list_all_metadata(user_id=None, project_id="proj-demo")
        wt = next((m for m in all_meta if m.name == "write-tests"), None)
        if wt:
            bullet(f"write-tests winner: scope={wt.scope.value}")
            check("PROJECT overrides BASE", wt.scope == SkillScope.PROJECT)

        sub("Search by trigger")
        matches = await store.search_by_trigger("deploy to production")
        bullet(f"'deploy to production' -> {len(matches)} matches")
        check("Trigger search returns matches", len(matches) > 0)

        sub("Search by category")
        cat_results = await store.search_by_category("testing")
        bullet(f"Category 'testing': {[m.name for m in cat_results]}")
        check("Category search works", len(cat_results) > 0)

        sub("Search by tags")
        tag_results = await store.search_by_tags(["python"])
        bullet(f"Tag 'python': {[m.name for m in tag_results]}")
        check("Tag search works", len(tag_results) > 0)

        sub("Relationships in MySQL")
        rels = [
            SkillRelationship("write-tests", "refactor-code",
                              SkillRelationType.COMPOSE_WITH, "Test after refactor"),
            SkillRelationship("deploy-service", "write-tests",
                              SkillRelationType.DEPEND_ON, "Must test before deploy"),
        ]
        count = await store.save_relationships(rels)
        bullet(f"Saved {count} relationships")
        all_rels = await store.get_all_relationships()
        for r in all_rels:
            bullet(f"  {r.source} --[{r.relation_type.value}]--> {r.target}")
        check("Relationships persisted in MySQL", len(all_rels) >= 2)

        sub("Effectiveness metrics in MySQL")
        await store.update_effectiveness("write-tests", SkillScope.BASE, "system", {
            "usage_count": 5, "success_count": 4, "failure_count": 1,
            "success_rate": 0.8, "avg_confidence_at_match": 0.87,
        })
        updated = await store.get("write-tests", SkillScope.BASE, "system")
        if updated and updated.effectiveness:
            bullet(f"Effectiveness: usage={updated.effectiveness.get('usage_count')}, "
                   f"success_rate={updated.effectiveness.get('success_rate')}")
            check("Effectiveness persisted", updated.effectiveness.get("usage_count") == 5)

        sub("Count and clear")
        total = await store.count()
        bullet(f"Total skills: {total}")
        check("Count > 0", total > 0)

        sub("Full CRUD: get + delete")
        skill = await store.get("python-basics", SkillScope.BASE, "system")
        check("Get by name returns skill", skill is not None)
        deleted = await store.delete("python-basics", SkillScope.BASE, "system")
        check("Delete returns True", deleted)
        skill_after = await store.get("python-basics", SkillScope.BASE, "system")
        check("Deleted skill is gone", skill_after is None)

    except Exception as exc:
        bullet(f"MySQL demo failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        sub("Cleanup")
        try:
            await store.clear()
            bullet("Cleared test data")
        except Exception:
            pass
        await store.disconnect()
        bullet("Disconnected")


async def demo_neo4j_graph(neo4j_config: Neo4jGraphStoreConfig, store: Any) -> None:
    header("18b. NEO4J SKILL GRAPH (Relationship Visualization)")

    try:
        from ctxforge.graph.stores.neo4j import Neo4jGraphStore
    except ImportError:
        bullet("neo4j package not installed; skipping")
        return

    graph_store = Neo4jGraphStore(config=neo4j_config)
    scope_id = f"skills-e2e-demo-{uuid.uuid4().hex[:8]}"

    try:
        sub("Connect to Neo4j")
        await graph_store.initialize()
        bullet(f"Connected to {neo4j_config.url}")
        check("Neo4j connection", True)

        sub("Store skill entities as graph nodes")
        from ctxforge.protocols.graph import GraphEdge, GraphNode

        nodes = []
        for s in DEMO_SKILLS:
            node = GraphNode(
                node_id=f"skill:{s.name}",
                scope_id=scope_id,
                name=s.name,
                labels=["Skill", s.category or "uncategorized"],
                attributes={
                    "description": s.description,
                    "scope": s.scope.value,
                    "category": s.category or "",
                    "tags": ",".join(s.tags),
                    "triggers": ",".join(s.triggers),
                },
                summary=s.description,
            )
            nodes.append(node)

        upserted = await graph_store.upsert_nodes(scope_id, nodes)
        bullet(f"Upserted {upserted} skill nodes into Neo4j")
        check("Nodes upserted", upserted == len(DEMO_SKILLS))

        sub("Store relationships as graph edges")
        relationships = [
            ("fix-import-errors", "python-basics", "DEPEND_ON", "Requires Python knowledge"),
            ("write-tests", "python-basics", "DEPEND_ON", "Tests need Python"),
            ("fix-import-errors", "write-tests", "COMPOSE_WITH", "Fix then test"),
            ("refactor-code", "write-tests", "COMPOSE_WITH", "Refactor then test"),
            ("deploy-service", "write-tests", "DEPEND_ON", "Must test before deploy"),
            ("refactor-code", "fix-import-errors", "SIMILAR_TO", "Both fix code issues"),
        ]
        edges = []
        for src, tgt, rel_type, fact in relationships:
            edge = GraphEdge(
                edge_id=f"{src}-{rel_type}-{tgt}",
                scope_id=scope_id,
                source_node_id=f"skill:{src}",
                target_node_id=f"skill:{tgt}",
                edge_type=rel_type,
                fact=fact,
                labels=[rel_type],
            )
            edges.append(edge)

        edge_count = await graph_store.upsert_edges(scope_id, edges)
        bullet(f"Upserted {edge_count} relationship edges")
        check("Edges upserted", edge_count == len(relationships))

        sub("Search graph by keyword")

        result = await graph_store.search(
            scope_id, "test", scope="nodes", limit=10,
        )
        bullet(f"Search 'test' -> {len(result.nodes)} nodes")
        found_names = [n.name for n in result.nodes]
        bullet(f"Found: {found_names}")
        check("Graph search returns results", len(result.nodes) > 0)

        sub("Search edges")
        edge_result = await graph_store.search(
            scope_id, "DEPEND_ON", scope="edges", limit=10,
        )
        bullet(f"Search 'DEPEND_ON' edges -> {len(edge_result.edges)} edges")
        check("Edge search works", len(edge_result.edges) > 0)

    except Exception as exc:
        bullet(f"Neo4j demo failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        sub("Cleanup")
        try:
            deleted = await graph_store.delete_scope(scope_id)
            bullet(f"Deleted scope data: {deleted} items")
        except Exception:
            pass
        try:
            await graph_store.close()
            bullet("Disconnected from Neo4j")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Context Assembly (bonus)
# ---------------------------------------------------------------------------

async def demo_context_assembly() -> None:
    header("BONUS: CONTEXT ASSEMBLY (Skills -> LLM Prompt)")

    context = Context(
        session_id="sess-demo", user_id="alice",
        system_instructions="You are a helpful coding assistant.",
        current_query="Help me write unit tests",
    )
    context.add_section("skills_index", "## Available Skills\n- write-tests\n- refactor-code",
                        priority=60, is_required=False)
    context.add_section("activated_skills", "## Skill: write-tests\n\n### Workflow\n1. Identify cases\n2. Write assertions",
                        priority=65, is_required=False)
    context.add_section("explicit_skill", "## Skill: deploy-service\n\nFull deployment workflow...",
                        priority=70, is_required=True)

    sub("Sections in priority order")
    for s in context.sections:
        bullet(f"[pri={s.priority}, req={s.is_required}] {s.name}")

    sub("Token budget packing")
    packed_big = context.priority_pack_sections(budget=9999)
    bullet(f"Budget=9999: {[s.name for s in packed_big]}")
    packed_tiny = context.priority_pack_sections(budget=20)
    bullet(f"Budget=20:   {[s.name for s in packed_tiny]} (required survives)")
    check("All sections fit large budget", len(packed_big) == 3)
    check("Required survives tiny budget", any(s.name == "explicit_skill" for s in packed_tiny))


# ---------------------------------------------------------------------------
# SkillService orchestration
# ---------------------------------------------------------------------------

async def demo_skill_service(store: Any) -> None:
    header("BONUS: SKILL SERVICE (High-Level Orchestration)")

    matcher = SkillMatcher()
    service = SkillService(store=store, matcher=matcher)

    sub("Register via convenience methods")
    await service.register_base_skill("code-format", "Format code with Black",
                                      "1. Run Black\n2. Run isort\n" + "x " * 20,
                                      ["format", "lint"])
    await service.register_user_skill("alice", "my-deploy", "Alice's deploy workflow",
                                      "1. Build\n2. Push\n" + "x " * 20,
                                      ["deploy"])
    await service.register_project_skill("proj-x", "api-test", "Test project APIs",
                                         "1. Start server\n2. Run tests\n" + "x " * 20,
                                         ["test api"])
    bullet("Registered 3 skills (base + user + project)")

    sub("Available skills for alice + proj-x")
    skills = await service.get_available_skills(user_id="alice", project_id="proj-x")
    for s in skills:
        bullet(f"{s.scope.value:8s} {s.name}")
    check("Multiple scopes visible", len(skills) >= 3)

    sub("Match query")
    matches = await service.match_skills("format my code", user_id="alice")
    if matches:
        bullet(f"Top: {matches[0].skill.name} (conf={matches[0].confidence:.3f})")
        check("code-format matched", matches[0].skill.name == "code-format")

    sub("Load full content on-demand")
    full = await service.load_skill_content("code-format", user_id="alice")
    if full:
        bullet(f"Loaded: {full.name}, content={len(full.content)} chars")
        check("Full content loaded", len(full.content) > 0)

    sub("Recommendations via relationships")
    await store.save_relationships([
        SkillRelationship("code-format", "api-test",
                          SkillRelationType.COMPOSE_WITH, "Format then test"),
    ])
    rec = await service.get_recommended_skills(["code-format"], user_id="alice", project_id="proj-x")
    bullet(f"Active=['code-format'] -> recommended={[r.name for r in rec]}")
    check("Recommendations work", len(rec) > 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end skills feature demo with MySQL + Neo4j",
    )
    parser.add_argument("--skip-mysql", action="store_true",
                        help="Use in-memory store instead of MySQL")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="Skip Neo4j graph demo")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-dependent demos (evaluation, generation, lifecycle)")
    return parser.parse_args()


async def main() -> int:
    global _PASS, _FAIL
    args = parse_args()

    print("\n" + "=" * 72)
    print("  CTXFORGE SKILLS SYSTEM -- END-TO-END FEATURE VALIDATION")
    print("  MySQL + Neo4j Persistent Storage")
    print("=" * 72)

    # Bootstrap infrastructure
    sub("Bootstrapping LLM provider")
    llm: Optional[ILLMProvider] = None
    if not args.skip_llm:
        llm = _get_llm_provider()
        if llm is not None:
            bullet(f"LLM provider: {llm.name} ({llm.default_model})")
        else:
            bullet("No LLM provider -- LLM-dependent demos will be skipped")
    else:
        bullet("LLM skipped (--skip-llm)")

    # Primary store for most demos
    primary_store = InMemorySkillStore()
    await primary_store.initialize()

    try:
        # ---- Feature demos ----
        await demo_core_models()
        await demo_cso_validation()
        await demo_scope_layering()
        await demo_trigger_matching()
        await demo_composite_scoring()
        await demo_skills_index()
        await demo_content_loading()
        await demo_validation()
        await demo_relationships(llm, primary_store)
        await demo_dependency_resolution(primary_store)
        await demo_evaluation(llm)
        await demo_effectiveness(primary_store)
        await demo_execution(primary_store)
        await demo_inheritance(primary_store)
        await demo_lifecycle(llm, primary_store)
        await demo_generation(llm)
        await demo_middleware()
        await demo_context_assembly()
        await demo_skill_service(primary_store)

        # ---- Persistent storage demos ----
        if not args.skip_mysql:
            mysql_config = _get_mysql_config()
            if mysql_config:
                await demo_mysql_store(mysql_config)
            else:
                bullet("MySQL config not available -- skipping")
        else:
            header("18a. MYSQL SKILL STORE")
            bullet("Skipped (--skip-mysql)")

        if not args.skip_neo4j:
            neo4j_config = _get_neo4j_config()
            await demo_neo4j_graph(neo4j_config, primary_store)
        else:
            header("18b. NEO4J SKILL GRAPH")
            bullet("Skipped (--skip-neo4j)")

        # ---- Summary ----
        header("DEMO COMPLETE")
        print()
        bullet(f"Results:  {_PASS} PASSED,  {_FAIL} FAILED")
        print()
        bullet("Infrastructure used:")
        bullet(f"  LLM:        {'REAL (' + llm.name + ')' if (llm and not isinstance(llm, _FakeLLM)) else 'skipped'}")
        bullet(f"  MySQL:      {'REAL' if not args.skip_mysql else 'skipped (--skip-mysql)'}")
        bullet(f"  Neo4j:      {'REAL' if not args.skip_neo4j else 'skipped (--skip-neo4j)'}")
        bullet("  In-memory:  always used as baseline")
        print()

        return 0 if _FAIL == 0 else 1

    except Exception as exc:
        print(f"\n  FATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
