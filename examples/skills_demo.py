#!/usr/bin/env python3
"""
End-to-End Demo: All Skill Features (Real LLM + Real Databases)

Exercises every feature of the ctxforge skill subsystem using real
infrastructure configured via examples/.env:

  - Azure OpenAI (or OpenAI) for LLM chat + embeddings
  - PostgreSQL for persistent skill storage
  - MySQL for persistent skill storage
  - In-memory as the always-available fallback

Features covered:
  1.  Core models          -- Skill, SkillMetadata, SkillContent, SkillScope
  2.  Scope layering       -- BASE < USER < PROJECT override semantics
  3.  Trigger matching     -- substring, word-overlap, regex
  4.  Composite scoring    -- weighted multi-signal ranking via SkillMatcher
  5.  Skills index         -- progressive-disclosure prompt injection
  6.  Content loading      -- SKILL.md parsing, directory layout, frontmatter
  7.  Skill relationships  -- LLM-inferred + manual, all 4 types
  8.  Dependency resolution -- topological sort of DEPEND_ON chains
  9.  Skill evaluation     -- real LLM 5-dimension quality scoring
 10.  Effectiveness tracking -- usage counts, success rate, ranking boost
 11.  Skill generation     -- real two-phase LLM extraction from session events
 12.  Middleware integration -- SkillsMiddleware, SkillRequestMiddleware,
                               SkillEffectivenessMiddleware
 13.  Context assembly     -- how skills land in the final LLM prompt
 14.  PostgreSQL skill store -- full CRUD + relationships + effectiveness
 15.  MySQL skill store    -- full CRUD + relationships + effectiveness

Usage:
    source ctxforge/venv/bin/activate
    python examples/skills_demo.py
"""

import asyncio
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

from ctxforge.config.base import EngineConfig, SkillGenerationConfig
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
)
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.services.skill_content_loader import SkillContentLoader
from ctxforge.engine.services.skill_effectiveness_service import (
    SkillEffectivenessService,
)
from ctxforge.engine.services.skill_evaluation_service import SkillEvaluationService
from ctxforge.engine.services.skill_generator_service import SkillGeneratorService
from ctxforge.engine.services.skill_matcher import (
    DEFAULT_WEIGHTS,
    RegexSkillMatcher,
    SkillMatcher,
)
from ctxforge.engine.services.skill_relationship_service import (
    SkillRelationshipService,
)
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.skill_effectiveness import SkillEffectivenessMiddleware
from ctxforge.middleware.skills import SkillRequestMiddleware, SkillsMiddleware
from ctxforge.protocols.llm import ILLMProvider
from ctxforge.storage.connection import MySQLConfig, PostgresConfig
from ctxforge.storage.memory.skill import InMemorySkillStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def sub(title: str) -> None:
    print(f"\n  --- {title} ---")


def bullet(text: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{text}")


# ---------------------------------------------------------------------------
# Infrastructure bootstrap
# ---------------------------------------------------------------------------

def _get_llm_provider() -> Optional[ILLMProvider]:
    """Create a real LLM provider from env config (cached via EngineFactory)."""
    try:
        factory = EngineFactory()
        cfg = _build_engine_config()
        provider = factory._create_llm_provider(cfg)
        return provider
    except Exception as exc:
        print(f"    [warn] Could not create LLM provider: {exc}")
        return None


def _build_engine_config() -> EngineConfig:
    """Build an EngineConfig with env overrides (same logic as examples/config.py)."""
    from ctxforge.config.loader import ConfigLoader
    loader = ConfigLoader()
    config_path = Path(__file__).parent / "engine_config.yaml"
    if config_path.exists():
        cfg = loader.load_from_file(str(config_path))
    else:
        cfg = EngineConfig()
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


def _get_postgres_config() -> Optional[PostgresConfig]:
    host = os.getenv("POSTGRES_HOST")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
    if not all([host, user, password, database]):
        return None
    return PostgresConfig(
        host=host, user=user, password=password, database=database,
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )


def _get_mysql_config() -> Optional[MySQLConfig]:
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    if not all([host, user, password, database]):
        return None
    return MySQLConfig(
        host=host, user=user, password=password, database=database,
        port=int(os.getenv("MYSQL_PORT", "3306")),
    )


# ---------------------------------------------------------------------------
# Test skill corpus (reused across demos)
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
            scripts={"run_tests.sh": "#!/bin/bash\npytest --cov -x"},
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
        content="# Python Basics\n\nPEP 8, type hints, virtual environments, pip.",
        triggers=["python basics"],
        category="coding", tags=["python"],
    ),
]


# ---------------------------------------------------------------------------
# 1. Core Models
# ---------------------------------------------------------------------------

async def demo_core_models() -> None:
    header("1. CORE MODELS")

    sub("SkillScope priorities")
    for scope in SkillScope:
        bullet(f"{scope.value:10s}  priority={SkillScope.priority(scope)}")

    skill = DEMO_SKILLS[0]
    sub("Skill with structured content")
    bullet(f"Name:          {skill.name}")
    bullet(f"Category:      {skill.category}")
    bullet(f"Tags:          {skill.tags}")
    bullet(f"Triggers:      {skill.triggers}")
    bullet(f"Has scripts:   {bool(skill.structured_content and skill.structured_content.scripts)}")

    sub("SkillMetadata (lightweight)")
    meta = skill.skill_metadata
    bullet(f"Name:        {meta.name}")
    bullet(f"Description: {meta.description}")
    bullet(f"Triggers:    {meta.triggers}")

    sub("SkillRelationship")
    rel = SkillRelationship(
        source="fix-import-errors", target="python-basics",
        relation_type=SkillRelationType.DEPEND_ON,
        reason="Requires Python knowledge", confidence=0.95,
    )
    bullet(f"{rel.source} --[{rel.relation_type.value}]--> {rel.target}")

    sub("SkillEvaluation scoring")
    score = SkillEvaluation.compute_overall_score(
        EvaluationLevel.GOOD, EvaluationLevel.GOOD,
        EvaluationLevel.AVERAGE, EvaluationLevel.GOOD, EvaluationLevel.GOOD,
    )
    bullet(f"GOOD/GOOD/AVG/GOOD/GOOD -> overall = {score}")


# ---------------------------------------------------------------------------
# 2. Scope Layering
# ---------------------------------------------------------------------------

async def demo_scope_layering() -> None:
    header("2. SCOPE LAYERING (BASE < USER < PROJECT)")

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
            content=f"# {desc}", triggers=["deploy"],
        ))
        bullet(f"Saved {scope.value:8s} deploy: {desc}")

    for label, uid, pid in [
        ("alice + proj-x", "alice", "proj-x"),
        ("alice only", "alice", None),
        ("anonymous", None, None),
    ]:
        meta = await store.list_all_metadata(user_id=uid, project_id=pid)
        d = next(m for m in meta if m.name == "deploy")
        sub(f"View for {label}")
        bullet(f"Winner: {d.scope.value} -> '{d.description}'")


# ---------------------------------------------------------------------------
# 3. Trigger Matching
# ---------------------------------------------------------------------------

async def demo_trigger_matching() -> None:
    header("3. TRIGGER MATCHING")

    meta = DEMO_SKILLS[3].skill_metadata
    sub("Substring matching")
    for q in ["How do I fix imports?", "import error on line 5", "What's the weather?"]:
        m = meta.matches_trigger(q)
        bullet(f"'{q}' -> {'MATCH: ' + m if m else 'no match'}")

    sub("RegexSkillMatcher")
    regex = RegexSkillMatcher()
    rmeta = SkillMetadata(
        name="fix-type-errors", description="Fix TypeErrors",
        scope=SkillScope.BASE, scope_id="system",
        triggers=[r"TypeError:\s+\w+", r"type\s+error"],
    )
    for q in ["TypeError: unsupported operand", "I have a type error", "sort a list"]:
        results = await regex.match(q, [rmeta], threshold=0.0)
        if results and results[0].confidence > 0:
            bullet(f"'{q}' -> MATCH (conf={results[0].confidence:.2f})")
        else:
            bullet(f"'{q}' -> no match")


# ---------------------------------------------------------------------------
# 4. Composite Scoring
# ---------------------------------------------------------------------------

async def demo_composite_scoring() -> None:
    header("4. COMPOSITE SCORING (SkillMatcher)")

    sub("Default weights")
    for signal, w in DEFAULT_WEIGHTS.items():
        bullet(f"{signal:15s} = {w:.2f}")

    matcher = SkillMatcher()
    metas = [s.skill_metadata for s in DEMO_SKILLS[:4]]

    sub("Matching queries")
    for q in [
        "Help me write unit tests for auth",
        "Refactor this messy function",
        "Deploy the payment service",
        "What's the meaning of life?",
    ]:
        results = await matcher.match(q, metas, threshold=0.0)
        if results:
            top = results[0]
            bullet(f"'{q[:50]}' -> {top.skill.name} (conf={top.confidence:.3f})")
        else:
            bullet(f"'{q[:50]}' -> no match")


# ---------------------------------------------------------------------------
# 5. Skills Index
# ---------------------------------------------------------------------------

async def demo_skills_index() -> None:
    header("5. SKILLS INDEX (PROGRESSIVE DISCLOSURE)")

    metas = [s.skill_metadata for s in DEMO_SKILLS[:4]]
    index = SkillsIndex(skills=metas)

    sub("Summary")
    bullet(f"Total: {index.total_count}")
    bullet(f"Compact: {index.format_compact()}")

    sub("Prompt format")
    for line in index.format_for_prompt().split("\n"):
        bullet(line, indent=6)


# ---------------------------------------------------------------------------
# 6. Content Loading
# ---------------------------------------------------------------------------

async def demo_content_loading() -> None:
    header("6. CONTENT LOADING (SkillContentLoader)")

    raw_md = textwrap.dedent("""\
        ---
        name: fix-import-errors
        category: debugging
        ---
        # Fix Import Errors

        1. Read the error traceback
        2. Identify the failing module
        3. Fix the cycle or add the dependency
    """)

    sub("Parse SKILL.md with frontmatter")
    content = SkillContentLoader.parse_skill_markdown(raw_md)
    bullet(f"Instructions: {content.instructions[:60]}...")
    fm = SkillContentLoader.parse_frontmatter(raw_md)
    bullet(f"Frontmatter: {fm}")

    sub("Load from directory")
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

    sub("Progressive disclosure levels")
    bullet(f"Level 1 (instructions): {len(SkillContentLoader.format_for_prompt(loaded))} chars")
    bullet(f"Level 2 (+scripts):     {len(SkillContentLoader.format_for_prompt(loaded, include_scripts=True))} chars")
    bullet(f"Level 3 (+refs):        {len(SkillContentLoader.format_for_prompt(loaded, include_scripts=True, include_references=True))} chars")


# ---------------------------------------------------------------------------
# 7. Skill Relationships (with real LLM when available)
# ---------------------------------------------------------------------------

async def demo_relationships(llm: Optional[ILLMProvider]) -> None:
    header("7. SKILL RELATIONSHIPS")

    store = InMemorySkillStore()
    await store.initialize()
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
    ]
    saved = await store.save_relationships(manual_rels)
    bullet(f"Saved {saved} manual relationships")

    sub("Full graph")
    for r in await store.get_all_relationships():
        bullet(f"{r.source} --[{r.relation_type.value}]--> {r.target}")

    rel_service = SkillRelationshipService(
        llm_provider=llm or _FakeLLM(), skill_store=store,
    )

    sub("Dependency chain for deploy-service")
    chain = await rel_service.resolve_dependency_chain("deploy-service")
    bullet(f"Chain: {' -> '.join(chain)}")

    sub("Composable skills for fix-import-errors")
    bullet(f"Composes with: {await rel_service.find_composable_skills('fix-import-errors')}")

    sub("Alternatives for refactor-code")
    bullet(f"Similar to: {await rel_service.find_alternatives('refactor-code')}")

    if llm is not None:
        sub("LLM-inferred relationships (real LLM)")
        try:
            metas = [s.skill_metadata for s in DEMO_SKILLS[:4]]
            inferred = await rel_service.analyze_relationships(metas)
            bullet(f"LLM inferred {len(inferred)} relationships:")
            for r in inferred:
                bullet(f"  {r.source} --[{r.relation_type.value}]--> {r.target} "
                       f"(conf={r.confidence:.2f}, reason='{r.reason[:60]}')")
        except Exception as exc:
            bullet(f"LLM relationship analysis failed: {exc}")
    else:
        sub("LLM-inferred relationships (skipped -- no LLM provider)")


# ---------------------------------------------------------------------------
# 8. Skill Evaluation (real LLM)
# ---------------------------------------------------------------------------

async def demo_evaluation(llm: Optional[ILLMProvider]) -> None:
    header("8. SKILL EVALUATION (5-Dimension Quality Scoring)")

    sub("Dimension weights")
    for dim, w in [("Safety", 25), ("Completeness", 25), ("Executability", 20),
                   ("Maintainability", 15), ("Cost-awareness", 15)]:
        bullet(f"{dim}: {w}%")

    if llm is not None:
        sub("Real LLM evaluation of 'write-tests' skill")
        eval_service = SkillEvaluationService(llm)
        try:
            result = await eval_service.evaluate(DEMO_SKILLS[0])
            bullet(f"Overall score: {result.overall_score}")
            for dim in ["safety", "completeness", "executability", "maintainability", "cost_awareness"]:
                level = getattr(result, dim)
                reason = getattr(result, f"{dim}_reason")
                bullet(f"  {dim:20s} = {level.value:8s}  {reason[:60]}")
        except Exception as exc:
            bullet(f"Evaluation failed: {exc}")
    else:
        sub("Real LLM evaluation (skipped -- no LLM provider)")

    sub("Score computation examples")
    for label, levels in [
        ("All GOOD", [EvaluationLevel.GOOD] * 5),
        ("All AVERAGE", [EvaluationLevel.AVERAGE] * 5),
        ("Mixed", [EvaluationLevel.GOOD, EvaluationLevel.GOOD,
                   EvaluationLevel.AVERAGE, EvaluationLevel.GOOD, EvaluationLevel.AVERAGE]),
    ]:
        score = SkillEvaluation.compute_overall_score(*levels)
        bullet(f"{label:15s} -> {score:.4f}")


# ---------------------------------------------------------------------------
# 9. Effectiveness Tracking
# ---------------------------------------------------------------------------

async def demo_effectiveness() -> None:
    header("9. EFFECTIVENESS TRACKING")

    store = InMemorySkillStore()
    await store.initialize()
    await store.save(DEMO_SKILLS[0])

    eff = SkillEffectivenessService(store)

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
              "avg_confidence_at_match", "sessions_used_in"]:
        bullet(f"{k}: {metrics[k]}")

    boost = await eff.get_ranking_boost("write-tests", SkillScope.BASE, "system")
    bullet(f"Ranking boost: {boost:.4f}")


# ---------------------------------------------------------------------------
# 10. Skill Generation (real LLM)
# ---------------------------------------------------------------------------

async def demo_generation(llm: Optional[ILLMProvider]) -> None:
    header("10. SKILL GENERATION (Two-Phase LLM)")

    if llm is None:
        bullet("Skipped -- no LLM provider available")
        return

    store = InMemorySkillStore()
    await store.initialize()
    service = SkillService(store=store)
    config = SkillGenerationConfig(enabled=True, min_session_events=3)
    generator = SkillGeneratorService(llm, service, config)

    sub("Generate from session events (two-phase)")
    events = [
        Event(type=EventType.USER, content="These tests keep failing randomly in CI"),
        Event(type=EventType.AGENT, content="Let me investigate the flaky tests"),
        Event(type=EventType.USER, content="It's the auth test, it fails 1 in 5 runs"),
        Event(type=EventType.AGENT, content="I found a race condition in the test setup fixture"),
        Event(type=EventType.USER, content="Great, can you fix it?"),
        Event(type=EventType.AGENT, content="Done. I added proper async locks and a retry decorator."),
    ]
    try:
        generated = await generator.generate_from_session(events, project_id="demo-proj")
        bullet(f"Generated {len(generated)} skill(s) from session:")
        for sk in generated:
            bullet(f"  name={sk.name}, category={sk.category}")
            bullet(f"  description: {sk.description}")
            bullet(f"  content: {sk.content[:80]}...")
    except Exception as exc:
        bullet(f"Session generation failed: {exc}")

    sub("Generate from prompt (single-phase)")
    try:
        prompt_skill = await generator.generate_from_prompt(
            "Create a skill for setting up Python virtual environments with pip and requirements.txt",
            project_id="demo-proj",
        )
        if prompt_skill:
            bullet(f"Generated: {prompt_skill.name}")
            bullet(f"  triggers: {prompt_skill.triggers}")
            bullet(f"  content: {prompt_skill.content[:80]}...")
        else:
            bullet("LLM returned invalid skill (name validation failed)")
    except Exception as exc:
        bullet(f"Prompt generation failed: {exc}")


# ---------------------------------------------------------------------------
# 11. Middleware Integration
# ---------------------------------------------------------------------------

async def demo_middleware() -> None:
    header("11. MIDDLEWARE INTEGRATION")

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
    for q in ["Help me write unit tests", "What's 2+2?", "Refactor this code"]:
        ctx = MiddlewareContext(user_input=q, user_id="alice", session_id="s1")
        result = await mw.process(ctx, noop)
        activated = result.modifications.get("activated_skills", [])
        sections = [s.name for s in result.context_sections]
        names = [a["name"] for a in activated] if activated else []
        bullet(f"'{q}' -> activated={names}, sections={sections}")

    sub("SkillRequestMiddleware -- explicit requests")
    req_mw = SkillRequestMiddleware(skill_service=service, user_id="alice")
    for q in ["use skill deploy-service", "@skill write-tests", "normal question"]:
        ctx = MiddlewareContext(user_input=q, user_id="alice", session_id="s1")
        result = await req_mw.process(ctx, noop)
        loaded = result.modifications.get("requested_skill")
        bullet(f"'{q}' -> {'loaded: ' + loaded['name'] if loaded else 'no skill requested'}")

    sub("SkillEffectivenessMiddleware -- usage tracking")
    eff_service = SkillEffectivenessService(store)
    eff_mw = SkillEffectivenessMiddleware(eff_service, SkillScope.BASE, "system")
    ctx = MiddlewareContext(user_input="test", user_id="alice", session_id="s1")
    ctx.add_flag("skills_auto_activated")
    ctx.modifications["skills"] = [{"activated_skills": ["write-tests"]}]
    await eff_mw.process(ctx, noop)
    m = await eff_service.get_effectiveness("write-tests", SkillScope.BASE, "system")
    bullet(f"After activation: usage_count={m['usage_count']}")


# ---------------------------------------------------------------------------
# 12. Context Assembly
# ---------------------------------------------------------------------------

async def demo_context_assembly() -> None:
    header("12. CONTEXT ASSEMBLY (Skills -> LLM Prompt)")

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

    sub("to_messages() preview")
    for msg in context.to_messages():
        bullet(f"{msg['role']:10s}: {msg['content'][:70].replace(chr(10), ' ')}...")


# ---------------------------------------------------------------------------
# 13 & 14. PostgreSQL and MySQL Skill Stores
# ---------------------------------------------------------------------------

async def _demo_db_store(store: Any, label: str) -> None:
    """Run full CRUD + relationships + effectiveness on a DB-backed store."""
    sub(f"{label}: Initialize")
    await store.initialize()
    bullet("Initialized")

    sub(f"{label}: Save skills")
    for s in DEMO_SKILLS[:3]:
        await store.save(s)
        bullet(f"Saved: {s.name} ({s.scope.value})")

    sub(f"{label}: Scope layering")
    await store.save(Skill(
        name="write-tests", description="Project-specific test workflow",
        scope=SkillScope.PROJECT, scope_id="proj-demo",
        content="# Project Tests\n1. Use project fixtures\n2. Run integration tests",
        triggers=["test"], category="testing",
    ))
    all_meta = await store.list_all_metadata(user_id=None, project_id="proj-demo")
    wt = next((m for m in all_meta if m.name == "write-tests"), None)
    if wt:
        bullet(f"write-tests winner: scope={wt.scope.value}, desc='{wt.description}'")

    sub(f"{label}: Search by trigger")
    matches = await store.search_by_trigger("deploy to production")
    bullet(f"'deploy to production' -> {len(matches)} matches")
    for m in matches[:3]:
        bullet(f"  {m.skill.name} (conf={m.confidence:.2f})")

    sub(f"{label}: Search by category")
    cat_results = await store.search_by_category("testing")
    bullet(f"Category 'testing': {[m.name for m in cat_results]}")

    sub(f"{label}: Search by tags")
    tag_results = await store.search_by_tags(["python"])
    bullet(f"Tag 'python': {[m.name for m in tag_results]}")

    sub(f"{label}: Relationships")
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

    sub(f"{label}: Effectiveness")
    skill = await store.get("write-tests", SkillScope.BASE, "system")
    if skill:
        await store.update_effectiveness("write-tests", SkillScope.BASE, "system", {
            "usage_count": 5, "success_count": 4, "failure_count": 1,
            "success_rate": 0.8, "avg_confidence_at_match": 0.87,
        })
        updated = await store.get("write-tests", SkillScope.BASE, "system")
        if updated and updated.effectiveness:
            bullet(f"Effectiveness: usage={updated.effectiveness.get('usage_count')}, "
                   f"success_rate={updated.effectiveness.get('success_rate')}")

    sub(f"{label}: Cleanup")
    for s in DEMO_SKILLS[:3]:
        await store.delete(s.name, s.scope, s.scope_id)
    await store.delete("write-tests", SkillScope.PROJECT, "proj-demo")
    bullet("Cleaned up test data")
    await store.disconnect()
    bullet("Disconnected")


async def demo_postgres_store() -> None:
    header("13. POSTGRESQL SKILL STORE")
    pg_config = _get_postgres_config()
    if pg_config is None:
        bullet("Skipped -- POSTGRES_HOST/USER/PASSWORD/DB not set in .env")
        return
    try:
        from ctxforge.storage.postgres.skill import PostgresSkillStore
        store = PostgresSkillStore(config=pg_config, table_name="skills_demo")
        await _demo_db_store(store, "PostgreSQL")
    except Exception as exc:
        bullet(f"PostgreSQL demo failed: {exc}")


async def demo_mysql_store() -> None:
    header("14. MYSQL SKILL STORE")
    mysql_config = _get_mysql_config()
    if mysql_config is None:
        bullet("Skipped -- MYSQL_HOST/USER/PASSWORD/DATABASE not set in .env")
        return
    try:
        from ctxforge.storage.mysql.skill import MySQLSkillStore
        store = MySQLSkillStore(config=mysql_config, table_name="skills_demo")
        await _demo_db_store(store, "MySQL")
    except Exception as exc:
        bullet(f"MySQL demo failed: {exc}")


# ---------------------------------------------------------------------------
# 15. SkillService orchestration
# ---------------------------------------------------------------------------

async def demo_skill_service() -> None:
    header("15. SKILL SERVICE (High-Level Orchestration)")

    store = InMemorySkillStore()
    await store.initialize()
    matcher = SkillMatcher()
    service = SkillService(store=store, matcher=matcher)

    sub("Register via convenience methods")
    await service.register_base_skill("code-format", "Format code with Black",
                                      "1. Run Black\n2. Run isort", ["format", "lint"])
    await service.register_user_skill("alice", "my-deploy", "Alice's deploy",
                                      "1. Build\n2. Push", ["deploy"])
    await service.register_project_skill("proj-x", "api-test", "Test project APIs",
                                         "1. Start server\n2. Run tests", ["test api"])
    bullet("Registered 3 skills (base + user + project)")

    sub("Available skills for alice + proj-x")
    skills = await service.get_available_skills(user_id="alice", project_id="proj-x")
    for s in skills:
        bullet(f"{s.scope.value:8s} {s.name}")

    sub("Match query")
    matches = await service.match_skills("format my code", user_id="alice")
    if matches:
        bullet(f"Top: {matches[0].skill.name} (conf={matches[0].confidence:.3f})")

    sub("Load full content on-demand")
    full = await service.load_skill_content("code-format", user_id="alice")
    if full:
        bullet(f"Loaded: {full.name}, content={len(full.content)} chars")

    sub("Recommendations via relationships")
    await store.save_relationships([
        SkillRelationship("code-format", "api-test",
                          SkillRelationType.COMPOSE_WITH, "Format then test"),
    ])
    rec = await service.get_recommended_skills(["code-format"], user_id="alice", project_id="proj-x")
    bullet(f"Active=['code-format'] -> recommended={[r.name for r in rec]}")


# ---------------------------------------------------------------------------
# Fake LLM fallback (only used when no real provider is available)
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Minimal fake so relationship service can be instantiated."""
    async def chat(self, **kw: Any) -> Any:
        from dataclasses import dataclass
        @dataclass
        class _R:
            content: str = "[]"
        return _R()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    print("\n" + "=" * 72)
    print("  CTXFORGE SKILLS SYSTEM -- COMPREHENSIVE END-TO-END DEMO")
    print("=" * 72)

    sub("Bootstrapping LLM provider from .env")
    llm = _get_llm_provider()
    if llm is not None:
        bullet(f"LLM provider: {llm.name} ({llm.default_model})")
    else:
        bullet("No LLM provider -- LLM-dependent demos will use fakes or be skipped")

    try:
        await demo_core_models()
        await demo_scope_layering()
        await demo_trigger_matching()
        await demo_composite_scoring()
        await demo_skills_index()
        await demo_content_loading()
        await demo_relationships(llm)
        await demo_evaluation(llm)
        await demo_effectiveness()
        await demo_generation(llm)
        await demo_middleware()
        await demo_context_assembly()
        await demo_postgres_store()
        await demo_mysql_store()
        await demo_skill_service()

        header("DEMO COMPLETE")
        print()
        bullet("All skill features demonstrated end-to-end.")
        print()
        bullet("Infrastructure used:")
        bullet(f"  LLM:        {'REAL (' + llm.name + ')' if llm else 'fake/skipped'}")
        bullet(f"  PostgreSQL: {'REAL' if _get_postgres_config() else 'skipped (not configured)'}")
        bullet(f"  MySQL:      {'REAL' if _get_mysql_config() else 'skipped (not configured)'}")
        bullet("  In-memory:  always used as baseline")
        print()
        return 0

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
