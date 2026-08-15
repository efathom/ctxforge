#!/usr/bin/env python3
"""
End-to-End Demo: Hierarchical Memory & Skills System

This script demonstrates all implemented features of the hierarchical
memory and skills system in ctxforge, including:

1. Scoped Memory:
   - Global, project, and session scopes
   - Memory categories (preference, convention, architecture, etc.)
   - Hierarchical override (session > project > global)
   - Merged memory retrieval with conflict resolution
   - Prompt formatting

2. Skills:
   - Skill registration at different scopes (base, user, project)
   - Scope layering (project > user > base)
   - Trigger-based matching
   - Skills index for progressive disclosure
   - Full skill content loading

3. Middleware:
   - ScopedMemoryMiddleware for prompt injection
   - ScopedMemoryAutoLearnMiddleware for preference extraction
   - SkillsMiddleware for skill index injection and auto-activation
   - SkillRequestMiddleware for explicit skill requests

4. Integration with CtxForge Engine:
   - Using engine API for scoped memories and skills
   - Factory-based service creation

Usage:
    python examples/hierarchical_memory_skills_demo.py
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Add ctxforge to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ctxforge.core.scoped_memory import (
    MemoryCategory,
    MemoryScope,
    ScopedMemory,
    ScopedMemoryQuery,
)
from ctxforge.core.skill import (
    Skill,
    SkillMetadata,
    SkillScope,
    SkillsIndex,
)
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.engine.services.skill_matcher import SkillMatcher
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.scoped_memory import (
    ScopedMemoryAutoLearnMiddleware,
    ScopedMemoryMiddleware,
)
from ctxforge.middleware.skills import SkillRequestMiddleware, SkillsMiddleware
from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore
from ctxforge.storage.memory.skill import InMemorySkillStore


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


async def demo_scoped_memory_models():
    """Demonstrate core scoped memory models."""
    print_header("1. SCOPED MEMORY MODELS")

    # Memory Scopes
    print_subheader("Memory Scopes")
    for scope in MemoryScope:
        priority = MemoryScope.priority(scope)
        print(f"  {scope.value}: priority={priority}")

    # Memory Categories
    print_subheader("Memory Categories")
    for category in MemoryCategory:
        display = MemoryCategory.get_display_name(category)
        print(f"  {category.value}: {display}")

    # Creating a scoped memory
    print_subheader("Creating Scoped Memories")
    memory = ScopedMemory(
        id=str(uuid.uuid4()),
        scope=MemoryScope.GLOBAL,
        scope_id="user-123",
        category=MemoryCategory.PREFERENCE,
        key="editor-preference",
        content="Always use VSCode for Python development",
        priority=10,
        metadata={"source": "user_stated"},
    )
    print(f"  Created: {memory.key}")
    print(f"    Scope: {memory.scope.value}")
    print(f"    Category: {memory.category.value}")
    print(f"    Priority: {memory.priority}")
    print(f"    Content: {memory.content}")

    # Serialization
    print_subheader("Serialization")
    memory_dict = memory.to_dict()
    restored = ScopedMemory.from_dict(memory_dict)
    print(f"  Serialized and restored: {restored.key} ✓")


async def demo_scoped_memory_store():
    """Demonstrate the in-memory scoped memory store."""
    print_header("2. SCOPED MEMORY STORE")

    store = InMemoryScopedMemoryStore()
    await store.initialize()
    print("  Store initialized ✓")

    # Save memories at different scopes
    print_subheader("Saving Memories at Different Scopes")

    global_mem = ScopedMemory(
        id=str(uuid.uuid4()),
        scope=MemoryScope.GLOBAL,
        scope_id="user-123",
        category=MemoryCategory.PREFERENCE,
        key="language",
        content="Use Python 3.11+",
    )
    await store.save(global_mem)
    print(f"  Saved GLOBAL: {global_mem.key} = '{global_mem.content}'")

    project_mem = ScopedMemory(
        id=str(uuid.uuid4()),
        scope=MemoryScope.PROJECT,
        scope_id="proj-abc",
        category=MemoryCategory.CONVENTION,
        key="testing",
        content="Use pytest with async fixtures",
    )
    await store.save(project_mem)
    print(f"  Saved PROJECT: {project_mem.key} = '{project_mem.content}'")

    session_mem = ScopedMemory(
        id=str(uuid.uuid4()),
        scope=MemoryScope.SESSION,
        scope_id="sess-xyz",
        category=MemoryCategory.CONTEXT,
        key="current-task",
        content="Implementing user authentication",
    )
    await store.save(session_mem)
    print(f"  Saved SESSION: {session_mem.key} = '{session_mem.content}'")

    # Retrieve by scope
    print_subheader("Retrieving by Scope")
    global_memories = await store.list_by_scope(MemoryScope.GLOBAL, "user-123")
    print(f"  GLOBAL memories for user-123: {len(global_memories)}")

    # Query across scopes (store returns a list)
    print_subheader("Query Across Scopes")
    query = ScopedMemoryQuery(
        user_id="user-123",
        project_id="proj-abc",
        session_id="sess-xyz",
    )
    memories = await store.query(query)
    print(f"  Total memories found: {len(memories)}")
    for mem in memories:
        print(f"    - [{mem.scope.value}] {mem.key}: {mem.content[:50]}...")

    # Demonstrate override
    print_subheader("Demonstrating Hierarchical Override")
    await store.save(ScopedMemory(
        id=str(uuid.uuid4()),
        scope=MemoryScope.GLOBAL,
        scope_id="user-123",
        category=MemoryCategory.PREFERENCE,
        key="database",
        content="Use PostgreSQL",
    ))
    print("  Saved GLOBAL: database = 'Use PostgreSQL'")

    await store.save(ScopedMemory(
        id=str(uuid.uuid4()),
        scope=MemoryScope.PROJECT,
        scope_id="proj-abc",
        category=MemoryCategory.PREFERENCE,
        key="database",
        content="Use SQLite for this project",
    ))
    print("  Saved PROJECT: database = 'Use SQLite for this project'")

    memories = await store.query(query)
    db_memory = next((m for m in memories if m.key == "database"), None)
    if db_memory:
        print(f"  After merge: database = '{db_memory.content}' (from {db_memory.scope.value})")
        print("  ✓ Project scope correctly overrides global!")

    # Count and clear
    print_subheader("Store Statistics")
    total = await store.count()
    print(f"  Total memories in store: {total}")


async def demo_scoped_memory_service():
    """Demonstrate the scoped memory service."""
    print_header("3. SCOPED MEMORY SERVICE")

    store = InMemoryScopedMemoryStore()
    await store.initialize()
    service = ScopedMemoryService(store=store)
    print("  Service initialized ✓")

    # Save memories using service convenience methods
    print_subheader("Using Service Convenience Methods")

    await service.save_global(
        user_id="user-123",
        key="code-style",
        content="Use Black formatter with 88 char line length",
        category=MemoryCategory.CONVENTION,
    )
    print("  Saved global preference via service")

    await service.save_project(
        project_id="myapp",
        key="architecture",
        content="Use hexagonal architecture with ports and adapters",
        category=MemoryCategory.ARCHITECTURE,
    )
    print("  Saved project architecture via service")

    await service.save_session(
        session_id="sess-001",
        key="focus",
        content="Currently working on API endpoints",
        category=MemoryCategory.CONTEXT,
    )
    print("  Saved session context via service")

    # Get merged memories
    print_subheader("Getting Merged Memories")
    merged = await service.get_merged_memories(
        user_id="user-123",
        project_id="myapp",
        session_id="sess-001",
    )
    active_scopes = len([s for s, c in merged.scope_counts.items() if c > 0])
    print(f"  Merged {merged.total_count} memories from {active_scopes} scopes")

    # Format for prompt
    print_subheader("Format for Prompt Injection")
    prompt_text = merged.format_for_prompt()
    print("  Generated prompt section:")
    for line in prompt_text.split("\n")[:10]:
        print(f"    {line}")
    if prompt_text.count("\n") > 10:
        print("    ...")


async def demo_skill_models():
    """Demonstrate core skill models."""
    print_header("4. SKILL MODELS")

    # Skill Scopes
    print_subheader("Skill Scopes")
    for scope in SkillScope:
        priority = SkillScope.priority(scope)
        print(f"  {scope.value}: priority={priority}")

    # Creating a skill
    print_subheader("Creating Skills")
    skill = Skill(
        name="sql-optimize",
        description="Optimize slow SQL queries step by step",
        scope=SkillScope.BASE,
        scope_id="system",
        content="""# SQL Query Optimization

## Prerequisites
- Access to EXPLAIN ANALYZE
- Query execution logs

## Steps
1. Analyze the current query plan
2. Identify full table scans
3. Add appropriate indexes
4. Rewrite subqueries as JOINs if needed
5. Test with EXPLAIN ANALYZE again
""",
        triggers=["slow query", "optimize sql", "query performance"],
        prerequisites=["sql-basics"],
        allowed_tools=["execute_sql", "explain_analyze"],
    )
    print(f"  Created skill: {skill.name}")
    print(f"    Description: {skill.description}")
    print(f"    Scope: {skill.scope.value}")
    print(f"    Triggers: {skill.triggers}")

    # Skill metadata (lightweight)
    print_subheader("Skill Metadata (Progressive Disclosure)")
    metadata = skill.skill_metadata
    print(f"  Metadata for {metadata.name}:")
    print(f"    Description: {metadata.description}")
    print(f"    Triggers: {metadata.triggers}")
    print("  (Note: Full content not loaded until needed)")

    # Skill matching
    print_subheader("Trigger Matching")
    test_queries = [
        "My query is running slow",
        "How do I optimize sql performance?",
        "What's the weather today?",
    ]
    for query in test_queries:
        matched = metadata.matches_trigger(query)
        status = f"✓ matched '{matched}'" if matched else "✗ no match"
        print(f"  '{query}' -> {status}")

    # Skills index
    print_subheader("Skills Index for Prompt")
    index = SkillsIndex(skills=[
        SkillMetadata(
            name="sql-optimize",
            description="Optimize slow SQL queries",
            scope=SkillScope.BASE,
            scope_id="system",
            triggers=["slow query", "optimize"],
        ),
        SkillMetadata(
            name="code-review",
            description="Perform thorough code review",
            scope=SkillScope.USER,
            scope_id="user-123",
            triggers=["review", "check code"],
        ),
    ])
    print(f"  Index contains {index.total_count} skills")
    print(f"  Compact format: {index.format_compact()}")


async def demo_skill_store():
    """Demonstrate the in-memory skill store."""
    print_header("5. SKILL STORE")

    store = InMemorySkillStore()
    await store.initialize()
    print("  Store initialized ✓")

    # Save skills at different scopes
    print_subheader("Saving Skills at Different Scopes")

    base_skill = Skill(
        name="debugging",
        description="Debug code systematically",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Debugging\n1. Reproduce\n2. Isolate\n3. Fix",
        triggers=["debug", "bug", "error"],
    )
    await store.save(base_skill)
    print(f"  Saved BASE skill: {base_skill.name}")

    user_skill = Skill(
        name="my-deploy",
        description="Custom deployment workflow",
        scope=SkillScope.USER,
        scope_id="user-123",
        content="# My Deploy\n1. Run tests\n2. Build\n3. Push",
        triggers=["deploy", "release"],
    )
    await store.save(user_skill)
    print(f"  Saved USER skill: {user_skill.name}")

    project_skill = Skill(
        name="api-test",
        description="Test API endpoints for this project",
        scope=SkillScope.PROJECT,
        scope_id="myproject",
        content="# API Testing\n1. Check endpoints\n2. Validate responses",
        triggers=["test api", "api test"],
    )
    await store.save(project_skill)
    print(f"  Saved PROJECT skill: {project_skill.name}")

    # Demonstrate scope layering
    print_subheader("Scope Layering (Override)")
    await store.save(Skill(
        name="debugging",  # Same name as base
        description="Project-specific debugging for microservices",
        scope=SkillScope.PROJECT,
        scope_id="myproject",
        content="# Microservice Debugging\n1. Check logs\n2. Trace requests",
        triggers=["debug"],
    ))
    print("  Saved PROJECT skill: debugging (overrides base)")

    all_skills = await store.list_all_metadata(
        user_id="user-123",
        project_id="myproject",
    )
    debug_skill = next((s for s in all_skills if s.name == "debugging"), None)
    if debug_skill:
        print(f"  After layering: debugging is from {debug_skill.scope.value} scope")
        print(f"    Description: {debug_skill.description}")

    # Search by trigger
    print_subheader("Search by Trigger")
    matches = await store.search_by_trigger(
        "I need to debug this error",
        user_id="user-123",
        project_id="myproject",
    )
    print("  Query: 'I need to debug this error'")
    print(f"  Found {len(matches)} matching skills:")
    for match in matches:
        print(f"    - {match.skill.name} (confidence: {match.confidence:.2f})")


async def demo_skill_service():
    """Demonstrate the skill service."""
    print_header("6. SKILL SERVICE")

    store = InMemorySkillStore()
    await store.initialize()
    matcher = SkillMatcher()
    service = SkillService(store=store, matcher=matcher)
    print("  Service initialized ✓")

    # Register skills
    print_subheader("Registering Skills via Service")

    await service.register_skill(Skill(
        name="code-format",
        description="Format code according to project standards",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Code Formatting\n\n1. Run Black\n2. Run isort\n3. Check flake8",
        triggers=["format", "lint", "style"],
    ))
    print("  Registered: code-format (base)")

    await service.register_skill(Skill(
        name="pr-review",
        description="Review pull requests thoroughly",
        scope=SkillScope.USER,
        scope_id="user-123",
        content="# PR Review\n\n1. Check tests\n2. Review logic\n3. Check style",
        triggers=["review pr", "pull request"],
    ))
    print("  Registered: pr-review (user)")

    # Get available skills
    print_subheader("Get Available Skills")
    skills = await service.get_available_skills(user_id="user-123")
    print(f"  Available skills for user-123: {len(skills)}")
    for skill in skills:
        print(f"    - {skill.name}: {skill.description}")

    # Match skills to query
    print_subheader("Match Skills to Query")
    test_queries = [
        "Please format my code",
        "Review this pull request",
        "What's 2+2?",
    ]
    for query in test_queries:
        matches = await service.match_skills(query, user_id="user-123")
        if matches:
            best = matches[0]
            print(f"  '{query}' -> {best.skill.name} ({best.confidence:.2f})")
        else:
            print(f"  '{query}' -> no match")

    # Load full skill content
    print_subheader("Load Full Skill Content")
    full_skill = await service.load_skill_content("code-format", user_id="user-123")
    if full_skill:
        print(f"  Loaded: {full_skill.name}")
        print(f"  Content preview: {full_skill.content[:50]}...")

    # Format skills index
    print_subheader("Format Skills Index for Prompt")
    all_skills = await service.get_available_skills(user_id="user-123")
    index_text = service.format_skills_index(all_skills)
    print("  Skills index:")
    for line in index_text.split("\n")[:8]:
        print(f"    {line}")


async def demo_scoped_memory_middleware():
    """Demonstrate scoped memory middleware."""
    print_header("7. SCOPED MEMORY MIDDLEWARE")

    # Setup
    store = InMemoryScopedMemoryStore()
    await store.initialize()
    service = ScopedMemoryService(store=store)

    # Add some memories
    await service.save_global(
        user_id="user-123",
        key="timezone",
        content="User is in PST timezone",
        category=MemoryCategory.PREFERENCE,
    )
    await service.save_project(
        project_id="proj-001",
        key="db-type",
        content="Using PostgreSQL 15",
        category=MemoryCategory.ARCHITECTURE,
    )

    # Create middleware
    middleware = ScopedMemoryMiddleware(
        memory_service=service,
        user_id="user-123",
        project_id="proj-001",
        enabled=True,
    )
    print(f"  Middleware: {middleware.name}")

    # Create context
    print_subheader("Injecting Memories into Context")
    context = MiddlewareContext(
        user_input="What database should I use?",
        user_id="user-123",
        session_id="sess-001",
        metadata={"project_id": "proj-001"},
    )

    # Process through middleware
    async def next_handler(ctx):
        return ctx

    result = await middleware.process(context, next_handler)

    # Check injected content
    if result.modifications:
        print("  Modifications made:")
        for key, value in result.modifications.items():
            if key == "injected_memories":
                print(f"    {key}: {len(value)} memories")
            elif key == "memory_prompt_section":
                lines = value.split("\n")
                print(f"    {key}:")
                for line in lines[:5]:
                    if line.strip():
                        print(f"      {line}")


async def demo_auto_learn_middleware():
    """Demonstrate auto-learn middleware."""
    print_header("8. AUTO-LEARN MIDDLEWARE")

    store = InMemoryScopedMemoryStore()
    await store.initialize()
    service = ScopedMemoryService(store=store)

    middleware = ScopedMemoryAutoLearnMiddleware(
        memory_service=service,
        user_id="user-123",
        enabled=True,
    )
    print(f"  Middleware: {middleware.name}")

    # Test preference extraction
    print_subheader("Extracting Preferences from User Input")
    test_inputs = [
        "I prefer using TypeScript over JavaScript",
        "Always use async/await for promises",
        "What's the weather today?",
    ]

    for user_input in test_inputs:
        context = MiddlewareContext(
            user_input=user_input,
            user_id="user-123",
            session_id="sess-001",
        )

        async def next_handler(ctx):
            return ctx

        await middleware.process(context, next_handler)

    # Check what was learned
    memories = await store.list_by_scope(MemoryScope.SESSION, "sess-001")
    print(f"\n  Learned {len(memories)} preferences:")
    for mem in memories:
        print(f"    - {mem.content}")


async def demo_skills_middleware():
    """Demonstrate skills middleware."""
    print_header("9. SKILLS MIDDLEWARE")

    store = InMemorySkillStore()
    await store.initialize()
    matcher = SkillMatcher()
    service = SkillService(store=store, matcher=matcher)

    # Register some skills
    await service.register_skill(Skill(
        name="unit-test",
        description="Write unit tests for functions",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Unit Testing\n\n1. Identify test cases\n2. Write assertions",
        triggers=["write test", "unit test", "test function"],
    ))

    await service.register_skill(Skill(
        name="refactor",
        description="Refactor code for better maintainability",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Refactoring\n\n1. Identify code smells\n2. Apply patterns",
        triggers=["refactor", "clean up", "improve code"],
    ))

    # Create middleware
    middleware = SkillsMiddleware(
        skill_service=service,
        user_id="user-123",
        auto_activate=True,
        max_auto_skills=2,
        confidence_threshold=0.5,
    )
    print(f"  Middleware: {middleware.name}")

    # Test auto-activation
    print_subheader("Auto-Activating Skills Based on Query")
    test_queries = [
        "Help me write unit tests for this function",
        "What's 2+2?",
        "Please refactor this code",
    ]

    for query in test_queries:
        context = MiddlewareContext(
            user_input=query,
            user_id="user-123",
            session_id="sess-001",
        )

        async def next_handler(ctx):
            return ctx

        result = await middleware.process(context, next_handler)

        activated = result.modifications.get("activated_skills", [])
        if activated:
            skill_names = [s["name"] for s in activated]
            print(f"  '{query[:40]}...' -> activated: {skill_names}")
        else:
            print(f"  '{query[:40]}...' -> no skills activated")


async def demo_skill_request_middleware():
    """Demonstrate skill request middleware."""
    print_header("10. SKILL REQUEST MIDDLEWARE")

    store = InMemorySkillStore()
    await store.initialize()
    matcher = SkillMatcher()
    service = SkillService(store=store, matcher=matcher)

    await service.register_skill(Skill(
        name="docker-deploy",
        description="Deploy using Docker containers",
        scope=SkillScope.BASE,
        scope_id="system",
        content="# Docker Deployment\n\n1. Build image\n2. Push to registry\n3. Deploy",
        triggers=["docker", "container"],
    ))

    middleware = SkillRequestMiddleware(
        skill_service=service,
        user_id="user-123",
    )
    print(f"  Middleware: {middleware.name}")

    # Test explicit skill requests
    print_subheader("Handling Explicit Skill Requests")
    test_queries = [
        "use skill docker-deploy",
        "@skill docker-deploy",
        "run docker-deploy skill",
        "Just a normal question",
    ]

    for query in test_queries:
        context = MiddlewareContext(
            user_input=query,
            user_id="user-123",
            session_id="sess-001",
        )

        async def next_handler(ctx):
            return ctx

        result = await middleware.process(context, next_handler)

        loaded = result.modifications.get("requested_skill")
        if loaded:
            print(f"  '{query}' -> loaded: {loaded['name']}")
        else:
            print(f"  '{query}' -> no skill requested")


async def demo_engine_integration():
    """Demonstrate integration with CtxForge engine."""
    print_header("11. CTXFORGE ENGINE INTEGRATION")

    from ctxforge.config.base import EngineConfig
    from ctxforge.engine.context_engine import CtxForge
    from ctxforge.storage.memory.memory import InMemoryMemoryStore
    from ctxforge.storage.memory.session import InMemorySessionStore

    # Create services
    scoped_store = InMemoryScopedMemoryStore()
    await scoped_store.initialize()
    scoped_service = ScopedMemoryService(store=scoped_store)

    skill_store = InMemorySkillStore()
    await skill_store.initialize()
    skill_service = SkillService(store=skill_store, matcher=SkillMatcher())

    # Create engine with services
    config = EngineConfig(
        name="demo-engine",
        scoped_memory={"enabled": True},
        skills={"enabled": True},
    )

    engine = CtxForge(
        config=config,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        scoped_memory_service=scoped_service,
        skill_service=skill_service,
    )
    print("  Engine initialized with hierarchical memory & skills ✓")

    # Use engine API for scoped memories
    print_subheader("Using Engine API for Scoped Memories")
    await engine.save_scoped_memory(
        scope="global",
        scope_id="user-demo",
        key="favorite-framework",
        content="FastAPI for APIs, React for frontend",
        category="preference",
    )
    print("  Saved global preference via engine")

    await engine.save_scoped_memory(
        scope="project",
        scope_id="demo-project",
        key="test-framework",
        content="Use pytest with coverage >= 80%",
        category="convention",
    )
    print("  Saved project convention via engine")

    # Get merged memories
    merged = await engine.get_merged_memories(
        user_id="user-demo",
        project_id="demo-project",
    )
    print(f"  Retrieved {merged.total_count} merged memories")

    # Use engine API for skills
    print_subheader("Using Engine API for Skills")
    await engine.register_skill(
        name="api-design",
        description="Design RESTful API endpoints",
        content="# API Design\n\n1. Define resources\n2. Choose HTTP methods\n3. Design responses",
        scope="base",
        scope_id="system",
        triggers=["api design", "rest api", "endpoint"],
    )
    print("  Registered skill via engine")

    skills = await engine.get_available_skills()
    print(f"  Available skills: {len(skills)}")

    matches = await engine.match_skills("Help me design a REST API")
    if matches:
        print(f"  Query matched: {matches[0].skill.name}")

    await engine.close()
    print("\n  Engine closed ✓")


async def demo_postgres_stores():
    """Demonstrate PostgreSQL database stores (optional - requires connection)."""
    print_header("12. POSTGRESQL DATABASE STORES (Optional)")

    import os

    from dotenv import load_dotenv

    # Try to load .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Check for PostgreSQL connection
    # Support both POSTGRES_DATABASE and POSTGRES_DB (from .env.example)
    pg_host = os.getenv("POSTGRES_HOST")
    pg_user = os.getenv("POSTGRES_USER")
    pg_password = os.getenv("POSTGRES_PASSWORD")
    pg_database = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")

    if not all([pg_host, pg_user, pg_password, pg_database]):
        print("  ⚠ PostgreSQL not configured. Skipping...")
        print("  Set POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB")
        print("  in examples/.env to enable this demo.")
        return

    try:
        from ctxforge.storage.connection import PostgresConfig
        from ctxforge.storage.postgres.scoped_memory import PostgresScopedMemoryStore
        from ctxforge.storage.postgres.skill import PostgresSkillStore

        pg_config = PostgresConfig(
            host=pg_host,
            user=pg_user,
            password=pg_password,
            database=pg_database,
            port=int(os.getenv("POSTGRES_PORT", "5432")),
        )

        # Test Scoped Memory Store
        print_subheader("PostgreSQL Scoped Memory Store")
        scoped_store = PostgresScopedMemoryStore(config=pg_config)
        await scoped_store.initialize()
        print("  Store initialized ✓")

        # Save a memory
        test_memory = ScopedMemory(
            id=str(uuid.uuid4()),
            scope=MemoryScope.GLOBAL,
            scope_id="pg-test-user",
            category=MemoryCategory.PREFERENCE,
            key="pg-test-key",
            content="This is a PostgreSQL test memory",
        )
        await scoped_store.save(test_memory)
        print(f"  Saved: {test_memory.key}")

        # Retrieve it by scope/scope_id/key
        retrieved = await scoped_store.get(
            test_memory.scope, test_memory.scope_id, test_memory.key
        )
        if retrieved:
            print(f"  Retrieved: {retrieved.content}")
        else:
            print("  ✗ Failed to retrieve memory")

        # Clean up
        await scoped_store.delete(
            test_memory.scope, test_memory.scope_id, test_memory.key
        )
        print("  Cleaned up test memory ✓")

        # Disconnect scoped memory store
        await scoped_store.disconnect()

        # Test Skill Store
        print_subheader("PostgreSQL Skill Store")
        skill_store = PostgresSkillStore(config=pg_config)
        await skill_store.initialize()
        print("  Store initialized ✓")

        # Save a skill
        test_skill = Skill(
            name="pg-test-skill",
            description="PostgreSQL test skill",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# PostgreSQL Test\n\nThis is a test skill.",
            triggers=["pg test"],
        )
        await skill_store.save(test_skill)
        print(f"  Saved: {test_skill.name}")

        # Retrieve it by name/scope/scope_id
        retrieved_skill = await skill_store.get(
            test_skill.name, test_skill.scope, test_skill.scope_id
        )
        if retrieved_skill:
            print(f"  Retrieved: {retrieved_skill.description}")
        else:
            print("  ✗ Failed to retrieve skill")

        # Clean up
        await skill_store.delete(
            test_skill.name, test_skill.scope, test_skill.scope_id
        )
        print("  Cleaned up test skill ✓")

        # Disconnect skill store
        await skill_store.disconnect()

        print("\n  PostgreSQL stores working correctly! ✓")

    except ImportError as e:
        print(f"  ⚠ PostgreSQL dependencies not installed: {e}")
    except Exception as e:
        print(f"  ⚠ PostgreSQL connection failed: {e}")
        print("  Check your connection settings in examples/.env")


async def demo_mysql_stores():
    """Demonstrate MySQL database stores (optional - requires connection)."""
    print_header("13. MYSQL DATABASE STORES (Optional)")

    import os

    from dotenv import load_dotenv

    # Try to load .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Check for MySQL connection
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_database = os.getenv("MYSQL_DATABASE")

    if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
        print("  ⚠ MySQL not configured. Skipping...")
        print("  Set MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")
        print("  in examples/.env to enable this demo.")
        return

    try:
        from ctxforge.storage.connection import MySQLConfig
        from ctxforge.storage.mysql.scoped_memory import MySQLScopedMemoryStore
        from ctxforge.storage.mysql.skill import MySQLSkillStore

        mysql_config = MySQLConfig(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
            port=int(os.getenv("MYSQL_PORT", "3306")),
        )

        # Test Scoped Memory Store
        print_subheader("MySQL Scoped Memory Store")
        scoped_store = MySQLScopedMemoryStore(config=mysql_config)
        await scoped_store.initialize()
        print("  Store initialized ✓")

        # Save a memory
        test_memory = ScopedMemory(
            id=str(uuid.uuid4()),
            scope=MemoryScope.GLOBAL,
            scope_id="mysql-test-user",
            category=MemoryCategory.PREFERENCE,
            key="mysql-test-key",
            content="This is a MySQL test memory",
        )
        await scoped_store.save(test_memory)
        print(f"  Saved: {test_memory.key}")

        # Retrieve it by scope/scope_id/key
        retrieved = await scoped_store.get(
            test_memory.scope, test_memory.scope_id, test_memory.key
        )
        if retrieved:
            print(f"  Retrieved: {retrieved.content}")
        else:
            print("  ✗ Failed to retrieve memory")

        # Clean up
        await scoped_store.delete(
            test_memory.scope, test_memory.scope_id, test_memory.key
        )
        print("  Cleaned up test memory ✓")

        # Disconnect scoped memory store
        await scoped_store.disconnect()

        # Test Skill Store
        print_subheader("MySQL Skill Store")
        skill_store = MySQLSkillStore(config=mysql_config)
        await skill_store.initialize()
        print("  Store initialized ✓")

        # Save a skill
        test_skill = Skill(
            name="mysql-test-skill",
            description="MySQL test skill",
            scope=SkillScope.BASE,
            scope_id="system",
            content="# MySQL Test\n\nThis is a test skill.",
            triggers=["mysql test"],
        )
        await skill_store.save(test_skill)
        print(f"  Saved: {test_skill.name}")

        # Retrieve it by name/scope/scope_id
        retrieved_skill = await skill_store.get(
            test_skill.name, test_skill.scope, test_skill.scope_id
        )
        if retrieved_skill:
            print(f"  Retrieved: {retrieved_skill.description}")
        else:
            print("  ✗ Failed to retrieve skill")

        # Clean up
        await skill_store.delete(
            test_skill.name, test_skill.scope, test_skill.scope_id
        )
        print("  Cleaned up test skill ✓")

        # Disconnect skill store
        await skill_store.disconnect()

        print("\n  MySQL stores working correctly! ✓")

    except ImportError as e:
        print(f"  ⚠ MySQL dependencies not installed: {e}")
    except Exception as e:
        print(f"  ⚠ MySQL connection failed: {e}")
        print("  Check your connection settings in examples/.env")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" HIERARCHICAL MEMORY & SKILLS SYSTEM - END-TO-END DEMO")
    print("=" * 70)

    try:
        # Core models
        await demo_scoped_memory_models()
        await demo_skill_models()

        # Storage
        await demo_scoped_memory_store()
        await demo_skill_store()

        # Services
        await demo_scoped_memory_service()
        await demo_skill_service()

        # Middleware
        await demo_scoped_memory_middleware()
        await demo_auto_learn_middleware()
        await demo_skills_middleware()
        await demo_skill_request_middleware()

        # Engine integration
        await demo_engine_integration()

        # Database stores (optional - require database connections)
        await demo_postgres_stores()
        await demo_mysql_stores()

        print_header("DEMO COMPLETE")
        print("\n  All features demonstrated successfully! ✓")
        print("\n  Key takeaways:")
        print("  - Scoped memories provide hierarchical context (global > project > session)")
        print("  - Skills offer progressive disclosure (metadata first, content on-demand)")
        print("  - Middleware components integrate with the CtxForge pipeline")
        print("  - The engine API provides high-level access to all features")
        print("  - PostgreSQL and MySQL stores provide persistent, distributed storage")
        print()

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
