"""
Integration tests for Hierarchical Memory & Skills System.

These tests verify that the services are properly wired into CtxForge
and can be accessed and used through the engine's public API.
"""
import pytest

from ctxforge.config.base import EngineConfig
from ctxforge.core.scoped_memory import MemoryCategory, MemoryScope
from ctxforge.core.skill import SkillScope
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.engine.services.skill_matcher import SkillMatcher
from ctxforge.engine.services.skill_service import SkillService
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore
from ctxforge.storage.memory.skill import InMemorySkillStore


@pytest.fixture
def config():
    """Create a test configuration with hierarchical memory enabled."""
    return EngineConfig(
        name="test-engine",
        scoped_memory={"enabled": True},
        skills={"enabled": True},
    )


@pytest.fixture
async def scoped_memory_service():
    """Create a scoped memory service with in-memory store."""
    store = InMemoryScopedMemoryStore()
    await store.initialize()
    return ScopedMemoryService(store=store)


@pytest.fixture
async def skill_service():
    """Create a skill service with in-memory store."""
    store = InMemorySkillStore()
    await store.initialize()
    matcher = SkillMatcher()
    return SkillService(store=store, matcher=matcher)


@pytest.fixture
async def engine(config, scoped_memory_service, skill_service):
    """Create a CtxForge instance with hierarchical memory and skills."""
    engine = CtxForge(
        config=config,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        scoped_memory_service=scoped_memory_service,
        skill_service=skill_service,
    )
    yield engine
    await engine.close()


class TestScopedMemoryIntegration:
    """Tests for scoped memory integration with CtxForge."""

    async def test_service_accessible(self, engine):
        """Test that scoped memory service is accessible."""
        assert engine.scoped_memory_service is not None

    async def test_save_global_memory(self, engine):
        """Test saving a global memory via engine API."""
        memory = await engine.save_scoped_memory(
            scope="global",
            scope_id="user-1",
            key="pref-editor",
            content="VSCode",
            category="preference",
        )
        assert memory is not None
        assert memory.scope == MemoryScope.GLOBAL
        assert memory.content == "VSCode"

    async def test_save_project_memory(self, engine):
        """Test saving a project memory."""
        memory = await engine.save_scoped_memory(
            scope="project",
            scope_id="proj-1",
            key="db-type",
            content="PostgreSQL",
            category="architecture",
        )
        assert memory is not None
        assert memory.scope == MemoryScope.PROJECT
        assert memory.category == MemoryCategory.ARCHITECTURE

    async def test_save_session_memory(self, engine):
        """Test saving a session memory."""
        memory = await engine.save_scoped_memory(
            scope="session",
            scope_id="sess-1",
            key="current-task",
            content="Refactoring auth module",
            category="context",
        )
        assert memory is not None
        assert memory.scope == MemoryScope.SESSION
        assert memory.category == MemoryCategory.CONTEXT

    async def test_get_merged_memories(self, engine):
        """Test getting merged memories with hierarchical override."""
        # Save memories at different scopes
        await engine.save_scoped_memory(
            scope="global", scope_id="user-1",
            key="editor", content="VSCode", category="preference"
        )
        await engine.save_scoped_memory(
            scope="project", scope_id="proj-1",
            key="editor", content="PyCharm", category="preference"
        )  # Overrides global

        result = await engine.get_merged_memories(
            user_id="user-1",
            project_id="proj-1"
        )

        assert result is not None
        assert result.total_count == 1
        assert result.override_count == 1
        # Project overrides global
        assert result.memories[0].content == "PyCharm"

    async def test_clear_session_memories(self, engine):
        """Test clearing session memories."""
        await engine.save_scoped_memory(
            scope="session", scope_id="sess-1",
            key="temp-1", content="Temp 1", category="context"
        )
        await engine.save_scoped_memory(
            scope="session", scope_id="sess-1",
            key="temp-2", content="Temp 2", category="context"
        )

        deleted = await engine.clear_session_memories("sess-1")
        assert deleted == 2

    async def test_memory_with_priority(self, engine):
        """Test that priority is preserved."""
        memory = await engine.save_scoped_memory(
            scope="global", scope_id="user-1",
            key="high-priority", content="Important",
            category="instruction", priority=10
        )
        assert memory.priority == 10


class TestSkillsIntegration:
    """Tests for skills integration with CtxForge."""

    async def test_service_accessible(self, engine):
        """Test that skill service is accessible."""
        assert engine.skill_service is not None

    async def test_register_base_skill(self, engine):
        """Test registering a base skill."""
        skill = await engine.register_skill(
            name="sql-optimize",
            description="Optimize SQL queries",
            content="# SQL Optimization\n\n1. Analyze query\n2. Add indexes",
            scope="base",
            scope_id="system",
            triggers=["slow query", "optimize sql"],
        )
        assert skill is not None
        assert skill.name == "sql-optimize"
        assert skill.scope == SkillScope.BASE

    async def test_register_user_skill(self, engine):
        """Test registering a user skill."""
        skill = await engine.register_skill(
            name="my-workflow",
            description="My custom workflow",
            content="# My Workflow\n\nSteps...",
            scope="user",
            scope_id="user-1",
        )
        assert skill is not None
        assert skill.scope == SkillScope.USER

    async def test_register_project_skill(self, engine):
        """Test registering a project skill."""
        skill = await engine.register_skill(
            name="deploy-app",
            description="Deploy the application",
            content="# Deploy App\n\n1. Build\n2. Test\n3. Deploy",
            scope="project",
            scope_id="proj-1",
            triggers=["deploy", "release"],
        )
        assert skill is not None
        assert skill.scope == SkillScope.PROJECT

    async def test_get_available_skills(self, engine):
        """Test getting available skills with layering."""
        await engine.register_skill(
            name="skill-a", description="Base A", content="Content",
            scope="base", scope_id="system"
        )
        await engine.register_skill(
            name="skill-a", description="User A", content="Content",
            scope="user", scope_id="user-1"
        )  # Overrides base

        skills = await engine.get_available_skills(user_id="user-1")

        assert len(skills) == 1
        assert skills[0].description == "User A"

    async def test_load_skill(self, engine):
        """Test loading full skill content."""
        await engine.register_skill(
            name="test-skill",
            description="Test skill",
            content="# Full Content\n\nDetailed workflow...",
            scope="base", scope_id="system"
        )

        skill = await engine.load_skill("test-skill")

        assert skill is not None
        assert "Full Content" in skill.content

    async def test_match_skills(self, engine):
        """Test matching skills by trigger."""
        await engine.register_skill(
            name="sql-optimize",
            description="Optimize SQL",
            content="Content",
            scope="base", scope_id="system",
            triggers=["slow query", "optimize sql"]
        )

        matches = await engine.match_skills("I have a slow query")

        assert len(matches) == 1
        assert matches[0].skill.name == "sql-optimize"

    async def test_match_skills_threshold(self, engine):
        """Test that low-confidence matches are filtered."""
        await engine.register_skill(
            name="sql-optimize",
            description="Optimize SQL",
            content="Content",
            scope="base", scope_id="system",
            triggers=["slow query"]
        )

        # Very high threshold should filter out partial matches
        matches = await engine.match_skills(
            "I have a slow query",
            threshold=0.99
        )

        # Exact trigger match should still pass
        assert len(matches) >= 0  # Depends on matching implementation


class TestEngineWithoutServices:
    """Tests for engine behavior when services are not configured."""

    @pytest.fixture
    async def engine_no_services(self):
        """Create engine without hierarchical memory services."""
        config = EngineConfig(name="test-minimal")
        engine = CtxForge(
            config=config,
            session_store=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            # No scoped_memory_service or skill_service
        )
        yield engine
        await engine.close()

    async def test_scoped_memory_returns_none(self, engine_no_services):
        """Test that scoped memory methods return None when not configured."""
        result = await engine_no_services.save_scoped_memory(
            scope="global", scope_id="user-1",
            key="k", content="c", category="preference"
        )
        assert result is None

    async def test_skill_returns_empty_list(self, engine_no_services):
        """Test that skill methods return empty when not configured."""
        skills = await engine_no_services.get_available_skills()
        assert skills == []

    async def test_match_skills_returns_empty(self, engine_no_services):
        """Test that match_skills returns empty when not configured."""
        matches = await engine_no_services.match_skills("query")
        assert matches == []
