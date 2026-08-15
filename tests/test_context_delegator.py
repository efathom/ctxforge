"""
Tests for ContextDelegator Service.

Tests that the delegator correctly prepares task-scoped context
packages for sub-agents with curated information.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.context import Context, ContextSection
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session
from ctxforge.engine.services.context_delegator_service import (
    ContextDelegator,
    DelegatedTask,
)


def _make_task(
    task_id="task-1",
    description="Implement user login",
    full_text="Create a login endpoint with JWT auth.",
    scene_setting="Part of the authentication module.",
    dependencies=None,
):
    return DelegatedTask(
        task_id=task_id,
        description=description,
        full_task_text=full_text,
        scene_setting=scene_setting,
        dependencies=dependencies or [],
    )


def _make_memory(content="User prefers REST over GraphQL"):
    """Helper to create a mock MemoryItem."""
    mock_mem = MagicMock(spec=MemoryItem)
    mock_mem.content = content
    mock_mem.memory_type = MemoryType.PREFERENCE
    mock_mem.to_prompt_format.return_value = content
    mock_mem.model_dump.return_value = {"content": content}
    return mock_mem


def _make_retriever(memories=None):
    """Helper to create a mock IRetriever that returns given memories."""
    retriever = AsyncMock()
    results = []
    for mem in (memories or []):
        result = MagicMock()
        result.memory = mem
        results.append(result)
    retriever.retrieve = AsyncMock(return_value=results)
    return retriever


class TestDelegatedTask:
    """Tests for the DelegatedTask data class."""

    def test_basic_creation(self):
        task = DelegatedTask(
            task_id="t-1",
            description="Do something",
            full_task_text="Full details here.",
        )
        assert task.task_id == "t-1"
        assert task.dependencies == []
        assert task.metadata == {}
        assert task.scene_setting == ""

    def test_with_dependencies(self):
        task = DelegatedTask(
            task_id="t-2",
            description="Step two",
            full_task_text="Details.",
            dependencies=["t-1"],
        )
        assert task.dependencies == ["t-1"]


class TestContextDelegatorBasic:
    """Tests for basic delegated context preparation."""

    @pytest.mark.asyncio
    async def test_basic_context_has_task_section(self):
        """The delegated context should contain the task specification."""
        delegator = ContextDelegator()
        task = _make_task()

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        assert isinstance(ctx, Context)
        task_section = ctx.get_section("task_specification")
        assert task_section is not None
        assert "Implement user login" in task_section.content
        assert "JWT auth" in task_section.content

    @pytest.mark.asyncio
    async def test_context_has_scene_setting(self):
        """Scene-setting context should be included when provided."""
        delegator = ContextDelegator()
        task = _make_task(scene_setting="This is the auth module.")

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        scene_section = ctx.get_section("scene_setting")
        assert scene_section is not None
        assert "auth module" in scene_section.content

    @pytest.mark.asyncio
    async def test_context_without_scene_setting(self):
        """No scene_setting section when not provided."""
        delegator = ContextDelegator()
        task = _make_task(scene_setting="")

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        assert ctx.get_section("scene_setting") is None

    @pytest.mark.asyncio
    async def test_context_has_dependency_info(self):
        """Dependency information should be included."""
        delegator = ContextDelegator()
        task = _make_task(dependencies=["task-0"])

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        dep_section = ctx.get_section("dependencies")
        assert dep_section is not None
        assert "task-0" in dep_section.content

    @pytest.mark.asyncio
    async def test_context_no_dependencies_no_section(self):
        """No dependencies section when task has no dependencies."""
        delegator = ContextDelegator()
        task = _make_task(dependencies=[])

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        assert ctx.get_section("dependencies") is None

    @pytest.mark.asyncio
    async def test_context_metadata(self):
        """Delegated context should carry task metadata."""
        delegator = ContextDelegator()
        task = _make_task(task_id="task-42")

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        assert ctx.metadata["delegated_task_id"] == "task-42"
        assert ctx.metadata["is_delegated"] is True

    @pytest.mark.asyncio
    async def test_context_session_id_is_task_scoped(self):
        """The session_id should be scoped to the task, not the parent."""
        delegator = ContextDelegator()
        task = _make_task(task_id="task-7")

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        assert ctx.session_id == "delegated-task-7"

    @pytest.mark.asyncio
    async def test_context_current_query_is_task_description(self):
        """The current_query should be set to the task description."""
        delegator = ContextDelegator()
        task = _make_task(description="Build the API endpoint")

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        assert ctx.current_query == "Build the API endpoint"

    @pytest.mark.asyncio
    async def test_system_instructions_passed_through(self):
        """System instructions should be included in the context."""
        delegator = ContextDelegator()
        task = _make_task()

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
            system_instructions="You are a backend engineer.",
        )

        assert ctx.system_instructions == "You are a backend engineer."


class TestContextDelegatorWithMemories:
    """Tests for memory retrieval in delegated contexts."""

    @pytest.mark.asyncio
    async def test_includes_task_relevant_memories(self):
        """Memories should be retrieved based on task description."""
        mem = _make_memory("User prefers REST")
        retriever = _make_retriever(memories=[mem])
        delegator = ContextDelegator(retriever=retriever)
        task = _make_task()

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
            include_memories=True,
        )

        assert len(ctx.memories) == 1
        retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_memories_disabled(self):
        """When include_memories=False, no retrieval should happen."""
        retriever = _make_retriever(memories=[_make_memory()])
        delegator = ContextDelegator(retriever=retriever)
        task = _make_task()

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
            include_memories=False,
        )

        assert len(ctx.memories) == 0
        retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_retriever_no_memories(self):
        """Without a retriever, memories should be empty even if requested."""
        delegator = ContextDelegator(retriever=None)
        task = _make_task()

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
            include_memories=True,
        )

        assert len(ctx.memories) == 0


class TestContextDelegatorExtraSections:
    """Tests for extra sections passed by the coordinator."""

    @pytest.mark.asyncio
    async def test_extra_sections_included(self):
        """Extra sections should appear in the delegated context."""
        delegator = ContextDelegator()
        task = _make_task()

        extra = ContextSection(
            name="coding_standards",
            content="Use 4-space indentation.",
            priority=80,
            is_required=True,
        )

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
            extra_sections=[extra],
        )

        section = ctx.get_section("coding_standards")
        assert section is not None
        assert "4-space indentation" in section.content


class TestContextDelegatorBatch:
    """Tests for batch context preparation."""

    @pytest.mark.asyncio
    async def test_batch_prepares_all_tasks(self):
        """prepare_batch should return a context for each task."""
        delegator = ContextDelegator()
        tasks = [
            _make_task(task_id="t-1", description="Task one"),
            _make_task(task_id="t-2", description="Task two"),
            _make_task(task_id="t-3", description="Task three"),
        ]

        results = await delegator.prepare_batch(
            tasks=tasks,
            user_id="user-1",
        )

        assert len(results) == 3
        assert "t-1" in results
        assert "t-2" in results
        assert "t-3" in results

        # Each context should be task-scoped
        assert results["t-1"].session_id == "delegated-t-1"
        assert results["t-2"].current_query == "Task two"

    @pytest.mark.asyncio
    async def test_batch_with_parent_session(self):
        """Batch contexts should reference the parent session."""
        delegator = ContextDelegator()
        parent = MagicMock(spec=Session)
        parent.session_id = "parent-sess-1"

        tasks = [_make_task(task_id="t-1")]
        results = await delegator.prepare_batch(
            tasks=tasks,
            user_id="user-1",
            parent_session=parent,
        )

        assert results["t-1"].metadata["parent_session_id"] == "parent-sess-1"


class TestSectionPriority:
    """Tests that sections are ordered by priority correctly."""

    @pytest.mark.asyncio
    async def test_task_spec_highest_priority(self):
        """Task specification should have the highest priority."""
        delegator = ContextDelegator()
        task = _make_task(
            scene_setting="Architecture context here.",
            dependencies=["dep-1"],
        )

        ctx = await delegator.prepare_delegated_context(
            task=task,
            user_id="user-1",
        )

        # Sections are sorted by priority descending
        priorities = [(s.name, s.priority) for s in ctx.sections]
        task_priority = next(p for n, p in priorities if n == "task_specification")
        scene_priority = next(p for n, p in priorities if n == "scene_setting")
        dep_priority = next(p for n, p in priorities if n == "dependencies")

        assert task_priority > scene_priority
        assert scene_priority > dep_priority
