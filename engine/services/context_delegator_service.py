"""
Context Delegator Service.

Prepares curated context packages for sub-tasks and sub-agents.
Instead of giving a sub-agent the full session context, the delegator
creates task-scoped context views with only the information relevant
to a specific sub-task, preventing context pollution across independent
work items.

Inspired by the Superpowers subagent-driven-development pattern where
the coordinator extracts all task text upfront and curates exactly what
context each sub-agent receives.
"""

import logging
from typing import Any, Dict, List, Optional

from ctxforge.core.context import Context, ContextBuilder, ContextSection
from ctxforge.core.memory import MemoryItem
from ctxforge.core.session import Session
from ctxforge.protocols.retriever import IRetriever
from ctxforge.protocols.storage import IMemoryStore

logger = logging.getLogger(__name__)


class DelegatedTask:
    """A self-contained task description with curated context.

    Attributes:
        task_id: Unique identifier for this sub-task.
        description: What the sub-agent should accomplish.
        scene_setting: Explains where this task fits in the larger picture.
        full_task_text: Complete task specification (not a file reference).
        dependencies: Other task IDs that must complete first.
        metadata: Arbitrary metadata for the task.
    """

    __slots__ = (
        "task_id",
        "description",
        "scene_setting",
        "full_task_text",
        "dependencies",
        "metadata",
    )

    def __init__(
        self,
        task_id: str,
        description: str,
        full_task_text: str,
        scene_setting: str = "",
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.description = description
        self.full_task_text = full_task_text
        self.scene_setting = scene_setting
        self.dependencies = dependencies or []
        self.metadata = metadata or {}


class ContextDelegator:
    """Prepares curated context packages for sub-tasks/sub-agents.

    Key principles (from Superpowers subagent-driven-development):
    1. Extract all task text upfront — sub-agents should not read files.
    2. Provide scene-setting context about the larger architecture.
    3. Each sub-agent gets a fresh, focused context (no pollution).
    4. Include only memories relevant to the specific sub-task.
    """

    def __init__(
        self,
        retriever: Optional[IRetriever] = None,
        memory_store: Optional[IMemoryStore] = None,
        default_token_budget: int = 8000,
        default_max_memories: int = 10,
    ):
        """Initialize the context delegator.

        Args:
            retriever: Memory retriever for task-scoped memory lookups.
            memory_store: Memory store for direct memory access.
            default_token_budget: Default token budget for delegated contexts.
            default_max_memories: Default max memories per delegated context.
        """
        self._retriever = retriever
        self._memory_store = memory_store
        self._default_token_budget = default_token_budget
        self._default_max_memories = default_max_memories

    async def prepare_delegated_context(
        self,
        task: DelegatedTask,
        user_id: str,
        parent_session: Optional[Session] = None,
        system_instructions: str = "",
        include_memories: bool = True,
        max_memories: Optional[int] = None,
        token_budget: Optional[int] = None,
        extra_sections: Optional[List[ContextSection]] = None,
    ) -> Context:
        """Create a focused context package for a sub-agent.

        The returned Context contains everything the sub-agent needs:
        - The full task text (no file references)
        - Scene-setting context about the architecture
        - Relevant memories (retrieved by task description)
        - Any extra sections the coordinator wants to include

        Args:
            task: The delegated task specification.
            user_id: User ID for memory retrieval scoping.
            parent_session: Optional parent session for shared state.
            system_instructions: System instructions for the sub-agent.
            include_memories: Whether to retrieve task-relevant memories.
            max_memories: Max memories to include (uses default if None).
            token_budget: Token budget for this context (uses default if None).
            extra_sections: Additional context sections to include.

        Returns:
            A Context object scoped to the specific sub-task.
        """
        budget = token_budget or self._default_token_budget
        max_mem = max_memories or self._default_max_memories

        # Build a session ID scoped to this sub-task
        session_id = f"delegated-{task.task_id}"

        builder = (
            ContextBuilder(session_id=session_id, user_id=user_id)
            .with_system_instructions(system_instructions)
            .with_token_budget(budget)
        )

        # 1. Task specification section (highest priority — always included)
        task_content = self._format_task_section(task)
        builder.with_section(
            name="task_specification",
            content=task_content,
            priority=100,
            is_required=True,
        )

        # 2. Scene-setting context (high priority)
        if task.scene_setting:
            builder.with_section(
                name="scene_setting",
                content=task.scene_setting,
                priority=90,
                is_required=True,
            )

        # 3. Dependency info
        if task.dependencies:
            dep_text = self._format_dependencies(task)
            builder.with_section(
                name="dependencies",
                content=dep_text,
                priority=85,
                is_required=True,
            )

        # 4. Task-relevant memories (retrieved by task description)
        if include_memories and self._retriever:
            memories = await self._retrieve_task_memories(
                query=task.description,
                user_id=user_id,
                max_results=max_mem,
            )
            if memories:
                builder.with_memories(memories)

        # 5. Extra sections from the coordinator
        if extra_sections:
            for section in extra_sections:
                builder.with_section(
                    name=section.name,
                    content=section.content,
                    priority=section.priority,
                    is_required=section.is_required,
                )

        # 6. Set the task description as the "current query" so
        #    downstream format converters include it properly.
        builder.with_current_query(task.description)

        # 7. Attach task metadata
        builder.with_metadata("delegated_task_id", task.task_id)
        builder.with_metadata("is_delegated", True)
        if parent_session:
            builder.with_metadata(
                "parent_session_id", parent_session.session_id
            )

        context = builder.build()
        logger.debug(
            "Prepared delegated context for task '%s' "
            "(%d sections, %d memories)",
            task.task_id,
            len(context.sections),
            len(context.memories),
        )
        return context

    async def prepare_batch(
        self,
        tasks: List[DelegatedTask],
        user_id: str,
        parent_session: Optional[Session] = None,
        system_instructions: str = "",
        include_memories: bool = True,
        max_memories: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> Dict[str, Context]:
        """Prepare delegated contexts for multiple tasks.

        Args:
            tasks: List of tasks to prepare contexts for.
            user_id: User ID for memory retrieval scoping.
            parent_session: Optional parent session.
            system_instructions: System instructions for sub-agents.
            include_memories: Whether to retrieve task-relevant memories.
            max_memories: Max memories per task.
            token_budget: Token budget per task context.

        Returns:
            Dict mapping task_id to its prepared Context.
        """
        results: Dict[str, Context] = {}
        for task in tasks:
            context = await self.prepare_delegated_context(
                task=task,
                user_id=user_id,
                parent_session=parent_session,
                system_instructions=system_instructions,
                include_memories=include_memories,
                max_memories=max_memories,
                token_budget=token_budget,
            )
            results[task.task_id] = context
        return results

    async def _retrieve_task_memories(
        self,
        query: str,
        user_id: str,
        max_results: int,
    ) -> List[MemoryItem]:
        """Retrieve memories relevant to a specific task.

        Args:
            query: The task description to search against.
            user_id: User scope for memory retrieval.
            max_results: Maximum number of memories to return.

        Returns:
            List of relevant MemoryItem objects.
        """
        if not self._retriever:
            return []
        try:
            from ctxforge.core.memory import MemoryQuery
            mq = MemoryQuery(
                query=query,
                user_id=user_id,
                limit=max_results,
            )
            results = await self._retriever.retrieve(mq)
            return [r.memory for r in results[:max_results]]
        except Exception as e:
            logger.warning("Failed to retrieve task memories: %s", e)
            return []

    @staticmethod
    def _format_task_section(task: DelegatedTask) -> str:
        """Format the task specification as a context section."""
        lines = [
            "## Task Specification",
            "",
            f"**Task ID:** {task.task_id}",
            f"**Description:** {task.description}",
            "",
            "### Full Task Details",
            "",
            task.full_task_text,
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_dependencies(task: DelegatedTask) -> str:
        """Format task dependency information."""
        lines = [
            "## Task Dependencies",
            "",
            "This task depends on the following tasks being completed first:",
            "",
        ]
        for dep_id in task.dependencies:
            lines.append(f"- `{dep_id}`")
        return "\n".join(lines)
