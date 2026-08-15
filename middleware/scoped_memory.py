"""
Scoped Memory Middleware.

Injects hierarchical memories into prompts based on
GLOBAL, PROJECT, and SESSION scopes.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ctxforge.core.scoped_memory import MemoryCategory
from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction

logger = logging.getLogger(__name__)


class ScopedMemoryMiddleware(BaseMiddleware):
    """
    Injects hierarchical memories into prompts.

    This middleware retrieves scoped memories (global, project, session)
    and injects them into the processed input or system context.
    """

    def __init__(
        self,
        memory_service: ScopedMemoryService,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        categories: Optional[List[MemoryCategory]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.

        In normal (factory-wired) usage, ``user_id`` and ``project_id``
        are resolved at runtime from the ``MiddlewareContext`` that
        carries per-request identifiers.  The constructor parameters
        exist only as optional overrides for single-user demos or tests.

        Args:
            memory_service: The scoped memory service
            user_id: Optional fixed user ID override (default: read from context)
            project_id: Optional fixed project ID override (default: read from context)
            categories: Optional list of categories to include
            enabled: Whether the middleware is enabled
        """
        super().__init__(enabled=enabled)
        self._memory_service = memory_service
        self._user_id = user_id
        self._project_id = project_id
        self._categories = categories

    @property
    def name(self) -> str:
        """Unique identifier for this middleware."""
        return "scoped_memory"

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID."""
        return self._user_id

    @user_id.setter
    def user_id(self, value: Optional[str]) -> None:
        """Set the user ID."""
        self._user_id = value

    @property
    def project_id(self) -> Optional[str]:
        """Get the project ID."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: Optional[str]) -> None:
        """Set the project ID."""
        self._project_id = value

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Inject scoped memories into the context.

        Args:
            context: The current middleware context
            next: Function to call the next middleware

        Returns:
            The processed context with memories injected
        """
        # Resolve IDs: prefer instance overrides, fall back to context values
        user_id = self._user_id or getattr(context, "user_id", None)
        session_id = context.session_id

        if not user_id:
            logger.debug("Skipping scoped memory injection: no user_id available")
            return await next(context)

        try:
            # Format memories for prompt injection
            memories_text = await self._memory_service.format_for_prompt(
                user_id=user_id,
                project_id=self._project_id,
                session_id=session_id,
                categories=self._categories,
            )

            if memories_text:
                # Inject as a context section (not into processed_input)
                context.add_section(
                    name="scoped_memories",
                    content=memories_text,
                    priority=55,
                    is_required=False,
                )

                # Record the modification
                context.record_modification(self.name, {
                    "action": "injected_memories",
                    "has_memories": True,
                    "session_id": session_id,
                })

                # Add a flag indicating memories were injected
                context.add_flag("scoped_memories_injected")

                logger.debug(
                    f"Injected scoped memories for user={user_id}, "
                    f"project={self._project_id}, session={session_id}"
                )
            else:
                context.record_modification(self.name, {
                    "action": "no_memories",
                    "has_memories": False,
                })

        except Exception as e:
            logger.warning(f"Failed to inject scoped memories: {e}")
            context.set_metadata(f"{self.name}_error", str(e))

        return await next(context)


class ScopedMemoryAutoLearnMiddleware(BaseMiddleware):
    """
    Middleware that auto-learns from conversation to create session memories.

    This is a post-processing middleware that analyzes agent responses
    and extracts potential session-level memories.
    """

    def __init__(
        self,
        memory_service: ScopedMemoryService,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.

        ``user_id`` and ``project_id`` are accepted for API symmetry
        with ``ScopedMemoryMiddleware`` but are currently unused —
        session memories are keyed by ``session_id`` from the context.

        Args:
            memory_service: The scoped memory service
            user_id: Unused (kept for API symmetry / future use)
            project_id: Unused (kept for API symmetry / future use)
            enabled: Whether the middleware is enabled
        """
        super().__init__(enabled=enabled)
        self._memory_service = memory_service
        self._user_id = user_id
        self._project_id = project_id
        # Patterns that might indicate a preference or instruction
        self._preference_patterns = [
            "i prefer",
            "i like",
            "i always want",
            "please always",
            "don't ever",
            "never",
        ]

    @property
    def name(self) -> str:
        """Unique identifier for this middleware."""
        return "scoped_memory_auto_learn"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Analyze input for potential memories to save.

        Args:
            context: The current middleware context
            next: Function to call the next middleware

        Returns:
            The processed context
        """
        # First, continue the chain
        result = await next(context)

        # Then, analyze the user input for preferences
        if context.user_input and context.session_id:
            await self._extract_session_memories(
                context.user_input,
                context.session_id,
            )

        return result

    async def _extract_session_memories(
        self,
        user_input: str,
        session_id: str,
    ) -> None:
        """
        Extract and save potential session memories from user input.

        This is a simple heuristic-based extraction. For more sophisticated
        extraction, use an LLM-based approach.
        """
        input_lower = user_input.lower()

        for pattern in self._preference_patterns:
            if pattern in input_lower:
                # Found a potential preference/instruction
                # Save as session memory with context category
                try:
                    key = f"extracted-{hash(user_input) % 10000}"
                    await self._memory_service.save_session(
                        session_id=session_id,
                        key=key,
                        content=user_input,
                        category=MemoryCategory.CONTEXT,
                        metadata={"source": "auto_learn", "pattern": pattern},
                    )
                    logger.debug(f"Auto-learned session memory: {key}")
                except Exception as e:
                    logger.warning(f"Failed to auto-learn memory: {e}")
                break
