from __future__ import annotations

"""
Assembly service.

This service owns context assembly logic:
- selecting/creating the configured assembler
- invoking it with best-effort backward-compatible signatures
- applying include_history/max_history_events trimming when needed
- optionally inserting extra sections (e.g., graph) after assembly
"""

from typing import Callable, List, Optional

from ctxforge.config.base import EngineConfig
from ctxforge.core.context import Context
from ctxforge.core.memory import MemoryItem
from ctxforge.core.session import Session
from ctxforge.protocols.compactor import IContextAssembler


class AssemblyService:
    """Owns assembler dependency and provides context assembly entrypoints."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        assembler_provider: Callable[[], Optional[IContextAssembler]],
        set_assembler: Callable[[IContextAssembler], None],
    ):
        self._cfg = config
        self._get_assembler = assembler_provider
        self._set_assembler = set_assembler

    async def assemble(
        self,
        *,
        session: Session,
        current_query: str,
        memories: List[MemoryItem],
        system_instructions: str,
        token_budget: int,
        include_history: bool,
        max_history_events: int,
        graph_section: Optional[str] = None,
        graph_section_mode: str = "flat",
        synthesized_memory_context: Optional[str] = None,
    ) -> Context:
        """
        Assemble a Context using the configured assembler (fallback to default).

        If `graph_section` is provided, it is appended as a non-required section
        after core assembly so the assembler remains unaware of graph internals.
        """
        assembler = self._get_assembler()
        if assembler is None:
            from ctxforge.compaction.assembler import DefaultContextAssembler

            assembler = DefaultContextAssembler()
            self._set_assembler(assembler)

        # Prefer an assembler signature that supports history controls if implemented.
        try:
            context = await assembler.assemble(
                session=session,
                current_query=current_query,
                memories=memories,
                system_instructions=system_instructions,
                token_budget=token_budget,
                include_history=include_history,
                max_history_events=max_history_events,
            )
        except TypeError:
            # Backward-compatible path (assembler without include_history args)
            context = await assembler.assemble(
                session=session,
                current_query=current_query,
                memories=memories,
                system_instructions=system_instructions,
                token_budget=token_budget,
            )
            if not include_history:
                context.events = []
            elif max_history_events >= 0:
                context.events = context.events[-max_history_events:]

        # Replace memory section with synthesized narrative if available
        if synthesized_memory_context is not None:
            for section in context.sections:
                if section.name == "memories":
                    section.content = synthesized_memory_context
                    break
            context.metadata["memory_synthesized"] = True

        if graph_section:
            context.add_section(
                name=self._cfg.graph.section_name,
                content=graph_section,
                priority=50,
                is_required=False,
            )
            # Tag the context so downstream consumers can detect the rendering mode.
            context.metadata["graph_section_mode"] = graph_section_mode

        return context


