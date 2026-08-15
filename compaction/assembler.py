"""
Context Assembler.

Assembles the final context from session, memories, expertise, and configuration.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ctxforge.core.context import Context, ContextBuilder
from ctxforge.core.events import Event, EventType
from ctxforge.core.expertise import ExpertiseItem
from ctxforge.core.intent_note import IntentNote
from ctxforge.core.memory import MemoryItem
from ctxforge.core.memory_index import DisclosureLevel, MemoryIndex, MemoryIndexEntry
from ctxforge.core.session import Session
from ctxforge.core.timeline import TimelineFilter, TimelineQuery, TimeRange
from ctxforge.engine.registry import registry
from ctxforge.engine.services.context_window_service import ContextWindowService
from ctxforge.protocols.compactor import IContextAssembler
from ctxforge.protocols.tokenizer import ITokenizerProvider


def estimate_tokens_simple(text: str) -> int:
    """Simple token estimation: ~4 chars per token."""
    return len(text) // 4 + 1


@registry.register_assembler("default")
class DefaultContextAssembler(IContextAssembler):
    """
    Default context assembler.
    
    Assembles context in a standard format:
    1. System instructions
    2. Session summary (if any)
    3. Relevant memories
    4. Conversation history
    5. Current query
    
    Example:
        >>> assembler = DefaultContextAssembler()
        >>> context = await assembler.assemble(
        ...     session=session,
        ...     current_query="What's my order status?",
        ...     memories=retrieved_memories,
        ...     system_instructions="You are a helpful assistant.",
        ... )
    """
    
    def __init__(
        self,
        memory_format: str = "bullet",
        include_timestamps: bool = False,
        max_memory_chars: int = 500,
        tokenizer_provider: Optional[ITokenizerProvider] = None,
        use_progressive_disclosure: bool = False,
        progressive_expand_top_n: int = 3,
    ):
        """
        Initialize assembler with formatting options.

        Args:
            memory_format: How to format memories
                ("bullet", "prose", "json", "progressive")
            include_timestamps: Include timestamps in messages
            max_memory_chars: Max characters per memory
            tokenizer_provider: Optional tokenizer for accurate budgeting
            use_progressive_disclosure: Use progressive disclosure for memories
            progressive_expand_top_n: Number of top memories to show in full
        """
        self._memory_format = memory_format
        self._include_timestamps = include_timestamps
        self._max_memory_chars = max_memory_chars
        self._tokenizer: Optional[ITokenizerProvider] = tokenizer_provider
        self._use_progressive = use_progressive_disclosure
        self._expand_top_n = progressive_expand_top_n

    def set_tokenizer_provider(self, tokenizer_provider: Optional[ITokenizerProvider]) -> None:
        """Inject a tokenizer provider for accurate budgeting/trimming."""
        self._tokenizer = tokenizer_provider
    
    @property
    def name(self) -> str:
        return "default"
    
    async def assemble(
        self,
        session: Session,
        current_query: str,
        memories: List[MemoryItem],
        system_instructions: str = "",
        token_budget: int = 8000,
        expertise_items: Optional[List[ExpertiseItem]] = None,
        expertise_id: Optional[str] = None,
        include_history: bool = True,
        max_history_events: int = 10,
    ) -> Context:
        """
        Assemble the context for LLM invocation.
        
        Arranges all components into a coherent context structure.
        
        Args:
            session: The session object
            current_query: The current user query
            memories: Retrieved memories to include
            system_instructions: System prompt
            token_budget: Maximum tokens
            expertise_items: Optional expertise items to include
            expertise_id: Optional expertise ID being used
        """
        builder = ContextBuilder(session.session_id, session.user_id)
        builder.with_token_budget(token_budget)
        builder.with_current_query(current_query)
        
        # 1. Add system instructions
        if system_instructions:
            builder.with_system_instructions(system_instructions)
            builder.with_section(
                name="system_instructions",
                content=system_instructions,
                priority=100,
                is_required=True,
            )
        
        # 2. Add session summary if present
        if session.summary:
            builder.with_section(
                name="session_summary",
                content=f"Previous conversation summary:\n{session.summary}",
                priority=90,
                is_required=False,
            )
        
        # 3. Add expertise items (high priority)
        if expertise_items:
            expertise_content = self._format_expertise(expertise_items)
            builder.with_section(
                name="expertise",
                content=expertise_content,
                priority=85,  # Between summary and memories
                is_required=False,
            )
            builder.with_expertise_items(expertise_items)
            if expertise_id:
                builder.with_expertise_id(expertise_id)
        
        # 4. Add relevant memories
        if memories:
            memory_content = self._format_memories(memories)
            builder.with_section(
                name="memories",
                content=memory_content,
                priority=80,
                is_required=False,
            )
            builder.with_memories(memories)
        
        # 5. Add conversation history (excluding current query)
        events_to_include: List[Event] = []
        if include_history:
            events_to_include = session.get_recent_events(max_history_events)

        history_content = self._format_history(events_to_include)
        if history_content:
            builder.with_section(
                name="conversation_history",
                content=history_content,
                priority=70,
                is_required=False,
            )
        builder.with_events(list(events_to_include))
        
        # Build the context
        context = builder.build()
        
        # Add metadata
        context.metadata.update({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "memory_count": len(memories),
            "event_count": len(session.events),
            "has_summary": bool(session.summary),
            "expertise_item_count": len(expertise_items) if expertise_items else 0,
            "expertise_id": expertise_id,
        })
        
        # Fit to budget if needed.
        # Prefer tokenizer-based check when available (budget is *input* budget).
        if self._tokenizer is not None:
            svc = ContextWindowService(tokenizer=self._tokenizer)
            available_budget = max(0, int(token_budget) - int(context.reserved_output_tokens))
            if svc.count_total_input_tokens(context) > available_budget:
                context = await self.fit_to_budget(context, token_budget)
        else:
            if context.estimate_total_tokens() > token_budget:
                context = await self.fit_to_budget(context, token_budget)
        
        return context
    
    def _format_expertise(self, items: List[ExpertiseItem]) -> str:
        """Format expertise items for the prompt."""
        if not items:
            return ""
        
        # Group by section
        by_section: Dict[str, List] = {}
        for item in items:
            if hasattr(item.section, 'to_display_name'):
                section_name = item.section.to_display_name()
            else:
                section_name = str(item.section)
            if section_name not in by_section:
                by_section[section_name] = []
            by_section[section_name].append(item)
        
        # Format output
        lines = ["Expertise Knowledge:"]
        for section_name, section_items in sorted(by_section.items()):
            lines.append(f"\n## {section_name}")
            for item in section_items:
                if hasattr(item, 'to_prompt_format'):
                    lines.append(f"• {item.to_prompt_format()}")
                else:
                    lines.append(f"• {item.content}")
        
        return "\n".join(lines)
    
    async def fit_to_budget(
        self,
        context: Context,
        budget: int,
    ) -> Context:
        """
        Fit context to a token budget.
        
        Removes lower-priority sections until within budget.
        """
        # Prefer tokenizer-based budgeting if available (more accurate than heuristics).
        if self._tokenizer is not None:
            svc = ContextWindowService(tokenizer=self._tokenizer)
            available_budget = max(0, int(budget) - int(context.reserved_output_tokens))
            before = svc.build_overview(context, total_budget=budget)
            if before.total_input_tokens <= available_budget:
                context.metadata["token_breakdown"] = svc.overview_to_metadata(before)
                return context
        else:
            if context.estimate_total_tokens() <= budget:
                return context
        
        # Sort sections by priority (lowest first for removal)
        sections = sorted(context.sections, key=lambda s: s.priority)
        
        current_tokens = context.estimate_total_tokens()
        sections_to_keep = list(context.sections)
        removed_sections: List[str] = []
        events_trimmed = 0

        if self._tokenizer is not None:
            svc = ContextWindowService(tokenizer=self._tokenizer)
            available_budget = max(0, int(budget) - int(context.reserved_output_tokens))
            current_tokens = svc.count_total_input_tokens(context)

        for section in sections:
            if self._tokenizer is not None:
                if current_tokens <= available_budget:
                    break
            else:
                if current_tokens <= budget:
                    break
            
            # Don't remove required sections unless necessary
            if self._tokenizer is not None:
                if section.is_required and current_tokens <= int(available_budget * 1.5):
                    continue
            else:
                if section.is_required and current_tokens <= budget * 1.5:
                    continue
            
            sections_to_keep.remove(section)
            removed_sections.append(section.name)

            if self._tokenizer is not None:
                tmp_ctx = context.model_copy(deep=True)
                tmp_ctx.sections = list(sections_to_keep)
                current_tokens = svc.count_total_input_tokens(tmp_ctx)
            else:
                section_tokens = estimate_tokens_simple(section.content)
                current_tokens -= section_tokens

        # If still over budget and tokenizer is available, trim history events oldest-first.
        tmp_events = context.events
        if self._tokenizer is not None:
            tmp_ctx = context.model_copy(deep=True)
            tmp_ctx.sections = list(sections_to_keep)
            while tmp_ctx.events and current_tokens > available_budget:
                tmp_ctx.events = tmp_ctx.events[1:]
                events_trimmed += 1
                current_tokens = svc.count_total_input_tokens(tmp_ctx)
            tmp_events = tmp_ctx.events
        
        # Rebuild context with remaining sections
        new_context = Context(
            session_id=context.session_id,
            user_id=context.user_id,
            sections=sections_to_keep,
            memories=context.memories,
            events=tmp_events,
            system_instructions=context.system_instructions,
            current_query=context.current_query,
            total_token_budget=context.total_token_budget,
            reserved_output_tokens=context.reserved_output_tokens,
            metadata=context.metadata.copy(),
            expertise_items=context.expertise_items,
            expertise_items_used=context.expertise_items_used,
            expertise_id=context.expertise_id,
        )
        new_context.metadata["budget_trimmed"] = True
        new_context.metadata["budget_trimmed_removed_sections"] = removed_sections
        if events_trimmed:
            new_context.metadata["budget_trimmed_events_removed"] = events_trimmed

        if self._tokenizer is not None:
            after = svc.build_overview(new_context, total_budget=budget)
            new_context.metadata["token_breakdown"] = svc.overview_to_metadata(after)
            new_context.metadata["original_tokens"] = before.total_input_tokens
            new_context.metadata["trimmed_tokens"] = after.total_input_tokens
        else:
            new_context.metadata["original_tokens"] = context.estimate_total_tokens()
        
        return new_context
    
    def _format_memories(self, memories: List[MemoryItem]) -> str:
        """Format memories according to the configured style."""
        if not memories:
            return ""

        # Use progressive disclosure if enabled
        if self._use_progressive or self._memory_format == "progressive":
            return self._format_memories_progressive(memories)

        if self._memory_format == "bullet":
            return self._format_memories_bullet(memories)
        elif self._memory_format == "prose":
            return self._format_memories_prose(memories)
        elif self._memory_format == "json":
            return self._format_memories_json(memories)
        else:
            return self._format_memories_bullet(memories)
    
    def _format_memories_bullet(self, memories: List[MemoryItem]) -> str:
        """Format memories as bullet points."""
        lines = ["Relevant information about the user:"]
        for memory in memories:
            content = memory.content[:self._max_memory_chars]
            if len(memory.content) > self._max_memory_chars:
                content += "..."
            lines.append(f"• {content}")
        return "\n".join(lines)
    
    def _format_memories_prose(self, memories: List[MemoryItem]) -> str:
        """Format memories as prose."""
        parts = ["Here's what I know about the user:"]
        for memory in memories:
            content = memory.content[:self._max_memory_chars]
            parts.append(content)
        return " ".join(parts)
    
    def _format_memories_json(self, memories: List[MemoryItem]) -> str:
        """Format memories as JSON-like structure."""
        data = []
        for memory in memories:
            data.append({
                "type": memory.type.value,
                "content": memory.content[:self._max_memory_chars],
                "confidence": memory.confidence_score,
            })
        return "User memories:\n" + json.dumps(data, indent=2)

    def _format_memories_progressive(
        self,
        memories: List[MemoryItem],
        token_budget: Optional[int] = None,
    ) -> str:
        """
        Format memories using progressive disclosure.

        Shows headlines for all memories, expands top N to full content.
        Uses stored headlines if available, falls back to truncation.

        Args:
            memories: List of memories to format
            token_budget: Optional token budget for memory section

        Returns:
            Formatted string with progressive disclosure
        """
        if not memories:
            return ""

        # Build memory index from memories
        index = MemoryIndex(total_memories=len(memories))
        for memory in memories:
            entry = MemoryIndexEntry.from_memory(memory)
            index.add(entry)

        # Determine how many to expand based on budget
        expand_n = self._expand_top_n

        if token_budget is not None:
            # Try to fit within budget by adjusting expand_n
            while expand_n > 0:
                estimated = index.estimate_tokens(
                    level=DisclosureLevel.HEADLINE,
                    expand_top_n=expand_n,
                    max_entries=len(memories),
                )
                if estimated <= token_budget:
                    break
                expand_n -= 1

        return index.to_prompt(
            level=DisclosureLevel.HEADLINE,
            expand_top_n=expand_n,
            max_entries=len(memories),
        )

    def format_memories_from_index(
        self,
        index: "MemoryIndex",
        token_budget: Optional[int] = None,
    ) -> str:
        """
        Format memories from a pre-built MemoryIndex.

        Useful when headlines have been pre-generated via HeadlineService.

        Args:
            index: Pre-built MemoryIndex with entries
            token_budget: Optional token budget for memory section

        Returns:
            Formatted string with progressive disclosure
        """
        if not index.entries:
            return ""

        expand_n = self._expand_top_n

        if token_budget is not None:
            while expand_n > 0:
                estimated = index.estimate_tokens(
                    level=DisclosureLevel.HEADLINE,
                    expand_top_n=expand_n,
                    max_entries=len(index.entries),
                )
                if estimated <= token_budget:
                    break
                expand_n -= 1

        return index.to_prompt(
            level=DisclosureLevel.HEADLINE,
            expand_top_n=expand_n,
            max_entries=len(index.entries),
        )

    def _format_history(self, events: List[Event]) -> str:
        """Format conversation history."""
        if not events:
            return ""
        
        lines: List[str] = []
        for event in events:
            if event.type == EventType.SYSTEM:
                continue  # System events are handled separately
            lines.append(self._format_single_event(event))
        
        return "\n".join(lines)

    def _format_history_with_timeline(
        self,
        events: List[Event],
        time_range: Optional["TimeRange"] = None,
        max_events: int = 10,
    ) -> str:
        """
        Format conversation history with timeline context.

        Args:
            events: List of events to format
            time_range: Optional time range to filter events
            max_events: Maximum events to include

        Returns:
            Formatted history string with time context
        """
        if time_range:
            query = TimelineQuery(time_range=time_range, max_events=max_events)
            result = TimelineFilter.filter_events(events, query)
            events = result.events

        if not events:
            return ""

        lines = []

        # Add time context header
        if events:
            first_ts = events[0].timestamp
            last_ts = events[-1].timestamp
            if first_ts and last_ts:
                start = first_ts.strftime("%H:%M")
                end = last_ts.strftime("%H:%M")
                lines.append(f"Recent conversation ({start} - {end}):")

        for event in events[-max_events:]:
            lines.append(self._format_single_event(event))

        return "\n".join(lines)

    def _format_single_event(self, event: Event) -> str:
        """Format a single event for display."""
        if event.type == EventType.USER:
            prefix = "User"
        elif event.type == EventType.AGENT:
            prefix = "Assistant"
        elif event.type == EventType.TOOL_CALL:
            prefix = "Tool Call"
        elif event.type == EventType.TOOL_OUTPUT:
            prefix = "Tool Result"
        elif event.type == EventType.SYSTEM:
            prefix = "System"
        else:
            prefix = event.type.value

        note = event.get_intent_note()
        if note is not None:
            rendered = self._format_intent_note(note)
        else:
            rendered = event.content

        if self._include_timestamps and event.timestamp:
            timestamp = event.timestamp.strftime("%H:%M")
            return f"[{timestamp}] {prefix}: {rendered}"
        else:
            return f"{prefix}: {rendered}"

    def _format_intent_note(self, note: IntentNote) -> str:
        """Render an intent note into a compact, human-readable line."""
        parts: List[str] = []
        if note.act:
            parts.append(f"[act={note.act}]")
        if note.target:
            parts.append(f"[target={note.target}]")
        head = " ".join(parts)
        if head:
            return f"{head} {note.note_text}".strip()
        return note.note_text


@registry.register_assembler("minimal")
class MinimalContextAssembler(IContextAssembler):
    """
    Minimal context assembler for testing/simple use cases.
    
    Just combines system instructions with the most recent messages.
    """
    
    @property
    def name(self) -> str:
        return "minimal"
    
    async def assemble(
        self,
        session: Session,
        current_query: str,
        memories: List[MemoryItem],
        system_instructions: str = "",
        token_budget: int = 8000,
    ) -> Context:
        """Assemble minimal context."""
        builder = ContextBuilder(session.session_id, session.user_id)
        builder.with_token_budget(token_budget)
        builder.with_current_query(current_query)
        
        if system_instructions:
            builder.with_system_instructions(system_instructions)
            builder.with_section(
                name="system",
                content=system_instructions,
                priority=100,
                is_required=True,
            )
        
        # Just add the last few messages
        recent_events = session.events[-5:]
        if recent_events:
            history = "\n".join(
                f"{e.type.value}: {e.content}"
                for e in recent_events
            )
            builder.with_section(
                name="history",
                content=history,
                priority=50,
                is_required=False,
            )
        
        return builder.build()
    
    async def fit_to_budget(
        self,
        context: Context,
        budget: int,
    ) -> Context:
        """Minimal doesn't do budget fitting."""
        return context

