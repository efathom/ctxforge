"""
Context Container - Assembled prompt context for LLM invocation.

The Context object represents the fully prepared context that will be
sent to the LLM, including system instructions, memories, expertise,
conversation history, and any other relevant information.
"""

import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ctxforge.core.events import Event
from ctxforge.core.expertise import ExpertiseItem
from ctxforge.core.memory import MemoryItem
from ctxforge.utils.budget_packer import budget_pack


class ContextSection(BaseModel):
    """A section of the context with its content and metadata."""
    
    name: str
    content: str
    priority: int = 0  # Higher priority sections appear first
    token_estimate: Optional[int] = None
    is_required: bool = True  # Whether this section can be pruned
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Context(BaseModel):
    """
    Assembled Context Container.
    
    Represents the complete context that will be assembled into a prompt
    for the LLM. Supports structured sections with priority ordering
    and token budget management.
    
    Attributes:
        session_id: The session this context belongs to
        user_id: The user this context is for
        sections: Ordered list of context sections
        current_query: The current user query
        memories: Retrieved memories to include
        events: Recent events from session history
        system_instructions: System prompt/instructions
        total_token_budget: Maximum tokens allowed
        reserved_output_tokens: Tokens reserved for response
        created_at: When the context was created
        metadata: Additional context metadata
    """
    
    session_id: str
    user_id: str
    sections: List[ContextSection] = Field(default_factory=list)
    current_query: str = ""
    memories: List[MemoryItem] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    system_instructions: str = ""
    total_token_budget: int = 8000
    reserved_output_tokens: int = 1000
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Expertise fields
    expertise_items: List[ExpertiseItem] = Field(default_factory=list)
    expertise_items_used: List[str] = Field(default_factory=list)  # Item IDs used in this turn
    expertise_id: Optional[str] = None  # ID of the expertise being used

    def add_section(
        self,
        name: str,
        content: str,
        priority: int = 0,
        is_required: bool = True,
        token_estimate: Optional[int] = None,
    ) -> None:
        """Add a section to the context."""
        section = ContextSection(
            name=name,
            content=content,
            priority=priority,
            is_required=is_required,
            token_estimate=token_estimate,
        )
        self.sections.append(section)
        # Re-sort by priority (descending)
        self.sections.sort(key=lambda s: s.priority, reverse=True)
    
    def get_section(self, name: str) -> Optional[ContextSection]:
        """Get a section by name."""
        for section in self.sections:
            if section.name == name:
                return section
        return None
    
    def remove_section(self, name: str) -> bool:
        """Remove a section by name. Returns True if found and removed."""
        for i, section in enumerate(self.sections):
            if section.name == name:
                self.sections.pop(i)
                return True
        return False
    
    def update_section(self, name: str, content: str) -> bool:
        """Update a section's content. Returns True if found and updated."""
        section = self.get_section(name)
        if section:
            section.content = content
            return True
        return False
    
    def get_available_tokens(self) -> int:
        """Calculate available tokens for content."""
        return self.total_token_budget - self.reserved_output_tokens
    
    def estimate_total_tokens(self) -> int:
        """Estimate total tokens used (rough approximation)."""
        total = 0
        
        # System instructions
        total += len(self.system_instructions.split()) * 1.3
        
        # Sections
        for section in self.sections:
            if section.token_estimate:
                total += section.token_estimate
            else:
                total += len(section.content.split()) * 1.3
        
        # Expertise items
        for item in self.expertise_items:
            content = item.content if hasattr(item, 'content') else str(item)
            total += len(content.split()) * 1.3
        
        # Memories
        for memory in self.memories:
            total += len(memory.content.split()) * 1.3
        
        # Events
        for event in self.events:
            total += len(event.content.split()) * 1.3
        
        # Current query
        total += len(self.current_query.split()) * 1.3
        
        return int(total)
    
    def is_within_budget(self) -> bool:
        """Check if the context is within the token budget."""
        return self.estimate_total_tokens() <= self.get_available_tokens()

    def priority_pack_sections(
        self,
        budget: int,
        token_fn: Optional[Callable[[str], int]] = None,
    ) -> List[ContextSection]:
        """Pack sections by priority into a token budget using greedy packing.

        Sections are sorted by priority (descending).  Required sections are
        always included first, then optional sections fill the remaining budget.

        Args:
            budget: Maximum token budget for sections.
            token_fn: Optional custom token counting function.

        Returns:
            Subset of sections that fit within the budget.
        """
        required = [s for s in self.sections if s.is_required]
        optional = sorted(
            [s for s in self.sections if not s.is_required],
            key=lambda s: s.priority,
            reverse=True,
        )

        # Required sections always included; deduct their cost first.
        count_fn = token_fn or (lambda t: len(t.split()))
        required_cost = sum(count_fn(s.content) for s in required)
        remaining = max(0, budget - required_cost)

        packed_optional = budget_pack(
            items=optional,
            budget=remaining,
            text_fn=lambda s: s.content,
            token_fn=token_fn,
        )

        return required + packed_optional
    
    def _format_expertise(self) -> str:
        """
        Format expertise items in ACE-style for prompt inclusion.
        
        Groups items by section and formats each as:
        [section] helpful=X harmful=Y :: content
        """
        if not self.expertise_items:
            return ""
        
        # Group by section
        by_section: Dict[str, List[ExpertiseItem]] = {}
        for item in self.expertise_items:
            section_key = item.section.to_display_name() if hasattr(item.section, 'to_display_name') else str(item.section)
            if section_key not in by_section:
                by_section[section_key] = []
            by_section[section_key].append(item)
        
        # Format output
        lines = ["Expertise Knowledge:"]
        for section_name, items in sorted(by_section.items()):
            lines.append(f"\n## {section_name}")
            for item in items:
                if hasattr(item, 'to_prompt_format'):
                    lines.append(f"- {item.to_prompt_format()}")
                else:
                    lines.append(f"- {item.content}")
        
        return "\n".join(lines)
    
    def mark_expertise_used(self, item_ids: List[str]) -> None:
        """Mark expertise items as used in this turn."""
        self.expertise_items_used.extend(item_ids)
    
    def to_prompt(self) -> str:
        """
        Assemble the context into a prompt string.
        
        This is a basic implementation that can be overridden by
        custom prompt builders.
        """
        parts = []
        
        # 1. System instructions
        if self.system_instructions:
            parts.append(f"System: {self.system_instructions}")
        
        # 2. Sections (in priority order)
        for section in self.sections:
            if section.content:
                parts.append(f"[{section.name}]\n{section.content}")
        
        # 3. Expertise items
        if self.expertise_items:
            parts.append(self._format_expertise())
        
        # 4. Memories
        if self.memories:
            parts.append("User Context (Long-term Memory):")
            for memory in self.memories:
                parts.append(f"- {memory.to_prompt_format()}")
        
        # 5. Recent events/history
        if self.events:
            parts.append("Recent History:")
            for event in self.events:
                parts.append(event.to_prompt_format())
        
        # 6. Current query
        if self.current_query:
            parts.append(f"USER: {self.current_query}")
            parts.append("AGENT:")
        
        return "\n".join(parts)
    
    def _build_system_content(self) -> str:
        """Build the system message content."""
        parts = []
        
        if self.system_instructions:
            parts.append(self.system_instructions)
        
        # Add sections
        for section in self.sections:
            if section.content:
                parts.append(f"[{section.name}]\n{section.content}")
        
        # Add expertise
        if self.expertise_items:
            parts.append(self._format_expertise())
        
        # Add memories
        if self.memories:
            memory_parts = ["User Context (Long-term Memory):"]
            for memory in self.memories:
                memory_parts.append(f"- {memory.to_prompt_format()}")
            parts.append("\n".join(memory_parts))
        
        return "\n\n".join(parts)
    
    def to_messages(self) -> List[Dict[str, str]]:
        """
        Convert context to generic chat message format.
        
        Returns a list of message dicts with 'role' and 'content' keys.
        Compatible with most LLM APIs.
        """
        messages = []
        
        # System message
        system_content = self._build_system_content()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # Conversation history
        for event in self.events:
            if event.type.value == "user":
                messages.append({"role": "user", "content": event.content})
            elif event.type.value == "agent":
                messages.append({"role": "assistant", "content": event.content})
            elif event.type.value == "tool_output":
                # Include tool outputs in history
                messages.append({
                    "role": "tool",
                    "content": event.content,
                    "tool_call_id": event.metadata.custom.get("tool_call_id", ""),
                })
        
        # Current query
        if self.current_query:
            messages.append({"role": "user", "content": self.current_query})
        
        return messages
    
    # =========================================================================
    # FORMAT CONVERTERS FOR DIFFERENT PROVIDERS
    # =========================================================================
    
    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """
        Convert to OpenAI chat completion format.
        
        Returns messages compatible with openai.chat.completions.create()
        
        Example:
            context = await engine.prepare_context(...)
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=context.to_openai_messages(),
            )
        """
        return self.to_messages()
    
    def to_anthropic_messages(self) -> tuple:
        """
        Convert to Anthropic messages format.
        
        Returns (system_prompt, messages) tuple for Anthropic API.
        Anthropic requires system prompt separate from messages.
        
        Example:
            system, messages = context.to_anthropic_messages()
            response = anthropic.messages.create(
                model="claude-3-opus-20240229",
                system=system,
                messages=messages,
            )
        """
        system_content = self._build_system_content()
        
        messages = []
        for event in self.events:
            if event.type.value == "user":
                messages.append({"role": "user", "content": event.content})
            elif event.type.value == "agent":
                messages.append({"role": "assistant", "content": event.content})
        
        if self.current_query:
            messages.append({"role": "user", "content": self.current_query})
        
        return system_content, messages
    
    def to_langchain_messages(self) -> List[Any]:
        """
        Convert to LangChain message format.
        
        Returns list of LangChain message objects.
        Requires langchain-core to be installed.
        
        Example:
            messages = context.to_langchain_messages()
            response = await llm.ainvoke(messages)
        """
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except ImportError:
            raise ImportError(
                "langchain-core is required for to_langchain_messages(). "
                "Install with: pip install langchain-core"
            ) from None
        
        messages = []
        
        # System message
        system_content = self._build_system_content()
        if system_content:
            messages.append(SystemMessage(content=system_content))
        
        # Conversation history
        for event in self.events:
            if event.type.value == "user":
                messages.append(HumanMessage(content=event.content))
            elif event.type.value == "agent":
                messages.append(AIMessage(content=event.content))
            elif event.type.value == "tool_output":
                messages.append(ToolMessage(
                    content=event.content,
                    tool_call_id=event.metadata.custom.get("tool_call_id", ""),
                ))
        
        # Current query
        if self.current_query:
            messages.append(HumanMessage(content=self.current_query))
        
        return messages
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to a dictionary.
        
        Useful for serialization or custom processing.
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "system_instructions": self.system_instructions,
            "sections": [s.model_dump() for s in self.sections],
            "memories": [m.model_dump() for m in self.memories],
            "events": [e.model_dump() for e in self.events],
            "current_query": self.current_query,
            "metadata": self.metadata,
            "expertise_id": self.expertise_id,
            "expertise_items": [i.model_dump() if hasattr(i, 'model_dump') else i for i in self.expertise_items],
            "expertise_items_used": self.expertise_items_used,
        }


class ContextBuilder:
    """Builder pattern for creating context objects with fluent API."""
    
    def __init__(self, session_id: str, user_id: str):
        self._session_id = session_id
        self._user_id = user_id
        self._sections: List[ContextSection] = []
        self._memories: List[MemoryItem] = []
        self._events: List[Event] = []
        self._system_instructions = ""
        self._current_query = ""
        self._token_budget = 8000
        self._output_tokens = 1000
        self._metadata: Dict[str, Any] = {}
        self._expertise_items: List[ExpertiseItem] = []
        self._expertise_items_used: List[str] = []
        self._expertise_id: Optional[str] = None
    
    def with_system_instructions(self, instructions: str) -> "ContextBuilder":
        """Set system instructions."""
        self._system_instructions = instructions
        return self
    
    def with_section(
        self,
        name: str,
        content: str,
        priority: int = 0,
        is_required: bool = True,
    ) -> "ContextBuilder":
        """Add a context section."""
        self._sections.append(ContextSection(
            name=name,
            content=content,
            priority=priority,
            is_required=is_required,
        ))
        return self
    
    def with_memories(self, memories: List[MemoryItem]) -> "ContextBuilder":
        """Add memories to context."""
        self._memories.extend(memories)
        return self
    
    def with_events(self, events: List[Event]) -> "ContextBuilder":
        """Add events to context."""
        self._events.extend(events)
        return self
    
    def with_current_query(self, query: str) -> "ContextBuilder":
        """Set the current user query."""
        self._current_query = query
        return self
    
    def with_token_budget(
        self,
        total_budget: int,
        reserved_output: int = 1000,
    ) -> "ContextBuilder":
        """Set token budget constraints."""
        self._token_budget = total_budget
        self._output_tokens = reserved_output
        return self
    
    def with_metadata(self, key: str, value: Any) -> "ContextBuilder":
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def with_expertise_items(self, items: List[ExpertiseItem]) -> "ContextBuilder":
        """Add expertise items to context."""
        self._expertise_items.extend(items)
        return self
    
    def with_expertise_id(self, expertise_id: str) -> "ContextBuilder":
        """Set the expertise ID being used."""
        self._expertise_id = expertise_id
        return self
    
    def with_expertise_items_used(self, item_ids: List[str]) -> "ContextBuilder":
        """Mark expertise items as used."""
        self._expertise_items_used.extend(item_ids)
        return self
    
    def build(self) -> Context:
        """Build and return the context."""
        context = Context(
            session_id=self._session_id,
            user_id=self._user_id,
            sections=sorted(self._sections, key=lambda s: s.priority, reverse=True),
            memories=self._memories,
            events=self._events,
            system_instructions=self._system_instructions,
            current_query=self._current_query,
            total_token_budget=self._token_budget,
            reserved_output_tokens=self._output_tokens,
            metadata=self._metadata,
            expertise_items=self._expertise_items,
            expertise_items_used=self._expertise_items_used,
            expertise_id=self._expertise_id,
        )
        return context

