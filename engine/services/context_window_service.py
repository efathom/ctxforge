"""
Context window service.

Provides:
- Token counting for an assembled `Context` using a pluggable `ITokenizerProvider`
- A compact "overview" breakdown suitable for logging/metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ctxforge.core.context import Context, ContextSection
from ctxforge.core.events import Event
from ctxforge.protocols.llm import ChatMessage
from ctxforge.protocols.tokenizer import ITokenizerProvider


@dataclass(frozen=True)
class ContextWindowOverview:
    """High-level token accounting for a Context."""

    # budgets
    total_budget: int
    reserved_output_tokens: int
    available_input_budget: int

    # totals
    total_input_tokens: int

    # breakdowns (best-effort)
    system_tokens: int
    history_tokens: int
    current_query_tokens: int

    # system sub-breakdown (best-effort)
    system_instructions_tokens: int
    section_tokens: Dict[str, int]
    memories_tokens: int
    expertise_tokens: int

    # metadata
    model: Optional[str] = None
    tokenizer_name: Optional[str] = None


class ContextWindowService:
    """Token counting and budgeting utilities for `Context`."""

    def __init__(self, *, tokenizer: ITokenizerProvider, model: Optional[str] = None):
        self._tokenizer = tokenizer
        self._model = model

    def _format_section_for_system(self, section: ContextSection) -> str:
        # Mirrors `Context._build_system_content()` behavior.
        return f"[{section.name}]\n{section.content}"

    def _format_memories_for_system(self, context: Context) -> str:
        # Mirrors `Context._build_system_content()` behavior.
        if not context.memories:
            return ""
        lines = ["User Context (Long-term Memory):"]
        for mem in context.memories:
            lines.append(f"- {mem.to_prompt_format()}")
        return "\n".join(lines)

    def _format_expertise_for_system(self, context: Context) -> str:
        # Mirrors `Context._build_system_content()` behavior.
        # Note: `_format_expertise` is an internal helper on Context, but it is stable within this repo.
        if not context.expertise_items:
            return ""
        return context._format_expertise()

    def _system_content_parts(self, context: Context) -> Tuple[str, List[Tuple[str, str]], str, str]:
        """
        Returns:
            - system_instructions
            - list of (section_name, formatted_section_str)
            - formatted_expertise_str
            - formatted_memories_str
        """
        sys_instr = context.system_instructions or ""
        formatted_sections: List[Tuple[str, str]] = []
        for sec in context.sections:
            if sec.content:
                formatted_sections.append((sec.name, self._format_section_for_system(sec)))
        expertise_str = self._format_expertise_for_system(context)
        memories_str = self._format_memories_for_system(context)
        return sys_instr, formatted_sections, expertise_str, memories_str

    def _events_to_chat_messages(self, events: Sequence[Event]) -> List[ChatMessage]:
        msgs: List[ChatMessage] = []
        for e in events:
            role = None
            if e.type.value == "user":
                role = "user"
            elif e.type.value == "agent":
                role = "assistant"
            elif e.type.value == "tool_output":
                role = "tool"
            if role is None:
                continue
            msgs.append(ChatMessage(role=role, content=e.content))
        return msgs

    def to_chat_messages(self, context: Context) -> List[ChatMessage]:
        """Convert `Context` into `ChatMessage` list for token counting."""
        messages: List[ChatMessage] = []

        system_content = context._build_system_content()
        if system_content:
            messages.append(ChatMessage(role="system", content=system_content))

        messages.extend(self._events_to_chat_messages(context.events))

        if context.current_query:
            messages.append(ChatMessage(role="user", content=context.current_query))

        return messages

    def count_total_input_tokens(self, context: Context) -> int:
        messages = self.to_chat_messages(context)
        return int(self._tokenizer.count_message_tokens(messages, model=self._model))

    def build_overview(self, context: Context, *, total_budget: Optional[int] = None) -> ContextWindowOverview:
        """
        Compute a token budget overview for a fully-assembled `Context`.

        Why this exists:
        - `total_input_tokens` is counted by tokenizing the *actual chat messages* we would send
          (system + history + current query). This is the most accurate number for budgeting.
        - The rest of the fields are a best-effort breakdown that is helpful for debugging and
          observability (e.g., "are sections or history dominating the prompt?").

        Notes:
        - Token counting is tokenizer/model-dependent and not perfectly additive across parts.
          Expect minor mismatches between `total_input_tokens` and (system + history + query).
        """
        # Budget math: the engine reserves some tokens for the *model's output*, so the input-side
        # budget is what remains after subtracting `reserved_output_tokens`.
        budget = int(total_budget if total_budget is not None else context.total_token_budget)
        reserved = int(context.reserved_output_tokens)
        available = max(0, budget - reserved)

        # Full-message totals (most accurate): count tokens for the exact message list that would
        # be sent to the model (system + events + current_query).
        total_input = self.count_total_input_tokens(context)

        # Best-effort breakdown: tokenize system + history + query separately.
        #
        # This helps answer questions like:
        # - "Is system content too large?"
        # - "Is conversation history dominating?"
        # - "Which sections are the biggest contributors?"
        system_content = context._build_system_content()
        system_tokens = int(self._tokenizer.count_tokens(system_content, model=self._model)) if system_content else 0

        history_msgs = self._events_to_chat_messages(context.events)
        history_tokens = int(self._tokenizer.count_message_tokens(history_msgs, model=self._model)) if history_msgs else 0

        current_query_tokens = int(self._tokenizer.count_tokens(context.current_query or "", model=self._model)) if context.current_query else 0

        # Break down the system message into its main parts. This mirrors `Context._build_system_content()`
        # (and intentionally duplicates that formatting) so the numbers map to what the model sees.
        sys_instr, formatted_sections, expertise_str, memories_str = self._system_content_parts(context)
        section_tokens: Dict[str, int] = {}
        for name, s in formatted_sections:
            section_tokens[name] = int(self._tokenizer.count_tokens(s, model=self._model)) if s else 0

        return ContextWindowOverview(
            total_budget=budget,
            reserved_output_tokens=reserved,
            available_input_budget=available,
            total_input_tokens=total_input,
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            current_query_tokens=current_query_tokens,
            system_instructions_tokens=int(self._tokenizer.count_tokens(sys_instr, model=self._model)) if sys_instr else 0,
            section_tokens=section_tokens,
            memories_tokens=int(self._tokenizer.count_tokens(memories_str, model=self._model)) if memories_str else 0,
            expertise_tokens=int(self._tokenizer.count_tokens(expertise_str, model=self._model)) if expertise_str else 0,
            model=self._model,
            tokenizer_name=getattr(self._tokenizer, "name", None),
        )

    def overview_to_metadata(self, overview: ContextWindowOverview) -> Dict[str, Any]:
        """Serialize overview to a JSON-friendly dict for context.metadata."""
        return {
            "total_budget": overview.total_budget,
            "reserved_output_tokens": overview.reserved_output_tokens,
            "available_input_budget": overview.available_input_budget,
            "total_input_tokens": overview.total_input_tokens,
            "system_tokens": overview.system_tokens,
            "history_tokens": overview.history_tokens,
            "current_query_tokens": overview.current_query_tokens,
            "system_instructions_tokens": overview.system_instructions_tokens,
            "section_tokens": dict(overview.section_tokens),
            "memories_tokens": overview.memories_tokens,
            "expertise_tokens": overview.expertise_tokens,
            "model": overview.model,
            "tokenizer": overview.tokenizer_name,
        }


