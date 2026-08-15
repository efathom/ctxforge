"""
Structured Summary - Schema for LLM-generated summaries.

Uses function calling to produce structured summaries with
specific fields for tracking conversation state.
"""

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType
from ctxforge.engine.registry import registry
from ctxforge.protocols.compactor import CompactionConfig


class StructuredSummary(BaseModel):
    """
    A structured representation summarizing conversation state.

    Used by StructuredSummarizingCondenser to produce typed summaries
    via LLM function calling.

    Fields are organized into categories:
    - Core context: user requirements, task tracking
    - Code-specific: files and functions modified
    - Testing: test status and errors
    - Session-specific: decisions and preferences

    Example:
        >>> summary = StructuredSummary(
        ...     user_context="User wants to build a REST API",
        ...     completed_tasks="Set up project structure, created models",
        ...     pending_tasks="Implement endpoints, add authentication",
        ... )
        >>> print(summary.to_prompt_format())
    """

    # Core context fields
    user_context: str = Field(
        default="",
        description="Essential user requirements, goals, and clarifications.",
    )

    completed_tasks: str = Field(
        default="",
        description="List of tasks completed so far with brief results.",
    )

    pending_tasks: str = Field(
        default="",
        description="List of tasks that still need to be done.",
    )

    current_state: str = Field(
        default="",
        description="Current variables, data structures, or relevant state.",
    )

    # Code-specific fields
    files_modified: str = Field(
        default="",
        description="List of files that have been created or modified.",
    )

    function_changes: str = Field(
        default="",
        description="List of functions that have been created or modified.",
    )

    # Test status fields
    tests_status: str = Field(
        default="",
        description="Whether tests are passing, failing, or unknown.",
    )

    error_messages: str = Field(
        default="",
        description="Key error messages encountered.",
    )

    # Session-specific fields
    key_decisions: str = Field(
        default="",
        description="Important decisions made during the conversation.",
    )

    user_preferences: str = Field(
        default="",
        description="User preferences or constraints mentioned.",
    )

    other_context: str = Field(
        default="",
        description="Any other important information.",
    )

    @classmethod
    def tool_definition(cls) -> Dict[str, Any]:
        """
        Generate OpenAI function/tool definition for structured generation.

        Returns dict compatible with OpenAI's tools parameter.

        Example:
            >>> tool = StructuredSummary.tool_definition()
            >>> response = await openai.chat.completions.create(
            ...     model="gpt-4",
            ...     messages=messages,
            ...     tools=[tool],
            ...     tool_choice={"type": "function", "function": {"name": "create_conversation_summary"}},
            ... )
        """
        properties = {}

        for field_name, field_info in cls.model_fields.items():
            description = field_info.description or ""
            properties[field_name] = {
                "type": "string",
                "description": description,
            }

        # Core fields are required
        required = ["user_context", "completed_tasks", "pending_tasks"]

        return {
            "type": "function",
            "function": {
                "name": "create_conversation_summary",
                "description": (
                    "Creates a structured summary of the conversation state "
                    "to preserve context when history grows too large."
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_format(self) -> str:
        """
        Format as markdown for prompt inclusion.

        Returns a human-readable markdown string suitable for
        including in LLM prompts.
        """
        sections = [
            "# Conversation Summary",
            "",
            "## Context",
            f"**User Context**: {self.user_context}",
            f"**Current State**: {self.current_state}",
            "",
            "## Tasks",
            f"**Completed**: {self.completed_tasks}",
            f"**Pending**: {self.pending_tasks}",
            "",
        ]

        # Add code sections if present
        if self.files_modified or self.function_changes:
            sections.extend([
                "## Code Changes",
                f"**Files Modified**: {self.files_modified}",
                f"**Functions Changed**: {self.function_changes}",
                "",
            ])

        # Add test status if present
        if self.tests_status or self.error_messages:
            sections.extend([
                "## Testing",
                f"**Status**: {self.tests_status}",
                f"**Errors**: {self.error_messages}",
                "",
            ])

        # Add decisions and preferences if present
        if self.key_decisions or self.user_preferences:
            sections.extend([
                "## Notes",
                f"**Key Decisions**: {self.key_decisions}",
                f"**User Preferences**: {self.user_preferences}",
                "",
            ])

        if self.other_context:
            sections.append(f"**Other**: {self.other_context}")

        return "\n".join(sections)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "StructuredSummary":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)

    def is_empty(self) -> bool:
        """Check if summary has no meaningful content."""
        return not any([
            self.user_context,
            self.completed_tasks,
            self.pending_tasks,
            self.current_state,
            self.files_modified,
            self.function_changes,
            self.tests_status,
            self.error_messages,
            self.key_decisions,
            self.user_preferences,
            self.other_context,
        ])


# Type alias for LLM function
LLMFunctionCallFunc = Any  # Callable that makes LLM function call


@registry.register_condenser("structured")
class StructuredSummarizingCondenser:
    """
    Condenser that produces structured summaries via function calling.

    Uses an LLM with function calling capability to generate a
    StructuredSummary from conversation events.

    Example:
        >>> async def call_llm(messages, tools):
        ...     response = await openai.chat.completions.create(
        ...         model="gpt-4",
        ...         messages=messages,
        ...         tools=tools,
        ...         tool_choice={"type": "function", "function": {"name": "create_conversation_summary"}},
        ...     )
        ...     return response.choices[0].message.tool_calls[0].function.arguments
        >>>
        >>> condenser = StructuredSummarizingCondenser(
        ...     llm_func=call_llm,
        ...     max_events=100,
        ... )
        >>> result = await condenser.condense(view, config)
    """

    def __init__(
        self,
        llm_func: Optional[LLMFunctionCallFunc] = None,
        max_events: int = 100,
        keep_first: int = 1,
        keep_last: int = 5,
        max_event_length: int = 10000,
    ):
        """
        Initialize the structured summarizing condenser.

        Args:
            llm_func: Async function to call LLM with function calling.
                      Should accept (messages, tools) and return JSON string.
                      If None, uses simple extraction fallback.
            max_events: Maximum events before triggering condensation
            keep_first: Number of first events to always keep (usually system)
            keep_last: Number of recent events to always keep
            max_event_length: Maximum length of each event content
        """
        self._llm = llm_func
        self._max_events = max_events
        self._keep_first = keep_first
        self._keep_last = keep_last
        self._max_event_length = max_event_length

    @property
    def name(self) -> str:
        return "structured_summarizing"

    def should_condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> bool:
        """
        Check if condensation is needed.

        Returns True if:
        - Event count exceeds max_events
        - View has an unhandled condensation request
        """
        if len(view) > self._max_events:
            return True
        if view.unhandled_condensation_request:
            return True
        return False

    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        """
        Condense using structured summary generation.

        Creates a StructuredSummary from the events to be condensed,
        keeps recent events, and returns the result.
        """
        config = config or CompactionConfig()
        events = list(view.events)

        if len(events) <= self._keep_first + self._keep_last:
            # Not enough events to condense
            return CondensationResult(
                view=view,
                summary_generated=False,
                tokens_saved=0,
                metadata={"strategy": self.name, "action": "no_op"},
            )

        # Split events into: keep_first, to_summarize, keep_last
        first_events = events[:self._keep_first]
        last_events = events[-self._keep_last:]
        middle_events = events[self._keep_first:-self._keep_last]

        if not middle_events:
            return CondensationResult(
                view=view,
                summary_generated=False,
                tokens_saved=0,
                metadata={"strategy": self.name, "action": "no_op"},
            )

        # Generate structured summary
        summary = await self._generate_summary(middle_events, view.summary)

        # Build new event list: first + kept last
        kept_events = first_events + last_events

        # Track forgotten events
        forgotten_ids = {e.event_id for e in middle_events}
        summary_text = summary.to_prompt_format()

        # Estimate tokens saved
        original_tokens = self._estimate_tokens(middle_events)
        summary_tokens = len(summary_text) // 4  # Rough estimate
        tokens_saved = max(0, original_tokens - summary_tokens)

        # Create new view
        new_view = view.with_forgotten(
            forgotten_ids,
            summary=summary_text,
        )
        new_view = new_view.with_events(kept_events)

        return CondensationResult(
            view=new_view,
            events_forgotten_start_id=middle_events[0].event_id,
            events_forgotten_end_id=middle_events[-1].event_id,
            summary_generated=True,
            tokens_saved=tokens_saved,
            metadata={
                "strategy": self.name,
                "events_summarized": len(middle_events),
                "structured_summary": summary.model_dump(),
            },
        )

    async def _generate_summary(
        self,
        events: List[Event],
        existing_summary: Optional[str] = None,
    ) -> StructuredSummary:
        """
        Generate a StructuredSummary from events.

        Uses LLM function calling if available, otherwise falls back
        to simple extraction.
        """
        if self._llm is not None:
            try:
                return await self._llm_generate_summary(events, existing_summary)
            except Exception:
                # Fall back to simple extraction on error
                pass

        return self._simple_extract_summary(events, existing_summary)

    async def _llm_generate_summary(
        self,
        events: List[Event],
        existing_summary: Optional[str] = None,
    ) -> StructuredSummary:
        """Generate summary using LLM function calling."""
        # Format events for LLM
        event_text = self._format_events_for_llm(events)

        # Build prompt
        system_message = (
            "You are a conversation summarizer. Analyze the conversation events "
            "and create a structured summary that preserves important context."
        )

        user_content = f"Summarize these conversation events:\n\n{event_text}"
        if existing_summary:
            user_content = (
                f"Previous summary:\n{existing_summary}\n\n"
                f"New events to incorporate:\n\n{event_text}"
            )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

        tools = [StructuredSummary.tool_definition()]

        # Call LLM
        result_json = await self._llm(messages, tools)

        # Parse result
        if isinstance(result_json, str):
            data = json.loads(result_json)
        else:
            data = result_json

        return StructuredSummary(**data)

    def _simple_extract_summary(
        self,
        events: List[Event],
        existing_summary: Optional[str] = None,
    ) -> StructuredSummary:
        """
        Simple extraction without LLM.

        Extracts key information from events using heuristics.
        """
        user_messages = []
        agent_messages = []
        tool_calls = []

        for event in events:
            content = event.content[:self._max_event_length]

            if event.type == EventType.USER:
                user_messages.append(content)
            elif event.type == EventType.AGENT:
                agent_messages.append(content)
            elif event.type == EventType.TOOL_CALL:
                tool_calls.append(content)

        # Extract user context from user messages
        user_context = ""
        if user_messages:
            # Take key parts from user messages
            user_context = "; ".join(
                msg[:200] for msg in user_messages[:5]
            )

        # Extract completed tasks from agent messages
        completed_tasks = ""
        if agent_messages:
            # Look for task-like content
            completed_tasks = f"Agent provided {len(agent_messages)} responses"

        # Pending tasks - hard to extract without LLM
        pending_tasks = ""

        # Files modified - look in tool calls
        files_modified = ""
        if tool_calls:
            files_modified = f"{len(tool_calls)} tool calls made"

        return StructuredSummary(
            user_context=user_context,
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
            files_modified=files_modified,
            other_context=existing_summary or "",
        )

    def _format_events_for_llm(self, events: List[Event]) -> str:
        """Format events as text for LLM consumption."""
        lines = []
        for event in events:
            role = event.type.value
            content = event.content[:self._max_event_length]
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    def _estimate_tokens(self, events: List[Event]) -> int:
        """Rough token estimate for events."""
        total_chars = sum(len(e.content) for e in events)
        return total_chars // 4  # Approximate tokens
