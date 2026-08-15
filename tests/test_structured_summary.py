"""
Tests for StructuredSummary and StructuredSummarizingCondenser.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.compaction.structured_summary import (
    StructuredSummarizingCondenser,
    StructuredSummary,
)
from ctxforge.compaction.view import CompactionView, CondensationResult
from ctxforge.core.events import Event, EventType

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    now = datetime.now(timezone.utc)
    return [
        Event(
            event_id="evt_0",
            type=EventType.SYSTEM,
            content="You are a helpful assistant.",
            timestamp=now - timedelta(minutes=20),
        ),
        Event(
            event_id="evt_1",
            type=EventType.USER,
            content="I want to build a REST API for my project.",
            timestamp=now - timedelta(minutes=19),
        ),
        Event(
            event_id="evt_2",
            type=EventType.AGENT,
            content="I'd be happy to help you build a REST API. What framework would you like to use?",
            timestamp=now - timedelta(minutes=18),
        ),
        Event(
            event_id="evt_3",
            type=EventType.USER,
            content="Let's use FastAPI with Python.",
            timestamp=now - timedelta(minutes=17),
        ),
        Event(
            event_id="evt_4",
            type=EventType.AGENT,
            content="Great choice! FastAPI is excellent for building APIs. Let's start with the project structure.",
            timestamp=now - timedelta(minutes=16),
        ),
        Event(
            event_id="evt_5",
            type=EventType.TOOL_CALL,
            content="create_file(path='main.py', content='from fastapi import FastAPI')",
            timestamp=now - timedelta(minutes=15),
        ),
        Event(
            event_id="evt_6",
            type=EventType.TOOL_OUTPUT,
            content="File created successfully: main.py",
            timestamp=now - timedelta(minutes=14),
        ),
        Event(
            event_id="evt_7",
            type=EventType.USER,
            content="Can you add authentication?",
            timestamp=now - timedelta(minutes=13),
        ),
        Event(
            event_id="evt_8",
            type=EventType.AGENT,
            content="Sure! I'll add JWT authentication to your API.",
            timestamp=now - timedelta(minutes=12),
        ),
        Event(
            event_id="evt_9",
            type=EventType.USER,
            content="Perfect, also make sure tests pass.",
            timestamp=now - timedelta(minutes=11),
        ),
    ]


@pytest.fixture
def sample_view(sample_events):
    """Create a CompactionView from sample events."""
    return CompactionView(events=tuple(sample_events))


@pytest.fixture
def many_events():
    """Create many events to trigger condensation."""
    now = datetime.now(timezone.utc)
    events = []
    for i in range(120):
        event_type = EventType.USER if i % 2 == 0 else EventType.AGENT
        events.append(
            Event(
                event_id=f"evt_{i}",
                type=event_type,
                content=f"Message content number {i}",
                timestamp=now - timedelta(minutes=120 - i),
            )
        )
    return events


# =============================================================================
# Test StructuredSummary Schema
# =============================================================================


class TestStructuredSummary:
    """Tests for StructuredSummary Pydantic model."""

    def test_defaults(self):
        """Default values are empty strings."""
        summary = StructuredSummary()

        assert summary.user_context == ""
        assert summary.completed_tasks == ""
        assert summary.pending_tasks == ""
        assert summary.current_state == ""
        assert summary.files_modified == ""
        assert summary.function_changes == ""
        assert summary.tests_status == ""
        assert summary.error_messages == ""
        assert summary.key_decisions == ""
        assert summary.user_preferences == ""
        assert summary.other_context == ""

    def test_with_values(self):
        """Can create summary with values."""
        summary = StructuredSummary(
            user_context="Build a REST API",
            completed_tasks="Set up project",
            pending_tasks="Add authentication",
            current_state="Project initialized",
            files_modified="main.py, models.py",
            tests_status="Passing",
        )

        assert summary.user_context == "Build a REST API"
        assert summary.completed_tasks == "Set up project"
        assert summary.pending_tasks == "Add authentication"
        assert summary.files_modified == "main.py, models.py"

    def test_is_empty(self):
        """is_empty returns True for empty summary."""
        empty = StructuredSummary()
        assert empty.is_empty() is True

        non_empty = StructuredSummary(user_context="Something")
        assert non_empty.is_empty() is False

    def test_to_json(self):
        """Can serialize to JSON."""
        summary = StructuredSummary(
            user_context="Build API",
            completed_tasks="Setup done",
        )

        json_str = summary.to_json()
        data = json.loads(json_str)

        assert data["user_context"] == "Build API"
        assert data["completed_tasks"] == "Setup done"

    def test_from_json(self):
        """Can deserialize from JSON."""
        json_str = json.dumps({
            "user_context": "Build API",
            "completed_tasks": "Setup done",
            "pending_tasks": "",
            "current_state": "",
            "files_modified": "",
            "function_changes": "",
            "tests_status": "",
            "error_messages": "",
            "key_decisions": "",
            "user_preferences": "",
            "other_context": "",
        })

        summary = StructuredSummary.from_json(json_str)

        assert summary.user_context == "Build API"
        assert summary.completed_tasks == "Setup done"


class TestStructuredSummaryToolDefinition:
    """Tests for StructuredSummary.tool_definition."""

    def test_tool_definition_structure(self):
        """Tool definition has correct structure."""
        tool = StructuredSummary.tool_definition()

        assert tool["type"] == "function"
        assert "function" in tool
        assert tool["function"]["name"] == "create_conversation_summary"
        assert "parameters" in tool["function"]

    def test_tool_definition_parameters(self):
        """Tool definition has all field parameters."""
        tool = StructuredSummary.tool_definition()
        params = tool["function"]["parameters"]

        assert params["type"] == "object"
        assert "properties" in params

        # Check all fields are present
        expected_fields = [
            "user_context",
            "completed_tasks",
            "pending_tasks",
            "current_state",
            "files_modified",
            "function_changes",
            "tests_status",
            "error_messages",
            "key_decisions",
            "user_preferences",
            "other_context",
        ]

        for field in expected_fields:
            assert field in params["properties"]
            assert params["properties"][field]["type"] == "string"
            assert "description" in params["properties"][field]

    def test_tool_definition_required_fields(self):
        """Tool definition has correct required fields."""
        tool = StructuredSummary.tool_definition()
        required = tool["function"]["parameters"]["required"]

        assert "user_context" in required
        assert "completed_tasks" in required
        assert "pending_tasks" in required


class TestStructuredSummaryPromptFormat:
    """Tests for StructuredSummary.to_prompt_format."""

    def test_basic_prompt_format(self):
        """Basic prompt format includes core sections."""
        summary = StructuredSummary(
            user_context="Build a REST API",
            completed_tasks="Set up project",
            pending_tasks="Add authentication",
            current_state="Project initialized",
        )

        output = summary.to_prompt_format()

        assert "# Conversation Summary" in output
        assert "## Context" in output
        assert "**User Context**: Build a REST API" in output
        assert "**Current State**: Project initialized" in output
        assert "## Tasks" in output
        assert "**Completed**: Set up project" in output
        assert "**Pending**: Add authentication" in output

    def test_prompt_format_with_code_changes(self):
        """Prompt format includes code sections when present."""
        summary = StructuredSummary(
            user_context="Build API",
            completed_tasks="Done",
            pending_tasks="More",
            files_modified="main.py, utils.py",
            function_changes="create_user, get_user",
        )

        output = summary.to_prompt_format()

        assert "## Code Changes" in output
        assert "**Files Modified**: main.py, utils.py" in output
        assert "**Functions Changed**: create_user, get_user" in output

    def test_prompt_format_without_code_changes(self):
        """Prompt format omits code section when empty."""
        summary = StructuredSummary(
            user_context="Build API",
            completed_tasks="Done",
            pending_tasks="More",
        )

        output = summary.to_prompt_format()

        assert "## Code Changes" not in output

    def test_prompt_format_with_tests(self):
        """Prompt format includes test sections when present."""
        summary = StructuredSummary(
            user_context="Build API",
            completed_tasks="Done",
            pending_tasks="More",
            tests_status="5 passing, 2 failing",
            error_messages="AssertionError in test_user",
        )

        output = summary.to_prompt_format()

        assert "## Testing" in output
        assert "**Status**: 5 passing, 2 failing" in output
        assert "**Errors**: AssertionError in test_user" in output

    def test_prompt_format_with_notes(self):
        """Prompt format includes notes when present."""
        summary = StructuredSummary(
            user_context="Build API",
            completed_tasks="Done",
            pending_tasks="More",
            key_decisions="Use PostgreSQL for database",
            user_preferences="Prefer async/await pattern",
        )

        output = summary.to_prompt_format()

        assert "## Notes" in output
        assert "**Key Decisions**: Use PostgreSQL" in output
        assert "**User Preferences**: Prefer async/await" in output

    def test_prompt_format_with_other_context(self):
        """Prompt format includes other context when present."""
        summary = StructuredSummary(
            user_context="Build API",
            completed_tasks="Done",
            pending_tasks="More",
            other_context="Using Docker for deployment",
        )

        output = summary.to_prompt_format()

        assert "**Other**: Using Docker for deployment" in output


# =============================================================================
# Test StructuredSummarizingCondenser
# =============================================================================


class TestStructuredSummarizingCondenser:
    """Tests for StructuredSummarizingCondenser."""

    def test_name_property(self):
        """Has correct name."""
        condenser = StructuredSummarizingCondenser()
        assert condenser.name == "structured_summarizing"

    def test_should_condense_by_event_count(self, sample_view):
        """should_condense returns True when over max_events."""
        condenser = StructuredSummarizingCondenser(max_events=5)

        assert len(sample_view) > 5
        assert condenser.should_condense(sample_view)

    def test_should_not_condense_under_max(self, sample_view):
        """should_condense returns False when under max_events."""
        condenser = StructuredSummarizingCondenser(max_events=100)

        assert len(sample_view) < 100
        assert not condenser.should_condense(sample_view)

    def test_should_condense_on_request(self, sample_view):
        """should_condense returns True for unhandled condensation request."""
        view_with_request = CompactionView(
            events=sample_view.events,
            unhandled_condensation_request=True,
        )
        condenser = StructuredSummarizingCondenser(max_events=100)

        assert condenser.should_condense(view_with_request)

    @pytest.mark.asyncio
    async def test_condense_returns_result(self, sample_view):
        """condense returns CondensationResult."""
        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(sample_view)

        assert isinstance(result, CondensationResult)

    @pytest.mark.asyncio
    async def test_condense_keeps_first_and_last(self, sample_view):
        """condense keeps first and last events."""
        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(sample_view)

        kept_events = list(result.view.events)

        # First event should be kept
        assert kept_events[0].event_id == "evt_0"

        # Last 2 events should be kept
        assert kept_events[-1].event_id == "evt_9"
        assert kept_events[-2].event_id == "evt_8"

    @pytest.mark.asyncio
    async def test_condense_generates_summary(self, sample_view):
        """condense generates a summary."""
        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(sample_view)

        assert result.summary_generated is True
        assert result.view.summary is not None
        assert len(result.view.summary) > 0

    @pytest.mark.asyncio
    async def test_condense_tracks_forgotten_events(self, sample_view):
        """condense tracks which events were forgotten."""
        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(sample_view)

        # Middle events should be forgotten
        assert result.events_forgotten_start_id is not None
        assert result.events_forgotten_end_id is not None

    @pytest.mark.asyncio
    async def test_condense_no_op_when_not_enough_events(self):
        """condense returns no-op when not enough events to condense."""
        events = [
            Event(event_id="1", type=EventType.USER, content="Hello"),
            Event(event_id="2", type=EventType.AGENT, content="Hi"),
        ]
        view = CompactionView(events=tuple(events))

        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(view)

        assert result.summary_generated is False
        assert result.metadata.get("action") == "no_op"

    @pytest.mark.asyncio
    async def test_condense_includes_metadata(self, sample_view):
        """condense includes metadata about condensation."""
        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(sample_view)

        assert "strategy" in result.metadata
        assert result.metadata["strategy"] == "structured_summarizing"
        assert "events_summarized" in result.metadata


class TestStructuredSummarizingCondenserWithMockLLM:
    """Tests for StructuredSummarizingCondenser with mock LLM."""

    @pytest.mark.asyncio
    async def test_uses_llm_when_provided(self, sample_view):
        """Uses LLM function when provided."""
        llm_called = False

        async def mock_llm(messages, tools):
            nonlocal llm_called
            llm_called = True
            return json.dumps({
                "user_context": "User wants to build a REST API with FastAPI",
                "completed_tasks": "Set up project structure",
                "pending_tasks": "Add authentication, write tests",
            })

        condenser = StructuredSummarizingCondenser(
            llm_func=mock_llm,
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(sample_view)

        assert llm_called is True
        assert "REST API" in result.view.summary
        assert "FastAPI" in result.view.summary

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self, sample_view):
        """Falls back to simple extraction on LLM error."""

        async def failing_llm(messages, tools):
            raise Exception("LLM API error")

        condenser = StructuredSummarizingCondenser(
            llm_func=failing_llm,
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        # Should not raise, should fall back
        result = await condenser.condense(sample_view)

        assert isinstance(result, CondensationResult)
        assert result.summary_generated is True

    @pytest.mark.asyncio
    async def test_llm_receives_correct_tool(self, sample_view):
        """LLM receives correct tool definition."""
        received_tools = None

        async def capture_llm(messages, tools):
            nonlocal received_tools
            received_tools = tools
            return json.dumps({
                "user_context": "Test",
                "completed_tasks": "Test",
                "pending_tasks": "Test",
            })

        condenser = StructuredSummarizingCondenser(
            llm_func=capture_llm,
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        await condenser.condense(sample_view)

        assert received_tools is not None
        assert len(received_tools) == 1
        assert received_tools[0]["function"]["name"] == "create_conversation_summary"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestStructuredSummaryEdgeCases:
    """Edge case tests."""

    def test_summary_with_special_characters(self):
        """Handles special characters in fields."""
        summary = StructuredSummary(
            user_context="Build API with <script> tags & \"quotes\"",
            completed_tasks="Done: 50% complete",
        )

        output = summary.to_prompt_format()

        assert "<script>" in output
        assert "&" in output

    def test_summary_with_unicode(self):
        """Handles unicode in fields."""
        summary = StructuredSummary(
            user_context="Build API für Benutzer 用户",
            completed_tasks="Done ✓",
        )

        output = summary.to_prompt_format()

        assert "für" in output
        assert "用户" in output
        assert "✓" in output

    @pytest.mark.asyncio
    async def test_condense_with_empty_view(self):
        """Handles empty view gracefully."""
        view = CompactionView(events=())
        condenser = StructuredSummarizingCondenser(max_events=5)

        result = await condenser.condense(view)

        assert isinstance(result, CondensationResult)
        assert result.summary_generated is False

    @pytest.mark.asyncio
    async def test_condense_preserves_view_metadata(self, sample_view):
        """Preserves existing view metadata."""
        view_with_data = CompactionView(
            events=sample_view.events,
            forgotten_event_ids=frozenset(["old_evt_1"]),
        )

        condenser = StructuredSummarizingCondenser(
            max_events=5,
            keep_first=1,
            keep_last=2,
        )

        result = await condenser.condense(view_with_data)

        # Original forgotten events should still be tracked
        assert "old_evt_1" in result.view.forgotten_event_ids
