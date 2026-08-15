#!/usr/bin/env python3
"""
Compaction Enhancements End-to-End Demo

This script demonstrates all the compaction enhancements implemented in ctxforge:

1. CompactionView Abstraction - Immutable views with forgotten event tracking
2. ICondenser Protocol - New condenser interface replacing earlier compactor naming
3. Individual Condensers - SlidingWindow, Summarizing, Importance, Structured
4. CondenserPipeline - Chain multiple condensers together
5. StructuredSummary - LLM function calling for structured summaries
6. Factory Configuration - YAML-based condenser configuration

Run:
    cd /path/to/ctxforge
    python -m examples.compaction_enhancements_demo
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Ensure ctxforge is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ctxforge.compaction.importance import ImportanceCondenser
from ctxforge.compaction.pipeline import CondenserPipeline
from ctxforge.compaction.sliding_window import SlidingWindowCondenser
from ctxforge.compaction.structured_summary import (
    StructuredSummarizingCondenser,
    StructuredSummary,
)
from ctxforge.compaction.summarizing import SummarizingCondenser

# Compaction imports
from ctxforge.compaction.view import CompactionView, CondensationResult  # noqa: F401
from ctxforge.config.base import (
    CompactionConfig as CompactionConfigSchema,
)
from ctxforge.config.base import (
    CompactionStrategyType,
    CondenserStepConfig,
    EngineConfig,
)
from ctxforge.core.events import Event, EventType
from ctxforge.core.session import Session

# Factory imports
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import registry
from ctxforge.protocols.compactor import CompactionConfig


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def create_sample_session() -> Session:
    """Create a session with realistic conversation events."""
    session = Session(session_id="demo_session", user_id="demo_user")
    now = datetime.now(timezone.utc)

    # System message
    session.add_event(Event(
        event_id="sys_1",
        type=EventType.SYSTEM,
        content="You are a helpful coding assistant specializing in Python.",
        timestamp=now - timedelta(minutes=60),
    ))

    # Conversation about building an API
    events_data = [
        (EventType.USER, "I want to build a REST API for managing tasks."),
        (EventType.AGENT, "Great! I can help you build a REST API. "
                          "What language and framework would you like to use?"),
        (EventType.USER, "Let's use Python with FastAPI."),
        (EventType.AGENT, "Excellent choice! FastAPI is modern, fast, and has great "
                          "documentation. Let me set up the project structure."),
        (EventType.TOOL_CALL, "create_directory(path='task_api')"),
        (EventType.TOOL_OUTPUT, "Directory created: task_api"),
        (EventType.TOOL_CALL, "create_file(path='task_api/main.py', content='...')"),
        (EventType.TOOL_OUTPUT, "File created: task_api/main.py"),
        (EventType.USER, "Can you add a Task model with title, description, and status?"),
        (EventType.AGENT, "Sure! I'll add a Pydantic model for Task with validation."),
        (EventType.TOOL_CALL, "create_file(path='task_api/models.py', content='...')"),
        (EventType.TOOL_OUTPUT, "File created: task_api/models.py"),
        (EventType.USER, "Now add CRUD endpoints for tasks."),
        (EventType.AGENT, "I'll add create, read, update, and delete endpoints."),
        (EventType.TOOL_CALL, "edit_file(path='task_api/main.py', content='...')"),
        (EventType.TOOL_OUTPUT, "File modified: task_api/main.py - added CRUD endpoints"),
        (EventType.USER, "Add input validation for the task title."),
        (EventType.AGENT, "I'll add Pydantic validators to ensure title is not empty."),
        (EventType.TOOL_CALL, "edit_file(path='task_api/models.py', content='...')"),
        (EventType.TOOL_OUTPUT, "File modified: task_api/models.py - added validation"),
        (EventType.USER, "Please add JWT authentication."),
        (EventType.AGENT, "I'll implement JWT authentication using python-jose."),
        (EventType.TOOL_CALL, "edit_file(path='task_api/auth.py', content='...')"),
        (EventType.TOOL_OUTPUT, "File created: task_api/auth.py - JWT auth implemented"),
        (EventType.USER, "Run the tests to make sure everything works."),
        (EventType.AGENT, "Running pytest on the task_api module..."),
        (EventType.TOOL_CALL, "run_command(cmd='pytest task_api/tests/')"),
        (EventType.TOOL_OUTPUT, "All 12 tests passed! Coverage: 94%"),
    ]

    for i, (event_type, content) in enumerate(events_data):
        session.add_event(Event(
            event_id=f"evt_{i+1}",
            type=event_type,
            content=content,
            timestamp=now - timedelta(minutes=59 - i),
        ))

    return session


def create_large_session(event_count: int = 100) -> Session:
    """Create a session with many events for stress testing."""
    session = Session(session_id="large_session", user_id="demo_user")
    now = datetime.now(timezone.utc)

    # System message
    session.add_event(Event(
        event_id="sys_1",
        type=EventType.SYSTEM,
        content="You are a helpful assistant.",
        timestamp=now - timedelta(hours=2),
    ))

    # Generate many events
    for i in range(event_count):
        event_type = EventType.USER if i % 2 == 0 else EventType.AGENT
        session.add_event(Event(
            event_id=f"evt_{i}",
            type=event_type,
            content=f"Message {i}: Lorem ipsum dolor sit amet, "
                    f"consectetur adipiscing elit. Iteration {i}.",
            timestamp=now - timedelta(minutes=event_count - i),
        ))

    return session


async def demo_compaction_view():
    """Demonstrate CompactionView abstraction."""
    print_header("1. CompactionView Abstraction")

    session = create_sample_session()
    print(f"Original session has {len(session.events)} events")

    # Create view from session
    view = CompactionView.from_session(session)
    print(f"Created CompactionView with {len(view)} events")

    # Demonstrate immutability - forgetting events creates new view
    print_subheader("Forgetting Events (Immutable)")
    events_to_forget = {e.event_id for e in view.events[5:15]}
    new_view = view.with_forgotten(
        events_to_forget,
        summary="User requested FastAPI REST API. Project structure created."
    )

    print(f"Original view still has: {len(view)} events")
    print(f"New view has: {len(new_view)} events")
    print(f"Forgotten events tracked: {len(new_view.forgotten_event_ids)}")
    print(f"Summary: {new_view.summary[:60]}...")

    # Check forgotten status
    print_subheader("Forgotten Event Tracking")
    print(f"Is 'evt_5' forgotten? {new_view.is_forgotten('evt_5')}")
    print(f"Is 'evt_1' forgotten? {new_view.is_forgotten('evt_1')}")

    # Convert to context events
    context_events = new_view.to_context_events()
    print(f"Context events for LLM: {len(context_events)}")

    return view


async def demo_sliding_window_condenser():
    """Demonstrate SlidingWindowCondenser."""
    print_header("2. SlidingWindowCondenser")

    session = create_sample_session()
    view = CompactionView.from_session(session)
    print(f"Input: {len(view)} events")

    condenser = SlidingWindowCondenser()
    config = CompactionConfig(keep_recent=8)

    # Check if should condense
    should = condenser.should_condense(view, config)
    print(f"Should condense? {should}")

    # Condense
    result = await condenser.condense(view, config)

    print_subheader("Result")
    print(f"Output events: {len(result.view)}")
    print(f"Events forgotten: {result.events_forgotten_count}")
    print(f"Tokens saved: ~{result.tokens_saved}")

    # Show remaining events
    print_subheader("Remaining Event Types")
    event_types = {}
    for e in result.view.events:
        event_types[e.type.value] = event_types.get(e.type.value, 0) + 1
    for t, count in event_types.items():
        print(f"  {t}: {count}")


async def demo_importance_condenser():
    """Demonstrate ImportanceCondenser."""
    print_header("3. ImportanceCondenser")

    session = create_sample_session()
    view = CompactionView.from_session(session)
    print(f"Input: {len(view)} events")

    condenser = ImportanceCondenser(min_importance=0.5)
    config = CompactionConfig(keep_recent=5)

    result = await condenser.condense(view, config)

    print_subheader("Result")
    print(f"Output events: {len(result.view)}")
    print(f"Events forgotten: {result.events_forgotten_count}")

    # Show which events were kept
    print_subheader("Kept Events (by type)")
    for e in result.view.events[:10]:
        print(f"  [{e.type.value}] {e.content[:50]}...")


async def demo_summarizing_condenser():
    """Demonstrate SummarizingCondenser with mock LLM."""
    print_header("4. SummarizingCondenser")

    session = create_sample_session()
    view = CompactionView.from_session(session)
    print(f"Input: {len(view)} events")

    # Mock summarization function
    async def mock_summarizer(text: str, existing: Optional[str] = None) -> str:
        return (
            "User requested a FastAPI REST API for task management. "
            "Created project structure with models.py, main.py, and auth.py. "
            "Implemented Task model with Pydantic validation, CRUD endpoints, "
            "and JWT authentication. All 12 tests passed with 94% coverage."
        )

    condenser = SummarizingCondenser(mock_summarizer)
    config = CompactionConfig(keep_recent=5)

    result = await condenser.condense(view, config)

    print_subheader("Result")
    print(f"Output events: {len(result.view)}")
    print(f"Summary generated: {result.summary_generated}")
    print("\nGenerated Summary:")
    print(f"  {result.view.summary}")


async def demo_structured_summary():
    """Demonstrate StructuredSummary and StructuredSummarizingCondenser."""
    print_header("5. StructuredSummary & StructuredSummarizingCondenser")

    # Show StructuredSummary schema
    print_subheader("StructuredSummary Schema")
    summary = StructuredSummary(
        user_context="Building a REST API for task management",
        completed_tasks="Project setup, Task model, CRUD endpoints, JWT auth",
        pending_tasks="Add database persistence, deploy to cloud",
        files_modified="task_api/main.py, task_api/models.py, task_api/auth.py",
        tests_status="All 12 tests passing, 94% coverage",
        key_decisions="Using FastAPI + Pydantic, JWT for auth",
    )

    print(f"Fields: {list(StructuredSummary.model_fields.keys())}")
    print("\nPrompt Format:")
    print(summary.to_prompt_format())

    # Show tool definition for LLM function calling
    print_subheader("OpenAI Tool Definition")
    tool = StructuredSummary.tool_definition()
    print(f"Function name: {tool['function']['name']}")
    print(f"Parameters: {list(tool['function']['parameters']['properties'].keys())}")

    # Demonstrate condenser with mock LLM
    print_subheader("StructuredSummarizingCondenser")

    session = create_sample_session()
    view = CompactionView.from_session(session)

    # Mock LLM that returns structured summary
    async def mock_llm(messages, tools):
        return json.dumps({
            "user_context": "Building FastAPI REST API for task management",
            "completed_tasks": "Project structure, Task model, CRUD, JWT auth",
            "pending_tasks": "Database persistence, cloud deployment",
            "files_modified": "main.py, models.py, auth.py",
            "tests_status": "12 tests passing, 94% coverage",
        })

    condenser = StructuredSummarizingCondenser(
        llm_func=mock_llm,
        max_events=20,
        keep_first=1,
        keep_last=5,
    )

    result = await condenser.condense(view)

    print(f"Input events: {len(view)}")
    print(f"Output events: {len(result.view)}")
    print(f"Summary generated: {result.summary_generated}")


async def demo_condenser_pipeline():
    """Demonstrate CondenserPipeline with multiple stages."""
    print_header("6. CondenserPipeline")

    session = create_large_session(150)
    view = CompactionView.from_session(session)
    print(f"Input: {len(view)} events")

    # Mock summarizer
    async def mock_summarizer(text: str, existing: Optional[str] = None) -> str:
        return "Summary of earlier conversation about various topics."

    # Create multi-stage pipeline
    pipeline = CondenserPipeline(
        ImportanceCondenser(min_importance=0.3),
        SummarizingCondenser(mock_summarizer),
        SlidingWindowCondenser(),
    )

    print(f"Pipeline name: {pipeline.name}")
    print(f"Stages: {len(pipeline._condensers)}")

    config = CompactionConfig(keep_recent=15)
    result = await pipeline.condense(view, config)

    print_subheader("Result")
    print(f"Output events: {len(result.view)}")
    print(f"Total forgotten: {result.events_forgotten_count}")
    print(f"Tokens saved: ~{result.tokens_saved}")

    # Show pipeline metadata
    print_subheader("Pipeline Metadata")
    if "stages" in result.metadata:
        for i, stage in enumerate(result.metadata["stages"]):
            if isinstance(stage, dict):
                print(f"  Stage {i+1}: {stage.get('name', 'unknown')} - "
                      f"forgot {stage.get('events_forgotten', 0)} events")
            else:
                print(f"  Stage {i+1}: {stage}")


async def demo_factory_configuration():
    """Demonstrate factory-based condenser configuration."""
    print_header("7. Factory Configuration")

    factory = EngineFactory()

    # Demo 1: Sliding window from config
    print_subheader("Strategy: sliding_window")
    config1 = EngineConfig(
        compaction=CompactionConfigSchema(
            strategy=CompactionStrategyType.SLIDING_WINDOW,
            keep_recent=10,
        )
    )
    condenser1 = factory._create_condenser(config1)
    print(f"Created: {condenser1.name}")

    # Demo 2: Structured from config
    print_subheader("Strategy: structured")
    config2 = EngineConfig(
        compaction=CompactionConfigSchema(
            strategy=CompactionStrategyType.STRUCTURED,
            structured_max_events=50,
            structured_keep_first=2,
            structured_keep_last=8,
        )
    )
    condenser2 = factory._create_condenser(config2)
    print(f"Created: {condenser2.name}")
    print(f"  max_events: {condenser2._max_events}")
    print(f"  keep_first: {condenser2._keep_first}")
    print(f"  keep_last: {condenser2._keep_last}")

    # Demo 3: Pipeline from config
    print_subheader("Strategy: pipeline")
    config3 = EngineConfig(
        compaction=CompactionConfigSchema(
            strategy=CompactionStrategyType.PIPELINE,
            pipeline=[
                CondenserStepConfig(type="importance"),
                CondenserStepConfig(
                    type="structured",
                    config={"max_events": 75, "keep_first": 1, "keep_last": 10},
                ),
                CondenserStepConfig(type="sliding_window"),
            ],
        )
    )
    condenser3 = factory._create_condenser(config3)
    print(f"Created: {condenser3.name}")
    print(f"  Stages: {len(condenser3._condensers)}")
    for i, c in enumerate(condenser3._condensers):
        print(f"    {i+1}. {c.name}")

    # Show YAML equivalent
    print_subheader("Equivalent YAML Configuration")
    yaml_config = """
compaction:
  strategy: pipeline
  pipeline:
    - type: importance
    - type: structured
      config:
        max_events: 75
        keep_first: 1
        keep_last: 10
    - type: sliding_window
"""
    print(yaml_config)


async def demo_registry():
    """Demonstrate registry with all registered condensers."""
    print_header("8. Registry")

    print("Registered Condensers:")
    condensers = ["sliding_window", "summarizing", "importance", "structured", "pipeline"]

    for name in condensers:
        cls = registry.get_condenser(name)
        if cls:
            print(f"  ✓ {name}: {cls.__name__}")
        else:
            print(f"  ✗ {name}: NOT FOUND")


async def demo_session_integration():
    """Demonstrate full session integration."""
    print_header("9. Full Session Integration")

    # Create session
    session = create_sample_session()
    original_count = len(session.events)
    print(f"Original session: {original_count} events")

    # Create view
    view = CompactionView.from_session(session)

    # Condense with pipeline
    async def mock_summarizer(text: str, existing: Optional[str] = None) -> str:
        return "FastAPI task API with CRUD, validation, and JWT auth."

    pipeline = CondenserPipeline(
        SummarizingCondenser(mock_summarizer),
        SlidingWindowCondenser(),
    )

    config = CompactionConfig(keep_recent=8)
    result = await pipeline.condense(view, config)

    # Apply back to session
    session.events.clear()
    session.events.extend(result.view.to_context_events())
    session.summary = result.view.summary

    print_subheader("After Condensation")
    print(f"Session events: {len(session.events)}")
    print(f"Session summary: {session.summary}")
    print(f"Events condensed: {original_count - len(session.events)}")

    # Show remaining events
    print_subheader("Remaining Events")
    for i, e in enumerate(session.events[:5]):
        print(f"  {i+1}. [{e.type.value}] {e.content[:40]}...")
    if len(session.events) > 5:
        print(f"  ... and {len(session.events) - 5} more")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  CTXFORGE COMPACTION ENHANCEMENTS DEMO")
    print("=" * 70)

    try:
        await demo_compaction_view()
        await demo_sliding_window_condenser()
        await demo_importance_condenser()
        await demo_summarizing_condenser()
        await demo_structured_summary()
        await demo_condenser_pipeline()
        await demo_factory_configuration()
        await demo_registry()
        await demo_session_integration()

        print_header("DEMO COMPLETE")
        print("\nAll compaction enhancements demonstrated successfully!")
        print("\nKey Features:")
        print("  ✓ CompactionView - Immutable views with forgotten tracking")
        print("  ✓ ICondenser - New protocol replacing ICompactor")
        print("  ✓ SlidingWindowCondenser - FIFO event removal")
        print("  ✓ ImportanceCondenser - Score-based retention")
        print("  ✓ SummarizingCondenser - LLM summarization")
        print("  ✓ StructuredSummary - Typed summary schema")
        print("  ✓ StructuredSummarizingCondenser - Function calling summaries")
        print("  ✓ CondenserPipeline - Multi-stage condensation")
        print("  ✓ Factory Configuration - YAML-based setup")
        print("  ✓ Registry Integration - All condensers registered")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
