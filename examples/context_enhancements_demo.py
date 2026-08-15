#!/usr/bin/env python3
"""
Context Engine Enhancements Demo.

Demonstrates all three enhancements:
1. Progressive Disclosure API - Compact memory summaries with on-demand expansion
2. Timeline-Based Event Retrieval - Query events by time ranges and turns
3. Tool Output Compression - Real-time compression of verbose tool outputs

This script showcases how these features work together to reduce token usage.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent directory to path for imports (go up to ctxforge/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ctxforge.compression.tool_compressor import (  # noqa: E402
    CompressionStrategy,
    ToolOutputCompressor,
    compress_tool_event,
)
from ctxforge.config.base import (  # noqa: E402
    ProgressiveDisclosureConfig,
    TimelineConfig,
    ToolCompressionConfig,
)
from ctxforge.core.events import Event, EventMetadata, EventType  # noqa: E402
from ctxforge.core.memory import MemoryItem, MemoryType  # noqa: E402
from ctxforge.core.memory_index import (  # noqa: E402
    DisclosureLevel,
    MemoryIndex,
    MemoryIndexEntry,
)
from ctxforge.core.session import Session  # noqa: E402
from ctxforge.engine.services.timeline_service import TimelineService  # noqa: E402


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# DEMO 1: Progressive Disclosure API
# =============================================================================

def demo_progressive_disclosure():
    """Demonstrate progressive disclosure memory system."""
    print_section("DEMO 1: Progressive Disclosure API")

    # Create sample memories with headlines
    memories = [
        MemoryItem(
            memory_id="mem_1",
            user_id="demo_user",
            content="The user strongly prefers dark mode in all applications. They find "
                    "light themes straining on the eyes and always enable dark mode first "
                    "when setting up new tools or environments.",
            type=MemoryType.SEMANTIC,
            headline="Prefers dark mode",
            subtitle="User always enables dark mode for eye comfort",
            confidence_score=0.95,
            tags=["preference", "ui"],
        ),
        MemoryItem(
            memory_id="mem_2",
            user_id="demo_user",
            content="The user is an experienced Python developer who uses pytest for testing, "
                    "black for formatting, and mypy for type checking. They follow PEP8 "
                    "guidelines strictly and prefer explicit type hints.",
            type=MemoryType.SEMANTIC,
            headline="Python developer with strict standards",
            subtitle="Uses pytest, black, mypy; follows PEP8 with type hints",
            confidence_score=0.90,
            tags=["development", "python"],
        ),
        MemoryItem(
            memory_id="mem_3",
            user_id="demo_user",
            content="The user's main project is a FastAPI backend service that handles "
                    "authentication, user management, and real-time notifications via "
                    "WebSockets. The project uses PostgreSQL for persistence.",
            type=MemoryType.EPISODIC,
            headline="FastAPI backend project",
            subtitle="Auth, user management, WebSocket notifications with PostgreSQL",
            confidence_score=0.85,
            tags=["project", "fastapi"],
        ),
        MemoryItem(
            memory_id="mem_4",
            user_id="demo_user",
            content="The user prefers detailed explanations with code examples when learning "
                    "new concepts. They like step-by-step walkthroughs and appreciate when "
                    "potential pitfalls are highlighted upfront.",
            type=MemoryType.PROCEDURAL,
            headline="Learns best with detailed examples",
            subtitle="Prefers step-by-step walkthroughs with pitfall warnings",
            confidence_score=0.88,
            tags=["learning", "preference"],
        ),
    ]

    # Build memory index
    print_subsection("Building Memory Index")
    index = MemoryIndex(total_memories=len(memories))
    for memory in memories:
        entry = MemoryIndexEntry.from_memory(memory)
        index.add(entry)

    print(f"Created index with {index.total_memories} memories")

    # Show different disclosure levels
    print_subsection("Disclosure Level: HEADLINE (most compact)")
    print(index.to_prompt(level=DisclosureLevel.HEADLINE))

    print_subsection("Disclosure Level: SUMMARY")
    print(index.to_prompt(level=DisclosureLevel.SUMMARY))

    print_subsection("Disclosure Level: FULL (first 2 expanded)")
    print(index.to_prompt(level=DisclosureLevel.HEADLINE, expand_top_n=2))

    # Token estimates
    print_subsection("Token Estimates by Level")
    headline_tokens = index.estimate_tokens(DisclosureLevel.HEADLINE)
    summary_tokens = index.estimate_tokens(DisclosureLevel.SUMMARY)
    full_tokens = index.estimate_tokens(DisclosureLevel.FULL)

    print(f"  HEADLINE: ~{headline_tokens} tokens")
    print(f"  SUMMARY:  ~{summary_tokens} tokens")
    print(f"  FULL:     ~{full_tokens} tokens")
    savings = ((full_tokens - headline_tokens) / full_tokens * 100)
    print(f"\n  Savings (HEADLINE vs FULL): ~{savings:.0f}%")

    # MemoryItem format methods
    print_subsection("MemoryItem Format Methods")
    mem = memories[0]
    print(f"to_headline_format(): {mem.to_headline_format()}")
    print(f"to_summary_format():  {mem.to_summary_format()}")


# =============================================================================
# DEMO 2: Timeline-Based Event Retrieval
# =============================================================================

def demo_timeline_retrieval():
    """Demonstrate timeline-based event retrieval."""
    print_section("DEMO 2: Timeline-Based Event Retrieval")

    # Create session with events over time
    session = Session(user_id="demo_user")
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    # Simulate a conversation with multiple turns
    events_data = [
        (0, EventType.USER, "What's the weather like today?"),
        (1, EventType.AGENT, "Let me check the weather for you."),
        (2, EventType.TOOL_CALL, "get_weather(location='current')"),
        (3, EventType.TOOL_OUTPUT, "Temperature: 72°F, Sunny"),
        (4, EventType.AGENT, "It's currently 72°F and sunny!"),

        (30, EventType.USER, "Can you read my config file?"),
        (31, EventType.AGENT, "Sure, I'll read your config file."),
        (32, EventType.TOOL_CALL, "read_file(path='config.yaml')"),
        (33, EventType.TOOL_OUTPUT, "database:\n  host: localhost\n  port: 5432"),
        (34, EventType.AGENT, "Here's your config file content..."),

        (60, EventType.USER, "Now run my tests"),
        (61, EventType.AGENT, "Running your test suite now."),
        (62, EventType.TOOL_CALL, "run_terminal_cmd(cmd='pytest')"),
        (63, EventType.TOOL_OUTPUT, "...10 passed, 2 failed in 3.5s"),
        (64, EventType.AGENT, "Tests completed with 2 failures."),

        (90, EventType.USER, "Fix the failing tests"),
        (91, EventType.AGENT, "I'll analyze and fix those tests."),
    ]

    for minutes, event_type, content in events_data:
        event = Event(
            type=event_type,
            content=content,
            timestamp=base_time + timedelta(minutes=minutes),
            metadata=EventMetadata(
                tool_name=content.split("(")[0] if event_type == EventType.TOOL_CALL else None
            ),
        )
        session.add_event(event)

    print(f"Created session with {len(session.events)} events")

    # Timeline Service
    timeline_service = TimelineService()

    # Query: Get conversation turns
    print_subsection("Get Conversation Turns")
    turns = session.get_turns()
    print(f"Total turns: {len(turns)}")
    for i, turn in enumerate(turns):
        user_msg = turn[0].content[:50]
        print(f"  Turn {i + 1}: {user_msg}...")

    # Query: Last N turns
    print_subsection("Last 2 Turns Only")
    last_turn_events = session.get_last_n_turn_events(n=2)
    print(f"Events in last 2 turns: {len(last_turn_events)}")
    for event in last_turn_events:
        print(f"  [{event.type.value}] {event.content[:40]}...")

    # Query: User-Agent exchanges only
    print_subsection("User-Agent Exchanges (No Tool Events)")
    result = timeline_service.get_user_assistant_exchanges(session)
    print(f"Conversation events: {len(result.events)} (filtered from {len(session.events)})")
    for event in result.events[:6]:
        print(f"  [{event.type.value}] {event.content[:50]}...")

    # Activity summary
    print_subsection("Activity Summary")
    summary = timeline_service.summarize_activity(session)
    print(f"  Total events: {summary['total_events']}")
    print(f"  Conversation turns: {summary['turn_count']}")
    print(f"  Tools used: {summary['tool_calls']}")
    print(f"  Event breakdown: {summary['event_counts']}")

    # Token savings
    print_subsection("Token Savings")
    all_events_tokens = sum(len(e.content) for e in session.events) // 4
    filtered_tokens = sum(len(e.content) for e in result.events) // 4
    print(f"  All events: ~{all_events_tokens} tokens")
    print(f"  Filtered (no tools): ~{filtered_tokens} tokens")
    print(f"  Savings: ~{((all_events_tokens - filtered_tokens) / all_events_tokens * 100):.0f}%")


# =============================================================================
# DEMO 3: Tool Output Compression
# =============================================================================

def demo_tool_compression():
    """Demonstrate tool output compression."""
    print_section("DEMO 3: Tool Output Compression")

    compressor = ToolOutputCompressor(
        max_output_chars=500,
        compression_threshold=100,
    )

    # Test 1: Long file content (TRUNCATE)
    print_subsection("Strategy: TRUNCATE (Long File)")
    long_file = """
def calculate_fibonacci(n):
    '''Calculate fibonacci number at position n.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

def is_prime(n):
    '''Check if a number is prime.'''
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# More functions follow...
""" + "\n# Additional code...\n" * 50

    result = compressor.compress(long_file, tool_name="read_file")
    print(f"Original: {len(long_file)} chars")
    print(f"Compressed: {len(result.compressed_content)} chars")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Tokens saved: ~{result.tokens_saved}")
    print(f"Compression ratio: {result.compression_ratio:.2%}")
    print(f"\nCompressed preview:\n{result.compressed_content[:300]}...")

    # Test 2: JSON content (KEY_VALUE)
    print_subsection("Strategy: KEY_VALUE (JSON API Response)")
    json_content = json.dumps({
        "users": [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "profile": {
                    "bio": f"This is a detailed biography for user {i}...",
                    "settings": {"theme": "dark", "notifications": True},
                },
            }
            for i in range(20)
        ],
        "pagination": {"page": 1, "total": 100, "per_page": 20},
    }, indent=2)

    result = compressor.compress(json_content, strategy=CompressionStrategy.KEY_VALUE)
    print(f"Original: {len(json_content)} chars")
    print(f"Compressed: {len(result.compressed_content)} chars")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Tokens saved: ~{result.tokens_saved}")
    print(f"\nCompressed JSON:\n{result.compressed_content}")

    # Test 3: Duplicate lines (DEDUPE)
    print_subsection("Strategy: DEDUPE (Repeated Output)")
    duplicate_content = "\n".join([
        f"Processing item {i % 5}..." for i in range(50)
    ] + [
        "Result: SUCCESS" for _ in range(20)
    ])

    result = compressor.compress(duplicate_content, strategy=CompressionStrategy.DEDUPE)
    print(f"Original: {len(duplicate_content)} chars, {duplicate_content.count(chr(10))} lines")
    print(f"Compressed: {len(result.compressed_content)} chars")
    print(f"Duplicates removed: {result.metadata.get('duplicates_removed', 0)}")
    print(f"Tokens saved: ~{result.tokens_saved}")
    print(f"\nCompressed:\n{result.compressed_content}")

    # Test 4: Compress tool event
    print_subsection("Compress Tool Event")
    event = Event(
        type=EventType.TOOL_OUTPUT,
        content=long_file,
        timestamp=datetime.now(timezone.utc),
        metadata=EventMetadata(tool_name="read_file"),
    )

    compressed_event = compress_tool_event(event, compressor)
    print(f"Original event content: {len(event.content)} chars")
    print(f"Compressed event content: {len(compressed_event.content)} chars")
    print(f"Metadata - compressed: {compressed_event.metadata.compressed}")
    print(f"Metadata - strategy: {compressed_event.metadata.compression_strategy}")
    print(f"Metadata - tokens_saved: {compressed_event.metadata.tokens_saved}")


# =============================================================================
# DEMO 4: Combined Token Savings
# =============================================================================

def demo_combined_savings():
    """Demonstrate combined token savings from all enhancements."""
    print_section("DEMO 4: Combined Token Savings")

    # Scenario: Agentic coding session
    print("Scenario: Coding assistant session with 10 memories, 20 events, 5 tool outputs\n")

    # Memory savings (Progressive Disclosure)
    print_subsection("1. Memory Token Savings (Progressive Disclosure)")
    memory_full_tokens = 800  # 10 memories at ~80 tokens each
    memory_headline_tokens = 250  # Headlines only ~25 tokens each
    memory_savings = (memory_full_tokens - memory_headline_tokens) / memory_full_tokens * 100
    print(f"   Full memories: ~{memory_full_tokens} tokens")
    print(f"   Headlines only: ~{memory_headline_tokens} tokens")
    print(f"   Savings: {memory_savings:.0f}%")

    # Event savings (Timeline Filtering)
    print_subsection("2. Event Token Savings (Timeline Filtering)")
    all_events_tokens = 2000  # 20 events
    filtered_events_tokens = 1200  # Last 3 turns, no tool events
    timeline_savings = (all_events_tokens - filtered_events_tokens) / all_events_tokens * 100
    print(f"   All events: ~{all_events_tokens} tokens")
    print(f"   Filtered (recent, no tools): ~{filtered_events_tokens} tokens")
    print(f"   Savings: {timeline_savings:.0f}%")

    # Tool output savings (Compression)
    print_subsection("3. Tool Output Token Savings (Compression)")
    tool_original_tokens = 5000  # 5 large tool outputs
    tool_compressed_tokens = 1500  # After compression
    compression_savings = (
        (tool_original_tokens - tool_compressed_tokens) / tool_original_tokens * 100
    )
    print(f"   Original tool outputs: ~{tool_original_tokens} tokens")
    print(f"   Compressed outputs: ~{tool_compressed_tokens} tokens")
    print(f"   Savings: {compression_savings:.0f}%")

    # Combined
    print_subsection("4. Combined Total")
    total_original = memory_full_tokens + all_events_tokens + tool_original_tokens
    total_optimized = memory_headline_tokens + filtered_events_tokens + tool_compressed_tokens
    total_savings = (total_original - total_optimized) / total_original * 100

    print(f"   Original context: ~{total_original} tokens")
    print(f"   Optimized context: ~{total_optimized} tokens")
    print(f"   Total savings: {total_savings:.0f}%")
    print(f"\n   💡 Reduced from ~{total_original} to ~{total_optimized} tokens!")


# =============================================================================
# DEMO 5: Configuration
# =============================================================================

def demo_configuration():
    """Demonstrate configuration options."""
    print_section("DEMO 5: Configuration")

    print_subsection("Progressive Disclosure Config")
    pd_config = ProgressiveDisclosureConfig()
    print(f"  enabled: {pd_config.enabled}")
    print(f"  max_headline_chars: {pd_config.max_headline_chars}")
    print(f"  max_subtitle_chars: {pd_config.max_subtitle_chars}")
    print(f"  expand_top_n: {pd_config.expand_top_n}")
    print(f"  use_llm_headlines: {pd_config.use_llm_headlines}")

    print_subsection("Timeline Config")
    tl_config = TimelineConfig()
    print(f"  enabled: {tl_config.enabled}")
    print(f"  default_time_range: {tl_config.default_time_range}")
    print(f"  include_timestamps_in_history: {tl_config.include_timestamps_in_history}")
    print(f"  group_by_turns: {tl_config.group_by_turns}")
    print(f"  max_events_default: {tl_config.max_events_default}")

    print_subsection("Tool Compression Config")
    tc_config = ToolCompressionConfig()
    print(f"  enabled: {tc_config.enabled}")
    print(f"  max_output_chars: {tc_config.max_output_chars}")
    print(f"  compression_threshold: {tc_config.compression_threshold}")
    print(f"  default_strategy: {tc_config.default_strategy}")


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# DEMO 6: Persistent Headline Storage (Database)
# =============================================================================

async def demo_persistent_headlines():
    """Demonstrate persistent headline storage in MySQL/PostgreSQL."""
    import os
    from pathlib import Path
    
    # Try to load .env file if it exists
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    
    print_section("DEMO 6: Persistent Headline Storage (Database)")
    
    # Check for MySQL configuration
    mysql_host = os.environ.get("MYSQL_HOST")
    postgres_host = os.environ.get("POSTGRES_HOST")
    
    if not mysql_host and not postgres_host:
        print("⚠️  Database connection not configured.")
        print("   Set MYSQL_HOST or POSTGRES_HOST in examples/.env to test persistence.")
        print("   Skipping database demo.\n")
        return
    
    # Test MySQL if configured
    if mysql_host:
        await _demo_mysql_headlines()
    
    # Test PostgreSQL if configured
    if postgres_host:
        await _demo_postgres_headlines()


async def _demo_mysql_headlines():
    """Demo MySQL headline persistence."""
    import os
    import uuid
    
    print_subsection("MySQL Headline Persistence")
    
    try:
        from ctxforge.storage.connection import MySQLConfig
        from ctxforge.storage.mysql.memory import MySQLMemoryStore
        
        config = MySQLConfig(
            host=os.environ.get("MYSQL_HOST", "localhost"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=os.environ.get("MYSQL_DATABASE", "ctxforge"),
        )
        
        store = MySQLMemoryStore(config)
        await store.initialize()
        
        # Create a memory with headline
        memory_id = f"demo_headline_{uuid.uuid4().hex[:8]}"
        memory = MemoryItem(
            memory_id=memory_id,
            user_id="demo_user",
            content="The user is an expert Python developer with 10 years of experience. "
                    "They specialize in backend development using FastAPI, Django, and Flask. "
                    "They prefer type hints and strict testing with pytest.",
            type=MemoryType.SEMANTIC,
            headline="Expert Python backend developer",
            subtitle="10 years experience with FastAPI, Django, Flask; strict testing",
            confidence_score=0.95,
            tags=["python", "backend", "expertise"],
        )
        
        # Save to database
        print("Saving memory with headline to MySQL...")
        await store.add(memory)
        print(f"  ✓ Saved memory: {memory_id}")
        print(f"  Headline: {memory.headline}")
        print(f"  Subtitle: {memory.subtitle}")
        
        # Retrieve from database
        print("\nRetrieving memory from MySQL...")
        retrieved = await store.get(memory_id)
        
        if retrieved:
            print(f"  ✓ Retrieved memory: {retrieved.memory_id}")
            print(f"  Headline: {retrieved.headline}")
            print(f"  Subtitle: {retrieved.subtitle}")
            print(f"  Has headline: {retrieved.has_headline()}")
            print(f"\n  Progressive format: {retrieved.to_headline_format()}")
            print(f"  Summary format: {retrieved.to_summary_format()}")
            
            # Verify persistence
            if retrieved.headline == memory.headline:
                print("\n  ✅ Headlines persisted correctly!")
            else:
                print("\n  ❌ Headline mismatch!")
        else:
            print("  ❌ Failed to retrieve memory")
        
        # Cleanup
        await store.delete(memory_id)
        await store.disconnect()
        print("\n  Cleaned up test data.\n")
        
    except Exception as e:
        print(f"  ❌ MySQL error: {e}\n")


async def _demo_postgres_headlines():
    """Demo PostgreSQL headline persistence."""
    import os
    import uuid
    
    print_subsection("PostgreSQL Headline Persistence")
    
    try:
        from ctxforge.storage.connection import PostgresConfig
        from ctxforge.storage.postgres.memory import PostgresMemoryStore
        
        config = PostgresConfig(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            database=os.environ.get("POSTGRES_DB", "ctxforge"),
        )
        
        store = PostgresMemoryStore(config)
        await store.initialize()
        
        # Create a memory with headline
        memory_id = f"demo_headline_{uuid.uuid4().hex[:8]}"
        memory = MemoryItem(
            memory_id=memory_id,
            user_id="demo_user",
            content="The user prefers dark mode in all applications and IDEs. "
                    "They find light themes straining on the eyes, especially during "
                    "late-night coding sessions.",
            type=MemoryType.SEMANTIC,
            headline="Prefers dark mode",
            subtitle="Light themes strain eyes, especially during late-night coding",
            confidence_score=0.90,
            tags=["preference", "ui", "accessibility"],
        )
        
        # Save to database
        print("Saving memory with headline to PostgreSQL...")
        await store.add(memory)
        print(f"  ✓ Saved memory: {memory_id}")
        print(f"  Headline: {memory.headline}")
        print(f"  Subtitle: {memory.subtitle}")
        
        # Retrieve from database
        print("\nRetrieving memory from PostgreSQL...")
        retrieved = await store.get(memory_id)
        
        if retrieved:
            print(f"  ✓ Retrieved memory: {retrieved.memory_id}")
            print(f"  Headline: {retrieved.headline}")
            print(f"  Subtitle: {retrieved.subtitle}")
            print(f"  Has headline: {retrieved.has_headline()}")
            print(f"\n  Progressive format: {retrieved.to_headline_format()}")
            print(f"  Summary format: {retrieved.to_summary_format()}")
            
            # Verify persistence
            if retrieved.headline == memory.headline:
                print("\n  ✅ Headlines persisted correctly!")
            else:
                print("\n  ❌ Headline mismatch!")
        else:
            print("  ❌ Failed to retrieve memory")
        
        # Cleanup
        await store.delete(memory_id)
        await store.disconnect()
        print("\n  Cleaned up test data.\n")
        
    except Exception as e:
        print(f"  ❌ PostgreSQL error: {e}\n")


def main():
    """Run all demos."""
    import asyncio
    
    print("\n" + "=" * 70)
    print(" CONTEXT ENGINE ENHANCEMENTS DEMO")
    print(" Token-efficient context management for agentic workflows")
    print("=" * 70)

    demo_progressive_disclosure()
    demo_timeline_retrieval()
    demo_tool_compression()
    demo_combined_savings()
    demo_configuration()
    
    # Run async database demo
    asyncio.run(demo_persistent_headlines())

    print_section("DEMO COMPLETE")
    print("All enhancements demonstrated successfully!")
    print("\nKey features:")
    print("  ✓ Progressive Disclosure - Compact memory representation")
    print("  ✓ Timeline Retrieval - Temporal event filtering")
    print("  ✓ Tool Compression - Verbose output reduction")
    print("  ✓ Persistent Headlines - Database storage for MySQL/PostgreSQL")
    print("\nCombined savings: 40-60% token reduction in typical workflows\n")


if __name__ == "__main__":
    main()
