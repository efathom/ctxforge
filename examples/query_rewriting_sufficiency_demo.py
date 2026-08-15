#!/usr/bin/env python3
"""
Query Rewriting and Sufficiency Judging Demo.

This script demonstrates the query rewriting and sufficiency judging features:
1. Query Rewriting - Transform ambiguous queries using conversation context
2. Sufficiency Judging - Evaluate if retrieved content adequately answers queries
3. Progressive Retrieval - Iteratively fetch more results until sufficient
4. Combined Usage - Query rewriting + sufficiency together
5. Database Storage - Test with PostgreSQL/MySQL when configured

Usage:
    python examples/query_rewriting_sufficiency_demo.py

Configuration:
    Copy examples/env.example to examples/.env and set:
    - OPENAI_API_KEY or AZURE_OPENAI_* for LLM
    - POSTGRES_* or MYSQL_* for database storage (optional)
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Add parent directory to path for imports (go up to ctxforge/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv  # noqa: E402
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}")
        print("  Copy env.example to .env and configure your API keys")
except ImportError:
    print("⚠ python-dotenv not installed, using environment variables only")

from ctxforge.config.base import QueryRewriteConfig  # noqa: E402
from ctxforge.core.events import Event, EventType  # noqa: E402
from ctxforge.core.memory import MemoryItem, MemoryType  # noqa: E402
from ctxforge.engine.services.query_rewriter_service import QueryRewriterService  # noqa: E402
from ctxforge.engine.services.sufficiency_service import (  # noqa: E402
    SufficiencyConfig,
    SufficiencyService,
)
from ctxforge.protocols.llm import ILLMProvider  # noqa: E402

# =============================================================================
# LLM Provider Setup
# =============================================================================

@dataclass
class MockLLMResponse:
    """Mock LLM response."""
    content: str


class DemoLLMProvider:
    """
    Demo LLM provider that simulates realistic responses.
    Used as fallback when no API key is configured.
    """

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "MockLLMProvider"

    @property
    def default_model(self) -> str:
        return "mock-model"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> MockLLMResponse:
        """Generate a mock response based on prompt content."""
        self._call_count += 1

        # Detect query rewriting prompts
        if "rewrite" in prompt.lower() or "rewritten_query" in prompt.lower():
            return MockLLMResponse(self._get_rewrite_response(prompt))

        # Detect sufficiency prompts
        if "sufficiency" in prompt.lower() or "verdict" in prompt.lower():
            return MockLLMResponse(self._get_sufficiency_response())

        return MockLLMResponse(self._get_sufficiency_response())

    def _get_rewrite_response(self, prompt: str) -> str:
        """Get rewrite response based on prompt content."""
        if "their" in prompt.lower() or "them" in prompt.lower():
            return """
<rewrite_response>
  <analysis>Query contains pronoun referring to John from conversation.</analysis>
  <rewritten_query>What food does John prefer for the Japan trip?</rewritten_query>
  <reason>pronoun</reason>
  <confidence>0.92</confidence>
  <resolved_entities>John, Japan trip</resolved_entities>
</rewrite_response>
"""
        elif "that" in prompt.lower() and "about that" in prompt.lower():
            return """
<rewrite_response>
  <analysis>Query refers to 'that' which is the Kyoto temple discussion.</analysis>
  <rewritten_query>Tell me more about the Kyoto temples John recommended</rewritten_query>
  <reason>reference</reason>
  <confidence>0.88</confidence>
  <resolved_entities>Kyoto temples, John</resolved_entities>
</rewrite_response>
"""
        elif "and the" in prompt.lower():
            return """
<rewrite_response>
  <analysis>Query starts with 'And' implying continuation of travel planning.</analysis>
  <rewritten_query>What are the best restaurants in Kyoto for the Japan trip?</rewritten_query>
  <reason>implicit</reason>
  <confidence>0.85</confidence>
  <resolved_entities>Kyoto, Japan trip, restaurants</resolved_entities>
</rewrite_response>
"""
        else:
            return """
<rewrite_response>
  <analysis>Query is already clear and self-contained.</analysis>
  <rewritten_query>What are the best sushi restaurants in Tokyo?</rewritten_query>
  <reason>no_change</reason>
  <confidence>0.95</confidence>
  <resolved_entities></resolved_entities>
</rewrite_response>
"""

    def _get_sufficiency_response(self) -> str:
        """Get sufficiency response based on call count."""
        if self._call_count <= 2:
            return """
<sufficiency_response>
  <consideration>Content provides some information but lacks specific details.</consideration>
  <verdict>MORE</verdict>
  <confidence>0.78</confidence>
  <missing_aspects>specific preferences, dietary restrictions</missing_aspects>
  <suggested_sources>memories, graph</suggested_sources>
</sufficiency_response>
"""
        else:
            return """
<sufficiency_response>
  <consideration>The retrieved content fully addresses the query.</consideration>
  <verdict>ENOUGH</verdict>
  <confidence>0.93</confidence>
  <missing_aspects></missing_aspects>
  <suggested_sources></suggested_sources>
</sufficiency_response>
"""

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


def create_llm_provider() -> ILLMProvider:
    """
    Create an LLM provider based on environment configuration.

    Checks for:
    1. Azure OpenAI (AZURE_OPENAI_API_KEY)
    2. OpenAI (OPENAI_API_KEY)
    3. Falls back to mock provider
    """
    # Check for Azure OpenAI
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    if azure_key and azure_endpoint and azure_deployment:
        try:
            from ctxforge.llm.azure_openai_provider import (
                AzureOpenAIConfig,
                AzureOpenAILLMProvider,
            )

            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            embed_deployment = os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
            )
            config = AzureOpenAIConfig(
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment=azure_deployment,
                embedding_deployment=embed_deployment,
            )
            print(f"✓ Using Azure OpenAI: {azure_endpoint}")
            print(f"  Deployment: {azure_deployment}")
            return AzureOpenAILLMProvider(config)
        except Exception as e:
            print(f"⚠ Failed to initialize Azure OpenAI: {e}")

    # Check for OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from ctxforge.llm.openai_provider import OpenAIConfig, OpenAILLMProvider

            model = os.getenv("OPENAI_MODEL", "gpt-4")
            config = OpenAIConfig(
                api_key=openai_key,
                model=model,
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            )
            print(f"✓ Using OpenAI: {model}")
            return OpenAILLMProvider(config)
        except Exception as e:
            print(f"⚠ Failed to initialize OpenAI: {e}")

    # Fall back to mock
    print("ℹ Using mock LLM provider (no API key configured)")
    print("  Set OPENAI_API_KEY or AZURE_OPENAI_* in .env for real LLM")
    return DemoLLMProvider()


# =============================================================================
# Demo Helpers
# =============================================================================

def create_event(event_type: EventType, content: str) -> Event:
    """Create a demo event."""
    return Event(
        event_id=f"demo-{event_type.value}-{len(content)}",
        type=event_type,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def create_conversation_history() -> List[Event]:
    """Create a sample conversation history for demo."""
    return [
        create_event(
            EventType.USER,
            "I'm planning a trip to Japan next month with my friend John"
        ),
        create_event(
            EventType.AGENT,
            "That sounds exciting! Japan has beautiful cherry blossoms in spring. "
            "What aspects of the trip would you like help with?"
        ),
        create_event(
            EventType.USER,
            "John recommended visiting Kyoto for the temples"
        ),
        create_event(
            EventType.AGENT,
            "Kyoto is wonderful! It has over 2,000 temples and shrines. "
            "The famous ones include Kinkaku-ji (Golden Pavilion) and Fushimi Inari."
        ),
    ]


def create_demo_memories() -> List[MemoryItem]:
    """Create demo memories for retrieval."""
    return [
        MemoryItem(
            memory_id="mem-1",
            user_id="demo-user",
            content="User prefers vegetarian food options",
            type=MemoryType.SEMANTIC,
            created_at=datetime.now(timezone.utc),
        ),
        MemoryItem(
            memory_id="mem-2",
            user_id="demo-user",
            content="User enjoys traditional Japanese cuisine, especially sushi",
            type=MemoryType.SEMANTIC,
            created_at=datetime.now(timezone.utc),
        ),
        MemoryItem(
            memory_id="mem-3",
            user_id="demo-user",
            content="User is allergic to shellfish",
            type=MemoryType.SEMANTIC,
            created_at=datetime.now(timezone.utc),
        ),
        MemoryItem(
            memory_id="mem-4",
            user_id="demo-user",
            content="User traveled to Tokyo last year and loved Shibuya",
            type=MemoryType.EPISODIC,
            created_at=datetime.now(timezone.utc),
        ),
        MemoryItem(
            memory_id="mem-5",
            user_id="demo-user",
            content="User prefers budget-friendly accommodations",
            type=MemoryType.SEMANTIC,
            created_at=datetime.now(timezone.utc),
        ),
    ]


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_query_rewriting(llm: ILLMProvider):
    """Demonstrate query rewriting capabilities."""
    print_header("DEMO 1: Query Rewriting")

    config = QueryRewriteConfig(enabled=True, max_history_turns=10)
    rewriter = QueryRewriterService(llm, config)

    history = create_conversation_history()

    print("\nConversation History:")
    for event in history:
        role = "User" if event.type == EventType.USER else "Assistant"
        content = event.content[:60] + "..." if len(event.content) > 60 else event.content
        print(f"  {role}: {content}")

    # Test cases
    test_queries = [
        ("What about their food preferences?", "Pronoun resolution"),
        ("Tell me more about that", "Reference resolution"),
        ("And the restaurants?", "Implicit context"),
        ("What are the best sushi restaurants in Tokyo?", "Clear query"),
    ]

    for query, description in test_queries:
        print_subheader(description)
        print(f"Original: \"{query}\"")

        result = await rewriter.rewrite(
            query=query,
            conversation_history=history,
        )

        if result.was_rewritten:
            print(f"Rewritten: \"{result.rewritten_query}\"")
            print(f"Reason: {result.reason.value}")
            print(f"Confidence: {result.confidence:.2f}")
            if result.resolved_entities:
                print(f"Resolved: {', '.join(result.resolved_entities)}")
        else:
            print("(No rewrite needed)")

    # Show heuristic detection
    print_subheader("Heuristic Detection")
    needs_rewrite_1 = rewriter.needs_rewriting('What about them?')
    print(f"'What about them?' needs rewriting: {needs_rewrite_1}")
    needs_rewrite_2 = rewriter.needs_rewriting('Show me Python docs')
    print(f"'Show me Python docs' needs rewriting: {needs_rewrite_2}")


async def demo_sufficiency_judging(llm: ILLMProvider):
    """Demonstrate sufficiency judging capabilities."""
    print_header("DEMO 2: Sufficiency Judging")

    config = SufficiencyConfig(enabled=True, max_iterations=3)
    service = SufficiencyService(llm, config)

    # Force "enough" response for mock provider
    if hasattr(llm, '_call_count'):
        llm._call_count = 10

    # Test with sufficient content
    print_subheader("Sufficient Content")
    query = "What food does the user prefer?"
    content = """
    1. User prefers vegetarian food options
    2. User enjoys traditional Japanese cuisine, especially sushi
    3. User is allergic to shellfish
    """

    result = await service.judge(query=query, retrieved_content=content)
    print(f"Query: \"{query}\"")
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning[:80]}...")

    # Test with insufficient content
    print_subheader("Insufficient Content")
    query = "What are the user's detailed travel preferences?"
    content = "User likes Japan."

    # Reset call count for "more" responses
    if hasattr(llm, 'reset_call_count'):
        llm.reset_call_count()

    result = await service.judge(query=query, retrieved_content=content)
    print(f"Query: \"{query}\"")
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.2f}")
    if result.missing_aspects:
        print(f"Missing: {', '.join(result.missing_aspects)}")
    if result.suggested_sources:
        print(f"Suggested sources: {', '.join(result.suggested_sources)}")

    # Test with empty content
    print_subheader("Empty Content (Fast Path)")
    result = await service.judge(query="Any query", retrieved_content="")
    print(f"Empty content verdict: {result.verdict.value}")
    print(f"Fast path reasoning: {result.reasoning}")


async def demo_progressive_retrieval(llm: ILLMProvider):
    """Demonstrate progressive retrieval."""
    print_header("DEMO 3: Progressive Retrieval")

    config = SufficiencyConfig(enabled=True, max_iterations=3)
    service = SufficiencyService(llm, config)

    memories = create_demo_memories()

    # Custom retriever that returns memories
    async def memory_retriever(limit: int) -> List[MemoryItem]:
        return memories[:limit]

    # Custom formatter
    def memory_formatter(items: List[MemoryItem]) -> str:
        return "\n".join(f"{i}. {m.content}" for i, m in enumerate(items, 1))

    print("\nAvailable Memories:")
    for m in memories:
        content = m.content[:50] + "..." if len(m.content) > 50 else m.content
        print(f"  - {content}")

    print_subheader("Progressive Retrieval Process")
    print("Query: \"What are the user's preferences?\"")
    print("Initial limit: 2, Max limit: 5")

    # Reset call count for mock provider
    if hasattr(llm, 'reset_call_count'):
        llm.reset_call_count()

    results, judgment, stats = await service.progressive_retrieve(
        query="What are the user's preferences?",
        retriever_func=memory_retriever,
        formatter_func=memory_formatter,
        initial_limit=2,
        max_limit=5,
    )

    print("\nResults:")
    print(f"  Total iterations: {stats.total_iterations}")
    print(f"  Initial results: {stats.initial_results}")
    print(f"  Final results: {stats.final_results}")
    print(f"  Results added: {stats.results_added}")
    print(f"  Final verdict: {judgment.verdict.value}")

    if stats.sources_tried:
        print(f"  Sources suggested: {', '.join(set(stats.sources_tried))}")

    print("\nRetrieved Memories:")
    for m in results:
        print(f"  - {m.content}")


async def demo_combined_usage(llm: ILLMProvider):
    """Demonstrate query rewriting + sufficiency together."""
    print_header("DEMO 4: Combined Query Rewriting + Sufficiency")

    # Set up services
    rewrite_config = QueryRewriteConfig(enabled=True)
    rewriter = QueryRewriterService(llm, rewrite_config)

    sufficiency_config = SufficiencyConfig(enabled=True)
    sufficiency_service = SufficiencyService(llm, sufficiency_config)

    history = create_conversation_history()
    memories = create_demo_memories()

    print("\nScenario: User asks an ambiguous follow-up question")
    print_subheader("Step 1: Query Rewriting")

    original_query = "What about their food preferences?"
    print(f"Original query: \"{original_query}\"")

    rewrite_result = await rewriter.rewrite(
        query=original_query,
        conversation_history=history,
    )

    effective_query = rewrite_result.rewritten_query
    print(f"Rewritten query: \"{effective_query}\"")
    if rewrite_result.resolved_entities:
        print(f"Resolved entities: {', '.join(rewrite_result.resolved_entities)}")

    print_subheader("Step 2: Retrieval with Sufficiency Check")

    # Simulate retrieval
    retrieved_content = "\n".join(
        f"- {m.content}" for m in memories[:3]
    )
    print("Retrieved content:")
    for m in memories[:3]:
        print(f"  - {m.content}")

    # Reset for fresh judgment and force "enough"
    if hasattr(llm, '_call_count'):
        llm._call_count = 10

    judgment = await sufficiency_service.judge(
        query=effective_query,
        retrieved_content=retrieved_content,
    )

    print_subheader("Step 3: Result")
    print(f"Sufficiency verdict: {judgment.verdict.value}")
    print(f"Confidence: {judgment.confidence:.2f}")

    if judgment.is_sufficient:
        print("\n✅ Query was successfully rewritten and sufficient content was retrieved!")
    else:
        print("\n⚠️ May need to retrieve more content")


async def demo_database_storage():
    """Demonstrate database storage for memories (optional)."""
    print_header("DEMO 5: Database Storage (Optional)")

    # Check PostgreSQL configuration
    pg_host = os.getenv("POSTGRES_HOST")
    pg_user = os.getenv("POSTGRES_USER")
    pg_password = os.getenv("POSTGRES_PASSWORD")
    pg_database = os.getenv("POSTGRES_DB") or os.getenv("POSTGRES_DATABASE")

    # Check MySQL configuration
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_database = os.getenv("MYSQL_DATABASE")

    if not (pg_host and pg_user and pg_database) and not (mysql_host and mysql_user):
        print("\n⚠ No database configured. Skipping database demo.")
        print("  Set POSTGRES_* or MYSQL_* in .env to test database storage.")
        return

    # Test PostgreSQL
    if pg_host and pg_user and pg_database:
        print_subheader("PostgreSQL Storage")
        try:
            from ctxforge.storage.connection import PostgresConfig
            from ctxforge.storage.postgres.memory import PostgresMemoryStore

            config = PostgresConfig(
                host=pg_host,
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=pg_user,
                password=pg_password or "",
                database=pg_database,
            )
            store = PostgresMemoryStore(config=config)
            await store.initialize()
            print(f"✓ Connected to PostgreSQL: {pg_host}:{os.getenv('POSTGRES_PORT', '5432')}")
            print(f"  Database: {pg_database}")

            # Test storage
            test_memory = MemoryItem(
                memory_id="demo-pg-memory",
                user_id="demo-user",
                content="Test memory for query rewriting demo",
                type=MemoryType.SEMANTIC,
                created_at=datetime.now(timezone.utc),
            )
            await store.add(test_memory)
            print("  ✓ Added test memory")

            retrieved = await store.get("demo-pg-memory")
            if retrieved:
                print(f"  ✓ Retrieved: {retrieved.content[:40]}...")

            await store.delete("demo-pg-memory")
            print("  ✓ Cleaned up test memory")

            await store.disconnect()
            print("  ✓ Disconnected")

        except Exception as e:
            print(f"  ✗ PostgreSQL error: {e}")

    # Test MySQL
    if mysql_host and mysql_user and mysql_database:
        print_subheader("MySQL Storage")
        try:
            from ctxforge.storage.connection import MySQLConfig
            from ctxforge.storage.mysql.memory import MySQLMemoryStore

            config = MySQLConfig(
                host=mysql_host,
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=mysql_user,
                password=mysql_password or "",
                database=mysql_database,
            )
            store = MySQLMemoryStore(config=config)
            await store.initialize()
            print(f"✓ Connected to MySQL: {mysql_host}:{os.getenv('MYSQL_PORT', '3306')}")
            print(f"  Database: {mysql_database}")

            # Test storage
            test_memory = MemoryItem(
                memory_id="demo-mysql-memory",
                user_id="demo-user",
                content="Test memory for query rewriting demo",
                type=MemoryType.SEMANTIC,
                created_at=datetime.now(timezone.utc),
            )
            await store.add(test_memory)
            print("  ✓ Added test memory")

            retrieved = await store.get("demo-mysql-memory")
            if retrieved:
                print(f"  ✓ Retrieved: {retrieved.content[:40]}...")

            await store.delete("demo-mysql-memory")
            print("  ✓ Cleaned up test memory")

            await store.disconnect()
            print("  ✓ Disconnected")

        except Exception as e:
            print(f"  ✗ MySQL error: {e}")


def demo_configuration():
    """Show configuration options."""
    print_header("DEMO 6: Configuration Options")

    print_subheader("Query Rewriting Config")
    config = QueryRewriteConfig(
        enabled=True,
        max_history_turns=10,
        min_confidence=0.7,
        cache_enabled=True,
        cache_ttl_seconds=300,
    )
    print(f"  enabled: {config.enabled}")
    print(f"  max_history_turns: {config.max_history_turns}")
    print(f"  min_confidence: {config.min_confidence}")
    print(f"  cache_enabled: {config.cache_enabled}")
    print(f"  cache_ttl_seconds: {config.cache_ttl_seconds}")

    print_subheader("Sufficiency Config")
    suff_config = SufficiencyConfig(
        enabled=True,
        max_iterations=3,
        min_confidence=0.7,
        fallback_sources=["memories", "graph", "expertise"],
    )
    print(f"  enabled: {suff_config.enabled}")
    print(f"  max_iterations: {suff_config.max_iterations}")
    print(f"  min_confidence: {suff_config.min_confidence}")
    print(f"  fallback_sources: {suff_config.fallback_sources}")

    print_subheader("Environment Variables")
    print("  LLM Configuration:")
    print(f"    OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    print(f"    OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'not set')}")
    print(f"    AZURE_OPENAI_*: {'set' if os.getenv('AZURE_OPENAI_API_KEY') else 'not set'}")
    print("  Database Configuration:")
    print(f"    POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'not set')}")
    print(f"    MYSQL_HOST: {os.getenv('MYSQL_HOST', 'not set')}")

    print_subheader("YAML Configuration Example")
    print("""
dynamic_context:
  query_rewriting:
    enabled: true
    max_history_turns: 10
    min_confidence: 0.7
    cache_enabled: true

  sufficiency:
    enabled: true
    max_iterations: 3
    fallback_sources:
      - memories
      - graph
      - expertise
""")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  QUERY REWRITING & SUFFICIENCY JUDGING DEMO")
    print("=" * 70)

    # Create LLM provider based on configuration
    print_subheader("LLM Provider Setup")
    llm = create_llm_provider()

    await demo_query_rewriting(llm)
    await demo_sufficiency_judging(llm)
    await demo_progressive_retrieval(llm)
    await demo_combined_usage(llm)
    await demo_database_storage()
    demo_configuration()

    print_header("DEMO COMPLETE")
    print("\nAll features demonstrated successfully!")

    is_mock = isinstance(llm, DemoLLMProvider)
    if is_mock:
        print("\n💡 To use a real LLM provider:")
        print("   1. Copy env.example to .env")
        print("   2. Set OPENAI_API_KEY or AZURE_OPENAI_* variables")
        print("   3. Run the demo again")


if __name__ == "__main__":
    asyncio.run(main())
