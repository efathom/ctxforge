#!/usr/bin/env python3
"""
Dynamic Context Demo.

Demonstrates the new Dynamic Context features:
1. Validated Knowledge Service - Direct save path for user-approved knowledge
2. Human-in-the-Loop Approval - Queuing knowledge for approval
3. Structured Knowledge Types - Classification and filtering
4. Semantic Model - Domain schema injection
5. Expertise Snapshots - Version tracking and diffing
6. Unified Retrieval - Cross-store search
7. Search-Before-Respond - Directive injection

Usage:
    python -m ctxforge.examples.dynamic_context_demo

    # With specific features
    python -m ctxforge.examples.dynamic_context_demo --feature snapshots
    python -m ctxforge.examples.dynamic_context_demo --feature unified

    # Test different store backends
    python -m ctxforge.examples.dynamic_context_demo --feature stores
    python -m ctxforge.examples.dynamic_context_demo --feature stores --store-type mysql
    python -m ctxforge.examples.dynamic_context_demo --feature stores --store-type file
"""

import argparse
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure `ctxforge` is importable when running as a script (not `python -m ...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Core imports
from ctxforge.core.expertise import Expertise, ExpertiseSection

# Knowledge types
from ctxforge.core.knowledge_types import (
    HeuristicKnowledgeClassifier,
    KnowledgeScope,
    KnowledgeType,
    LLMKnowledgeClassifier,
    StructuredKnowledge,
)

# Semantic model
from ctxforge.core.semantic_model import (
    EntityDefinition,
    FileBasedSemanticModelStore,
    InMemorySemanticModelStore,
    RelationshipDefinition,
    SemanticModel,
)

# Snapshots
from ctxforge.engine.services.expertise_snapshot_service import (
    ExpertiseSnapshotService,
    FileBasedSnapshotStore,
    InMemorySnapshotStore,
)
from ctxforge.engine.services.semantic_model_service import SemanticModelService

# Approval middleware
from ctxforge.middleware.approval import (
    ApprovalRequest,
    ApprovalStatus,
    InMemoryApprovalStore,
)

# Search-before-respond
from ctxforge.middleware.search_before_respond import (
    HybridIntentClassifier,
    LLMIntentClassifier,
    SearchBeforeRespondMiddleware,
    SearchIntentClassifier,
)
from ctxforge.retrieval.filters.knowledge_type import KnowledgeTypeFilter

# Unified retrieval
from ctxforge.retrieval.unified_retriever import (
    ResultSource,
    RetrievalResult,
    UnifiedRetriever,
)


async def demo_knowledge_types():
    """Demonstrate structured knowledge types and classification."""
    print("\n" + "=" * 60)
    print("📚 DEMO: Structured Knowledge Types")
    print("=" * 60)
    
    # Classify various content with improved heuristic classifier
    examples = [
        "Always validate user input before processing",
        "SELECT * FROM users WHERE id = ?",
        "Don't forget to close database connections",
        "Step 1: Open settings. Step 2: Click save.",
        "A transaction means an atomic database operation",
        "Limit results to 100 rows maximum",
        "For example, use a prepared statement like this",
    ]
    
    classifier = HeuristicKnowledgeClassifier()
    
    print("\n🔍 Auto-classifying knowledge content (with confidence):\n")
    for content in examples:
        kt, confidence = await classifier.classify(content)
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        print(f"  [{kt.value.upper():12}] {conf_bar} {confidence:.0%}  {content[:45]}...")
    
    # Create structured knowledge
    print("\n📋 Creating structured knowledge item:")
    rule = StructuredKnowledge(
        knowledge_type=KnowledgeType.RULE,
        name="Input Validation",
        content="Always validate user input before processing to prevent injection attacks",
        scope=KnowledgeScope.GLOBAL,
        priority=10,
        tags=["security", "validation"],
    )
    print(f"  {rule.to_prompt_format()}")
    
    # Knowledge type filter
    print("\n🔧 Filtering by knowledge type:")
    filter = KnowledgeTypeFilter(
        include_types=[KnowledgeType.RULE, KnowledgeType.GOTCHA],
        priority_types=[KnowledgeType.RULE],
    )
    print("  Include types: RULE, GOTCHA")
    print(f"  Priority boost for RULE: {filter.get_priority_boost(KnowledgeType.RULE)}")
    print(f"  Priority boost for INSIGHT: {filter.get_priority_boost(KnowledgeType.INSIGHT)}")


async def demo_semantic_model():
    """Demonstrate semantic model context anchor."""
    print("\n" + "=" * 60)
    print("🗺️  DEMO: Semantic Model Context Anchor")
    print("=" * 60)
    
    # Create a semantic model
    model = SemanticModel(
        name="Customer Service KB",
        description="Knowledge base for customer service agents",
        version="1.0.0",
        entities=[
            EntityDefinition(
                name="customers",
                description="Customer profiles and account information",
                use_cases=["Find customer by ID", "Update contact info"],
            ),
            EntityDefinition(
                name="orders",
                description="Order history and status tracking",
                use_cases=["Check order status", "Process refunds"],
            ),
            EntityDefinition(
                name="products",
                description="Product catalog and inventory",
                use_cases=["Search products", "Check availability"],
            ),
        ],
        relationships=[
            RelationshipDefinition(
                name="customer_orders",
                from_entity="customers",
                to_entity="orders",
                description="Customer's order history",
                cardinality="one_to_many",
            ),
        ],
        global_rules=[
            "Always verify customer identity before sharing account details",
            "Escalate refund requests over $500 to supervisor",
        ],
        common_gotchas=[
            "Check both email and phone when looking up customers",
            "Order status may have 24-hour delay in some regions",
        ],
    )
    
    print("\n📄 Semantic model in compact format:\n")
    print(model.to_context_string(compact=True))
    
    # Service demo
    store = InMemorySemanticModelStore()
    await store.save("customer-service", model)
    
    service = SemanticModelService(store=store, default_model=model)
    
    print("\n✅ Model saved to store")
    models = await service.list_models()
    print(f"   Available models: {models}")


async def demo_snapshots():
    """Demonstrate expertise snapshots and diffing."""
    print("\n" + "=" * 60)
    print("📸 DEMO: Expertise Snapshots with Diffing")
    print("=" * 60)
    
    # Create an expertise
    expertise = Expertise(
        expertise_id="sql-best-practices",
        name="SQL Best Practices",
        domain="database",
    )
    expertise.add_item(
        ExpertiseSection.STRATEGIES, "Use parameterized queries to prevent SQL injection"
    )
    expertise.add_item(ExpertiseSection.HEURISTICS, "Index columns used in WHERE clauses")
    
    # Create snapshot service
    store = InMemorySnapshotStore()
    service = ExpertiseSnapshotService(store=store)
    
    # Create version 1.0
    print("\n📸 Creating snapshot v1.0...")
    snap_v1 = await service.create_snapshot(
        expertise=expertise,
        version="1.0.0",
        created_by="admin",
        description="Initial version",
    )
    print(f"   Snapshot ID: {snap_v1.snapshot_id}")
    print(f"   Content hash: {snap_v1.content_hash}")
    print(f"   Items: {len(snap_v1.items)}")
    
    # Add more items and create version 1.1
    expertise.add_item(ExpertiseSection.STRATEGIES, "Use EXPLAIN to analyze query performance")
    expertise.add_item(ExpertiseSection.COMMON_MISTAKES, "Avoid SELECT * in production code")
    
    print("\n📸 Creating snapshot v1.1 (with new items)...")
    snap_v2 = await service.create_snapshot(
        expertise=expertise,
        version="1.1.0",
        created_by="admin",
        description="Added performance tips",
    )
    print(f"   Snapshot ID: {snap_v2.snapshot_id}")
    print(f"   Items: {len(snap_v2.items)}")
    
    # Compare versions
    print("\n🔍 Comparing v1.0 → v1.1:")
    diff = await service.diff_versions("sql-best-practices", "1.0.0", "1.1.0")
    if diff:
        print(f"   Added: {diff.items_added}")
        print(f"   Removed: {diff.items_removed}")
        print(f"   Modified: {diff.items_modified}")
        print(f"\n📋 Changelog:\n{diff.to_changelog()}")


async def demo_unified_retrieval():
    """Demonstrate unified cross-store retrieval."""
    print("\n" + "=" * 60)
    print("🔎 DEMO: Unified Cross-Store Retrieval")
    print("=" * 60)
    
    # Create retriever with weighted sources
    retriever = UnifiedRetriever(
        merge_strategy="interleave",
        score_weights={
            ResultSource.EXPERTISE: 1.2,  # Boost expertise
            ResultSource.MEMORY: 1.0,
            ResultSource.GRAPH: 0.8,
        },
    )
    
    # Create mock adapters
    class MockExpertiseAdapter:
        async def search(self, query: str, limit: int = 10, **kwargs):
            return [
                RetrievalResult(
                    content="Use parameterized queries for SQL",
                    score=0.9,
                    source=ResultSource.EXPERTISE,
                    knowledge_type="rule",
                ),
                RetrievalResult(
                    content="Always validate input data",
                    score=0.85,
                    source=ResultSource.EXPERTISE,
                    knowledge_type="rule",
                ),
            ]
    
    class MockMemoryAdapter:
        async def search(self, query: str, limit: int = 10, **kwargs):
            return [
                RetrievalResult(
                    content="User prefers detailed explanations",
                    score=0.95,
                    source=ResultSource.MEMORY,
                    knowledge_type="preference",
                ),
                RetrievalResult(
                    content="User asked about SQL optimization yesterday",
                    score=0.7,
                    source=ResultSource.MEMORY,
                ),
            ]
    
    # Register stores
    retriever.register_store(
        name="expertise",
        source=ResultSource.EXPERTISE,
        adapter=MockExpertiseAdapter(),
        priority=10,
    )
    retriever.register_store(
        name="memory",
        source=ResultSource.MEMORY,
        adapter=MockMemoryAdapter(),
        priority=5,
    )
    
    print("\n🔍 Searching across all stores...")
    results = await retriever.search("SQL best practices", max_results=5)
    
    print(f"\n📊 Found {len(results)} results:\n")
    formatted = retriever.format_results(results, include_score=True)
    print(formatted)
    
    # Filter by source
    print("\n🔍 Filtering to expertise only...")
    expertise_results = await retriever.search(
        "SQL best practices",
        sources=[ResultSource.EXPERTISE],
    )
    print(f"   Found {len(expertise_results)} expertise results")


async def demo_approval_workflow():
    """Demonstrate human-in-the-loop approval."""
    print("\n" + "=" * 60)
    print("✋ DEMO: Human-in-the-Loop Approval")
    print("=" * 60)
    
    # Create approval store
    store = InMemoryApprovalStore()
    
    # Create approval request
    request = ApprovalRequest(
        session_id="session-123",
        user_id="user-456",
        knowledge_type="expertise_item",
        proposed_content="Always use HTTPS for API calls",
        source_question="How should I secure my API endpoints?",
        reasoning="This is a common security best practice",
    )
    
    await store.save_request(request)
    print(f"\n📝 Created approval request: {request.request_id[:8]}...")
    print(f"   Type: {request.knowledge_type}")
    print(f"   Content: {request.proposed_content}")
    print(f"   Status: {request.status}")
    
    # Get pending requests
    pending = await store.get_pending_for_user("user-456")
    print(f"\n⏳ Pending requests for user: {len(pending)}")
    
    # Approve the request
    updated = await store.update_status(
        request.request_id,
        ApprovalStatus.APPROVED,
    )
    print(f"\n✅ Request approved at: {updated.resolved_at}")
    
    # Check pending again
    pending = await store.get_pending_for_user("user-456")
    print(f"   Pending requests now: {len(pending)}")


async def demo_search_before_respond():
    """Demonstrate search-before-respond middleware."""
    print("\n" + "=" * 60)
    print("🔍 DEMO: Search-Before-Respond Directive")
    print("=" * 60)
    
    # Create middleware
    middleware = SearchBeforeRespondMiddleware(
        knowledge_domains=["SQL patterns", "best practices", "gotchas"],
    )
    
    # Test intent detection
    queries = [
        "How do I query users by email?",
        "What's the best way to handle joins?",
        "Hello!",
        "Show me the order history",
        "Thanks for your help",
    ]
    
    print("\n🔍 Testing query intent detection:\n")
    for query in queries:
        should_inject = middleware.should_inject_directive(query)
        status = "✅ SEARCH" if should_inject else "⏭️  SKIP"
        print(f"  {status}: \"{query}\"")
    
    # Intent classifier
    print("\n🎯 Intent classification:")
    classifier = SearchIntentClassifier()
    
    test_queries = [
        "How do I write a SQL query for this?",
        "What is the difference between INNER and OUTER join?",
        "Is this query correct?",
    ]
    
    for query in test_queries:
        result = await classifier.classify(query)
        domains = classifier.get_search_domains(result.intents)
        print(f"\n  Query: \"{query}\"")
        print(f"  Intents: {result.intents or 'none'}")
        print(f"  Search domains: {domains}")
    
    # Show generated directive
    print("\n📋 Generated directive:\n")
    directive = middleware.generate_directive()
    print(f"  {directive}")


async def demo_store_backends(store_type: str = "memory"):
    """Demonstrate different storage backends for semantic models and snapshots."""
    print("\n" + "=" * 60)
    print(f"💾 DEMO: Storage Backends ({store_type.upper()})")
    print("=" * 60)

    # =========================================================================
    # Semantic Model Stores
    # =========================================================================
    print("\n📦 SEMANTIC MODEL STORES\n")

    # Create a sample model
    model = SemanticModel(
        name="Test Model",
        description="Model for testing storage backends",
        version="1.0.0",
        entities=[
            EntityDefinition(
                name="users",
                description="User accounts",
                use_cases=["Find user", "Update profile"],
            ),
        ],
        global_rules=["Validate input", "Log access"],
    )

    if store_type == "memory":
        print("  Using: InMemorySemanticModelStore")
        model_store = InMemorySemanticModelStore()

    elif store_type == "file":
        temp_dir = tempfile.mkdtemp(prefix="ctxforge_models_")
        print("  Using: FileBasedSemanticModelStore")
        print(f"  Directory: {temp_dir}")
        model_store = FileBasedSemanticModelStore(temp_dir)

    elif store_type == "mysql":
        print("  Using: MySQLSemanticModelStore")
        try:
            from ctxforge.storage.connection import MySQLConfig
            from ctxforge.storage.mysql import MySQLSemanticModelStore
            config = MySQLConfig(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                database=os.getenv("MYSQL_DATABASE", "ctxforge_test"),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
            )
            model_store = MySQLSemanticModelStore(config, table_name="demo_semantic_models")
            await model_store.initialize()
            print(f"  Connected to: {config.host}:{config.port}/{config.database}")
        except ImportError:
            print("  ❌ aiomysql not installed. Run: pip install aiomysql")
            return
        except Exception as e:
            print(f"  ❌ MySQL connection failed: {e}")
            return

    elif store_type == "postgres":
        print("  Using: PostgresSemanticModelStore")
        try:
            from ctxforge.storage.connection import PostgresConfig
            from ctxforge.storage.postgres import PostgresSemanticModelStore
            config = PostgresConfig(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DATABASE", "ctxforge_test"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
            )
            model_store = PostgresSemanticModelStore(config, table_name="demo_semantic_models")
            await model_store.initialize()
            print(f"  Connected to: {config.host}:{config.port}/{config.database}")
        except ImportError:
            print("  ❌ asyncpg not installed. Run: pip install asyncpg")
            return
        except Exception as e:
            print(f"  ❌ PostgreSQL connection failed: {e}")
            return
    else:
        print(f"  ❌ Unknown store type: {store_type}")
        return

    # Test save and load
    print("\n  📝 Saving model 'test-model'...")
    await model_store.save("test-model", model)
    print("  ✅ Saved!")

    print("\n  📖 Loading model...")
    loaded = await model_store.load("test-model")
    if loaded:
        print(f"  ✅ Loaded: {loaded.name} v{loaded.version}")
        print(f"     Entities: {[e.name for e in loaded.entities]}")
    else:
        print("  ❌ Failed to load model")

    print("\n  📋 Listing models...")
    models = await model_store.list_models()
    print(f"  ✅ Found: {models}")

    # =========================================================================
    # Expertise Snapshot Stores
    # =========================================================================
    print("\n" + "-" * 40)
    print("\n📸 EXPERTISE SNAPSHOT STORES\n")

    # Create sample expertise
    expertise = Expertise(
        expertise_id="demo-expertise",
        name="Demo Expertise",
        domain="testing",
    )
    expertise.add_item(ExpertiseSection.STRATEGIES, "Use unit tests")
    expertise.add_item(ExpertiseSection.HEURISTICS, "Test edge cases")

    if store_type == "memory":
        print("  Using: InMemorySnapshotStore")
        snapshot_store = InMemorySnapshotStore()

    elif store_type == "file":
        temp_dir = tempfile.mkdtemp(prefix="ctxforge_snapshots_")
        print("  Using: FileBasedSnapshotStore")
        print(f"  Directory: {temp_dir}")
        snapshot_store = FileBasedSnapshotStore(temp_dir)

    elif store_type == "mysql":
        print("  Using: MySQLSnapshotStore")
        try:
            from ctxforge.storage.connection import MySQLConfig
            from ctxforge.storage.mysql import MySQLSnapshotStore
            config = MySQLConfig(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                database=os.getenv("MYSQL_DATABASE", "ctxforge_test"),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
            )
            snapshot_store = MySQLSnapshotStore(config, table_name="demo_snapshots")
            await snapshot_store.initialize()
            print(f"  Connected to: {config.host}:{config.port}/{config.database}")
        except ImportError:
            print("  ❌ aiomysql not installed. Run: pip install aiomysql")
            return
        except Exception as e:
            print(f"  ❌ MySQL connection failed: {e}")
            return

    elif store_type == "postgres":
        print("  Using: PostgresSnapshotStore")
        try:
            from ctxforge.storage.connection import PostgresConfig
            from ctxforge.storage.postgres import PostgresSnapshotStore
            config = PostgresConfig(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DATABASE", "ctxforge_test"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
            )
            snapshot_store = PostgresSnapshotStore(config, table_name="demo_snapshots")
            await snapshot_store.initialize()
            print(f"  Connected to: {config.host}:{config.port}/{config.database}")
        except ImportError:
            print("  ❌ asyncpg not installed. Run: pip install asyncpg")
            return
        except Exception as e:
            print(f"  ❌ PostgreSQL connection failed: {e}")
            return

    # Create snapshot service
    service = ExpertiseSnapshotService(store=snapshot_store)

    # Create version 1.0
    print("\n  📸 Creating snapshot v1.0...")
    snap_v1 = await service.create_snapshot(
        expertise=expertise,
        version="1.0.0",
        created_by="demo",
        description="Initial version",
    )
    print(f"  ✅ Created: {snap_v1.snapshot_id}")
    print(f"     Hash: {snap_v1.content_hash}")
    print(f"     Items: {len(snap_v1.items)}")

    # Add items and create version 1.1
    expertise.add_item(ExpertiseSection.COMMON_MISTAKES, "Skipping integration tests")

    print("\n  📸 Creating snapshot v1.1...")
    snap_v2 = await service.create_snapshot(
        expertise=expertise,
        version="1.1.0",
        created_by="demo",
        description="Added mistake warning",
    )
    print(f"  ✅ Created: {snap_v2.snapshot_id}")
    print(f"     Items: {len(snap_v2.items)}")

    # List versions
    print("\n  📋 Listing versions...")
    versions = await snapshot_store.list_versions("demo-expertise")
    print(f"  ✅ Found: {versions}")

    # Diff versions
    print("\n  🔍 Comparing v1.0 → v1.1...")
    diff = await service.diff_versions("demo-expertise", "1.0.0", "1.1.0")
    if diff:
        print(f"  ✅ Changes: +{diff.items_added} -{diff.items_removed} ~{diff.items_modified}")

    # Cleanup for database stores
    if store_type in ("mysql", "postgres"):
        print("\n  🧹 Cleaning up database tables...")
        try:
            if hasattr(model_store, '_manager'):
                await model_store._manager.execute("DROP TABLE IF EXISTS demo_semantic_models")
            if hasattr(snapshot_store, '_manager'):
                await snapshot_store._manager.execute("DROP TABLE IF EXISTS demo_snapshots")
            print("  ✅ Cleanup complete")
        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")

        # Disconnect
        if hasattr(model_store, 'disconnect'):
            await model_store.disconnect()
        if hasattr(snapshot_store, 'disconnect'):
            await snapshot_store.disconnect()

    print("\n  ✅ Store backend demo complete!")


async def demo_llm_classifiers():
    """Demonstrate LLM-based classifiers for knowledge types and search intents."""
    print("\n" + "=" * 60)
    print("🤖 DEMO: LLM-Based Classifiers")
    print("=" * 60)

    # Create a mock LLM provider that simulates intelligent classification
    class MockLLMProvider:
        """Mock LLM that simulates intelligent classification responses."""

        async def complete(self, prompt: str) -> str:
            # Detect if this is knowledge type or intent classification
            if "RULE" in prompt or "GOTCHA" in prompt or "knowledge type" in prompt.lower():
                return await self._classify_knowledge(prompt)
            else:
                return await self._classify_intent(prompt)

        async def _classify_knowledge(self, prompt: str) -> str:
            import json
            import re

            # Extract content from prompt
            match = re.search(r'Content:\s*["\']?(.+?)["\']?\s*(?:Respond|$)', prompt, re.DOTALL)
            content = match.group(1).lower() if match else prompt.lower()

            # Intelligent classification based on content
            if any(w in content for w in ['always', 'never', 'must', 'should']):
                return json.dumps({
                    "knowledge_type": "rule",
                    "confidence": 0.92,
                    "reasoning": "Contains prescriptive language (always/must/should)"
                })
            elif any(w in content for w in ["don't forget", "watch out", "careful", "gotcha"]):
                return json.dumps({
                    "knowledge_type": "gotcha",
                    "confidence": 0.88,
                    "reasoning": "Contains warning/caution language"
                })
            elif any(w in content for w in ['select', 'from', 'where', 'join', 'insert']):
                return json.dumps({
                    "knowledge_type": "pattern",
                    "confidence": 0.95,
                    "reasoning": "Contains SQL query pattern"
                })
            elif any(w in content for w in ['step 1', 'step 2', 'first', 'then', 'finally']):
                return json.dumps({
                    "knowledge_type": "procedure",
                    "confidence": 0.90,
                    "reasoning": "Contains step-by-step instructions"
                })
            elif any(w in content for w in ['means', 'defined as', 'refers to', 'is a']):
                return json.dumps({
                    "knowledge_type": "definition",
                    "confidence": 0.87,
                    "reasoning": "Contains definitional language"
                })
            else:
                return json.dumps({
                    "knowledge_type": "insight",
                    "confidence": 0.75,
                    "reasoning": "General observation or insight"
                })

        async def _classify_intent(self, prompt: str) -> str:
            import json
            import re

            # Extract query from prompt
            match = re.search(r'User query:\s*["\'](.+?)["\']', prompt, re.IGNORECASE)
            query = match.group(1).lower() if match else prompt.lower()

            intents = []
            confidence = 0.85

            if any(w in query for w in ['query', 'sql', 'select', 'find', 'get data']):
                intents.append('query')
                confidence = 0.95
            if any(w in query for w in ['what is', 'explain', 'tell me about', 'describe']):
                intents.append('lookup')
            if any(w in query for w in ['how do', 'how can', 'steps', 'guide', 'walk me']):
                intents.append('procedure')
            if any(w in query for w in ['difference', 'compare', 'vs', 'versus']):
                intents.append('comparison')
            if any(w in query for w in ['error', 'fix', 'not working', 'debug', 'failing']):
                intents.append('troubleshooting')
                confidence = 0.92
            if any(w in query for w in ['example', 'sample', 'show me', 'demonstrate']):
                intents.append('example')
            if any(w in query for w in ['correct', 'valid', 'right', 'verify', 'check']):
                intents.append('validation')

            if not intents:
                return json.dumps({
                    "intents": [],
                    "confidence": 0.3,
                    "reasoning": "No clear intent detected"
                })

            return json.dumps({
                "intents": intents,
                "confidence": confidence,
                "reasoning": f"Detected {len(intents)} intent(s) from query patterns"
            })

    mock_llm = MockLLMProvider()

    # =========================================================================
    # Knowledge Type Classification
    # =========================================================================
    print("\n📚 KNOWLEDGE TYPE CLASSIFICATION")
    print("-" * 50)

    knowledge_examples = [
        "Always validate user input before processing",
        "SELECT * FROM users WHERE status = 'active'",
        "Don't forget to close database connections after use",
        "Step 1: Open settings. Step 2: Click save. Step 3: Confirm.",
        "A transaction is defined as an atomic unit of work",
        "Consider using connection pooling for better performance",
        "Maximum 100 rows per query result",
    ]

    # Compare classifiers
    heuristic_clf = HeuristicKnowledgeClassifier()
    llm_clf = LLMKnowledgeClassifier(llm_provider=mock_llm)

    print(f"\n{'Content':<45} | {'Heuristic':<12} | {'LLM':<12} | {'LLM Conf':>8}")
    print("-" * 85)

    for content in knowledge_examples:
        h_type, h_conf = await heuristic_clf.classify(content)
        l_type, l_conf = await llm_clf.classify(content)

        h_str = f"{h_type.value}"
        l_str = f"{l_type.value}"

        print(f"{content[:43]:<45} | {h_str:<12} | {l_str:<12} | {l_conf:>7.0%}")

    # =========================================================================
    # Search Intent Classification
    # =========================================================================
    print("\n\n🔍 SEARCH INTENT CLASSIFICATION")
    print("-" * 50)

    intent_examples = [
        "How do I write a SQL query to find inactive users?",
        "What is the difference between INNER and OUTER join?",
        "Getting an error: connection timeout",
        "Show me an example of a GROUP BY with HAVING",
        "Is this query correct?",
        "Explain what a foreign key is",
    ]

    # Compare classifiers
    pattern_clf = SearchIntentClassifier()
    llm_intent_clf = LLMIntentClassifier(llm_provider=mock_llm)

    print(f"\n{'Query':<45} | {'Pattern':<20} | {'LLM':<20}")
    print("-" * 90)

    for query in intent_examples:
        p_result = await pattern_clf.classify(query)
        l_result = await llm_intent_clf.classify(query)

        p_intents = ', '.join(sorted(p_result.intents)) if p_result.intents else '-'
        l_intents = ', '.join(sorted(l_result.intents)) if l_result.intents else '-'

        print(f"{query[:43]:<45} | {p_intents:<20} | {l_intents:<20}")

    # =========================================================================
    # Hybrid Classifier Demo
    # =========================================================================
    print("\n\n🔀 HYBRID CLASSIFIER (Pattern + LLM Fallback)")
    print("-" * 50)

    hybrid_intent_clf = HybridIntentClassifier(
        embedding_provider=None,
        pattern_confidence_threshold=0.85,
        combine_results=True,
    )

    # Test a few queries
    test_queries = [
        "How do I optimize this slow query?",
        "What happens when connection pool is exhausted?",
        "Fix the syntax error in my SQL",
    ]

    for query in test_queries:
        result = await hybrid_intent_clf.classify(query)
        intents = ', '.join(sorted(result.intents)) if result.intents else '-'
        print(f"  [{result.method:<18}] {result.confidence:.0%} | {intents:<25} | {query[:35]}")

    print("\n  ✅ LLM classifiers demo complete!")


async def main():
    """Run the dynamic context demos."""
    parser = argparse.ArgumentParser(description="Dynamic Context Demo")
    parser.add_argument(
        "--feature",
        choices=["types", "semantic", "snapshots", "unified", "approval", "search", "stores",
                 "llm", "all"],
        default="all",
        help="Which feature to demo",
    )
    parser.add_argument(
        "--store-type",
        choices=["memory", "file", "mysql", "postgres"],
        default="memory",
        help="Storage backend type (for --feature stores)",
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🚀 CTXFORGE DYNAMIC CONTEXT DEMO")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
    
    feature = args.feature
    
    if feature in ("types", "all"):
        await demo_knowledge_types()
    
    if feature in ("semantic", "all"):
        await demo_semantic_model()
    
    if feature in ("snapshots", "all"):
        await demo_snapshots()
    
    if feature in ("unified", "all"):
        await demo_unified_retrieval()
    
    if feature in ("approval", "all"):
        await demo_approval_workflow()
    
    if feature in ("search", "all"):
        await demo_search_before_respond()

    if feature == "stores":
        await demo_store_backends(args.store_type)

    if feature == "llm":
        await demo_llm_classifiers()

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
