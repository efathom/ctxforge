"""
Default configuration values for the ctxforge framework.

These defaults provide sensible starting points for development
and testing. Production deployments should customize as needed.
"""

from ctxforge.config.base import (
    CompactionConfig,
    CompactionStrategyType,
    EmbeddingConfig,
    EngineConfig,
    ExpertiseConfig,
    ExpertiseRetrievalConfig,
    ExpertiseStoreConfig,
    ExpertiseVectorStoreConfig,
    ExtractionConfig,
    GraphCommunitiesConfig,
    GraphConfig,
    GraphEmbeddingsConfig,
    GraphExtractionConfig,
    GraphInvalidationConfig,
    GraphOntologyConfig,
    GraphRetrievalConfig,
    GraphStoreConfig,
    GraphTemporalConfig,
    LLMConfig,
    LLMProviderType,
    LogLevel,
    MemoryStoreConfig,
    MemoryVectorStoreConfig,
    MiddlewareItemConfig,
    Neo4jGraphStoreConfig,
    ObservabilityConfig,
    PipelineConfig,
    PipelinesConfig,
    PluginsConfig,
    PromptConfig,
    RetrievalConfig,
    RetrievalStrategyType,
    ScopesConfig,
    SessionStoreConfig,
    SkillsConfig,
    StorageBackendType,
    StorageConfig,
    VectorStoreType,
)

# Default configuration for development/testing
DEFAULT_CONFIG = EngineConfig(
    name="ctxforge",
    version="0.1.0",
    debug=False,
    
    llm=LLMConfig(
        provider=LLMProviderType.MOCK,
        model="gpt-4",
        temperature=0.7,
        max_tokens=4096,
        timeout=30.0,
        max_retries=3,
    ),
    
    storage=StorageConfig(
        session=SessionStoreConfig(
            backend=StorageBackendType.MEMORY,
            ttl_seconds=86400,
            max_sessions_per_user=100,
        ),
        memory=MemoryStoreConfig(
            store_backend=StorageBackendType.MEMORY,
            vector=MemoryVectorStoreConfig(
                backend=VectorStoreType.MEMORY,
                index_name="agent_memories",
                embedding=EmbeddingConfig(
                    provider="openai",
                    model="text-embedding-3-small",
                    dimension=1536,
                    batch_size=100,
                ),
            ),
        ),
    ),

    expertise=ExpertiseConfig(
        enabled=False,
        store=ExpertiseStoreConfig(backend=StorageBackendType.MEMORY),
        vectorstore=ExpertiseVectorStoreConfig(
            backend=VectorStoreType.MEMORY,
            index_name="expertise_items",
            embedding=EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
                batch_size=100,
            ),
        ),
        retrieval=ExpertiseRetrievalConfig(
            enabled=True,
            default_limit=10,
            min_score=0.0,
            min_effectiveness=0.0,
            rerank_enabled=False,
            reranker="effectiveness",
            rerank_model=None,
            rerank_top_k=10,
        ),
    ),
    
    retrieval=RetrievalConfig(
        # Default must work without embeddings/vector DBs
        strategy=RetrievalStrategyType.KEYWORD,
        default_limit=5,
        min_score=0.0,
        semantic_weight=0.7,
        keyword_weight=0.3,
        rerank_enabled=False,
        reranker="llm",
        rerank_model=None,
        rerank_top_k=10,
    ),
    
    compaction=CompactionConfig(
        strategy=CompactionStrategyType.SUMMARIZE,
        event_threshold=10,
        token_threshold=4000,
        keep_recent=5,
        max_summary_tokens=500,
        async_compaction=True,
        include_tool_calls=True,
    ),
    
    skills=SkillsConfig(
        enabled=True,
    ),

    pipelines=PipelinesConfig(
        prepare=PipelineConfig(
            chain=[
                MiddlewareItemConfig(
                    type="pii",
                    enabled=True,
                    priority=100,
                    phases=["prepare_input"],
                    config={
                        "redact": True,
                        "redact_input": True,
                        "redact_response": True,
                    },
                ),
                MiddlewareItemConfig(
                    type="scoped_memory",
                    enabled=True,
                    priority=950,
                    phases=["prepare_input"],
                    config={},
                ),
                MiddlewareItemConfig(
                    type="skill_request",
                    enabled=True,
                    priority=910,
                    phases=["prepare_input"],
                    config={},
                ),
                MiddlewareItemConfig(
                    type="skills",
                    enabled=True,
                    priority=900,
                    phases=["prepare_input"],
                    config={},
                ),
                MiddlewareItemConfig(
                    type="query_rewriter",
                    enabled=True,
                    priority=800,
                    phases=["prepare_input"],
                    config={},
                ),
            ],
        ),
        record=PipelineConfig(
            chain=[
            MiddlewareItemConfig(
                    type="pii",
                enabled=True,
                priority=100,
                    phases=["record_input_output"],
                config={
                        "redact": True,
                        "redact_input": True,
                        "redact_response": True,
                },
            ),
        ],
        ),
    ),
    
    extraction=ExtractionConfig(
        enabled=True,
        async_processing=True,
        extract_semantic=True,
        extract_episodic=True,
        extract_procedural=False,
        min_confidence=0.5,
        max_candidates=10,
        use_llm=True,
        use_patterns=True,
        update_planning_enabled=False,
        update_planning_model=None,
        update_planning_candidates_per_item=5,
    ),
    
    observability=ObservabilityConfig(
        log_level=LogLevel.INFO,
        log_format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        log_to_file=False,
        tracing_enabled=False,
        metrics_enabled=False,
    ),
    
    prompts=PromptConfig(
        system_template="You are a helpful AI assistant.",
        memory_section_name="User Context",
        history_section_name="Recent History",
        max_history_events=10,
    ),

    scopes=ScopesConfig(
        enable_global=True,
        global_scope_id="global",
        global_retrieval_limit=5,
        global_score_weight=0.8,
        allow_global_writes=False,
    ),

    plugins=PluginsConfig(
        modules=[],
        registrations=[],
    ),

    graph=GraphConfig(
        enabled=False,
        store=GraphStoreConfig(
            backend="memory",
            neo4j=Neo4jGraphStoreConfig(
                url="bolt://localhost:7687",
                username="neo4j",
                password="neo4j",
                database=None,
                create_indexes=True,
                entity_label="__Entity__",
                fulltext_entity_index_name="ce_entity_fulltext",
                fulltext_edge_index_name="ce_edge_fulltext",
            ),
        ),
        ontology=GraphOntologyConfig(module=None, attr_name="GRAPH_ONTOLOGY"),
        extraction=GraphExtractionConfig(enabled=True, model=None),
        embeddings=GraphEmbeddingsConfig(
            enabled=False,
            embedding=EmbeddingConfig(),
            max_concurrency=8,
        ),
        invalidation=GraphInvalidationConfig(
            enabled=False,
            model=None,
            candidate_limit=25,
            max_concurrency=4,
        ),
        temporal=GraphTemporalConfig(
            enabled=False,
            model=None,
            max_concurrency=4,
        ),
        communities=GraphCommunitiesConfig(
            enabled=False,
            max_communities=5,
            rebuild_every_n_episodes=25,
            min_cluster_size=2,
            merge_similar_enabled=False,
            merge_similarity_threshold=0.92,
            attach_orphans_enabled=False,
            attach_similarity_threshold=0.85,
            max_nodes=2000,
            max_edges=4000,
            max_concurrency=4,
            model=None,
        ),
        retrieval=GraphRetrievalConfig(
            enabled=True,
            max_facts=20,
            max_entities=20,
            max_episodes=10,
            include_entities=True,
            include_episodes=False,
            valid_only=True,
            methods=["keyword", "bfs", "semantic"],
            seed_k=8,
            bfs_max_depth=1,
            bfs_edges_per_node=12,
            rerank_enabled=False,
            reranker="rrf",
            rerank_top_k=30,
        ),
        section_name="Graph Memory",
    ),
)


# Production-ready configuration template
PRODUCTION_CONFIG = DEFAULT_CONFIG.merge_with({
    "debug": False,
    "llm": {
        "provider": "openai",
        "max_retries": 5,
        "timeout": 60.0,
    },
    "storage": {
        "session": {
            "backend": "redis",
        },
        "memory": {
            "backend": "pinecone",
        },
    },
    "observability": {
        "log_level": "WARNING",
        "tracing_enabled": True,
        "metrics_enabled": True,
    },
})


# Development configuration with verbose logging
DEVELOPMENT_CONFIG = DEFAULT_CONFIG.merge_with({
    "debug": True,
    "observability": {
        "log_level": "DEBUG",
    },
})


# Testing configuration with mock providers
TESTING_CONFIG = DEFAULT_CONFIG.merge_with({
    "debug": True,
    "llm": {
        "provider": "mock",
    },
    "storage": {
        "session": {
            "backend": "memory",
            "ttl_seconds": 3600,
        },
        "memory": {
            "backend": "memory",
        },
    },
    "extraction": {"async_processing": False},  # Sync for easier testing
    "compaction": {"async_compaction": False},  # Sync for easier testing
})

