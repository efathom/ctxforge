"""
Configuration Schemas for the ctxforge framework.

All configuration is defined using Pydantic models for
type safety, validation, and easy serialization.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LogLevel(str, Enum):
    """Logging levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProviderType(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"
    MOCK = "mock"
    OPENROUTER = "openrouter"


class StorageBackendType(str, Enum):
    """Supported storage backends."""
    
    MEMORY = "memory"
    REDIS = "redis"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class VectorStoreType(str, Enum):
    """Supported vector stores."""
    
    MEMORY = "memory"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    CHROMADB = "chromadb"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class RetrievalStrategyType(str, Enum):
    """Supported retrieval strategies."""
    
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    TEMPORAL = "temporal"
    SALIENCE = "salience"


class CompactionStrategyType(str, Enum):
    """Supported compaction strategies."""
    
    SUMMARIZE = "summarize"
    PRUNE = "prune"
    SLIDING_WINDOW = "sliding_window"
    IMPORTANCE = "importance"
    STRUCTURED = "structured"
    PIPELINE = "pipeline"


# =============================================================================
# Component Configurations
# =============================================================================

class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: LLMProviderType = LLMProviderType.MOCK
    model: str = "gpt-4"
    api_key: Optional[str] = None  # Can be set via environment
    api_base: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    timeout: float = Field(default=30.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    
    # Additional parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v, info):
        """API key can be None for mock provider."""
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""
    
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    dimension: int = 1536
    batch_size: int = 100
    # Optional override of the OpenAI-compatible endpoint (e.g. a local
    # Text Embeddings Inference / Ollama / vLLM server).
    base_url: Optional[str] = None


class SessionStoreConfig(BaseModel):
    """Configuration for session storage."""
    
    backend: StorageBackendType = StorageBackendType.MEMORY
    connection_string: Optional[str] = None
    ttl_seconds: int = Field(default=86400, ge=0)  # 24 hours
    max_sessions_per_user: int = Field(default=100, ge=1)
    
    # Backend-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class MemoryVectorStoreConfig(BaseModel):
    """Configuration for the memory vector index backend."""

    backend: VectorStoreType = VectorStoreType.MEMORY
    connection_string: Optional[str] = None
    index_name: str = "agent_memories"

    # Embedding settings
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Backend-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class MemoryStoreConfig(BaseModel):
    """
    Configuration for memory persistence + (optional) vector indexing.

    New shape:
    - store_backend: persistence backend (IMemoryStore)
    - vector: vector index backend (IVectorStore used by MemoryIndexer)

    Backward compatibility:
    - legacy keys under storage.memory.* are accepted and translated:
      - storage.memory.backend:
        - if it matches StorageBackendType -> store_backend
        - if it matches VectorStoreType -> vector.backend
      - storage.memory.{connection_string,index_name,embedding,extra_params} -> vector.*
    """

    # Persistence (IMemoryStore)
    store_backend: StorageBackendType = StorageBackendType.MEMORY
    store_connection_string: Optional[str] = None
    store_extra_params: Dict[str, Any] = Field(default_factory=dict)

    # Vector index (IVectorStore)
    vector: MemoryVectorStoreConfig = Field(default_factory=MemoryVectorStoreConfig)

    # Legacy keys (accepted as input; excluded from dumps)
    backend: Optional[str] = Field(default=None, exclude=True)
    connection_string: Optional[str] = Field(default=None, exclude=True)
    index_name: Optional[str] = Field(default=None, exclude=True)
    embedding: Optional[EmbeddingConfig] = Field(default=None, exclude=True)
    extra_params: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _translate_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # If the new fields are already present, prefer them but still allow minor fixes.
        vector = dict(d.get("vector") or {})

        legacy_backend = d.get("backend")
        if legacy_backend is not None:
            backend_s = str(legacy_backend).lower()
            # If legacy backend matches a persistence backend, treat it as store_backend.
            if backend_s in {e.value for e in StorageBackendType}:
                d.setdefault("store_backend", backend_s)
                # If vector backend not explicitly provided, keep vector disabled by default.
                vector.setdefault("backend", vector.get("backend", VectorStoreType.MEMORY.value))
            # If legacy backend matches a vector store backend, treat it as vector.backend.
            elif backend_s in {e.value for e in VectorStoreType}:
                # Legacy `backend` historically referred to the vector backend; treat it as an override.
                vector["backend"] = backend_s
                d.setdefault("store_backend", StorageBackendType.MEMORY.value)

        # Legacy vector fields promoted into vector.*
        if "connection_string" in d and d.get("connection_string") is not None:
            vector["connection_string"] = d.get("connection_string")
        if "index_name" in d and d.get("index_name") is not None:
            vector["index_name"] = d.get("index_name")
        if "embedding" in d and d.get("embedding") is not None:
            vector["embedding"] = d.get("embedding")
        if "extra_params" in d and d.get("extra_params") is not None:
            vector["extra_params"] = d.get("extra_params")

        if vector:
            d["vector"] = vector

        return d


class StorageConfig(BaseModel):
    """Combined storage configuration."""
    
    session: SessionStoreConfig = Field(default_factory=SessionStoreConfig)
    memory: MemoryStoreConfig = Field(default_factory=MemoryStoreConfig)


class ExpertiseStoreConfig(BaseModel):
    """Configuration for expertise persistence store."""

    backend: StorageBackendType = StorageBackendType.MEMORY
    connection_string: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ExpertiseVectorStoreConfig(BaseModel):
    """
    Configuration for expertise vector store.
    
    Separate from memory vector store (Option B).
    """

    backend: VectorStoreType = VectorStoreType.MEMORY
    connection_string: Optional[str] = None
    index_name: str = "expertise_items"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ExpertiseRetrievalConfig(BaseModel):
    """Configuration for expertise retrieval."""

    enabled: bool = True
    default_limit: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_effectiveness: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_enabled: bool = False
    reranker: str = "effectiveness"
    rerank_model: Optional[str] = None
    rerank_top_k: int = 10


class ExpertiseConfig(BaseModel):
    """Top-level expertise configuration."""

    enabled: bool = False
    store: ExpertiseStoreConfig = Field(default_factory=ExpertiseStoreConfig)
    vectorstore: ExpertiseVectorStoreConfig = Field(default_factory=ExpertiseVectorStoreConfig)
    retrieval: ExpertiseRetrievalConfig = Field(default_factory=ExpertiseRetrievalConfig)


class RetrievalConfig(BaseModel):
    """Configuration for memory retrieval."""
    
    strategy: RetrievalStrategyType = RetrievalStrategyType.SEMANTIC
    default_limit: int = Field(default=5, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Hybrid search settings
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Reranking settings
    rerank_enabled: bool = False
    reranker: str = "llm"
    rerank_model: Optional[str] = None
    rerank_top_k: int = 10


class CondenserStepConfig(BaseModel):
    """Configuration for a single condenser in a pipeline."""

    type: str  # "sliding_window", "summarizing", "importance", "structured"
    config: Dict[str, Any] = Field(default_factory=dict)


class ContextHealthConfig(BaseModel):
    """Configuration for context window health monitoring."""

    enabled: bool = True
    info_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    warning_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    critical_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    inject_warnings: bool = False
    inject_at_level: str = "warning"


class CompactionConfig(BaseModel):
    """Configuration for context compaction."""

    strategy: CompactionStrategyType = CompactionStrategyType.SUMMARIZE

    # Thresholds
    event_threshold: int = Field(default=10, ge=1)
    token_threshold: int = Field(default=4000, ge=100)

    # Retention
    keep_recent: int = Field(default=5, ge=1)

    # Summarization
    summarization_model: Optional[str] = None
    max_summary_tokens: int = Field(default=500, ge=50)

    # Options
    async_compaction: bool = True
    include_tool_calls: bool = True

    # Pipeline configuration (used when strategy=PIPELINE)
    pipeline: List[CondenserStepConfig] = Field(default_factory=list)

    # Structured summary options (used when strategy=STRUCTURED)
    structured_max_events: int = Field(default=100, ge=10)
    structured_keep_first: int = Field(default=1, ge=0)
    structured_keep_last: int = Field(default=10, ge=1)

    # Context health monitoring
    health: ContextHealthConfig = Field(default_factory=ContextHealthConfig)


class MiddlewareItemConfig(BaseModel):
    """Configuration for a single middleware."""
    
    type: str
    enabled: bool = True
    priority: int = 0
    phases: Optional[List[str]] = None  # None => run in all phases
    config: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """
    Configuration for an ordered middleware pipeline.
    
    Middlewares are executed in descending priority order (higher first).
    """
    
    chain: List[MiddlewareItemConfig] = Field(default_factory=list)


class PipelinesConfig(BaseModel):
    """
    Configuration for engine pipelines.
    
    - prepare: executed during ctxforge.prepare_context()
    - record: executed during ctxforge.record_turn()
    """
    
    prepare: PipelineConfig = Field(default_factory=PipelineConfig)
    record: PipelineConfig = Field(default_factory=PipelineConfig)


class ExtractionConfig(BaseModel):
    """Configuration for memory extraction."""
    
    enabled: bool = True
    async_processing: bool = True
    
    # What to extract
    extract_semantic: bool = True
    extract_episodic: bool = True
    extract_procedural: bool = False
    extract_preference: bool = True
    extract_tool: bool = False

    # Quality settings
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_candidates: int = Field(default=10, ge=1)
    
    # LLM-based extraction
    use_llm: bool = True
    extraction_model: Optional[str] = None

    # LLM-guided update planning (ADD/UPDATE/DELETE/NONE)
    update_planning_enabled: bool = False
    update_planning_model: Optional[str] = None
    update_planning_candidates_per_item: int = Field(default=5, ge=0)
    
    # Pattern-based extraction
    use_patterns: bool = True
    custom_patterns: Dict[str, str] = Field(default_factory=dict)

    # Batch extraction (sliding window)
    batch_window_size: int = Field(default=10, ge=1)
    batch_overlap: int = Field(default=2, ge=0)

    # Parallel extraction concurrency
    max_extraction_concurrency: int = Field(default=3, ge=1)

    # Multi-stage integration pipeline
    integration_pipeline_enabled: bool = False
    integration_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    integration_detect_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    integration_model: Optional[str] = None

    # Gist extraction (atomic, timestamped, self-contained statements)
    extract_gists: bool = False
    gist_model: Optional[str] = None
    gist_enhanced_facts: bool = False  # two-phase: gists inform fact extraction

    # Preference evolution tracking
    preference_evolution_enabled: bool = False
    preference_contradiction_similarity_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0
    )
    preference_auto_supersede: bool = True
    preference_importance_decay_on_supersede: float = Field(
        default=0.1, ge=0.0, le=1.0
    )

    # Memory narrative synthesis
    synthesis_enabled: bool = False
    synthesis_min_memories: int = Field(default=3, ge=1)
    synthesis_max_tokens: int = Field(default=300, ge=1)
    synthesis_model: Optional[str] = None

    # Personalization effectiveness metrics
    personalization_metrics_enabled: bool = False
    personalization_memory_hit_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0
    )


class ObservabilityConfig(BaseModel):
    """Configuration for observability."""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    
    # Tracing
    tracing_enabled: bool = False
    tracing_exporter: str = "console"  # console, jaeger, otlp
    tracing_endpoint: Optional[str] = None
    
    # Metrics
    metrics_enabled: bool = False
    metrics_port: int = 9090


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""
    
    system_template: str = "You are a helpful AI assistant."
    memory_section_name: str = "User Context"
    history_section_name: str = "Recent History"
    max_history_events: int = 10
    
    # Custom templates (Jinja2)
    custom_templates: Dict[str, str] = Field(default_factory=dict)


class FusionConfig(BaseModel):
    """
    Configuration for optional answer fusion workflows.

    Note: ctxforge remains inference-agnostic by default; fusion is only used when
    explicitly invoked by a caller that provides an `ILLMProvider`.
    """

    enabled: bool = False
    mode: str = "two_step"
    synthesis_model: Optional[str] = None
    max_tokens: int = Field(default=800, ge=1)


class RetrievalControllerSourcesConfig(BaseModel):
    """Control which retrieval sources the optional retrieval controller may use."""

    memory: bool = True
    graph: bool = True
    expertise: bool = True


class RetrievalControllerConfig(BaseModel):
    """
    Optional iterative retrieval controller configuration.

    This controller can run a bounded loop that gathers evidence from memory, graph, and/or expertise,
    updates an evidence+gaps state, and decides whether to retrieve more or answer.
    """

    enabled: bool = False

    # Online-safe bounds
    max_iterations: int = Field(default=2, ge=1)
    max_llm_calls: int = Field(default=4, ge=1)
    time_budget_ms: int = Field(default=0, ge=0)  # 0 disables wall-clock budget enforcement

    # Router model for decisions (defaults to provider default when not set).
    router_model: Optional[str] = None
    router_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    router_max_tokens: int = Field(default=350, ge=1)

    # Early stop + novelty controls
    early_stop_on_no_gaps: bool = True
    novelty_masking: bool = True
    min_new_items_to_continue: int = Field(default=1, ge=0)

    # Per-iteration source limits
    memory_limit_per_iter: int = Field(default=5, ge=0)
    expertise_limit_per_iter: int = Field(default=5, ge=0)

    # Default expertise id (callers may override per request).
    default_expertise_id: Optional[str] = None

    sources: RetrievalControllerSourcesConfig = Field(default_factory=RetrievalControllerSourcesConfig)

    # Query planning: decompose complex questions into sub-queries.
    enable_query_planning: bool = True
    max_sub_queries: int = Field(default=4, ge=1, le=8)

    # Retrieval reflection: coverage-based stopping.
    min_coverage_percentage: float = Field(default=0.7, ge=0.0, le=1.0)

    # Parallel retrieval of sub-queries.
    max_parallel_queries: int = Field(default=4, ge=1)


class ConsolidationConfig(BaseModel):
    """Background memory maintenance: decay, merge, prune."""

    enabled: bool = False
    decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    max_age_days: int = Field(default=30, ge=1)
    merge_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    min_importance: float = Field(default=0.1, ge=0.0, le=1.0)


class ScopesConfig(BaseModel):
    """
    Configuration for memory scopes.

    Supports a user scope (default) and an optional global scope.
    The global scope is intended for shared knowledge and is read-only by default.
    """

    enable_global: bool = True
    global_scope_id: str = "global"
    global_retrieval_limit: int = Field(default=5, ge=0)
    global_score_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    allow_global_writes: bool = False


class PluginRegistration(BaseModel):
    """
    A single plugin registration loaded from configuration.

    Registers a component by import path without requiring core-code changes.
    """

    component_type: str
    name: str
    class_path: str


class PluginsConfig(BaseModel):
    """
    Plugin configuration.

    - modules: imported for side effects OR may expose register(registry) callable
    - registrations: explicit class-path registrations
    """

    modules: List[str] = Field(default_factory=list)
    registrations: List[PluginRegistration] = Field(default_factory=list)


class TaskModelRoutingConfig(BaseModel):
    """Configuration for task-specific model routing."""

    enabled: bool = False
    default_model: Optional[str] = None
    extraction_model: Optional[str] = None
    planning_model: Optional[str] = None
    synthesis_model: Optional[str] = None
    judge_model: Optional[str] = None


class EntropyGateConfig(BaseModel):
    """Configuration for entropy-aware extraction gating."""

    enabled: bool = False
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    min_chars: int = Field(default=20, ge=0)
    recent_window_size: int = Field(default=20, ge=1)
    embedding_cache_size: int = Field(default=256, ge=1)


class ConsolidationQualityConfig(BaseModel):
    """Configuration for merge/add/ignore consolidation behavior."""

    enabled: bool = False
    semantic_merge_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    keyword_overlap_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    contradiction_check_enabled: bool = True
    contradiction_policy: str = "preserve_both"  # preserve_both | prefer_new | prefer_existing
    # LLM-based contradiction detection (more accurate but slower/costlier)
    use_llm_contradiction_check: bool = False
    llm_contradiction_model: Optional[str] = None  # Model override for contradiction checks


class RetrievalFastPathConfig(BaseModel):
    """Configuration for lightweight retrieval fast-paths."""

    enabled: bool = False
    detect_count_queries: bool = True
    detect_list_queries: bool = True
    detect_relation_queries: bool = True
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class GraphPathMiningConfig(BaseModel):
    """Configuration for bridge discovery and path mining."""

    enabled: bool = False
    bridge_discovery_enabled: bool = False
    max_path_depth: int = Field(default=3, ge=1)
    max_paths: int = Field(default=10, ge=0)
    min_path_length: int = Field(default=2, ge=2)
    max_nodes: int = Field(default=25, ge=0)
    min_nodes: int = Field(default=8, ge=0)
    temporal_flow_hours: float = Field(default=6.0, ge=0)
    temporal_window_hours: int = Field(default=168, ge=1)
    bridge_search_top_k: int = Field(default=5, ge=1)
    bridge_proximity_hours: float = Field(default=24.0, ge=0)
    node_score_threshold_pct: float = Field(default=0.10, ge=0.0, le=1.0)


class AnswerNormalizationConfig(BaseModel):
    """Configuration for post-answer normalization."""

    enabled: bool = False
    normalize_lists: bool = True
    normalize_dates: bool = True
    normalize_yes_no: bool = True
    normalize_plurality: bool = True


class MemoryQualityConfig(BaseModel):
    """Top-level quality controls for memory and retrieval behavior."""

    model_routing: TaskModelRoutingConfig = Field(default_factory=TaskModelRoutingConfig)
    entropy_gate: EntropyGateConfig = Field(default_factory=EntropyGateConfig)
    consolidation: ConsolidationQualityConfig = Field(default_factory=ConsolidationQualityConfig)
    retrieval_fast_path: RetrievalFastPathConfig = Field(default_factory=RetrievalFastPathConfig)
    graph_path_mining: GraphPathMiningConfig = Field(default_factory=GraphPathMiningConfig)
    answer_normalization: AnswerNormalizationConfig = Field(default_factory=AnswerNormalizationConfig)


class Neo4jGraphStoreConfig(BaseModel):
    url: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j"
    database: Optional[str] = None
    create_indexes: bool = True
    entity_label: str = "__Entity__"
    vector_index_name: str = "ce_entity_name_embedding"
    vector_dimensions: Optional[int] = None
    fulltext_entity_index_name: str = "ce_entity_fulltext"
    fulltext_edge_index_name: str = "ce_edge_fulltext"


class GraphOntologyConfig(BaseModel):
    module: Optional[str] = None
    attr_name: str = "GRAPH_ONTOLOGY"


class GraphExtractionConfig(BaseModel):
    enabled: bool = True
    model: Optional[str] = None


class GraphVectorFusedTripletsConfig(BaseModel):
    """
    Configuration for graph+vector fused triplet scoring retrieval.

    This recipe borrows a key idea from Cognee: use semantic similarity signals to seed graph candidates,
    then score edges (triplets) by combining endpoint-node relevance and edge relevance.
    """

    enabled: bool = False

    # Seed selection (best-effort; exact behavior depends on backend capabilities).
    wide_search_top_k: int = Field(default=100, ge=0)
    seed_node_k: int = Field(default=50, ge=0)
    seed_edge_k: int = Field(default=50, ge=0)

    # Candidate reduction
    max_candidate_edges: int = Field(default=500, ge=0)
    max_output_edges: int = Field(default=20, ge=0)

    # Scoring weights
    w_node: float = Field(default=1.0, ge=0.0)
    w_edge: float = Field(default=1.0, ge=0.0)
    w_distance_penalty: float = Field(default=0.0, ge=0.0)

    # Edge scoring source (MVP: keyword scoring; future: fact embeddings).
    edge_score_mode: str = "relation_keyword"  # relation_keyword | fact_embedding | none


class GraphPPRConfig(BaseModel):
    """
    Configuration for Personalized PageRank (PPR) reranking over a bounded candidate subgraph.

    This runs only over the already-retrieved candidate edges/nodes, then reorders nodes/edges
    before token budgeting. Default is off to preserve behavior.
    """

    enabled: bool = False

    # PPR parameters
    damping: float = Field(default=0.85, ge=0.0, le=1.0)
    max_iters: int = Field(default=50, ge=1)
    tol: float = Field(default=1e-6, ge=0.0)

    # Graph construction
    directed: bool = False
    use_edge_weight_attr: str = "weight"  # edge.attributes[key] if present, else 1.0

    # Reset / personalization vector behavior
    reset_mode: str = "node_scores"  # node_scores | uniform | query_embedding
    reset_top_k_nodes: int = Field(default=50, ge=0)  # 0 => use all nodes
    reset_min_weight: float = Field(default=0.0, ge=0.0)

    # Output shaping
    rerank_nodes: bool = True
    rerank_edges: bool = True
    edge_score_mode: str = "endpoint_sum"  # endpoint_sum | existing

    # Safety
    min_nodes: int = Field(default=3, ge=0)
    min_edges: int = Field(default=2, ge=0)


class TopologySerializationConfig(BaseModel):
    """
    Configuration for topology-aware context serialization.

    When enabled, the graph context is rendered as labeled facts with inline
    evidence, annotated reasoning paths, and bridge connection summaries
    instead of flat lists.
    """

    enabled: bool = False
    fact_label_prefix: str = "F"
    max_fact_content_chars: int = Field(default=300, ge=0)
    max_evidence_per_fact: int = Field(default=2, ge=0)
    max_evidence_chars: int = Field(default=200, ge=0)
    max_reasoning_paths: int = Field(default=10, ge=0)
    max_bridge_summaries: int = Field(default=5, ge=0)
    include_edge_types_in_paths: bool = True
    include_timestamps: bool = True


class GraphRetrievalConfig(BaseModel):
    """
    Graph retrieval configuration.

    This controls how the engine selects entities/edges to include in the rendered graph
    context section.
    """

    enabled: bool = True
    max_facts: int = Field(default=20, ge=0)
    max_entities: int = Field(default=20, ge=0)
    max_episodes: int = Field(default=10, ge=0)
    include_entities: bool = True
    include_episodes: bool = False
    valid_only: bool = True

    # Hybrid retrieval recipes.
    # - keyword: keyword/fulltext-based seeding (fallback when semantic is disabled/unavailable)
    # - bfs: bounded neighborhood expansion from the top seed entities
    # - semantic: semantic seeding via `search_nodes_semantic` when embeddings are enabled
    methods: List[str] = Field(default_factory=lambda: ["keyword", "bfs", "semantic"])
    seed_k: int = Field(default=8, ge=0)
    bfs_max_depth: int = Field(default=1, ge=0)
    bfs_edges_per_node: int = Field(default=12, ge=0)

    # Reranking: deterministic best-effort reranking/merging (default) or future pluggable rerankers.
    rerank_enabled: bool = False
    reranker: str = "rrf"
    rerank_top_k: int = Field(default=30, ge=0)

    # Personalized PageRank retrieval recipe.
    ppr_enabled: bool = False
    ppr_damping: float = Field(default=0.5, ge=0.0, le=1.0)
    ppr_seed_top_k: int = Field(default=20, ge=1)
    ppr_result_top_k: int = Field(default=10, ge=1)

    # ------------------------------------------------------------------
    # Planner + evidence + token budgeting
    # ------------------------------------------------------------------
    # Planner mode:
    # - "auto": choose local/global/hybrid based on extracted keywords + fallbacks
    # - "local": entity-first neighborhood expansion
    # - "global": relation/fact-first seeding
    # - "hybrid": combine both
    planner_mode: str = "auto"

    # Keyword extraction knobs (used by the planner when planner_mode="auto").
    max_low_level_keywords: int = Field(default=10, ge=0)
    max_high_level_keywords: int = Field(default=8, ge=0)

    # Evidence linking: optionally attach supporting episodes/chunks for the retrieved subgraph.
    evidence_enabled: bool = False
    max_evidence_items: int = Field(default=10, ge=0)

    # Token budgets (0 => disabled / fallback to count-based limits).
    # These budgets are enforced only when a tokenizer provider is available.
    max_entity_tokens: int = Field(default=0, ge=0)
    max_relation_tokens: int = Field(default=0, ge=0)
    max_evidence_tokens: int = Field(default=0, ge=0)
    max_total_tokens: int = Field(default=0, ge=0)

    # Graph+vector fused triplet scoring retrieval.
    vector_fused_triplets: GraphVectorFusedTripletsConfig = Field(default_factory=GraphVectorFusedTripletsConfig)

    # Optional: Personalized PageRank reranking over the bounded candidate subgraph.
    ppr: GraphPPRConfig = Field(default_factory=GraphPPRConfig)

    # Topology-aware context serialization.
    topology_aware: TopologySerializationConfig = Field(default_factory=TopologySerializationConfig)


class GraphStoreConfig(BaseModel):
    backend: str = "memory"  # memory | neo4j
    neo4j: Neo4jGraphStoreConfig = Field(default_factory=Neo4jGraphStoreConfig)


class GraphEmbeddingsConfig(BaseModel):
    enabled: bool = False
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    max_concurrency: int = Field(default=8, ge=1)


class GraphInvalidationConfig(BaseModel):
    enabled: bool = False
    model: Optional[str] = None
    candidate_limit: int = Field(default=25, ge=1)
    max_concurrency: int = Field(default=4, ge=1)


class GraphEntityLinkingConfig(BaseModel):
    """Configuration for KNN-based entity linking (SAME_AS edges)."""

    enabled: bool = False
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_neighbors: int = Field(default=5, ge=1)
    run_on_ingest: bool = True
    batch_size: int = Field(default=100, ge=1)


class GraphTemporalConfig(BaseModel):
    enabled: bool = False
    model: Optional[str] = None
    max_concurrency: int = Field(default=4, ge=1)


class GraphCommunitiesConfig(BaseModel):
    """
    Community-layer configuration.

    Communities are derived clusters of entity nodes used to provide higher-level structure
    in the graph context without per-query summarization.
    """

    enabled: bool = False
    max_communities: int = Field(default=5, ge=0)
    rebuild_every_n_episodes: int = Field(default=25, ge=0)
    min_cluster_size: int = Field(default=2, ge=1)

    # Optional refinement steps (gated; default off).
    merge_similar_enabled: bool = False
    merge_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    attach_orphans_enabled: bool = False
    attach_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # Limits for rebuild operations.
    max_nodes: int = Field(default=2000, ge=0)
    max_edges: int = Field(default=4000, ge=0)
    max_concurrency: int = Field(default=4, ge=1)
    model: Optional[str] = None


class GraphConfig(BaseModel):
    enabled: bool = False
    store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    ontology: GraphOntologyConfig = Field(default_factory=GraphOntologyConfig)
    extraction: GraphExtractionConfig = Field(default_factory=GraphExtractionConfig)
    embeddings: GraphEmbeddingsConfig = Field(default_factory=GraphEmbeddingsConfig)
    invalidation: GraphInvalidationConfig = Field(default_factory=GraphInvalidationConfig)
    temporal: GraphTemporalConfig = Field(default_factory=GraphTemporalConfig)
    entity_linking: GraphEntityLinkingConfig = Field(default_factory=GraphEntityLinkingConfig)
    communities: GraphCommunitiesConfig = Field(default_factory=GraphCommunitiesConfig)
    retrieval: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)
    section_name: str = "Graph Memory"


# =============================================================================
# Hierarchical Memory & Skills Configurations
# =============================================================================

class ScopedMemoryStoreConfig(BaseModel):
    """Configuration for scoped memory storage."""
    
    backend: StorageBackendType = StorageBackendType.MEMORY
    connection_string: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ScopedMemoryConfig(BaseModel):
    """Configuration for hierarchical scoped memory."""
    
    enabled: bool = False
    store: ScopedMemoryStoreConfig = Field(default_factory=ScopedMemoryStoreConfig)
    
    # Injection settings
    auto_inject: bool = True  # Auto-inject into prepare_context
    injection_template: Optional[str] = None  # Custom template for prompt injection
    
    # Auto-learn settings (extract preferences from user input)
    auto_learn_enabled: bool = False
    preference_patterns: List[str] = Field(
        default_factory=lambda: [
            r"I prefer (.+)",
            r"always (.+)",
            r"never (.+)",
            r"use (.+) for (.+)",
        ]
    )


class SkillStoreConfig(BaseModel):
    """Configuration for skill storage."""
    
    backend: StorageBackendType = StorageBackendType.MEMORY
    connection_string: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class SkillEvaluationConfig(BaseModel):
    """Configuration for LLM-based skill quality evaluation."""
    enabled: bool = False
    auto_evaluate_on_register: bool = False
    model: Optional[str] = None


class SkillRelationshipConfig(BaseModel):
    """Configuration for skill relationship analysis."""
    enabled: bool = True
    auto_analyze: bool = False
    auto_resolve_dependencies: bool = True


class SkillGenerationConfig(BaseModel):
    """Configuration for session-to-skill generation."""
    enabled: bool = True
    auto_generate_from_sessions: bool = False
    min_session_events: int = Field(default=5, ge=1)
    model: Optional[str] = None


class SkillEffectivenessConfig(BaseModel):
    """Configuration for skill effectiveness tracking."""
    enabled: bool = True
    track_usage: bool = True
    weight_in_ranking: float = Field(default=0.3, ge=0.0, le=1.0)


class SkillGraduationConfig(BaseModel):
    """Configuration for skill graduation criteria."""
    enabled: bool = False
    min_usage_count: int = Field(default=5, ge=1)
    min_success_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    auto_graduate: bool = False


class SkillInheritanceConfig(BaseModel):
    """Configuration for cross-scope skill inheritance."""
    enabled: bool = False
    include_inherited_in_index: bool = True
    graduation: SkillGraduationConfig = Field(
        default_factory=SkillGraduationConfig,
    )


class ExecutableRuntimeConfig(BaseModel):
    """Configuration for the executable skill runtime."""
    enabled: bool = False
    timeout_sec: float = Field(default=10.0, gt=0)
    max_concurrent: int = Field(default=3, ge=1)
    sandbox: bool = True
    max_output_chars: int = Field(default=4000, ge=100)
    # Execution boundary: "subprocess" (isolated process + resource limits) or
    # "inprocess" (worker thread; required for the call_tool bridge).
    isolation: str = "subprocess"
    cpu_time_limit_sec: int = Field(default=0, ge=0)
    memory_limit_mb: int = Field(default=0, ge=0)
    blocked_tools: List[str] = Field(default_factory=lambda: [
        "save_skill", "execute_skill", "list_skills", "get_skill",
    ])


class SkillsConfig(BaseModel):
    """Configuration for the skills system."""
    
    enabled: bool = True
    store: SkillStoreConfig = Field(default_factory=SkillStoreConfig)
    
    # Skill injection settings
    auto_inject_index: bool = True  # Inject skills index into system prompt
    auto_activate: bool = True  # Auto-activate skills based on triggers
    max_auto_skills: int = Field(default=2, ge=1)  # Max skills to auto-activate
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Skill matching settings
    use_regex_triggers: bool = False  # Enable regex-based trigger matching
    use_embedding_matching: bool = False  # Use semantic matching for skills

    # Advanced skill sub-configs
    evaluation: SkillEvaluationConfig = Field(default_factory=SkillEvaluationConfig)
    relationships: SkillRelationshipConfig = Field(default_factory=SkillRelationshipConfig)
    generation: SkillGenerationConfig = Field(default_factory=SkillGenerationConfig)
    effectiveness: SkillEffectivenessConfig = Field(default_factory=SkillEffectivenessConfig)
    inheritance: SkillInheritanceConfig = Field(default_factory=SkillInheritanceConfig)
    executable_runtime: ExecutableRuntimeConfig = Field(default_factory=ExecutableRuntimeConfig)


# =============================================================================
# Dynamic Context Configurations
# =============================================================================

class ApprovalConfig(BaseModel):
    """Configuration for human-in-the-loop approval middleware."""
    
    enabled: bool = False
    knowledge_types: List[str] = Field(
        default_factory=lambda: ["expertise_item", "validated_query", "procedural_memory"]
    )
    stop_on_pending: bool = False
    expiry_hours: int = Field(default=24, ge=1)


class SemanticModelConfig(BaseModel):
    """Configuration for semantic model context anchor."""
    
    enabled: bool = False
    default_model_id: Optional[str] = None
    compact_format: bool = True
    auto_inject: bool = True  # Auto-inject into prepare_context


class SnapshotConfig(BaseModel):
    """Configuration for expertise snapshots."""
    
    enabled: bool = False
    auto_snapshot_on_change: bool = False
    max_snapshots_per_expertise: int = Field(default=100, ge=1)


class UnifiedRetrievalConfig(BaseModel):
    """Configuration for unified cross-store retrieval."""
    
    enabled: bool = False
    merge_strategy: str = "interleave"  # interleave, score_only, source_first
    max_results: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Source weights (1.0 = neutral)
    expertise_weight: float = Field(default=1.0, ge=0.0)
    memory_weight: float = Field(default=1.0, ge=0.0)
    graph_weight: float = Field(default=1.0, ge=0.0)


class SearchBeforeRespondConfig(BaseModel):
    """Configuration for search-before-respond middleware."""
    
    enabled: bool = False
    knowledge_domains: List[str] = Field(
        default_factory=lambda: ["domain knowledge", "past experiences", "documented patterns"]
    )
    trigger_keywords: List[str] = Field(
        default_factory=lambda: [
            "how", "what", "why", "where", "when", "which",
            "explain", "describe", "tell me", "find", "get",
            "query", "show", "list", "calculate", "determine",
        ]
    )
    auto_search: bool = False


class ProgressiveDisclosureConfig(BaseModel):
    """Configuration for progressive memory disclosure."""

    enabled: bool = False
    max_headline_chars: int = Field(default=80, ge=20)
    max_subtitle_chars: int = Field(default=150, ge=50)
    expand_top_n: int = Field(default=3, ge=0)
    use_llm_headlines: bool = True  # Use LLM to generate headlines


class TimelineConfig(BaseModel):
    """Configuration for timeline-based event retrieval."""

    enabled: bool = True
    default_time_range: str = Field(
        default="this_session",
        description="Default time range: last_hour, last_24h, last_7d, today, this_session"
    )
    include_timestamps_in_history: bool = False
    group_by_turns: bool = False
    max_events_default: int = Field(default=100, ge=1)


class ToolCompressionConfig(BaseModel):
    """Configuration for tool output compression."""

    enabled: bool = True
    max_output_chars: int = Field(default=2000, ge=100)
    compression_threshold: int = Field(
        default=500,
        ge=0,
        description="Don't compress content below this length"
    )
    default_strategy: str = Field(
        default="auto",
        description="Compression strategy: auto, truncate, key_value, dedupe"
    )

    # Tool-specific strategy overrides
    tool_configs: Dict[str, str] = Field(
        default_factory=dict,
        description="Tool name to strategy mapping"
    )


class QueryRewriteConfig(BaseModel):
    """Configuration for query rewriting."""

    enabled: bool = False
    max_history_turns: int = Field(default=10, ge=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    llm_profile: str = "default"
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=0)

    # Heuristic trigger patterns (before calling LLM)
    check_pronouns: bool = True
    check_references: bool = True
    check_implicit: bool = True
    min_words_for_skip: int = Field(
        default=8, ge=1,
        description="Queries longer than this are less likely to need rewriting"
    )


class SufficiencyConfig(BaseModel):
    """Configuration for sufficiency judging."""

    enabled: bool = False
    max_iterations: int = Field(default=3, ge=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    llm_profile: str = "default"
    fallback_sources: List[str] = Field(
        default_factory=lambda: ["memories", "graph", "expertise"]
    )
    initial_limit: int = Field(default=5, ge=1)
    max_limit: int = Field(default=20, ge=1)


class DynamicContextConfig(BaseModel):
    """Configuration for dynamic context features."""

    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    semantic_model: SemanticModelConfig = Field(default_factory=SemanticModelConfig)
    snapshots: SnapshotConfig = Field(default_factory=SnapshotConfig)
    unified_retrieval: UnifiedRetrievalConfig = Field(default_factory=UnifiedRetrievalConfig)
    search_before_respond: SearchBeforeRespondConfig = Field(
        default_factory=SearchBeforeRespondConfig
    )
    progressive_disclosure: ProgressiveDisclosureConfig = Field(
        default_factory=ProgressiveDisclosureConfig
    )
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    tool_compression: ToolCompressionConfig = Field(
        default_factory=ToolCompressionConfig
    )
    query_rewriting: QueryRewriteConfig = Field(
        default_factory=QueryRewriteConfig
    )
    sufficiency: SufficiencyConfig = Field(
        default_factory=SufficiencyConfig
    )


# =============================================================================
# Main Engine Configuration
# =============================================================================

class EngineConfig(BaseModel):
    """
    Main configuration for the ctxforge.
    
    This is the top-level configuration that includes all component configs.
    Can be loaded from YAML, JSON, environment variables, or programmatically.
    """
    
    # Engine identity
    name: str = "ctxforge"
    version: str = "0.1.0"
    debug: bool = False
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    expertise: ExpertiseConfig = Field(default_factory=ExpertiseConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    pipelines: PipelinesConfig = Field(default_factory=PipelinesConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    retrieval_controller: RetrievalControllerConfig = Field(default_factory=RetrievalControllerConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    scopes: ScopesConfig = Field(default_factory=ScopesConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    memory_quality: MemoryQualityConfig = Field(default_factory=MemoryQualityConfig)
    dynamic_context: DynamicContextConfig = Field(default_factory=DynamicContextConfig)
    scoped_memory: ScopedMemoryConfig = Field(default_factory=ScopedMemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    
    # Custom extensions
    extensions: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Create configuration from a dictionary."""
        return cls.model_validate(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return self.model_dump()
    
    def merge_with(self, overrides: Dict[str, Any]) -> "EngineConfig":
        """Create a new config with overrides applied."""
        current = self.model_dump()
        self._deep_merge(current, overrides)
        return EngineConfig.model_validate(current)
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        """Deep merge override into base (in place)."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                EngineConfig._deep_merge(base[key], value)
            else:
                base[key] = value

