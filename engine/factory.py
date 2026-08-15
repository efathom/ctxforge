"""
Factory for creating ctxforge instances.

M1 refactor:
- Removes staged processor pipeline wiring (pre_processors/post_processors).
- Adds config-driven middleware pipelines (prepare/record) using middleware chain.
"""

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ctxforge.compaction.view import ICondenser
from ctxforge.config.base import EngineConfig
from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.config.loader import load_config
from ctxforge.engine.context_engine import CtxForge
from ctxforge.engine.deps import EngineDeps
from ctxforge.engine.registry import ComponentRegistry, registry
from ctxforge.extraction.entropy_gate import EntropyGate
from ctxforge.extraction.update_planner import LLMMemoryUpdatePlanner
from ctxforge.graph.communities.builder import CommunityBuilder
from ctxforge.graph.default_ontology import GRAPH_ONTOLOGY
from ctxforge.graph.extraction.llm import LLMGraphExtractor
from ctxforge.graph.maintenance.invalidation import LLMGraphContradictionDetector
from ctxforge.graph.maintenance.temporal import LLMEdgeTemporalExtractor
from ctxforge.graph.ontology import load_ontology_from_module
from ctxforge.graph.stores.memory import InMemoryGraphStore
from ctxforge.llm.task_model_resolver import TaskModelResolver
from ctxforge.protocols.compactor import IContextAssembler
from ctxforge.protocols.expertise import IExpertiseRetriever, IExpertiseStore
from ctxforge.protocols.extractor import IMemoryExtractor
from ctxforge.protocols.llm import IEmbeddingProvider, ILLMProvider
from ctxforge.protocols.retriever import IRetriever
from ctxforge.protocols.storage import IMemoryStore, ISessionStore
from ctxforge.protocols.tokenizer import ITokenizerProvider
from ctxforge.protocols.vectorstore import IVectorStore
from ctxforge.retrieval.fast_path_retriever import FastPathRetriever

logger = logging.getLogger(__name__)


class EngineFactory:
    """
    Factory for creating ctxforge instances.
    
    The factory handles:
    - Loading configuration from files/dicts
    - Resolving component implementations from registry
    - Constructing and wiring all dependencies
    
    Example:
        # From config file
        engine = await EngineFactory.from_config_file("config.yaml")
        
        # From dict
        engine = await EngineFactory.from_dict({
            "storage": {"session": {"backend": "redis"}},
        })
        
        # With custom components
        engine = await EngineFactory.create(
            config=my_config,
            session_store=MyCustomSessionStore(),
        )
    """
    
    def __init__(self, component_registry: Optional[ComponentRegistry] = None):
        """
        Initialize the factory.
        
        Args:
            component_registry: Optional custom registry (uses global if not provided)
        """
        self._registry = component_registry or registry
        # Cache for LLM/embedding providers to avoid creating multiple clients
        self._llm_provider_cache: Dict[str, "ILLMProvider"] = {}
        self._embedding_provider_cache: Dict[str, "IEmbeddingProvider"] = {}
    
    async def create(
        self,
        config: EngineConfig,
        session_store: Optional[ISessionStore] = None,
        memory_store: Optional[IMemoryStore] = None,
        retriever: Optional[IRetriever] = None,
        condenser: Optional[ICondenser] = None,
        assembler: Optional[IContextAssembler] = None,
        extractor: Optional[IMemoryExtractor] = None,
        embedding_provider: Optional[IEmbeddingProvider] = None,
        vector_store: Optional[IVectorStore] = None,
        expertise_store: Optional[IExpertiseStore] = None,
        expertise_embedding_provider: Optional[IEmbeddingProvider] = None,
        expertise_vector_store: Optional[IVectorStore] = None,
        expertise_retriever: Optional[IExpertiseRetriever] = None,
        reflector: Optional[Any] = None,
        curator: Optional[Any] = None,
    ) -> CtxForge:
        """
        Create a ctxforge with the given configuration.
        
        Components can be provided directly or resolved from the registry
        based on configuration.
        
        Args:
            config: Engine configuration
            session_store: Optional session store (created from config if not provided)
            memory_store: Optional memory store
            retriever: Optional retriever
            condenser: Optional condenser
            extractor: Optional extractor
            assembler: Optional context assembler
            
        Returns:
            Configured ctxforge instance
        """
        owned_resources: List[Any] = []

        # Resolve components from registry if not provided
        owns_session_store = session_store is None
        if session_store is None:
            session_store = self._create_session_store(config)
        if owns_session_store:
            await self._maybe_initialize(session_store)
            owned_resources.append(session_store)
        
        owns_memory_store = memory_store is None
        if memory_store is None:
            memory_store = self._create_memory_store(config)
        if owns_memory_store:
            await self._maybe_initialize(memory_store)
            owned_resources.append(memory_store)

        # Tokenizer provider (for accurate context budgeting). Best-effort:
        # - Use an explicit tokenizer provider if registered by plugins (future)
        # - Otherwise adapt the configured LLM provider's counting methods.
        tokenizer_provider: Optional[ITokenizerProvider] = None
        try:
            llm_for_tokenizer = self._create_llm_provider(config)
            if llm_for_tokenizer is not None:
                from ctxforge.llm.tokenizer_provider import LLMTokenizerProvider

                tokenizer_provider = LLMTokenizerProvider(llm_for_tokenizer)
        except Exception:
            tokenizer_provider = None

        if embedding_provider is None:
            embedding_provider = self._create_embedding_provider(config)

        if vector_store is None:
            vector_store = await self._create_vector_store(config, embedding_provider=embedding_provider)

        memory_indexer = None
        if vector_store is not None and embedding_provider is not None:
            from ctxforge.retrieval.indexers.memory import MemoryIndexer
            memory_indexer = MemoryIndexer(vector_store, embedding_provider)

        expertise_indexer = None
        # Expertise wiring (Option B: separate config.expertise.vectorstore)
        if config.expertise.enabled:
            owns_expertise_store = expertise_store is None
            if expertise_store is None:
                expertise_store = await self._create_expertise_store(config)
            if owns_expertise_store and expertise_store is not None:
                await self._maybe_initialize(expertise_store)
                owned_resources.append(expertise_store)

            if expertise_embedding_provider is None:
                expertise_embedding_provider = self._create_embedding_provider_from_embedding_config(
                    embedding_config=config.expertise.vectorstore.embedding,
                    llm_config=config.llm,
                )

            if expertise_vector_store is None:
                expertise_vector_store = await self._create_vector_store_from_spec(
                    backend=config.expertise.vectorstore.backend.value,
                    index_name=config.expertise.vectorstore.index_name,
                    connection_string=config.expertise.vectorstore.connection_string,
                    embedding_dim=config.expertise.vectorstore.embedding.dimension,
                    extra_params=config.expertise.vectorstore.extra_params,
                )

            if expertise_vector_store is not None and expertise_embedding_provider is not None:
                from ctxforge.retrieval.indexers.expertise import ExpertiseIndexer
                expertise_indexer = ExpertiseIndexer(expertise_vector_store, expertise_embedding_provider)

            if expertise_retriever is None and expertise_store is not None and expertise_indexer is not None:
                from ctxforge.retrieval.retrievers.expertise import ExpertiseRetriever
                exp_reranker = None
                if getattr(config.expertise.retrieval, "rerank_enabled", False):
                    reranker_name = (getattr(config.expertise.retrieval, "reranker", None) or "effectiveness").lower()
                    if reranker_name == "llm":
                        llm = self._create_llm_provider(config)
                        if llm is not None:
                            from ctxforge.retrieval.rerankers.llm import LLMContextReranker
                            model = config.expertise.retrieval.rerank_model or llm.default_model
                            exp_reranker = LLMContextReranker(llm_provider=llm, model=model)
                    else:
                        # built-in expertise rerankers (non-LLM)
                        if reranker_name == "effectiveness":
                            from ctxforge.retrieval.rerankers.expertise import EffectivenessReranker
                            exp_reranker = EffectivenessReranker()
                        elif reranker_name == "usage_recency":
                            from ctxforge.retrieval.rerankers.expertise import UsageRecencyReranker
                            exp_reranker = UsageRecencyReranker()
                        elif reranker_name == "diversity":
                            from ctxforge.retrieval.rerankers.expertise import (
                                ExpertiseDiversityReranker,
                            )
                            exp_reranker = ExpertiseDiversityReranker()

                expertise_retriever = ExpertiseRetriever(expertise_store, expertise_indexer, reranker=exp_reranker)
        
        if retriever is None:
            retriever = self._create_retriever(config, memory_store, memory_indexer)
        
        if condenser is None:
            condenser = self._create_condenser(config)
        
        if assembler is None:
            assembler = self._create_assembler(config)
        # Inject tokenizer into assembler when supported (no-op for custom assemblers).
        try:
            set_tok = getattr(assembler, "set_tokenizer_provider", None)
            if callable(set_tok):
                set_tok(tokenizer_provider)
        except Exception:
            pass
        
        if extractor is None:
            extractor = self._create_extractor(config)

        # Task model resolver for routing models by task type
        task_resolver = TaskModelResolver(
            getattr(getattr(config, "memory_quality", None), "model_routing", None)
        )

        update_planner = None
        if getattr(config.extraction, "update_planning_enabled", False):
            llm = self._create_llm_provider(config)
            if llm is not None:
                # Prefer explicit config, then task resolver, then provider default
                model = (
                    config.extraction.update_planning_model
                    or task_resolver.resolve_planning(llm)
                    or llm.default_model
                )
                update_planner = LLMMemoryUpdatePlanner(
                    llm_provider=llm,
                    default_model=model,
                )

        graph_store = None
        graph_extractor = None
        graph_ontology = None
        graph_embedding_provider: Optional[IEmbeddingProvider] = None
        graph_contradiction_detector = None
        graph_edge_temporal_extractor = None
        graph_community_builder = None
        if getattr(config, "graph", None) is not None and getattr(config.graph, "enabled", False):
            # Ontology
            graph_ontology = GRAPH_ONTOLOGY
            ontology_module = getattr(config.graph.ontology, "module", None)
            if ontology_module:
                try:
                    graph_ontology = load_ontology_from_module(
                        ontology_module,
                        attr_name=getattr(config.graph.ontology, "attr_name", "GRAPH_ONTOLOGY"),
                    )
                except Exception:
                    graph_ontology = GRAPH_ONTOLOGY

            # Store
            backend = (getattr(config.graph.store, "backend", None) or "memory").lower()
            if backend == "memory":
                graph_store = InMemoryGraphStore()
            elif backend == "neo4j":
                try:
                    from ctxforge.graph.stores.neo4j import Neo4jGraphStore
                    # Configure vector index dimensions if embeddings are enabled.
                    if getattr(config.graph, "embeddings", None) is not None and getattr(
                        config.graph.embeddings, "enabled", False
                    ):
                        try:
                            config.graph.store.neo4j.vector_dimensions = int(
                                config.graph.embeddings.embedding.dimension
                            )
                        except Exception:
                            pass
                    graph_store = Neo4jGraphStore(config.graph.store.neo4j)
                except Exception:
                    graph_store = None
            else:
                graph_store = None

            # Embeddings (for node semantic search)
            if getattr(config.graph, "embeddings", None) is not None and getattr(
                config.graph.embeddings, "enabled", False
            ):
                graph_embedding_provider = self._create_embedding_provider_from_embedding_config(
                    embedding_config=config.graph.embeddings.embedding,
                    llm_config=config.llm,
                )

            # Extractor
            if graph_store is not None and getattr(config.graph.extraction, "enabled", True):
                llm = self._create_llm_provider(config)
                if llm is not None:
                    # Prefer explicit config, then task resolver, then provider default
                    model = (
                        config.graph.extraction.model
                        or task_resolver.resolve_extraction(llm)
                        or llm.default_model
                    )
                    graph_extractor = LLMGraphExtractor(llm_provider=llm, default_model=model)

            # Contradiction detection / invalidation
            if graph_store is not None and getattr(getattr(config.graph, "invalidation", None), "enabled", False):
                llm = self._create_llm_provider(config)
                if llm is not None:
                    model = getattr(config.graph.invalidation, "model", None) or llm.default_model
                    graph_contradiction_detector = LLMGraphContradictionDetector(
                        llm_provider=llm,
                        default_model=model,
                    )

            # Temporal enrichment
            if graph_store is not None and getattr(getattr(config.graph, "temporal", None), "enabled", False):
                llm = self._create_llm_provider(config)
                if llm is not None:
                    model = getattr(config.graph.temporal, "model", None) or llm.default_model
                    graph_edge_temporal_extractor = LLMEdgeTemporalExtractor(
                        llm_provider=llm,
                        default_model=model,
                    )

            # Community layer (derived)
            if graph_store is not None and getattr(getattr(config.graph, "communities", None), "enabled", False):
                llm = self._create_llm_provider(config)
                graph_community_builder = CommunityBuilder(
                    llm_provider=llm,
                    embedding_provider=graph_embedding_provider,
                )
        
        # Entity linking (SAME_AS edges via KNN)
        graph_entity_linker = None
        el_cfg = getattr(config.graph, "entity_linking", None) if getattr(config, "graph", None) else None
        if el_cfg is not None and getattr(el_cfg, "enabled", False):
            from ctxforge.graph.maintenance.entity_linking import EntityLinker

            graph_entity_linker = EntityLinker(
                similarity_threshold=float(getattr(el_cfg, "similarity_threshold", 0.85)),
                max_neighbors=int(getattr(el_cfg, "max_neighbors", 5)),
                batch_size=int(getattr(el_cfg, "batch_size", 100)),
            )

        # Dynamic context services
        validated_knowledge_service = None
        semantic_model_service = None
        snapshot_service = None
        unified_retriever = None
        approval_store = None
        
        dc_config = getattr(config, "dynamic_context", None)
        if dc_config is not None:
            # Validated knowledge service
            from ctxforge.engine.services.validated_knowledge_service import (
                ValidatedKnowledgeService,
            )
            validated_knowledge_service = ValidatedKnowledgeService(
                expertise_store=expertise_store,
                expertise_indexer=expertise_indexer,
                memory_store=memory_store,
                memory_indexer=memory_indexer,
            )
            
            # Semantic model service
            if getattr(dc_config.semantic_model, "enabled", False):
                from ctxforge.core.semantic_model import InMemorySemanticModelStore
                from ctxforge.engine.services.semantic_model_service import SemanticModelService
                semantic_model_service = SemanticModelService(
                    store=InMemorySemanticModelStore(),
                )
            
            # Snapshot service
            if getattr(dc_config.snapshots, "enabled", False):
                from ctxforge.engine.services.expertise_snapshot_service import (
                    ExpertiseSnapshotService,
                    InMemorySnapshotStore,
                )
                snapshot_service = ExpertiseSnapshotService(
                    store=InMemorySnapshotStore(),
                )
            
            # Unified retriever
            if getattr(dc_config.unified_retrieval, "enabled", False):
                from ctxforge.retrieval.events_intent_adapter import EventsIntentAdapter
                from ctxforge.retrieval.unified_retriever import ResultSource, UnifiedRetriever
                ur_config = dc_config.unified_retrieval
                unified_retriever = UnifiedRetriever(
                    merge_strategy=ur_config.merge_strategy,
                    score_weights={
                        ResultSource.EXPERTISE: ur_config.expertise_weight,
                        ResultSource.MEMORY: ur_config.memory_weight,
                        ResultSource.GRAPH: ur_config.graph_weight,
                    },
                )
                # Register events-intent retrieval (session events with persisted intent notes).
                unified_retriever.register_store(
                    name="events_intent",
                    source=ResultSource.EVENTS_INTENT,
                    adapter=EventsIntentAdapter(session_store=session_store),
                    priority=6,
                )
            
            # Approval store
            if getattr(dc_config.approval, "enabled", False):
                from ctxforge.middleware.approval import InMemoryApprovalStore
                approval_store = InMemoryApprovalStore()
        
        # Scoped memory service
        scoped_memory_service = None
        sm_config = getattr(config, "scoped_memory", None)
        if sm_config is not None and getattr(sm_config, "enabled", False):
            scoped_memory_store = await self._create_scoped_memory_store(config)
            if scoped_memory_store is not None:
                await self._maybe_initialize(scoped_memory_store)
                owned_resources.append(scoped_memory_store)
                from ctxforge.engine.services.scoped_memory_service import ScopedMemoryService
                scoped_memory_service = ScopedMemoryService(store=scoped_memory_store)
        
        # Skill service
        skill_service = None
        skills_config = getattr(config, "skills", None)
        if skills_config is not None and getattr(skills_config, "enabled", False):
            skill_store = await self._create_skill_store(config)
            if skill_store is not None:
                await self._maybe_initialize(skill_store)
                owned_resources.append(skill_store)
                from ctxforge.engine.services.skill_matcher import RegexSkillMatcher, SkillMatcher
                from ctxforge.engine.services.skill_service import SkillService
                
                # Choose matcher based on config
                if getattr(skills_config, "use_regex_triggers", False):
                    skill_matcher = RegexSkillMatcher()
                else:
                    skill_matcher = SkillMatcher()
                
                skill_service = SkillService(store=skill_store, matcher=skill_matcher)
        
        llm_provider = self._create_llm_provider(config)

        # Entropy gate for pre-extraction filtering
        entropy_gate = None
        mq_config = getattr(config, "memory_quality", None)
        eg_config = getattr(mq_config, "entropy_gate", None) if mq_config else None
        if eg_config is not None and getattr(eg_config, "enabled", False):
            entropy_gate = EntropyGate(
                config=eg_config,
                embedding_provider=embedding_provider,
            )

        # Fast-path retriever for O(1) cache lookups
        fast_path_retriever = None
        fp_config = getattr(mq_config, "retrieval_fast_path", None) if mq_config else None
        if fp_config is not None and getattr(fp_config, "enabled", False):
            fast_path_retriever = FastPathRetriever(config=fp_config)

        # --- Memory integration services ---
        integration_pipeline = None
        preference_evolution_service = None
        memory_synthesizer = None
        personalization_metrics_service = None

        ext_cfg = config.extraction

        # Preference evolution service
        if getattr(ext_cfg, "preference_evolution_enabled", False) and llm_provider is not None:
            from ctxforge.engine.services.preference_evolution_service import (
                PreferenceEvolutionService,
            )
            from ctxforge.extraction.integration_config import PreferenceEvolutionConfig

            pref_config = PreferenceEvolutionConfig(
                enabled=True,
                contradiction_similarity_threshold=ext_cfg.preference_contradiction_similarity_threshold,
                auto_supersede=ext_cfg.preference_auto_supersede,
                importance_decay_on_supersede=ext_cfg.preference_importance_decay_on_supersede,
            )
            preference_evolution_service = PreferenceEvolutionService(
                llm=llm_provider,
                memory_store=memory_store,
                config=pref_config,
            )

        # Integration pipeline
        if getattr(ext_cfg, "integration_pipeline_enabled", False) and llm_provider is not None:
            from ctxforge.extraction.integration_config import IntegrationConfig
            from ctxforge.extraction.integration_pipeline import MemoryIntegrationPipeline

            int_config = IntegrationConfig(
                enabled=True,
                detect_threshold=ext_cfg.integration_detect_threshold,
                similarity_threshold=ext_cfg.integration_similarity_threshold,
                model=ext_cfg.integration_model,
            )
            integration_pipeline = MemoryIntegrationPipeline(
                llm=llm_provider,
                memory_store=memory_store,
                embedding_provider=embedding_provider,
                config=int_config,
                preference_evolution_service=preference_evolution_service,
            )

        # Memory synthesizer
        if getattr(ext_cfg, "synthesis_enabled", False) and llm_provider is not None:
            from ctxforge.engine.services.memory_synthesizer_service import MemorySynthesizerService
            from ctxforge.extraction.integration_config import SynthesizerConfig

            syn_config = SynthesizerConfig(
                enabled=True,
                min_memories_to_synthesize=ext_cfg.synthesis_min_memories,
                max_synthesis_tokens=ext_cfg.synthesis_max_tokens,
                model=ext_cfg.synthesis_model,
            )
            memory_synthesizer = MemorySynthesizerService(
                llm=llm_provider,
                config=syn_config,
            )

        # Personalization metrics
        if getattr(ext_cfg, "personalization_metrics_enabled", False):
            from ctxforge.engine.services.personalization_metrics_service import (
                PersonalizationMetricsService,
            )
            from ctxforge.extraction.integration_config import PersonalizationMetricsConfig

            pm_config = PersonalizationMetricsConfig(
                enabled=True,
                memory_hit_threshold=ext_cfg.personalization_memory_hit_threshold,
            )
            personalization_metrics_service = PersonalizationMetricsService(
                config=pm_config,
            )

        # Note: GraphService is constructed inside CtxForge when graph deps are provided,
        # so it is not available at factory time.
        deps = EngineDeps(
            config=config,
            session_store=session_store,
            memory_store=memory_store,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            tokenizer_provider=tokenizer_provider,
            vector_store=vector_store,
            memory_indexer=memory_indexer,
            expertise_store=expertise_store,
            expertise_retriever=expertise_retriever,
            reflector=reflector,
            curator=curator,
            scoped_memory_service=scoped_memory_service,
            skill_service=skill_service,
            graph_service=None,
        )
        prepare_chain = self._create_pipeline(config, pipeline="prepare", deps=deps)
        record_chain = self._create_pipeline(config, pipeline="record", deps=deps)
        
        return CtxForge(
            config=config,
            session_store=session_store,
            memory_store=memory_store,
            retriever=retriever,
            condenser=condenser,
            assembler=assembler,
            extractor=extractor,
            prepare_chain=prepare_chain,
            record_chain=record_chain,
            memory_indexer=memory_indexer,
            vector_store=vector_store,
            expertise_indexer=expertise_indexer,
            expertise_vector_store=expertise_vector_store,
            expertise_store=expertise_store,
            expertise_retriever=expertise_retriever,
            reflector=reflector,
            curator=curator,
            update_planner=update_planner,
            graph_store=graph_store,
            graph_extractor=graph_extractor,
            graph_ontology=graph_ontology,
            graph_embedding_provider=graph_embedding_provider,
            tokenizer_provider=tokenizer_provider,
            graph_contradiction_detector=graph_contradiction_detector,
            graph_edge_temporal_extractor=graph_edge_temporal_extractor,
            graph_community_builder=graph_community_builder,
            graph_entity_linker=graph_entity_linker,
            # Dynamic context services
            validated_knowledge_service=validated_knowledge_service,
            semantic_model_service=semantic_model_service,
            snapshot_service=snapshot_service,
            unified_retriever=unified_retriever,
            approval_store=approval_store,
            # Hierarchical memory & skills
            scoped_memory_service=scoped_memory_service,
            skill_service=skill_service,
            owned_resources=owned_resources,
            # Entropy gate
            entropy_gate=entropy_gate,
            # Fast-path retriever
            fast_path_retriever=fast_path_retriever,
            # Memory integration services
            integration_pipeline=integration_pipeline,
            memory_synthesizer=memory_synthesizer,
            personalization_metrics_service=personalization_metrics_service,
        )

    # ==========================================================================
    # DI-style entrypoint + validation (M3)
    # ==========================================================================

    @dataclass
    class ValidationResult:
        errors: List[str]
        warnings: List[str]

        @property
        def ok(self) -> bool:
            return not self.errors

    def validate_config(self, config: EngineConfig) -> "EngineFactory.ValidationResult":
        """
        Validate cross-field requirements that Pydantic won't catch:
        - vector store backends requiring API keys
        - expertise enabled requiring a non-memory vectorstore for semantic retrieval (warning)
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Memory vector store requirements
        mem_backend = config.storage.memory.vector.backend.value
        if mem_backend == "pinecone":
            api_key = (config.storage.memory.vector.extra_params or {}).get("api_key")
            if not api_key:
                errors.append("storage.memory.vector.backend=pinecone requires storage.memory.vector.extra_params.api_key")

        # Expertise requirements
        if config.expertise.enabled:
            exp_vs_backend = config.expertise.vectorstore.backend.value
            if exp_vs_backend == "memory":
                warnings.append("expertise.enabled=true but expertise.vectorstore.backend=memory (no semantic retrieval/indexing)")
            if exp_vs_backend == "pinecone":
                api_key = (config.expertise.vectorstore.extra_params or {}).get("api_key")
                if not api_key:
                    errors.append("expertise.vectorstore.backend=pinecone requires expertise.vectorstore.extra_params.api_key")

        return EngineFactory.ValidationResult(errors=errors, warnings=warnings)

    async def build(self, config: EngineConfig, **overrides) -> CtxForge:
        """
        Primary DI-style entrypoint.
        Runs config validation then delegates to create().
        """
        # Load plugins (module imports + class-path registrations) before wiring components.
        self._load_plugins(config)

        validation = self.validate_config(config)
        if validation.errors:
            raise ValueError("Invalid EngineConfig:\n- " + "\n- ".join(validation.errors))
        return await self.create(config=config, **overrides)

    def _load_plugins(self, config: EngineConfig) -> None:
        """
        Load plugins configured on the EngineConfig.

        - Imports modules listed in config.plugins.modules. If a module exposes
          a callable `register(registry)`, it will be invoked with this factory's registry.
          Otherwise, importing for side effects is expected.
        - Registers any class-path registrations listed in config.plugins.registrations.
        """
        plugins = getattr(config, "plugins", None)
        if plugins is None:
            return

        # Module imports
        for mod_path in list(getattr(plugins, "modules", []) or []):
            module = importlib.import_module(mod_path)
            register_fn = getattr(module, "register", None)
            if callable(register_fn):
                register_fn(self._registry)

        # Explicit registrations
        for reg in list(getattr(plugins, "registrations", []) or []):
            self._registry.register_component_class_path(
                component_type=reg.component_type,
                name=reg.name,
                class_path=reg.class_path,
            )

    async def _maybe_initialize(self, component: Any) -> None:
        """
        Best-effort async initialization for factory-owned components.

        - Prefer `initialize()` when present.
        - Fall back to `connect()` for legacy components (e.g., Redis stores).
        """
        init_fn = getattr(component, "initialize", None)
        if callable(init_fn):
            result = init_fn()
            if inspect.isawaitable(result):
                await result
            return

        connect_fn = getattr(component, "connect", None)
        if callable(connect_fn):
            result = connect_fn()
            if inspect.isawaitable(result):
                await result
            return

    def _create_llm_provider(self, config: EngineConfig) -> Optional[ILLMProvider]:
        """
        Create an ILLMProvider from config.llm (cached).

        Supports built-in providers registered in the registry. Some providers
        require special construction (e.g., OpenAI uses OpenAIConfig dataclass).
        
        The provider is cached by a key derived from its configuration to avoid
        creating multiple clients (which can cause connection pool exhaustion or
        rate limiting issues with Azure OpenAI).
        """
        provider_name = (getattr(config.llm.provider, "value", None) or str(config.llm.provider)).lower()
        
        # Build cache key from relevant config fields
        extra = dict(config.llm.extra_params or {})
        cache_key = f"{provider_name}|{config.llm.api_base or ''}|{config.llm.model}|{extra.get('azure_endpoint', '')}|{extra.get('api_version', '')}"
        
        if cache_key in self._llm_provider_cache:
            return self._llm_provider_cache[cache_key]

        if provider_name == "openai":
            from ctxforge.llm.openai_provider import OpenAIConfig, OpenAILLMProvider

            api_key = config.llm.api_key or ""
            openai_cfg = OpenAIConfig(
                api_key=api_key,
                model=config.llm.model,
                embedding_model=config.storage.memory.vector.embedding.model,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
            )
            provider = OpenAILLMProvider(openai_cfg)
            self._llm_provider_cache[cache_key] = provider
            return provider

        if provider_name in ("azure", "azure_openai"):
            from ctxforge.llm.azure_openai_provider import AzureOpenAIConfig, AzureOpenAILLMProvider

            extra = dict(config.llm.extra_params or {})
            azure_endpoint = (
                str(extra.get("azure_endpoint") or extra.get("endpoint") or (config.llm.api_base or "")).strip()
            )
            api_version = str(extra.get("api_version") or extra.get("azure_api_version") or "2024-02-15-preview").strip()

            # In Azure, config.llm.model is typically the deployment name.
            deployment = str(extra.get("deployment") or extra.get("chat_deployment") or config.llm.model).strip()
            embedding_deployment = str(
                extra.get("embedding_deployment") or config.storage.memory.vector.embedding.model
            ).strip()

            azure_cfg = AzureOpenAIConfig(
                api_key=config.llm.api_key or "",
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment=deployment,
                embedding_deployment=embedding_deployment,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                timeout=float(config.llm.timeout),
                max_retries=int(config.llm.max_retries),
            )
            provider = AzureOpenAILLMProvider(azure_cfg)
            self._llm_provider_cache[cache_key] = provider
            return provider

        if provider_name == "mock":
            from ctxforge.llm.mock_provider import MockLLMProvider
            provider = MockLLMProvider(config=dict(config.llm.extra_params or {}))
            self._llm_provider_cache[cache_key] = provider
            return provider

        if provider_name == "openrouter":
            from ctxforge.llm.openrouter_provider import OpenRouterConfig, OpenRouterLLMProvider

            extra = dict(config.llm.extra_params or {})
            openrouter_cfg = OpenRouterConfig(
                api_key=config.llm.api_key or extra.get("api_key") or "",
                model=config.llm.model,
                base_url=extra.get("base_url") or "https://openrouter.ai/api/v1",
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                http_referer=extra.get("http_referer"),
                site_title=extra.get("site_title"),
            )
            provider = OpenRouterLLMProvider(openrouter_cfg)
            self._llm_provider_cache[cache_key] = provider
            return provider

        provider_cls = self._registry.get_llm(provider_name)
        if provider_cls is None:
            # Fallback to mock
            from ctxforge.llm.mock_provider import MockLLMProvider
            provider = MockLLMProvider(config=dict(config.llm.extra_params or {}))
            self._llm_provider_cache[cache_key] = provider
            return provider

        # Generic: try passing dict config, then no-arg.
        try:
            provider = provider_cls(dict(config.llm.extra_params or {}))
        except TypeError:
            provider = provider_cls()
        self._llm_provider_cache[cache_key] = provider
        return provider
    
    def _create_session_store(self, config: EngineConfig) -> ISessionStore:
        """Create session store from configuration."""
        backend = config.storage.session.backend.value
        store_class = self._registry.get_session_store(backend)
        
        if store_class is None:
            # Fall back to in-memory store
            from ctxforge.storage.memory.session import InMemorySessionStore
            return InMemorySessionStore()
        
        return store_class(config.storage.session)
    
    def _create_memory_store(self, config: EngineConfig) -> IMemoryStore:
        """Create memory store from configuration."""
        backend = config.storage.memory.store_backend.value
        store_class = self._registry.get_memory_store(backend)
        
        if store_class is None:
            # Fall back to in-memory store
            from ctxforge.storage.memory.memory import InMemoryMemoryStore
            return InMemoryMemoryStore()

        # Build backend-specific configs (stores generally expect RedisConfig/PostgresConfig/MySQLConfig).
        if backend == "redis":
            cfg = self._redis_config_from_memory_store_config(config)
            return store_class(cfg)
        if backend == "postgres":
            cfg = self._postgres_config_from_memory_store_config(config)
            return store_class(cfg)
        if backend == "mysql":
            cfg = self._mysql_config_from_memory_store_config(config)
            return store_class(cfg)

        # Generic fallback: pass the config object (or dict) if accepted.
        try:
            return store_class(config.storage.memory)
        except TypeError:
            return store_class(dict(config.storage.memory.store_extra_params or {}))
    
    def _create_retriever(
        self, 
        config: EngineConfig, 
        memory_store: IMemoryStore,
        memory_indexer=None,
    ) -> Optional[IRetriever]:
        """Create retriever from configuration."""
        strategy = config.retrieval.strategy.value
        retriever_class = self._registry.get_retriever(strategy)
        
        if retriever_class is None:
            # Fall back to simple retriever
            from ctxforge.retrieval.retrievers.base import SimpleRetriever
            base = SimpleRetriever(memory_store)
            return self._maybe_wrap_with_reranker(config, base)
        
        # Different retrievers have different initialization requirements
        if strategy in ("simple", "keyword"):
            from ctxforge.retrieval.retrievers.base import SimpleRetriever
            base = SimpleRetriever(memory_store)
            return self._maybe_wrap_with_reranker(config, base)
        elif strategy in ("semantic", "hybrid", "temporal"):
            # Prefer vectorstore-backed retrievers when an indexer is available
            if memory_indexer is not None:
                if strategy == "semantic":
                    from ctxforge.retrieval.retrievers.vectorstore_semantic import (
                        VectorStoreSemanticRetriever,
                    )
                    base = VectorStoreSemanticRetriever(memory_store, memory_indexer)
                    return self._maybe_wrap_with_reranker(config, base)
                if strategy == "hybrid":
                    from ctxforge.retrieval.retrievers.vectorstore_hybrid import (
                        VectorStoreHybridRetriever,
                    )
                    base = VectorStoreHybridRetriever(
                        memory_store,
                        memory_indexer,
                        semantic_weight=config.retrieval.semantic_weight,
                        keyword_weight=config.retrieval.keyword_weight,
                    )
                    return self._maybe_wrap_with_reranker(config, base)
                if strategy == "temporal":
                    from ctxforge.retrieval.retrievers.vectorstore_temporal import (
                        VectorStoreTemporalRetriever,
                    )
                    base = VectorStoreTemporalRetriever(memory_store, memory_indexer)
                    return self._maybe_wrap_with_reranker(config, base)

            # Otherwise fall back to embedding-in-memory retrievers (requires memory.embedding present)
            async def noop_embedding(text: str) -> list:
                return []
            base = retriever_class(memory_store, noop_embedding)
            return self._maybe_wrap_with_reranker(config, base)
        else:
            # Try direct instantiation
            try:
                base = retriever_class(memory_store)
                return self._maybe_wrap_with_reranker(config, base)
            except TypeError:
                base = retriever_class(config.retrieval)
                return self._maybe_wrap_with_reranker(config, base)

    def _maybe_wrap_with_reranker(self, config: EngineConfig, base: IRetriever) -> IRetriever:
        """Wrap a retriever with an IReranker if enabled."""
        if not getattr(config.retrieval, "rerank_enabled", False):
            return base

        reranker_name = (getattr(config.retrieval, "reranker", None) or "llm").lower()
        llm = self._create_llm_provider(config)

        reranker = None
        if reranker_name == "llm":
            if llm is None:
                return base
            from ctxforge.retrieval.rerankers.llm import LLMReranker
            model = config.retrieval.rerank_model or llm.default_model
            reranker = LLMReranker(llm_provider=llm, model=model)
        else:
            cls = self._registry.get_reranker(reranker_name)
            if cls is not None:
                try:
                    reranker = cls()
                except TypeError:
                    reranker = None

        if reranker is None:
            return base

        from ctxforge.retrieval.retrievers.reranking import RerankingRetriever
        return RerankingRetriever(base=base, reranker=reranker)

    def _create_embedding_provider(self, config: EngineConfig) -> Optional[IEmbeddingProvider]:
        """Create memory embedding provider from configuration (best-effort)."""
        return self._create_embedding_provider_from_embedding_config(
            embedding_config=config.storage.memory.vector.embedding,
            llm_config=config.llm,
        )

    def _create_embedding_provider_from_embedding_config(
        self,
        embedding_config,
        llm_config,
    ) -> Optional[IEmbeddingProvider]:
        """Create embedding provider from EmbeddingConfig."""
        provider_name = (embedding_config.provider or "").lower()

        if provider_name == "openai":
            from ctxforge.llm.openai_provider import OpenAIConfig, OpenAIEmbeddingProvider
            openai_cfg = OpenAIConfig(
                api_key=embedding_config.api_key or (llm_config.api_key or ""),
                model=llm_config.model,
                embedding_model=embedding_config.model,
                base_url=getattr(embedding_config, "base_url", None),
            )
            return OpenAIEmbeddingProvider(openai_cfg)

        if provider_name in ("azure", "azure_openai"):
            from ctxforge.llm.azure_openai_provider import (
                AzureOpenAIConfig,
                AzureOpenAIEmbeddingProvider,
            )

            extra = dict(llm_config.extra_params or {})
            azure_endpoint = (
                str(extra.get("azure_endpoint") or extra.get("endpoint") or (llm_config.api_base or "")).strip()
            )
            api_version = str(extra.get("api_version") or extra.get("azure_api_version") or "2024-02-15-preview").strip()

            # For embeddings, `embedding_config.model` is typically the embedding deployment name.
            embedding_deployment = str(embedding_config.model).strip()
            # For chat deployment, fall back to llm_config.model (deployment name).
            deployment = str(extra.get("deployment") or extra.get("chat_deployment") or llm_config.model).strip()

            azure_cfg = AzureOpenAIConfig(
                api_key=embedding_config.api_key or (llm_config.api_key or ""),
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment=deployment,
                embedding_deployment=embedding_deployment,
                timeout=float(llm_config.timeout),
                max_retries=int(llm_config.max_retries),
            )
            return AzureOpenAIEmbeddingProvider(azure_cfg)

        provider_cls = self._registry.get_embedding(provider_name)
        if provider_cls is None:
            from ctxforge.llm.mock_provider import MockEmbeddingProvider
            return MockEmbeddingProvider(dimension=embedding_config.dimension)

        try:
            return provider_cls(embedding_config)
        except TypeError:
            return provider_cls()

    async def _create_vector_store(
        self,
        config: EngineConfig,
        embedding_provider: Optional[IEmbeddingProvider] = None,
    ) -> Optional[IVectorStore]:
        """Create and initialize a vector store based on storage.memory.vector backend."""
        backend = config.storage.memory.vector.backend.value
        if backend == "memory":
            return None

        dimension = config.storage.memory.vector.embedding.dimension
        derived = await self._derive_embedding_dimension(embedding_provider)
        if derived > 0:
            dimension = derived

        return await self._create_vector_store_from_spec(
            backend=backend,
            index_name=config.storage.memory.vector.index_name,
            connection_string=config.storage.memory.vector.connection_string,
            embedding_dim=dimension,
            extra_params=config.storage.memory.vector.extra_params,
        )

    async def _derive_embedding_dimension(
        self,
        embedding_provider: Optional[IEmbeddingProvider],
    ) -> int:
        """Best-effort: ask a provider that can report its dimension.

        Only providers exposing ``get_dimension()`` (e.g. the local
        sentence-transformers provider) are consulted; for others the configured
        ``embedding.dimension`` is authoritative.
        """
        if embedding_provider is None:
            return 0
        getter = getattr(embedding_provider, "get_dimension", None)
        if not callable(getter):
            return 0
        try:
            result = getter()
            if inspect.isawaitable(result):
                result = await result
            return int(result or 0)
        except Exception as exc:
            logger.warning("Failed to derive embedding dimension: %s", exc)
            return 0

    async def _create_vector_store_from_spec(
        self,
        backend: str,
        index_name: str,
        connection_string: Optional[str],
        embedding_dim: int,
        extra_params: Optional[Dict[str, Any]],
    ) -> Optional[IVectorStore]:
        """Create and initialize a vector store from a generic spec."""
        if backend == "memory":
            return None

        extra = dict(extra_params or {})
        conn = connection_string

        store: Optional[IVectorStore] = None
        if backend == "chromadb":
            from ctxforge.vectorstores.chroma_store import ChromaConfig, ChromaDBStore
            cfg = ChromaConfig(
                collection_name=index_name,
                persist_directory=conn or extra.get("persist_directory"),
                dimension=embedding_dim,
                **{k: v for k, v in extra.items() if k in {
                    "host", "port", "ssl", "tenant", "database", "auth_token", "create_collection_if_missing",
                }},
            )
            store = ChromaDBStore(cfg)
        elif backend == "pinecone":
            from ctxforge.vectorstores.pinecone_store import PineconeConfig, PineconeStore
            api_key = extra.get("api_key") or ""
            if not api_key:
                raise ValueError("Pinecone requires extra_params.api_key")
            cfg = PineconeConfig(
                api_key=api_key,
                index_name=index_name,
                dimension=embedding_dim,
                environment=extra.get("environment"),
                host=extra.get("host"),
                create_index_if_missing=bool(extra.get("create_index_if_missing", False)),
            )
            store = PineconeStore(cfg)
        elif backend == "weaviate":
            from ctxforge.vectorstores.weaviate_store import WeaviateConfig, WeaviateStore
            cfg = WeaviateConfig(
                url=conn or extra.get("url") or "http://localhost:8080",
                api_key=extra.get("api_key"),
                class_name=extra.get("class_name", "Memory"),
                dimension=embedding_dim,
            )
            store = WeaviateStore(cfg)

        if store is None:
            return None

        await store.initialize()
        return store

    async def _create_expertise_store(self, config: EngineConfig) -> Optional[IExpertiseStore]:
        """Create expertise store from configuration."""
        backend = config.expertise.store.backend.value
        cls = self._registry.get_expertise_store(backend)
        if cls is None:
            from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
            return InMemoryExpertiseStore()

        # memory store takes dict config; postgres/redis have their own config types
        if backend == "postgres":
            from ctxforge.storage.postgres.expertise import PostgresExpertiseStore
            pg = self._postgres_config_from_expertise_config(config)
            return PostgresExpertiseStore(config=pg)
        if backend == "redis":
            from ctxforge.storage.redis.expertise import RedisExpertiseStore
            rc = self._redis_config_from_expertise_config(config)
            return RedisExpertiseStore(config=rc)

        # fallback
        try:
            return cls(config.expertise.store.extra_params)
        except TypeError:
            return cls()

    def _postgres_config_from_expertise_config(self, config: EngineConfig):
        from ctxforge.storage.connection import PostgresConfig

        extra = dict(config.expertise.store.extra_params or {})
        conn = config.expertise.store.connection_string
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("postgres"):
                qs = parse_qs(u.query or "")
                sslmode = (qs.get("sslmode", ["prefer"])[0] or "prefer").lower()
                pg = PostgresConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 5432,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "postgres",
                    password=u.password,
                    ssl=(sslmode in ("require", "verify-ca", "verify-full")),
                )
                # allow overrides
                for k, v in extra.items():
                    if hasattr(pg, k):
                        setattr(pg, k, v)
                return pg

        # No connection string -> use extra params as overrides on defaults
        pg = PostgresConfig()
        for k, v in extra.items():
            if hasattr(pg, k):
                setattr(pg, k, v)
        return pg

    def _redis_config_from_expertise_config(self, config: EngineConfig):
        from ctxforge.storage.connection import RedisConfig

        extra = dict(config.expertise.store.extra_params or {})
        conn = config.expertise.store.connection_string
        if conn:
            u = urlparse(conn)
            if u.scheme in ("redis", "rediss"):
                db = 0
                if (u.path or "").lstrip("/").isdigit():
                    db = int((u.path or "").lstrip("/"))
                rc = RedisConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 6379,
                    db=db,
                    password=u.password,
                    ssl=(u.scheme == "rediss"),
                )
                for k, v in extra.items():
                    if hasattr(rc, k):
                        setattr(rc, k, v)
                return rc

        rc = RedisConfig()
        for k, v in extra.items():
            if hasattr(rc, k):
                setattr(rc, k, v)
        return rc

    def _postgres_config_from_memory_store_config(self, config: EngineConfig):
        from ctxforge.storage.connection import PostgresConfig

        extra = dict(config.storage.memory.store_extra_params or {})
        conn = config.storage.memory.store_connection_string
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("postgres"):
                qs = parse_qs(u.query or "")
                sslmode = (qs.get("sslmode", ["prefer"])[0] or "prefer").lower()
                pg = PostgresConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 5432,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "postgres",
                    password=u.password,
                    ssl=(sslmode in ("require", "verify-ca", "verify-full")),
                )
                for k, v in extra.items():
                    if hasattr(pg, k):
                        setattr(pg, k, v)
                return pg

        pg = PostgresConfig()
        for k, v in extra.items():
            if hasattr(pg, k):
                setattr(pg, k, v)
        return pg

    def _mysql_config_from_memory_store_config(self, config: EngineConfig):
        from ctxforge.storage.connection import MySQLConfig

        extra = dict(config.storage.memory.store_extra_params or {})
        conn = config.storage.memory.store_connection_string
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("mysql"):
                mc = MySQLConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 3306,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "root",
                    password=u.password,
                )
                for k, v in extra.items():
                    if hasattr(mc, k):
                        setattr(mc, k, v)
                return mc

        mc = MySQLConfig()
        for k, v in extra.items():
            if hasattr(mc, k):
                setattr(mc, k, v)
        return mc

    def _redis_config_from_memory_store_config(self, config: EngineConfig):
        from ctxforge.storage.connection import RedisConfig

        extra = dict(config.storage.memory.store_extra_params or {})
        conn = config.storage.memory.store_connection_string
        if conn:
            u = urlparse(conn)
            if u.scheme in ("redis", "rediss"):
                db = 0
                if (u.path or "").lstrip("/").isdigit():
                    db = int((u.path or "").lstrip("/"))
                rc = RedisConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 6379,
                    db=db,
                    password=u.password,
                    ssl=(u.scheme == "rediss"),
                )
                for k, v in extra.items():
                    if hasattr(rc, k):
                        setattr(rc, k, v)
                return rc

        rc = RedisConfig()
        for k, v in extra.items():
            if hasattr(rc, k):
                setattr(rc, k, v)
        return rc
    
    async def _create_scoped_memory_store(self, config: EngineConfig):
        """Create scoped memory store from configuration."""
        sm_config = getattr(config, "scoped_memory", None)
        if sm_config is None:
            return None
        
        backend = sm_config.store.backend.value
        
        if backend == "memory":
            from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore
            return InMemoryScopedMemoryStore()
        
        if backend == "postgres":
            from ctxforge.storage.connection import PostgresConnectionManager
            from ctxforge.storage.postgres.scoped_memory import PostgresScopedMemoryStore
            
            pg_config = self._postgres_config_from_scoped_memory_config(config)
            manager = PostgresConnectionManager(pg_config)
            return PostgresScopedMemoryStore(connection_manager=manager)
        
        if backend == "mysql":
            from ctxforge.storage.connection import MySQLConnectionManager
            from ctxforge.storage.mysql.scoped_memory import MySQLScopedMemoryStore
            
            mysql_config = self._mysql_config_from_scoped_memory_config(config)
            manager = MySQLConnectionManager(mysql_config)
            return MySQLScopedMemoryStore(connection_manager=manager)
        
        # Fallback to in-memory
        from ctxforge.storage.memory.scoped_memory import InMemoryScopedMemoryStore
        return InMemoryScopedMemoryStore()
    
    async def _create_skill_store(self, config: EngineConfig):
        """Create skill store from configuration."""
        skills_config = getattr(config, "skills", None)
        if skills_config is None:
            return None
        
        backend = skills_config.store.backend.value
        
        if backend == "memory":
            from ctxforge.storage.memory.skill import InMemorySkillStore
            return InMemorySkillStore()
        
        if backend == "postgres":
            from ctxforge.storage.connection import PostgresConnectionManager
            from ctxforge.storage.postgres.skill import PostgresSkillStore
            
            pg_config = self._postgres_config_from_skill_config(config)
            manager = PostgresConnectionManager(pg_config)
            return PostgresSkillStore(connection_manager=manager)
        
        if backend == "mysql":
            from ctxforge.storage.connection import MySQLConnectionManager
            from ctxforge.storage.mysql.skill import MySQLSkillStore
            
            mysql_config = self._mysql_config_from_skill_config(config)
            manager = MySQLConnectionManager(mysql_config)
            return MySQLSkillStore(connection_manager=manager)
        
        # Fallback to in-memory
        from ctxforge.storage.memory.skill import InMemorySkillStore
        return InMemorySkillStore()
    
    def _postgres_config_from_scoped_memory_config(self, config: EngineConfig):
        """Create PostgresConfig from scoped memory configuration."""
        from ctxforge.storage.connection import PostgresConfig
        
        sm_config = config.scoped_memory
        extra = dict(sm_config.store.extra_params or {})
        conn = sm_config.store.connection_string
        
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("postgres"):
                qs = parse_qs(u.query or "")
                sslmode = (qs.get("sslmode", ["prefer"])[0] or "prefer").lower()
                pg = PostgresConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 5432,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "postgres",
                    password=u.password,
                    ssl=(sslmode in ("require", "verify-ca", "verify-full")),
                )
                for k, v in extra.items():
                    if hasattr(pg, k):
                        setattr(pg, k, v)
                return pg
        
        pg = PostgresConfig()
        for k, v in extra.items():
            if hasattr(pg, k):
                setattr(pg, k, v)
        return pg
    
    def _mysql_config_from_scoped_memory_config(self, config: EngineConfig):
        """Create MySQLConfig from scoped memory configuration."""
        from ctxforge.storage.connection import MySQLConfig
        
        sm_config = config.scoped_memory
        extra = dict(sm_config.store.extra_params or {})
        conn = sm_config.store.connection_string
        
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("mysql"):
                mc = MySQLConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 3306,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "root",
                    password=u.password,
                )
                for k, v in extra.items():
                    if hasattr(mc, k):
                        setattr(mc, k, v)
                return mc
        
        mc = MySQLConfig()
        for k, v in extra.items():
            if hasattr(mc, k):
                setattr(mc, k, v)
        return mc
    
    def _postgres_config_from_skill_config(self, config: EngineConfig):
        """Create PostgresConfig from skill configuration."""
        from ctxforge.storage.connection import PostgresConfig
        
        skills_config = config.skills
        extra = dict(skills_config.store.extra_params or {})
        conn = skills_config.store.connection_string
        
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("postgres"):
                qs = parse_qs(u.query or "")
                sslmode = (qs.get("sslmode", ["prefer"])[0] or "prefer").lower()
                pg = PostgresConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 5432,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "postgres",
                    password=u.password,
                    ssl=(sslmode in ("require", "verify-ca", "verify-full")),
                )
                for k, v in extra.items():
                    if hasattr(pg, k):
                        setattr(pg, k, v)
                return pg
        
        pg = PostgresConfig()
        for k, v in extra.items():
            if hasattr(pg, k):
                setattr(pg, k, v)
        return pg
    
    def _mysql_config_from_skill_config(self, config: EngineConfig):
        """Create MySQLConfig from skill configuration."""
        from ctxforge.storage.connection import MySQLConfig
        
        skills_config = config.skills
        extra = dict(skills_config.store.extra_params or {})
        conn = skills_config.store.connection_string
        
        if conn:
            u = urlparse(conn)
            if u.scheme.startswith("mysql"):
                mc = MySQLConfig(
                    host=u.hostname or "localhost",
                    port=u.port or 3306,
                    database=(u.path or "/context_engine").lstrip("/") or "context_engine",
                    user=u.username or "root",
                    password=u.password,
                )
                for k, v in extra.items():
                    if hasattr(mc, k):
                        setattr(mc, k, v)
                return mc
        
        mc = MySQLConfig()
        for k, v in extra.items():
            if hasattr(mc, k):
                setattr(mc, k, v)
        return mc
    
    def _create_condenser(self, config: EngineConfig) -> Optional[ICondenser]:
        """Create condenser from configuration."""
        from ctxforge.config.base import CompactionStrategyType

        strategy = config.compaction.strategy

        # Handle pipeline strategy - compose multiple condensers
        if strategy == CompactionStrategyType.PIPELINE:
            return self._create_condenser_pipeline(config)

        # Handle structured strategy - may need LLM function
        if strategy == CompactionStrategyType.STRUCTURED:
            return self._create_structured_condenser(config)

        # Standard single condenser
        condenser_class = self._registry.get_condenser(strategy.value)
        if condenser_class is None:
            return None

        # Note: condensers don't take config in __init__, they receive it in condense()
        return condenser_class()

    def _create_condenser_pipeline(self, config: EngineConfig) -> Optional[ICondenser]:
        """Create a condenser pipeline from config.pipeline settings."""
        from ctxforge.compaction.pipeline import CondenserPipeline

        if not config.compaction.pipeline:
            # Empty pipeline - fall back to default
            return None

        condensers = []
        for step in config.compaction.pipeline:
            condenser_class = self._registry.get_condenser(step.type)
            if condenser_class is None:
                continue

            # Instantiate with step-specific config if needed
            step_config = step.config or {}
            if step_config:
                try:
                    condenser = condenser_class(**step_config)
                except TypeError:
                    # Class doesn't accept config in __init__
                    condenser = condenser_class()
            else:
                condenser = condenser_class()

            condensers.append(condenser)

        if not condensers:
            return None

        return CondenserPipeline(*condensers)

    def _create_structured_condenser(self, config: EngineConfig) -> Optional[ICondenser]:
        """Create a structured summarizing condenser."""
        from ctxforge.compaction.structured_summary import (
            StructuredSummarizingCondenser,
        )

        return StructuredSummarizingCondenser(
            max_events=config.compaction.structured_max_events,
            keep_first=config.compaction.structured_keep_first,
            keep_last=config.compaction.structured_keep_last,
        )

    def _create_assembler(self, config: EngineConfig) -> IContextAssembler:
        """Create context assembler from configuration."""
        assembler_type = "default"
        assembler_class = self._registry.get_assembler(assembler_type)
        if assembler_class is None:
            from ctxforge.compaction.assembler import DefaultContextAssembler
            return DefaultContextAssembler()
        return assembler_class()
    
    def _create_extractor(self, config: EngineConfig) -> Optional[IMemoryExtractor]:
        """Create extractor from configuration."""
        if not config.extraction.enabled:
            return None

        base_extractor: Optional[IMemoryExtractor] = None

        # Try to get LLM extractor if configured
        if config.extraction.use_llm:
            extractor_class = self._registry.get_extractor("llm")
            if extractor_class:
                base_extractor = extractor_class(config.extraction)

        # Fall back to pattern extractor
        if base_extractor is None and config.extraction.use_patterns:
            extractor_class = self._registry.get_extractor("pattern")
            if extractor_class:
                base_extractor = extractor_class(config.extraction)

        if base_extractor is None:
            return None

        # Wrap with gist-enhanced extraction if configured
        if config.extraction.extract_gists:
            llm = self._create_llm_provider(config)
            if llm is not None:
                from ctxforge.extraction.gist_enhanced_extractor import GistEnhancedExtractor
                from ctxforge.extraction.gist_extractor import GistExtractor

                gist_ext = GistExtractor(
                    llm_provider=llm,
                    model=config.extraction.gist_model,
                )
                if config.extraction.gist_enhanced_facts and hasattr(base_extractor, "_do_extract"):
                    base_extractor = GistEnhancedExtractor(
                        gist_extractor=gist_ext,
                        fact_extractor=base_extractor,
                    )

        return base_extractor
    
    def _create_pipeline(self, config: EngineConfig, pipeline: str, deps: Optional[EngineDeps] = None):
        """
        Create a MiddlewareChain for the given pipeline name.
        
        Args:
            pipeline: "prepare" or "record"
        """
        from ctxforge.middleware.base import MiddlewareChain

        chain_cfg = getattr(config.pipelines, pipeline).chain
        # Sort by priority desc
        entries = sorted(chain_cfg, key=lambda m: m.priority, reverse=True)
        chain = MiddlewareChain()
        for entry in entries:
            if not entry.enabled:
                continue
            middleware = self._instantiate_middleware(entry.type, entry.config, deps=deps)
            if middleware is None:
                continue
            # Phase filtering wrapper if configured
            if entry.phases:
                middleware = _PhaseFilteredMiddleware(
                    inner=middleware,
                    phases=set(entry.phases),
                )
            chain.add(middleware)
        return chain

    def _instantiate_middleware(
        self,
        middleware_type: str,
        cfg: Dict[str, Any],
        deps: Optional[EngineDeps],
    ):
        """
        Instantiate middleware by type using built-in factories.
        
        For M1 we support a core set of built-ins. More middleware types
        can be added by extending this method or moving to a registry-based
        constructor convention.
        """
        middleware_type_l = middleware_type.lower()
        cfg = dict(cfg or {})

        # Preferred: middleware factory (dependency-aware) registered in the registry.
        factory = self._registry.get_middleware_factory(middleware_type_l)
        if factory is not None and deps is not None:
            try:
                created = factory.create(config=cfg, deps=deps)
            except TypeError:
                # Support factories registered as callables: factory(config=..., deps=...)
                created = factory(config=cfg, deps=deps)
            return created

        # Temporary: legacy built-in wiring kept for compatibility.
        if middleware_type_l == "pii":
            from ctxforge.middleware.pii.middleware import PIIMiddleware
            return PIIMiddleware(**cfg)
        if middleware_type_l == "audit":
            from ctxforge.middleware.audit.middleware import AuditMiddleware
            return AuditMiddleware(**cfg)
        if middleware_type_l == "rate_limit":
            from ctxforge.middleware.ratelimit.limiter import TokenBucketLimiter
            from ctxforge.middleware.ratelimit.middleware import RateLimitMiddleware
            rate = float(cfg.pop("rate", 10.0))
            capacity = int(cfg.pop("capacity", 100))
            limiter = TokenBucketLimiter(rate=rate, capacity=capacity)
            return RateLimitMiddleware(limiter=limiter, **cfg)
        if middleware_type_l == "content_filter":
            from ctxforge.middleware.content.middleware import ContentFilterMiddleware
            return ContentFilterMiddleware(**cfg)
        if middleware_type_l == "expertise_retrieval":
            from ctxforge.middleware.expertise.middleware import ExpertiseRetrievalMiddleware
            retriever = getattr(deps, "expertise_retriever", None) if deps is not None else None
            store = getattr(deps, "expertise_store", None) if deps is not None else None
            if retriever is None:
                return None
            return ExpertiseRetrievalMiddleware(
                retriever=retriever,
                expertise_store=store,
                **cfg,
            )
        if middleware_type_l == "expertise_evolution":
            from ctxforge.middleware.expertise.middleware import ExpertiseEvolutionMiddleware
            reflector = getattr(deps, "reflector", None) if deps is not None else None
            store = getattr(deps, "expertise_store", None) if deps is not None else None
            curator = getattr(deps, "curator", None) if deps is not None else None
            if reflector is None or store is None:
                return None
            return ExpertiseEvolutionMiddleware(
                reflector=reflector,
                expertise_store=store,
                curator=curator,
                **cfg,
            )
        if middleware_type_l == "expertise_audit":
            # Optional: not wired in demo yet
            return None
        if middleware_type_l == "skills":
            skill_service = getattr(deps, "skill_service", None) if deps is not None else None
            if skill_service is None:
                return None
            from ctxforge.middleware.skills import SkillsMiddleware
            skills_cfg = getattr(deps.config, "skills", None) if deps is not None else None
            return SkillsMiddleware(
                skill_service=skill_service,
                auto_activate=getattr(skills_cfg, "auto_activate", True) if skills_cfg else cfg.get("auto_activate", True),
                max_auto_skills=getattr(skills_cfg, "max_auto_skills", 2) if skills_cfg else cfg.get("max_auto_skills", 2),
                confidence_threshold=getattr(skills_cfg, "confidence_threshold", 0.7) if skills_cfg else cfg.get("confidence_threshold", 0.7),
                **{k: v for k, v in cfg.items() if k not in ("auto_activate", "max_auto_skills", "confidence_threshold")},
            )
        if middleware_type_l == "skill_request":
            skill_service = getattr(deps, "skill_service", None) if deps is not None else None
            if skill_service is None:
                return None
            from ctxforge.middleware.skills import SkillRequestMiddleware
            return SkillRequestMiddleware(skill_service=skill_service, **cfg)
        if middleware_type_l == "scoped_memory":
            scoped_memory_service = getattr(deps, "scoped_memory_service", None) if deps is not None else None
            if scoped_memory_service is None:
                return None
            from ctxforge.middleware.scoped_memory import ScopedMemoryMiddleware
            return ScopedMemoryMiddleware(memory_service=scoped_memory_service, **cfg)
        if middleware_type_l == "scoped_memory_auto_learn":
            scoped_memory_service = getattr(deps, "scoped_memory_service", None) if deps is not None else None
            if scoped_memory_service is None:
                return None
            from ctxforge.middleware.scoped_memory import ScopedMemoryAutoLearnMiddleware
            return ScopedMemoryAutoLearnMiddleware(memory_service=scoped_memory_service, **cfg)
        if middleware_type_l == "tool_compression":
            from ctxforge.middleware.tool_compression import ToolCompressionMiddleware
            return ToolCompressionMiddleware(**cfg)
        if middleware_type_l == "selective_tool_compression":
            from ctxforge.middleware.tool_compression import SelectiveCompressionMiddleware
            return SelectiveCompressionMiddleware(**cfg)
        if middleware_type_l == "query_rewriter":
            from ctxforge.middleware.query_rewriter import QueryRewriterMiddleware
            llm = getattr(deps, "llm_provider", None) if deps is not None else None
            return QueryRewriterMiddleware(llm_provider=llm, **cfg)
        if middleware_type_l == "intent_notes":
            llm = getattr(deps, "llm_provider", None) if deps is not None else None
            if llm is None:
                return None
            from ctxforge.core.scoped_memory import MemoryCategory
            from ctxforge.engine.services.intent_note_service import (
                IntentNoteService,
                IntentNoteServiceConfig,
            )
            from ctxforge.middleware.intent_notes import IntentNotesMiddleware

            project_id = cfg.pop("project_id", None)
            allow_overwrite = bool(cfg.pop("allow_overwrite", False))
            generate_for_event_types = cfg.pop("generate_for_event_types", None)
            include_tool_events = bool(cfg.pop("include_tool_events", False))
            min_content_length = cfg.pop("min_content_length", None)
            max_history_events_for_prompt = cfg.pop("max_history_events_for_prompt", None)
            model = cfg.pop("model", None)
            functional_seed_key = cfg.pop(
                "functional_seed_scoped_memory_key",
                "intent_note.functional_type_seeds",
            )
            functional_seed_category = cfg.pop("functional_seed_category", "convention")
            try:
                seed_category = MemoryCategory(functional_seed_category)
            except Exception:
                seed_category = MemoryCategory.CONVENTION

            svc_cfg = IntentNoteServiceConfig(
                allow_overwrite=allow_overwrite,
                include_tool_events=include_tool_events,
                model=model,
                min_content_length=int(min_content_length) if min_content_length is not None else 20,
                max_history_events_for_prompt=(
                    int(max_history_events_for_prompt)
                    if max_history_events_for_prompt is not None
                    else 10
                ),
            )
            svc = IntentNoteService(llm_provider=llm, config=svc_cfg)

            scoped_memory_service = getattr(deps, "scoped_memory_service", None) if deps is not None else None
            return IntentNotesMiddleware(
                intent_note_service=svc,
                scoped_memory_service=scoped_memory_service,
                project_id=project_id,
                allow_overwrite=allow_overwrite,
                generate_for_event_types=generate_for_event_types,
                functional_seed_scoped_memory_key=functional_seed_key,
                functional_seed_category=seed_category,
                **cfg,
            )

        # Try registry lookup as a last resort (expects constructor compatible with cfg)
        cls = self._registry.get_middleware(middleware_type_l)
        if cls:
            try:
                return cls(**cfg)
            except TypeError:
                return cls(cfg)
        return None
    
    @classmethod
    async def from_config(cls, config: EngineConfig) -> CtxForge:
        """Create an engine from an EngineConfig."""
        factory = cls()
        return await factory.build(config)
    
    @classmethod
    async def from_config_file(cls, path: str) -> CtxForge:
        """Create an engine from a configuration file."""
        config = load_config(file_path=path)
        return await cls.from_config(config)
    
    @classmethod
    async def from_dict(
        cls,
        data: Dict[str, Any],
        use_env: bool = True,
    ) -> CtxForge:
        """Create an engine from a configuration dictionary."""
        config = load_config(overrides=data, use_env=use_env)
        return await cls.from_config(config)
    
    @classmethod
    async def create_default(cls) -> CtxForge:
        """Create an engine with default configuration."""
        return await cls.from_config(DEFAULT_CONFIG)
    
    @classmethod
    async def create_minimal(
        cls,
        session_store: Optional[ISessionStore] = None,
        memory_store: Optional[IMemoryStore] = None,
    ) -> CtxForge:
        """Create a minimal engine with just storage."""
        from ctxforge.storage.memory.memory import InMemoryMemoryStore
        from ctxforge.storage.memory.session import InMemorySessionStore
        
        return CtxForge(
            config=DEFAULT_CONFIG,
            session_store=session_store or InMemorySessionStore(),
            memory_store=memory_store or InMemoryMemoryStore(),
        )


class _PhaseFilteredMiddleware:
    """
    Wrapper middleware that only runs the inner middleware on configured phases.
    """

    def __init__(self, inner, phases: set[str]):
        self._inner = inner
        self._phases = phases

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", self.__class__.__name__)

    async def process(self, context, next):
        phase = getattr(context, "phase", None)
        if phase is not None and phase not in self._phases:
            return await next(context)
        return await self._inner.process(context, next)
