"""
The Context Engine - Framework for LLM context management.

This is the core class that manages context for LLM agents without
being coupled to any specific LLM provider. It handles session management,
memory retrieval, context assembly, expertise management, and turn recording.

The engine produces Context objects that can be used with ANY LLM or
agent framework (LangGraph, LangChain, custom agents, etc.)
"""

import asyncio
import inspect
import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from ctxforge.compaction.view import CompactionView, CondensationResult, ICondenser
from ctxforge.config.base import EngineConfig
from ctxforge.core.context import Context
from ctxforge.core.events import Event
from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    TurnOutcome,
)
from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.core.session import Session
from ctxforge.engine.services.assembly_service import AssemblyService
from ctxforge.engine.services.compaction_service import CompactionService
from ctxforge.engine.services.expertise_service import ExpertiseService
from ctxforge.engine.services.fusion_service import TwoStepInputs
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.engine.services.memory_service import MemoryService
from ctxforge.engine.services.memory_synthesizer_service import MemorySynthesizerService
from ctxforge.engine.services.memory_update_service import MemoryUpdateService
from ctxforge.engine.services.personalization_metrics_service import PersonalizationMetricsService
from ctxforge.engine.services.retrieval_controller_service import (
    RetrievalControllerResult,
    RetrievalControllerService,
)
from ctxforge.engine.services.session_service import SessionService
from ctxforge.engine.services.turn_recording_service import TurnRecordingService
from ctxforge.extraction.entropy_gate import EntropyGate
from ctxforge.extraction.integration_pipeline import MemoryIntegrationPipeline
from ctxforge.graph.communities.builder import CommunityBuilder
from ctxforge.graph.ontology import GraphOntology
from ctxforge.middleware import MiddlewareChain, MiddlewareContext
from ctxforge.protocols.compactor import IContextAssembler
from ctxforge.protocols.expertise import (
    ICurator,
    IExpertiseRetriever,
    IExpertiseStore,
    IReflector,
)
from ctxforge.protocols.extractor import ExtractionConfig, IMemoryExtractor
from ctxforge.protocols.graph import IGraphExtractor, IGraphStore
from ctxforge.protocols.graph_maintenance import (
    IGraphContradictionDetector,
    IGraphEdgeTemporalExtractor,
)
from ctxforge.protocols.llm import ChatMessage, IEmbeddingProvider, ILLMProvider
from ctxforge.protocols.retriever import IRetriever
from ctxforge.protocols.storage import IMemoryStore, ISessionStore
from ctxforge.protocols.tokenizer import ITokenizerProvider
from ctxforge.protocols.update_planner import IMemoryUpdatePlanner
from ctxforge.protocols.vectorstore import IVectorStore
from ctxforge.retrieval.aggregation_builder import AggregationBuilder
from ctxforge.retrieval.fast_path_retriever import FastPathRetriever
from ctxforge.retrieval.indexers.memory import MemoryIndexer

logger = logging.getLogger(__name__)


class CtxForge:
    """
    Context Engine - Framework for LLM context management.
    
    This engine manages the full lifecycle of context for LLM agents:
    
    DOES:
    ✅ Session management (load, save, versioning, TTL)
    ✅ Memory storage and retrieval (semantic, hybrid, temporal)
    ✅ Context assembly (prompt building with sections)
    ✅ Compaction (summarization, pruning)
    ✅ Memory extraction from conversations
    ✅ Middleware pipelines (PII redaction, validation, rate limiting, audit, etc.)
    
    DOES NOT:
    ❌ LLM inference (bring your own LLM)
    ❌ Agent loops (use LangGraph, LangChain, etc.)
    ❌ Tool execution (agent's responsibility)
    
    Usage Pattern:
        # 1. Prepare context (before LLM call)
        context = await engine.prepare_context(
            session_id="sess_123",
            user_id="user_456",
            user_input="Hello!",
        )
        
        # 2. Use with any LLM
        messages = context.to_openai_messages()
        response = await my_llm.chat(messages)
        
        # 3. Record the turn (after LLM call)
        await engine.record_turn(
            session_id="sess_123",
            user_id="user_456",
            user_input="Hello!",
            assistant_response=response.content,
        )
    """
    
    def __init__(
        self,
        config: EngineConfig,
        session_store: ISessionStore,
        memory_store: IMemoryStore,
        retriever: Optional[IRetriever] = None,
        condenser: Optional[ICondenser] = None,
        assembler: Optional[IContextAssembler] = None,
        extractor: Optional[IMemoryExtractor] = None,
        prepare_chain: Optional[MiddlewareChain] = None,
        record_chain: Optional[MiddlewareChain] = None,
        memory_indexer: Optional[MemoryIndexer] = None,
        vector_store: Optional[IVectorStore] = None,
        expertise_indexer: Optional[Any] = None,
        expertise_vector_store: Optional[IVectorStore] = None,
        # Expertise components
        expertise_store: Optional[IExpertiseStore] = None,
        expertise_retriever: Optional[IExpertiseRetriever] = None,
        reflector: Optional[IReflector] = None,
        curator: Optional[ICurator] = None,
        update_planner: Optional[IMemoryUpdatePlanner] = None,
        graph_store: Optional[IGraphStore] = None,
        graph_extractor: Optional[IGraphExtractor] = None,
        graph_ontology: Optional[GraphOntology] = None,
        graph_embedding_provider: Optional[IEmbeddingProvider] = None,
        tokenizer_provider: Optional[ITokenizerProvider] = None,
        graph_contradiction_detector: Optional[IGraphContradictionDetector] = None,
        graph_edge_temporal_extractor: Optional[IGraphEdgeTemporalExtractor] = None,
        graph_community_builder: Optional[CommunityBuilder] = None,
        graph_entity_linker: Optional[Any] = None,
        graph_service: Optional[GraphService] = None,
        memory_update_service: Optional[MemoryUpdateService] = None,
        retrieval_controller_service: Optional[RetrievalControllerService] = None,
        # Dynamic context services
        validated_knowledge_service: Optional[Any] = None,
        semantic_model_service: Optional[Any] = None,
        snapshot_service: Optional[Any] = None,
        unified_retriever: Optional[Any] = None,
        approval_store: Optional[Any] = None,
        # Hierarchical memory & skills services
        scoped_memory_service: Optional[Any] = None,
        skill_service: Optional[Any] = None,
        skill_lifecycle_service: Optional[Any] = None,
        owned_resources: Optional[List[Any]] = None,
        # Entropy gate for pre-extraction filtering
        entropy_gate: Optional[EntropyGate] = None,
        # Fast-path retriever for O(1) cache lookups
        fast_path_retriever: Optional[FastPathRetriever] = None,
        # Memory integration pipeline
        integration_pipeline: Optional[MemoryIntegrationPipeline] = None,
        # Memory synthesis service
        memory_synthesizer: Optional[MemorySynthesizerService] = None,
        # Personalization metrics service
        personalization_metrics_service: Optional[PersonalizationMetricsService] = None,
    ):
        """
        Initialize the Context Engine with injected dependencies.
        
        Args:
            config: Engine configuration
            session_store: Session storage implementation
            memory_store: Memory storage implementation
            retriever: Optional memory retriever (uses basic search if not provided)
            condenser: Optional context condenser
            extractor: Optional memory extractor for background extraction
            prepare_chain: Optional middleware chain executed during prepare_context
            record_chain: Optional middleware chain executed during record_turn
            expertise_store: Optional expertise storage for persistence
            expertise_retriever: Optional retriever for expertise items
            reflector: Optional reflector for turn analysis
            curator: Optional curator for expertise evolution
        """
        self.config = config
        self._session_store = session_store
        self._memory_store = memory_store
        self._retriever = retriever
        self._condenser = condenser
        self._assembler = assembler
        self._extractor = extractor
        self._prepare_chain = prepare_chain
        self._record_chain = record_chain
        self._memory_indexer = memory_indexer
        self._vector_store = vector_store
        self._expertise_indexer = expertise_indexer
        self._expertise_vector_store = expertise_vector_store
        
        # Expertise components
        self._expertise_store = expertise_store
        self._expertise_retriever = expertise_retriever
        self._reflector = reflector
        self._curator = curator
        self._update_planner = update_planner

        # Memory update planning service (owned dependency).
        self._memory_update_service: Optional[MemoryUpdateService] = memory_update_service
        if self._memory_update_service is None and self._update_planner is not None:
            self._memory_update_service = MemoryUpdateService(
                config=config,
                memory_store=self._memory_store,
                update_planner=self._update_planner,
                memory_indexer=self._memory_indexer,
            )

        # Memory service (owned dependency).
        self._memory_service: MemoryService = MemoryService(
            config=self.config,
            memory_store=self._memory_store,
            memory_indexer=self._memory_indexer,
            memory_retriever_provider=lambda: self._retriever,
        )

        # Background task tracking (used for async extraction/compaction/graph ingestion).
        self._background_tasks: set = set()

        # Session service (owned dependency).
        self._session_service: SessionService = SessionService(session_store=self._session_store)

        # Note: memory retrieval is owned by MemoryService (not a separate retrieval service).

        # Compaction service (owned dependency).
        self._compaction_service: Optional[CompactionService] = None
        if self._condenser is not None:
            self._compaction_service = CompactionService(
                config=self.config,
                condenser=self._condenser,
                session_service=self._session_service,
            )

        # Assembly service (owned dependency).
        self._assembly_service: AssemblyService = AssemblyService(
            config=self.config,
            assembler_provider=lambda: self._assembler,
            set_assembler=lambda a: setattr(self, "_assembler", a),
        )

        # Turn recording service (owned dependency).
        self._turn_recording_service: TurnRecordingService = TurnRecordingService(
            config=self.config,
            session_service=self._session_service,
            record_chain_provider=lambda: self._record_chain,
            run_chain=self._run_chain,
            background_tasks=self._background_tasks,
            extraction_enabled_provider=lambda: self._extractor is not None,
            run_extraction=self._run_extraction,
            compaction_service_provider=lambda: self._compaction_service,
            graph_service_provider=lambda: self._graph_service,
        )

        # Expertise service (owned dependency).
        self._expertise_service: ExpertiseService = ExpertiseService(
            config=self.config,
            expertise_store_provider=lambda: self._expertise_store,
            expertise_retriever_provider=lambda: self._expertise_retriever,
            reflector_provider=lambda: self._reflector,
            curator_provider=lambda: self._curator,
            memory_service_provider=lambda: self._memory_service,
            record_turn=self.record_turn,
            prepare_context=self.prepare_context,
        )

        # Graph subsystem (owned by GraphService).
        #
        # - If a fully-constructed service is provided, we use it as-is.
        # - Otherwise, if raw graph dependencies are provided, we construct GraphService here.
        self._graph_service: Optional[GraphService] = graph_service
        if self._graph_service is None:
            if graph_store is not None:
                # Allow graph retrieval even when ingestion is not wired (extractor/ontology missing).
                self._graph_service = GraphService(
                    config=config,
                    graph_store=graph_store,
                    graph_extractor=graph_extractor,
                    graph_ontology=graph_ontology,
                    embedding_provider=graph_embedding_provider,
                    tokenizer_provider=tokenizer_provider,
                    contradiction_detector=graph_contradiction_detector,
                    temporal_extractor=graph_edge_temporal_extractor,
                    community_builder=graph_community_builder,
                    entity_linker=graph_entity_linker,
                    background_tasks=self._background_tasks,
                )

        # Retrieval controller (optional; owned dependency).
        self._retrieval_controller_service: Optional[RetrievalControllerService] = retrieval_controller_service
        if self._retrieval_controller_service is None and getattr(self.config, "retrieval_controller", None) is not None:
            self._retrieval_controller_service = RetrievalControllerService(
                config=self.config,
                memory_service=self._memory_service,
                graph_service=self._graph_service,
                expertise_service=self._expertise_service,
                assembly_service=self._assembly_service,
                fast_path_retriever=fast_path_retriever,
            )
        
        # Dynamic context services
        self._validated_knowledge_service = validated_knowledge_service
        self._semantic_model_service = semantic_model_service
        self._snapshot_service = snapshot_service
        self._unified_retriever = unified_retriever
        self._approval_store = approval_store
        
        # Hierarchical memory & skills services
        self._scoped_memory_service = scoped_memory_service
        self._skill_service = skill_service
        self._skill_lifecycle = skill_lifecycle_service

        # Entropy gate for pre-extraction filtering
        self._entropy_gate: Optional[EntropyGate] = entropy_gate

        # Fast-path retriever for O(1) cache lookups
        self._fast_path_retriever: Optional[FastPathRetriever] = fast_path_retriever
        self._fast_path_index_loaded = False  # lazy-load flag
        # Keep a reference to graph_store for index persistence
        self._graph_store: Optional[IGraphStore] = graph_store
        self._graph_ontology: Optional[GraphOntology] = graph_ontology

        # Memory integration pipeline
        self._integration_pipeline: Optional[MemoryIntegrationPipeline] = integration_pipeline

        # Memory synthesis service
        self._memory_synthesizer: Optional[MemorySynthesizerService] = memory_synthesizer

        # Personalization metrics service
        self._personalization_metrics_service: Optional[PersonalizationMetricsService] = (
            personalization_metrics_service
        )

        # Factory-owned resources that should be closed when the engine closes.
        self._owned_resources: List[Any] = list(owned_resources or [])
        
        logger.info(f"ctxforge initialized: {config.name} v{config.version}")
    
    @property
    def session_store(self) -> ISessionStore:
        """Get the session store."""
        return self._session_store
    
    @property
    def memory_store(self) -> IMemoryStore:
        """Get the memory store."""
        return self._memory_store

    @property
    def graph_service(self) -> Optional[GraphService]:
        """Get the graph service, if configured."""
        return self._graph_service
    
    # =========================================================================
    # Fast-path index lazy-load
    # =========================================================================

    async def _maybe_load_fast_path_index(self, scope_id: str) -> None:
        """Lazy-load the enhanced memory index from the graph store on first call."""
        if self._fast_path_index_loaded:
            return
        self._fast_path_index_loaded = True
        if self._fast_path_retriever is None or self._graph_store is None:
            return
        try:
            index = await self._graph_store.load_enhanced_index(scope_id)
            if index is not None:
                self._fast_path_retriever.set_enhanced_index(index)
                logger.debug("Fast-path enhanced index loaded for scope=%s", scope_id)
        except Exception as exc:
            logger.warning("Failed to load fast-path enhanced index: %s", exc)

    # =========================================================================
    # CONTEXT PREPARATION (Before LLM call)
    # =========================================================================

    async def prepare_context(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        system_instructions: Optional[str] = None,
        include_memories: bool = True,
        include_history: bool = True,
        include_graph: bool = True,
        max_history_events: Optional[int] = None,
        max_memories: Optional[int] = None,
        *,
        # --- expertise (opt-in) ---
        expertise_id: Optional[str] = None,
        max_expertise_items: int = 5,
        # --- controller (opt-in, requires llm) ---
        use_controller: bool = False,
        llm: Optional[ILLMProvider] = None,
        model: Optional[str] = None,
        # --- session return (opt-in) ---
        return_session: bool = False,
        # --- two-step fusion inputs (opt-in) ---
        return_two_step_inputs: bool = False,
        **options,
    ) -> Union[Context, Tuple[Context, Session], TwoStepInputs]:
        """
        Prepare context for an LLM call.

        This is the single entry point for all context preparation. Optional
        keyword arguments enable additional features composably:

        - ``expertise_id``: retrieve and attach expertise items to the context.
        - ``use_controller=True`` + ``llm``: use the iterative retrieval controller.
        - ``return_session=True``: return ``(Context, Session)`` instead of ``Context``.
        - ``return_two_step_inputs=True``: return ``TwoStepInputs`` for the two-step
          fusion workflow.

        Args:
            session_id: The session identifier
            user_id: The user identifier
            user_input: The user's input message
            system_instructions: Optional system prompt (uses config default if not provided)
            include_memories: Whether to retrieve and include memories
            include_history: Whether to include conversation history
            include_graph: Whether to include graph data
            max_history_events: Max history events (uses config default if not provided)
            max_memories: Max memories to retrieve (uses config default if not provided)
            expertise_id: Optional expertise ID to retrieve items from
            max_expertise_items: Maximum expertise items to include (default 5)
            use_controller: Use the iterative retrieval controller (requires ``llm``)
            llm: LLM provider (required when ``use_controller=True``)
            model: Optional model override for the controller
            return_session: If True, return ``(Context, Session)``
            return_two_step_inputs: If True, return ``TwoStepInputs``
            **options: Additional options

        Returns:
            Context, (Context, Session), or TwoStepInputs depending on flags.

        Example:
            context = await engine.prepare_context(
                session_id="sess_123",
                user_id="user_456",
                user_input="What restaurants did I like?",
            )

            # Use with OpenAI
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=context.to_openai_messages(),
            )

            # With expertise
            context = await engine.prepare_context(
                session_id="sess_123",
                user_id="user_456",
                user_input="Hello!",
                expertise_id="my-expertise",
            )

            # With controller
            context = await engine.prepare_context(
                session_id="sess_123",
                user_id="user_456",
                user_input="Hello!",
                use_controller=True,
                llm=my_llm,
            )
        """
        # --- two-step fusion inputs mode ---
        if return_two_step_inputs:
            return await self._prepare_two_step_inputs(
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
                system_instructions=system_instructions,
                include_memories=include_memories,
                include_history=include_history,
                max_history_events=max_history_events,
                max_memories=max_memories,
            )

        # --- controller mode ---
        if use_controller:
            if llm is None:
                raise ValueError("use_controller=True requires llm to be provided.")
            context = await self._prepare_context_with_controller(
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
                llm=llm,
                system_instructions=system_instructions,
                include_history=include_history,
                max_history_events=max_history_events,
                expertise_id=expertise_id,
                model=model,
            )
            if return_session:
                session = await self._session_service.fetch(session_id=session_id, user_id=user_id)
                return context, session
            return context

        # --- standard path ---
        context = await self._prepare_context_core(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            system_instructions=system_instructions,
            include_memories=include_memories,
            include_history=include_history,
            include_graph=include_graph,
            max_history_events=max_history_events,
            max_memories=max_memories,
        )

        # --- expertise enrichment ---
        if expertise_id is not None:
            expertise_items = await self._expertise_service.retrieve_expertise_items(
                expertise_id=expertise_id,
                query=user_input,
                limit=max_expertise_items,
            )
            context.expertise_items = expertise_items
            context.expertise_id = expertise_id
            context.expertise_items_used = [item.item_id for item in expertise_items]
            context.metadata["expertise_id"] = expertise_id
            context.metadata["expertise_item_count"] = len(expertise_items)

        # --- session return ---
        if return_session:
            session = await self._session_service.fetch(session_id=session_id, user_id=user_id)
            return context, session

        return context

    async def _prepare_context_core(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        system_instructions: Optional[str] = None,
        include_memories: bool = True,
        include_history: bool = True,
        include_graph: bool = True,
        max_history_events: Optional[int] = None,
        max_memories: Optional[int] = None,
    ) -> Context:
        """Core context preparation logic (no expertise, no controller)."""
        start_time = time.time()

        mw_ctx = MiddlewareContext(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
        )
        mw_ctx.phase = "prepare_input"
        mw_ctx = await self._run_chain(self._prepare_chain, mw_ctx)

        session = await self._session_service.fetch(session_id=session_id, user_id=user_id)
        mw_ctx.session = session
        mw_ctx.phase = "prepare_session_loaded"
        mw_ctx = await self._run_chain(self._prepare_chain, mw_ctx)

        processed_input = mw_ctx.processed_input or user_input

        await self._maybe_load_fast_path_index(scope_id=user_id)

        fast_path_memories: List[MemoryItem] = []
        if self._fast_path_retriever is not None and include_memories:
            result = self._fast_path_retriever.try_fast_path(processed_input)
            if result.hit:
                fast_path_memories = result.memories
                logger.debug(
                    "Fast-path hit (%s): skipping full retrieval",
                    result.query_type,
                )

        memories: List[MemoryItem] = []
        if fast_path_memories:
            memories = fast_path_memories
        elif include_memories:
            limit = max_memories or self.config.retrieval.default_limit
            memories = await self._memory_service.search(
                user_id=user_id,
                query=processed_input,
                limit=limit,
            )

        # Record retrieval metrics
        if self._personalization_metrics_service is not None and memories:
            self._personalization_metrics_service.record_retrieval(
                session_id=session_id,
                user_id=user_id,
                memories=memories,
            )

        # Memory synthesis: consolidate retrieved memories into narrative
        synthesized_memory_context: Optional[str] = None
        if self._memory_synthesizer is not None and memories:
            synthesis_min = self.config.extraction.synthesis_min_memories
            if len(memories) >= synthesis_min:
                synthesized_memory_context = await self._memory_synthesizer.synthesize(
                    memories=memories,
                    query=processed_input,
                    max_tokens=self.config.extraction.synthesis_max_tokens,
                )

        graph_section = None
        if include_graph and getattr(self.config, "graph", None) is not None and getattr(self.config.graph, "enabled", False):
            if self._graph_service is not None:
                graph_section = await self._graph_service.build_section(user_id=user_id, query=processed_input)

        if self._compaction_service is not None:
            await self._compaction_service.maybe_compact(session=session)

        mw_ctx.metadata["memories"] = memories
        mw_ctx.phase = "prepare_retrieval"
        mw_ctx = await self._run_chain(self._prepare_chain, mw_ctx)

        system_prompt = system_instructions or self.config.prompts.system_template
        history_limit = max_history_events or self.config.prompts.max_history_events

        mw_ctx.phase = "prepare_assembly"
        mw_ctx = await self._run_chain(self._prepare_chain, mw_ctx)

        context = await self._assembly_service.assemble(
            session=session,
            current_query=processed_input,
            memories=memories,
            system_instructions=system_prompt,
            token_budget=self.config.compaction.token_threshold,
            include_history=include_history,
            max_history_events=history_limit,
            graph_section=graph_section,
            synthesized_memory_context=synthesized_memory_context,
        )

        # Inject middleware-contributed context sections
        for section in mw_ctx.context_sections:
            context.add_section(
                name=section.name,
                content=section.content,
                priority=section.priority,
                is_required=section.is_required,
                token_estimate=section.token_estimate,
            )

        elapsed_ms = (time.time() - start_time) * 1000
        context.metadata["preparation_time_ms"] = elapsed_ms
        context.metadata["memory_count"] = len(memories)
        context.metadata["history_event_count"] = len(context.events)
        context.metadata["graph_enabled"] = bool(getattr(self.config.graph, "enabled", False)) if getattr(self.config, "graph", None) is not None else False

        logger.debug(
            f"Context prepared in {elapsed_ms:.2f}ms "
            f"(memories={len(memories)}, history={len(context.events)})"
        )

        mw_ctx.metadata["context"] = context
        mw_ctx.phase = "prepare_done"
        await self._run_chain(self._prepare_chain, mw_ctx)

        return context

    async def _prepare_two_step_inputs(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        system_instructions: Optional[str] = None,
        include_memories: bool = True,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        max_memories: Optional[int] = None,
    ) -> TwoStepInputs:
        """Internal: prepare inputs for the two-step fusion workflow."""
        kg_section = None
        if getattr(self.config, "graph", None) is not None and getattr(self.config.graph, "enabled", False):
            if self._graph_service is not None:
                kg_section = await self._graph_service.build_section(user_id=user_id, query=user_input)

        mem_ctx = await self._prepare_context_core(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            system_instructions=system_instructions,
            include_memories=include_memories,
            include_history=include_history,
            include_graph=not bool(kg_section),
            max_history_events=max_history_events,
            max_memories=max_memories,
        )
        mem_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in mem_ctx.to_messages()]

        return TwoStepInputs(
            kg_section=kg_section,
            memory_context=mem_ctx,
            memory_messages=mem_messages,
        )

    async def _prepare_context_with_controller(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        llm: ILLMProvider,
        system_instructions: Optional[str] = None,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        expertise_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Context:
        """Internal: prepare context using the iterative retrieval controller."""
        if self._retrieval_controller_service is None or not getattr(self.config.retrieval_controller, "enabled", False):
            raise ValueError(
                "Retrieval controller is disabled. Enable config.retrieval_controller.enabled to use it."
            )

        mw_ctx = MiddlewareContext(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
        )
        mw_ctx.phase = "prepare_input"
        mw_ctx = await self._run_chain(self._prepare_chain, mw_ctx)

        session = await self._session_service.fetch(session_id=session_id, user_id=user_id)
        mw_ctx.session = session
        mw_ctx.phase = "prepare_session_loaded"
        mw_ctx = await self._run_chain(self._prepare_chain, mw_ctx)

        processed_input = mw_ctx.processed_input or user_input

        result: RetrievalControllerResult = await self._retrieval_controller_service.prepare_context(
            session=session,
            user_id=user_id,
            question=processed_input,
            llm=llm,
            system_instructions=system_instructions,
            include_history=include_history,
            max_history_events=max_history_events,
            expertise_id=expertise_id,
            model=model,
        )

        # Inject middleware-contributed context sections
        for section in mw_ctx.context_sections:
            result.context.add_section(
                name=section.name,
                content=section.content,
                priority=section.priority,
                is_required=section.is_required,
                token_estimate=section.token_estimate,
            )

        mw_ctx.metadata["context"] = result.context
        mw_ctx.metadata["retrieval_controller_result"] = {
            "iterations": [it.__dict__ for it in result.iterations],
            "evidence": result.evidence,
            "gaps": result.gaps,
        }
        mw_ctx.phase = "prepare_done"
        await self._run_chain(self._prepare_chain, mw_ctx)

        return result.context

    # =========================================================================
    # DEPRECATED PREPARE VARIANTS (backward compatibility)
    # =========================================================================

    async def prepare_two_step_inputs(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        system_instructions: Optional[str] = None,
        include_memories: bool = True,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        max_memories: Optional[int] = None,
    ) -> TwoStepInputs:
        """
        .. deprecated::
            Use ``prepare_context(return_two_step_inputs=True)`` instead.
        """
        warnings.warn(
            "prepare_two_step_inputs() is deprecated. "
            "Use prepare_context(return_two_step_inputs=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.prepare_context(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            system_instructions=system_instructions,
            include_memories=include_memories,
            include_history=include_history,
            max_history_events=max_history_events,
            max_memories=max_memories,
            return_two_step_inputs=True,
        )

    async def prepare_context_with_controller(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        llm: ILLMProvider,
        system_instructions: Optional[str] = None,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        expertise_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Context:
        """
        .. deprecated::
            Use ``prepare_context(use_controller=True, llm=...)`` instead.
        """
        warnings.warn(
            "prepare_context_with_controller() is deprecated. "
            "Use prepare_context(use_controller=True, llm=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.prepare_context(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            system_instructions=system_instructions,
            include_history=include_history,
            max_history_events=max_history_events,
            use_controller=True,
            llm=llm,
            expertise_id=expertise_id,
            model=model,
        )

    async def prepare_context_with_session(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        **kwargs,
    ) -> Tuple[Context, Session]:
        """
        .. deprecated::
            Use ``prepare_context(return_session=True)`` instead.
        """
        warnings.warn(
            "prepare_context_with_session() is deprecated. "
            "Use prepare_context(return_session=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.prepare_context(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            return_session=True,
            **kwargs,
        )

    async def answer_two_step(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        llm: ILLMProvider,
        system_instructions: Optional[str] = None,
        include_memories: bool = True,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        max_memories: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Convenience method: run a two-step answer workflow and return the synthesized answer.

        .. deprecated::
            Use ``ctxforge.helpers.answer_two_step(engine, llm, ...)`` instead.
        """
        warnings.warn(
            "answer_two_step() is deprecated. "
            "Use ctxforge.helpers.answer_two_step(engine, llm, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ctxforge.helpers import answer_two_step
        return await answer_two_step(
            engine=self,
            llm=llm,
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            system_instructions=system_instructions,
            include_memories=include_memories,
            include_history=include_history,
            max_history_events=max_history_events,
            max_memories=max_memories,
            model=model,
        )

    async def answer_with_controller(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        llm: ILLMProvider,
        system_instructions: Optional[str] = None,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        expertise_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Convenience method: run the iterative retrieval controller and return the final answer.

        .. deprecated::
            Use ``ctxforge.helpers.answer_with_controller(engine, llm, ...)`` instead.
        """
        warnings.warn(
            "answer_with_controller() is deprecated. "
            "Use ctxforge.helpers.answer_with_controller(engine, llm, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ctxforge.helpers import answer_with_controller
        return await answer_with_controller(
            engine=self,
            llm=llm,
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            system_instructions=system_instructions,
            include_history=include_history,
            max_history_events=max_history_events,
            expertise_id=expertise_id,
            model=model,
        )
    
    # =========================================================================
    # TURN RECORDING (After LLM call)
    # =========================================================================

    async def record_turn(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        pipeline_metadata: Optional[Dict[str, Any]] = None,
        *,
        # --- expertise feedback (opt-in) ---
        expertise_items_used: Optional[List[str]] = None,
        outcome: Optional[TurnOutcome] = None,
        expertise_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Optional[Expertise]:
        """
        Record a completed conversation turn.

        Call this AFTER your agent gets the LLM response. This will:
        1. Add user and assistant events to the session
        2. Save the session
        3. Trigger background memory extraction (if configured)
        4. Trigger background compaction (if needed)
        5. (Optional) Run expertise reflection/curation when ``expertise_items_used``
           and ``outcome`` are provided.

        Args:
            session_id: The session identifier
            user_id: The user identifier
            user_input: The user's input (should match what was passed to prepare_context)
            assistant_response: The LLM's response
            metadata: Optional metadata to attach to events
            pipeline_metadata: Optional metadata passed through the record middleware
            expertise_items_used: IDs of expertise items used (enables feedback)
            outcome: The turn outcome (enables feedback)
            expertise_id: Optional expertise ID for curation
            ground_truth: Optional ground truth for comparison

        Returns:
            Updated Expertise if feedback was processed, None otherwise.

        Example:
            # Basic recording
            await engine.record_turn(
                session_id="sess_123",
                user_id="user_456",
                user_input="Hello!",
                assistant_response=response.content,
            )

            # With expertise feedback
            updated = await engine.record_turn(
                session_id="sess_123",
                user_id="user_456",
                user_input="Hello!",
                assistant_response=response.content,
                expertise_items_used=["item-1", "item-2"],
                outcome=TurnOutcome.SUCCESS,
                expertise_id="my-expertise",
            )
        """
        await self._turn_recording_service.record_turn(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata,
            pipeline_metadata=pipeline_metadata,
        )

        if expertise_items_used is not None and outcome is not None:
            return await self._expertise_service.reflect_and_curate(
                session_id=session_id,
                user_input=user_input,
                assistant_response=assistant_response,
                expertise_items_used=expertise_items_used,
                outcome=outcome,
                expertise_id=expertise_id,
                ground_truth=ground_truth,
            )

        return None
    
    async def record_user_message(
        self,
        session_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Record only a user message (without assistant response).
        
        Useful for streaming scenarios or multi-step interactions.
        """
        return await self._turn_recording_service.record_user_message(
            session_id=session_id,
            user_id=user_id,
            content=content,
            metadata=metadata,
        )
    
    async def record_assistant_message(
        self,
        session_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Record only an assistant message.
        
        Useful for streaming scenarios or multi-step interactions.
        """
        return await self._turn_recording_service.record_assistant_message(
            session_id=session_id,
            user_id=user_id,
            content=content,
            metadata=metadata,
        )
    
    async def record_tool_use(
        self,
        session_id: str,
        user_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: str,
        tool_call_id: Optional[str] = None,
    ) -> Tuple[Event, Event]:
        """
        Record a tool call and its result.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            tool_name: Name of the tool
            tool_input: Tool input arguments
            tool_output: Tool output/result
            tool_call_id: Optional tool call ID for correlation
            
        Returns:
            Tuple of (tool_call_event, tool_output_event)
        """
        return await self._turn_recording_service.record_tool_use(
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            tool_call_id=tool_call_id,
        )
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    async def get_session(self, session_id: str, user_id: str) -> Session:
        """Get a session by ID."""
        return await self._session_service.fetch(session_id=session_id, user_id=user_id)
    
    async def update_session_state(
        self,
        session_id: str,
        user_id: str,
        **data,
    ) -> None:
        """Update session state data."""
        session = await self._session_service.fetch(session_id=session_id, user_id=user_id)
        session.state.update(data)
        await self._session_service.save(session)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return await self._session_service.delete(session_id=session_id)
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Session]:
        """List sessions for a user."""
        return await self._session_service.list(user_id=user_id, limit=limit)
    
    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================
    
    async def add_memory(self, memory: MemoryItem) -> str:
        """
        Add a memory for a user.
        
        Args:
            memory: The memory to add
            
        Returns:
            The memory ID
        """
        return await self._memory_service.add(memory)
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            The memory if found, None otherwise
        """
        return await self._memory_service.get(memory_id)
    
    async def update_memory(self, memory: MemoryItem) -> bool:
        """
        Update an existing memory.
        
        Use this to:
        - Correct extracted facts
        - Update confidence scores
        - Add/remove tags
        - Modify metadata
        
        Args:
            memory: The memory with updated fields
            
        Returns:
            True if the memory was updated, False if not found
        """
        return await self._memory_service.update(memory)
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """
        Search memories for a user.
        
        Args:
            user_id: The user ID
            query: The search query
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        return await self._memory_service.search(user_id=user_id, query=query, limit=limit)

    def get_session_metrics(self, session_id: str) -> Optional[Any]:
        """Get personalization metrics for a session."""
        if self._personalization_metrics_service is None:
            return None
        return self._personalization_metrics_service.get_session_metrics(session_id)

    def get_user_metrics(self, user_id: str) -> Optional[List[Any]]:
        """Get personalization metrics across sessions for a user."""
        if self._personalization_metrics_service is None:
            return None
        return self._personalization_metrics_service.get_user_metrics(user_id)

    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """Get all memories for a user."""
        return await self._memory_service.get_by_user(user_id=user_id, limit=limit)

    async def search_memories_by_query(
        self,
        query: MemoryQuery,
    ) -> List[MemoryItem]:
        """
        Search memories using a full MemoryQuery.

        Supports filtering by tags, types, confidence, active status, etc.
        Goes directly to the store (bypasses retriever and global scopes).

        Example::

            from ctxforge.core.memory import MemoryQuery

            results = await engine.search_memories_by_query(
                MemoryQuery(
                    user_id="incident-123",
                    tags=["__suggested"],
                    min_confidence=0.3,
                    limit=100,
                )
            )
        """
        return await self._memory_service.search_by_query(query)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory (hard delete)."""
        return await self._memory_service.delete(memory_id)

    async def delete_all_user_memories(
        self,
        user_id: str,
        *,
        include_inactive: bool = True,
        batch_size: int = 1000,
    ) -> int:
        """
        Delete all memories for a user (hard delete).

        This is useful for demos/tests or when you want to fully reset a user's memory state.
        If a vector index is configured, this also clears the user's index namespace.

        Args:
            user_id: The user ID
            include_inactive: Whether to include deactivated memories
            batch_size: Fetch/delete batch size

        Returns:
            Number of deleted memories
        """
        return await self._memory_service.delete_all_user_memories(
            user_id=user_id,
            include_inactive=include_inactive,
            batch_size=batch_size,
        )
    
    async def deactivate_memory(self, memory_id: str) -> bool:
        """
        Deactivate a memory (soft delete).
        
        The memory is kept but excluded from searches.
        Use this instead of delete when you want to preserve history.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            True if the memory was deactivated, False if not found
        """
        return await self._memory_service.deactivate(memory_id)
    
    # =========================================================================
    # EXPERTISE MANAGEMENT
    # =========================================================================
    
    @property
    def expertise_store(self) -> Optional[IExpertiseStore]:
        """Get the expertise store."""
        return self._expertise_store
    
    @property
    def expertise_retriever(self) -> Optional[IExpertiseRetriever]:
        """Get the expertise retriever."""
        return self._expertise_retriever

    @property
    def memory_indexer(self) -> Optional[MemoryIndexer]:
        """Get the memory indexer (vectorstore-backed) if configured."""
        return self._memory_indexer

    @property
    def expertise_indexer(self):
        """Get the expertise indexer (vectorstore-backed) if configured."""
        return self._expertise_indexer
    
    async def create_expertise(
        self,
        expertise_id: str,
        name: str,
        domain: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Expertise:
        """
        Create a new expertise.
        
        Args:
            expertise_id: Unique identifier for the expertise
            name: Display name for the expertise
            domain: Optional domain/category
            description: Optional description
            
        Returns:
            The created Expertise object
        """
        return await self._expertise_service.create_expertise(
            expertise_id=expertise_id,
            name=name,
            domain=domain,
            description=description,
        )
    
    async def load_expertise(self, expertise_id: str) -> Optional[Expertise]:
        """
        Load an expertise by ID.
        
        Args:
            expertise_id: The expertise ID
            
        Returns:
            The Expertise if found, None otherwise
        """
        return await self._expertise_service.load_expertise(expertise_id=expertise_id)
    
    async def save_expertise(self, expertise: Expertise) -> None:
        """
        Save an expertise.
        
        Args:
            expertise: The expertise to save
        """
        await self._expertise_service.save_expertise(expertise=expertise)
    
    async def add_expertise_item(
        self,
        expertise_id: str,
        section: ExpertiseSection,
        content: str,
        source: Optional[str] = None,
    ) -> Optional[ExpertiseItem]:
        """
        Add an item to an expertise.
        
        Args:
            expertise_id: The expertise ID
            section: The section for the item
            content: The item content
            source: Optional source identifier
            
        Returns:
            The created ExpertiseItem, or None if expertise not found
        """
        return await self._expertise_service.add_expertise_item(
            expertise_id=expertise_id,
            section=section,
            content=content,
            source=source,
        )
    
    async def retrieve_expertise_items(
        self,
        expertise_id: str,
        query: str,
        limit: int = 5,
    ) -> List[ExpertiseItem]:
        """
        Retrieve relevant expertise items for a query.
        
        Args:
            expertise_id: The expertise ID
            query: The query to match against
            limit: Maximum items to retrieve
            
        Returns:
            List of relevant ExpertiseItem objects
        """
        return await self._expertise_service.retrieve_expertise_items(
            expertise_id=expertise_id,
            query=query,
            limit=limit,
        )
    
    async def record_turn_with_feedback(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        assistant_response: str,
        expertise_items_used: List[str],
        outcome: TurnOutcome,
        expertise_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Optional[Expertise]:
        """
        .. deprecated::
            Use ``record_turn(expertise_items_used=..., outcome=...)`` instead.
        """
        warnings.warn(
            "record_turn_with_feedback() is deprecated. "
            "Use record_turn(expertise_items_used=..., outcome=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.record_turn(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            assistant_response=assistant_response,
            expertise_items_used=expertise_items_used,
            outcome=outcome,
            expertise_id=expertise_id,
            ground_truth=ground_truth,
        )

    async def prepare_context_with_expertise(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        expertise_id: str,
        max_expertise_items: int = 5,
        **kwargs,
    ) -> Context:
        """
        .. deprecated::
            Use ``prepare_context(expertise_id=...)`` instead.
        """
        warnings.warn(
            "prepare_context_with_expertise() is deprecated. "
            "Use prepare_context(expertise_id=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.prepare_context(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            expertise_id=expertise_id,
            max_expertise_items=max_expertise_items,
            **kwargs,
        )
    
    # =========================================================================
    # MANUAL OPERATIONS
    # =========================================================================
    
    async def compact_session(self, session_id: str, user_id: str) -> CondensationResult:
        """
        Manually trigger session compaction.

        Args:
            session_id: The session to compact
            user_id: The user ID

        Returns:
            CondensationResult with details of what was done
        """
        if self._compaction_service is None:
            # Return an empty no-op result
            return CondensationResult(
                view=CompactionView(events=[], summary=None),
                summary_generated=False,
                tokens_saved=0,
                metadata={"error": "Compaction service not configured"},
            )

        return await self._compaction_service.compact_session(
            session_id=session_id, user_id=user_id
        )
    
    async def extract_memories_from_text(
        self,
        user_id: str,
        text: str,
        auto_save: bool = True,
    ) -> List[MemoryItem]:
        """
        Manually extract memories from text.
        
        Args:
            user_id: The user ID
            text: The text to extract from
            auto_save: Whether to automatically save extracted memories
            
        Returns:
            List of extracted memories
        """
        if not self._extractor:
            return []
        
        config = ExtractionConfig(
            extract_semantic=self.config.extraction.extract_semantic,
            extract_episodic=self.config.extraction.extract_episodic,
            min_confidence=self.config.extraction.min_confidence,
        )
        
        result = await self._extractor.extract_from_text(text, config)
        
        memories = []
        for candidate in result.filter_by_confidence(config.min_confidence):
            memory = candidate.to_memory_item(user_id)
            if auto_save:
                await self.add_memory(memory)
            memories.append(memory)
        
        return memories
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    async def _run_chain(
        self,
        chain: Optional[MiddlewareChain],
        context: MiddlewareContext,
    ) -> MiddlewareContext:
        """Run a middleware chain if configured."""
        if not chain:
            return context
        result = await chain.execute(context)
        # Even if stopped/failed, we return the best-effort context for now.
        return result.context
    
    
    async def _run_extraction(
        self,
        session: Session,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Run memory extraction in the background."""
        if not self._extractor:
            return

        # Entropy gate: skip extraction for low-novelty turns
        if self._entropy_gate is not None and self._entropy_gate.enabled:
            gate_result = await self._entropy_gate.evaluate(user_input, assistant_response)
            if not gate_result.should_extract:
                logger.debug(
                    f"Extraction skipped by entropy gate: {gate_result.reason}"
                    + (f" (similarity={gate_result.similarity_score:.3f})" if gate_result.similarity_score else "")
                )
                return
        
        try:
            config = ExtractionConfig(
                extract_semantic=self.config.extraction.extract_semantic,
                extract_episodic=self.config.extraction.extract_episodic,
                extract_procedural=self.config.extraction.extract_procedural,
                min_confidence=self.config.extraction.min_confidence,
                max_candidates=self.config.extraction.max_candidates,
            )
            
            result = await self._extractor.extract(
                user_input=user_input,
                agent_response=assistant_response,
                session=session,
                config=config,
            )

            filtered_candidates = list(result.filter_by_confidence(config.min_confidence))

            new_items: List[MemoryItem] = []

            if (
                self.config.extraction.integration_pipeline_enabled
                and self._integration_pipeline is not None
                and filtered_candidates
            ):
                # Multi-stage integration pipeline
                integration_results = await self._integration_pipeline.process(
                    candidates=filtered_candidates,
                    user_id=session.user_id,
                    query=user_input,
                )
                new_items = [
                    r.memory_item for r in integration_results if r.memory_item
                ]
                pref_changes = sum(
                    1 for r in integration_results if r.preference_changed
                )
                if self._personalization_metrics_service is not None:
                    self._personalization_metrics_service.record_extraction(
                        session_id=session.session_id,
                        extraction_count=len(new_items),
                        preference_changes=pref_changes,
                    )
            elif (
                self.config.extraction.update_planning_enabled
                and self._update_planner is not None
                and filtered_candidates
            ):
                new_items = [
                    c.to_memory_item(session.user_id)
                    for c in filtered_candidates
                ]
                if self._memory_update_service is not None:
                    await self._memory_update_service.plan_and_apply(
                        user_id=session.user_id,
                        query=user_input,
                        new_items=new_items,
                    )
                if self._personalization_metrics_service is not None:
                    self._personalization_metrics_service.record_extraction(
                        session_id=session.session_id,
                        extraction_count=len(new_items),
                    )
            else:
                new_items = [
                    c.to_memory_item(session.user_id)
                    for c in filtered_candidates
                ]
                for memory in new_items:
                    await self.add_memory(memory)
                if self._personalization_metrics_service is not None:
                    self._personalization_metrics_service.record_extraction(
                        session_id=session.session_id,
                        extraction_count=len(new_items),
                    )
            
            if result.count > 0:
                logger.debug(f"Extracted {result.count} memories")

            # Rebuild the fast-path enhanced index after new memories are stored.
            if new_items and self._fast_path_retriever is not None:
                await self._rebuild_enhanced_index(
                    user_id=session.user_id,
                    new_memories=new_items,
                )

        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")

    async def _rebuild_enhanced_index(
        self,
        user_id: str,
        new_memories: List[MemoryItem],
    ) -> None:
        """Rebuild the enhanced memory index from newly extracted memories."""
        try:
            builder = AggregationBuilder(ontology=self._graph_ontology)
            index = builder.build_aggregations(new_memories)

            # If a graph store is available, persist the index.
            if self._graph_store is not None:
                existing = await self._graph_store.load_enhanced_index(user_id)
                if existing is not None:
                    # Merge new entities/relations into the existing index.
                    for name, agg in index.entities.items():
                        if name in existing.entities:
                            ea = existing.entities[name]
                            for action, count in agg.event_counts.items():
                                ea.event_counts[action] = ea.event_counts.get(action, 0) + count
                            for attr_key, attr_vals in agg.attribute_sets.items():
                                if attr_key not in ea.attribute_sets:
                                    ea.attribute_sets[attr_key] = set()
                                ea.attribute_sets[attr_key].update(attr_vals)
                            ea.evidence_memory_ids.extend(agg.evidence_memory_ids)
                        else:
                            existing.entities[name] = agg
                    existing.relations.extend(index.relations)
                    for date_key, mem_ids in index.temporal_index.items():
                        if date_key not in existing.temporal_index:
                            existing.temporal_index[date_key] = []
                        existing.temporal_index[date_key].extend(mem_ids)
                    index = existing

                await self._graph_store.save_enhanced_index(user_id, index)

            self._fast_path_retriever.set_enhanced_index(index)
            logger.debug("Fast-path enhanced index rebuilt for user=%s", user_id)
        except Exception as exc:
            logger.warning("Failed to rebuild fast-path enhanced index: %s", exc)

    # =========================================================================
    # DYNAMIC CONTEXT (Validated Knowledge, Snapshots, Unified Retrieval)
    # =========================================================================
    
    @property
    def validated_knowledge_service(self):
        """Get the validated knowledge service."""
        return self._validated_knowledge_service
    
    @property
    def semantic_model_service(self):
        """Get the semantic model service."""
        return self._semantic_model_service
    
    @property
    def snapshot_service(self):
        """Get the expertise snapshot service."""
        return self._snapshot_service
    
    @property
    def unified_retriever(self):
        """Get the unified cross-store retriever."""
        return self._unified_retriever
    
    @property
    def approval_store(self):
        """Get the approval store for human-in-the-loop workflow."""
        return self._approval_store
    
    @property
    def scoped_memory_service(self):
        """Get the scoped memory service for hierarchical memories."""
        return self._scoped_memory_service
    
    @property
    def skill_service(self):
        """Get the skill service for workflow management."""
        return self._skill_service

    @property
    def skill_lifecycle_service(self):
        """Get the skill lifecycle service for end-to-end skill management."""
        return self._skill_lifecycle

    async def create_skill_from_github(
        self,
        github_url: str,
        project_id: str = "default",
        github_token: Optional[str] = None,
    ) -> Optional[Any]:
        """Generate, validate, evaluate, and persist a skill from a GitHub repo."""
        if not self._skill_lifecycle:
            raise RuntimeError("SkillLifecycleService not configured")
        return await self._skill_lifecycle.create_from_github(
            github_url, project_id, github_token,
        )

    async def create_skill_from_document(
        self,
        file_path: str,
        project_id: str = "default",
    ) -> Optional[Any]:
        """Generate, validate, evaluate, and persist a skill from a document."""
        if not self._skill_lifecycle:
            raise RuntimeError("SkillLifecycleService not configured")
        return await self._skill_lifecycle.create_from_document(
            file_path, project_id,
        )

    async def save_validated_knowledge(
        self,
        content: str,
        name: str,
        expertise_id: Optional[str] = None,
        user_id: Optional[str] = None,
        section: Optional[str] = None,
        source_question: Optional[str] = None,
        summary: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Save user-validated knowledge directly to the knowledge base.
        
        This bypasses the reflection/curation loop for knowledge that
        the user has explicitly approved.
        
        Args:
            content: The knowledge content to save
            name: Human-readable name for this knowledge
            expertise_id: Optional expertise ID (saves to expertise store)
            user_id: Optional user ID (saves to memory store)
            section: Optional expertise section
            source_question: The original question context
            summary: Brief summary
            notes: Additional notes/caveats
            tags: Optional tags
            
        Returns:
            The ID of the saved knowledge item, or None if not saved
        """
        if self._validated_knowledge_service is None:
            logger.warning("Validated knowledge service not configured")
            return None
        
        from ctxforge.engine.services.validated_knowledge_service import ValidatedKnowledgeEntry
        
        entry = ValidatedKnowledgeEntry(
            name=name,
            question=source_question or "",
            content=content,
            summary=summary,
            notes=notes,
            tags=tags or [],
        )
        
        return await self._validated_knowledge_service.save_validated_entry(
            entry=entry,
            expertise_id=expertise_id,
            user_id=user_id,
            section=section,
        )
    
    async def create_expertise_snapshot(
        self,
        expertise_id: str,
        version: str,
        created_by: Optional[str] = None,
        description: str = "",
    ):
        """
        Create a snapshot of the current expertise state.
        
        Snapshots enable version tracking and diffing for expertise domains.
        
        Args:
            expertise_id: The expertise to snapshot
            version: Version identifier (e.g., "1.0.0")
            created_by: Who created this snapshot
            description: Optional description
            
        Returns:
            The created snapshot, or None if snapshot service not configured
        """
        if self._snapshot_service is None:
            logger.warning("Snapshot service not configured")
            return None
        
        if self._expertise_store is None:
            logger.warning("Expertise store not configured")
            return None
        
        expertise = await self._expertise_store.load(expertise_id)
        if expertise is None:
            return None
        
        return await self._snapshot_service.create_snapshot(
            expertise=expertise,
            version=version,
            created_by=created_by,
            description=description,
        )
    
    async def compare_expertise_versions(
        self,
        expertise_id: str,
        from_version: str,
        to_version: str,
    ):
        """
        Compare two expertise versions and generate a diff.
        
        Args:
            expertise_id: The expertise ID
            from_version: Earlier version
            to_version: Later version
            
        Returns:
            ExpertiseDiff object, or None if versions not found
        """
        if self._snapshot_service is None:
            logger.warning("Snapshot service not configured")
            return None
        
        return await self._snapshot_service.diff_versions(
            expertise_id=expertise_id,
            from_version=from_version,
            to_version=to_version,
        )
    
    async def search_unified(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_results: int = 10,
        sources: Optional[List[str]] = None,
        knowledge_types: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Any]:
        """
        Search across all knowledge stores using unified retrieval.
        
        This provides a single search interface across expertise, memories,
        and other knowledge sources.
        
        Args:
            query: The search query
            user_id: Optional user ID for personalized results
            max_results: Maximum total results
            sources: Optional filter by source types
            knowledge_types: Optional filter by knowledge types
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        if self._unified_retriever is None:
            logger.warning("Unified retriever not configured")
            return []
        
        return await self._unified_retriever.search(
            query=query,
            user_id=user_id,
            max_results=max_results,
            sources=sources,
            knowledge_types=knowledge_types,
            min_score=min_score,
        )
    
    # =========================================================================
    # HIERARCHICAL MEMORY (Scoped Memories)
    # =========================================================================
    
    async def save_scoped_memory(
        self,
        scope: str,
        scope_id: str,
        key: str,
        content: str,
        category: str = "preference",
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save a scoped memory.
        
        Args:
            scope: Memory scope ("global", "project", "session")
            scope_id: The ID for the scope (user_id, project_id, session_id)
            key: Unique key within the scope
            content: The memory content
            category: Memory category (preference, convention, architecture, etc.)
            priority: Priority for ordering (higher = loaded first)
            metadata: Optional additional metadata
            
        Returns:
            The saved ScopedMemory, or None if service not configured
        """
        if self._scoped_memory_service is None:
            logger.warning("Scoped memory service not configured")
            return None
        
        from ctxforge.core.scoped_memory import MemoryCategory, MemoryScope
        
        scope_enum = MemoryScope(scope)
        category_enum = MemoryCategory(category)
        
        if scope_enum == MemoryScope.GLOBAL:
            return await self._scoped_memory_service.save_global(
                user_id=scope_id, key=key, content=content,
                category=category_enum, priority=priority, metadata=metadata,
            )
        elif scope_enum == MemoryScope.PROJECT:
            return await self._scoped_memory_service.save_project(
                project_id=scope_id, key=key, content=content,
                category=category_enum, priority=priority, metadata=metadata,
            )
        else:  # SESSION
            return await self._scoped_memory_service.save_session(
                session_id=scope_id, key=key, content=content,
                category=category_enum, priority=priority, metadata=metadata,
            )
    
    async def get_merged_memories(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ):
        """
        Get merged memories across scopes with hierarchical override.
        
        Session memories override project, which override global.
        
        Args:
            user_id: The user ID (for global scope)
            project_id: Optional project ID
            session_id: Optional session ID
            categories: Optional filter by categories
            
        Returns:
            MergedMemoryResult, or None if service not configured
        """
        if self._scoped_memory_service is None:
            logger.warning("Scoped memory service not configured")
            return None
        
        from ctxforge.core.scoped_memory import MemoryCategory
        
        cat_enums = None
        if categories:
            cat_enums = [MemoryCategory(c) for c in categories]
        
        return await self._scoped_memory_service.get_merged_memories(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            categories=cat_enums,
        )
    
    async def clear_session_memories(self, session_id: str) -> int:
        """
        Clear all memories for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Number of memories deleted, or 0 if service not configured
        """
        if self._scoped_memory_service is None:
            logger.warning("Scoped memory service not configured")
            return 0
        
        return await self._scoped_memory_service.clear_session(session_id)
    
    # =========================================================================
    # SKILLS MANAGEMENT
    # =========================================================================
    
    async def register_skill(
        self,
        name: str,
        description: str,
        content: str,
        scope: str = "base",
        scope_id: str = "system",
        triggers: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ):
        """
        Register a skill.
        
        Args:
            name: Skill name (lowercase, hyphens only)
            description: Brief description (max 256 chars)
            content: Full workflow content (markdown)
            scope: Skill scope ("base", "user", "project")
            scope_id: Scope identifier
            triggers: Keywords/patterns that activate the skill
            prerequisites: Other skills required
            allowed_tools: Tools this skill can use
            metadata: Additional metadata
            version: Skill version
            
        Returns:
            The registered Skill, or None if service not configured
        """
        if self._skill_service is None:
            logger.warning("Skill service not configured")
            return None
        
        from ctxforge.core.skill import Skill, SkillScope
        
        skill = Skill(
            name=name,
            description=description,
            scope=SkillScope(scope),
            scope_id=scope_id,
            content=content,
            triggers=triggers or [],
            prerequisites=prerequisites or [],
            allowed_tools=allowed_tools or [],
            metadata=metadata or {},
            version=version,
        )
        
        await self._skill_service.register_skill(skill)
        return skill
    
    async def get_available_skills(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Get available skills with scope layering.
        
        Returns skill metadata (not full content) for progressive disclosure.
        
        Args:
            user_id: Optional user ID for user-scoped skills
            project_id: Optional project ID for project-scoped skills
            
        Returns:
            List of SkillMetadata, or empty list if service not configured
        """
        if self._skill_service is None:
            logger.warning("Skill service not configured")
            return []
        
        return await self._skill_service.get_available_skills(
            user_id=user_id,
            project_id=project_id,
        )
    
    async def load_skill(
        self,
        name: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Load full skill content by name.
        
        Respects scope layering (project > user > base).
        
        Args:
            name: Skill name
            user_id: Optional user ID
            project_id: Optional project ID
            
        Returns:
            Full Skill object, or None if not found
        """
        if self._skill_service is None:
            logger.warning("Skill service not configured")
            return None
        
        return await self._skill_service.load_skill_content(
            name=name,
            user_id=user_id,
            project_id=project_id,
        )
    
    async def match_skills(
        self,
        query: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        threshold: float = 0.7,
    ) -> List[Any]:
        """
        Find skills that match a query based on triggers.
        
        Args:
            query: The user query
            user_id: Optional user ID
            project_id: Optional project ID
            threshold: Minimum confidence threshold
            
        Returns:
            List of SkillMatch objects, or empty list if service not configured
        """
        if self._skill_service is None:
            logger.warning("Skill service not configured")
            return []
        
        return await self._skill_service.match_skills(
            query=query,
            user_id=user_id,
            project_id=project_id,
            threshold=threshold,
        )
    
    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================
    
    async def wait_for_background_tasks(self, timeout: float = 30.0) -> None:
        """
        Wait for all background tasks to complete.
        
        Useful for testing and graceful shutdown.
        """
        if self._background_tasks:
            await asyncio.wait(self._background_tasks, timeout=timeout)
    
    async def close(self) -> None:
        """
        Close the engine and clean up resources.
        """
        await self.wait_for_background_tasks(timeout=5.0)
        closed_ids: set[int] = set()
        if self._graph_service is not None:
            try:
                await self._graph_service.close()
                closed_ids.add(id(self._graph_service))
            except Exception as e:
                logger.warning(f"Graph service close failed: {e}")
        if self._vector_store is not None:
            try:
                await self._vector_store.close()
                closed_ids.add(id(self._vector_store))
            except Exception as e:
                logger.warning(f"Vector store close failed: {e}")
        if self._expertise_vector_store is not None:
            try:
                await self._expertise_vector_store.close()
                closed_ids.add(id(self._expertise_vector_store))
            except Exception as e:
                logger.warning(f"Expertise vector store close failed: {e}")

        # Best-effort close for any factory-owned resources (e.g., DB-backed stores).
        for res in list(self._owned_resources):
            if id(res) in closed_ids:
                continue
            close_fn = getattr(res, "close", None)
            if callable(close_fn):
                try:
                    result = close_fn()
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    logger.warning(f"Owned resource close failed: {e}")
                continue

            # Legacy alias: some components expose disconnect() instead of close().
            disconnect_fn = getattr(res, "disconnect", None)
            if callable(disconnect_fn):
                try:
                    result = disconnect_fn()
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    logger.warning(f"Owned resource disconnect failed: {e}")
        logger.info("ctxforge closed")
    
    # =========================================================================
    # FACTORY METHODS
    # =========================================================================
    
    @classmethod
    async def create(
        cls,
        config_path: Optional[str] = None,
        config: Optional[EngineConfig] = None,
        **overrides,
    ) -> "CtxForge":
        """
        Create a CtxForge engine from configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            config: EngineConfig object (alternative to config_path)
            **overrides: Configuration overrides
            
        Returns:
            Configured CtxForge instance
        """
        from ctxforge.engine.factory import EngineFactory
        
        if config_path:
            return await EngineFactory.from_config_file(config_path)
        elif config:
            return await EngineFactory.from_config(config)
        else:
            return await EngineFactory.create_default()
