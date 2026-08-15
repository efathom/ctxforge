"""
Expertise Middleware implementations.

Provides middleware for integrating expertise into the ctxforge middleware chain:

- ExpertiseEvolutionMiddleware: Handles reflection and curation after turns
- ExpertiseRetrievalMiddleware: Retrieves relevant expertise items before processing
- ExpertiseAuditMiddleware: Logs expertise usage and evolution events
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ctxforge.core.expertise import (
    CompletedTurn,
    Expertise,
    ReflectionResult,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction
from ctxforge.protocols.expertise import (
    ICurator,
    IExpertiseRetriever,
    IExpertiseStore,
    IReflector,
)

logger = logging.getLogger(__name__)


# Metadata keys for expertise information in MiddlewareContext
EXPERTISE_ID_KEY = "expertise_id"
EXPERTISE_ITEMS_KEY = "expertise_items"
EXPERTISE_ITEMS_USED_KEY = "expertise_items_used"
TURN_OUTCOME_KEY = "turn_outcome"
GROUND_TRUTH_KEY = "ground_truth"
REFLECTION_RESULT_KEY = "reflection_result"
CURATION_PLAN_KEY = "curation_plan"


class ExpertiseRetrievalMiddleware(BaseMiddleware):
    """
    Middleware that retrieves relevant expertise items before processing.
    
    This middleware runs early in the chain to inject expertise items
    into the context, making them available for subsequent middleware
    and the main processing logic.
    
    Example:
        middleware = ExpertiseRetrievalMiddleware(
            retriever=expertise_retriever,
            default_expertise_id="math-expertise",
            top_k=10,
        )
        chain.add(middleware)
    """
    
    def __init__(
        self,
        retriever: IExpertiseRetriever,
        expertise_store: Optional[IExpertiseStore] = None,
        default_expertise_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.0,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            retriever: Retriever for finding relevant expertise items
            expertise_store: Store for loading expertise (if not cached in retriever)
            default_expertise_id: Default expertise ID if not in context
            top_k: Number of items to retrieve
            min_score: Minimum relevance score for items
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled)
        
        self._retriever = retriever
        self._expertise_store = expertise_store
        self._default_expertise_id = default_expertise_id
        self._top_k = top_k
        self._min_score = min_score
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "expertise_retrieval"
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Retrieve expertise items and add them to context.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context with expertise items
        """
        # Get expertise ID from context or use default
        expertise_id = context.get_metadata(EXPERTISE_ID_KEY) or self._default_expertise_id
        
        if not expertise_id:
            # No expertise configured, skip
            return await next(context)
        
        # Ensure expertise ID is set in context for downstream middleware
        if not context.get_metadata(EXPERTISE_ID_KEY):
            context.set_metadata(EXPERTISE_ID_KEY, expertise_id)
        
        try:
            # Get the query from processed input
            query = context.processed_input or context.user_input
            
            # Retrieve relevant items
            # IExpertiseRetriever.retrieve expects `limit` (not top_k)
            items = await self._retriever.retrieve(
                query=query,
                expertise_id=expertise_id,
                limit=self._top_k,
            )
            
            # Store items in context
            context.set_metadata(EXPERTISE_ITEMS_KEY, items)
            context.add_flag("expertise_retrieved")
            
            logger.debug(
                f"Retrieved {len(items)} expertise items for query: {query[:50]}..."
            )
            
        except Exception as e:
            logger.warning(f"Failed to retrieve expertise: {e}")
            context.set_metadata(f"{self.name}_error", str(e))
        
        return await next(context)


class ExpertiseEvolutionMiddleware(BaseMiddleware):
    """
    Middleware that handles expertise evolution after turns.
    
    This middleware runs after the main processing to:
    1. Reflect on the turn outcome using the Reflector
    2. Update expertise item counts based on feedback
    3. Optionally run the Curator to evolve expertise
    
    Example:
        middleware = ExpertiseEvolutionMiddleware(
            reflector=reflector,
            curator=curator,
            expertise_store=expertise_store,
            evolve_on_success=True,
            evolve_on_failure=True,
        )
        chain.add(middleware)
    """
    
    def __init__(
        self,
        reflector: IReflector,
        expertise_store: IExpertiseStore,
        curator: Optional[ICurator] = None,
        enabled: bool = True,
        evolve_on_success: bool = True,
        evolve_on_failure: bool = True,
        auto_curate: bool = True,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the middleware.
        
        Args:
            reflector: Reflector for analyzing turn outcomes
            expertise_store: Store for persisting expertise changes
            curator: Optional curator for evolving expertise
            enabled: Whether middleware is enabled
            evolve_on_success: Whether to evolve on successful turns
            evolve_on_failure: Whether to evolve on failed turns
            auto_curate: Whether to automatically run curator
            min_confidence: Minimum reflection confidence to act on
        """
        super().__init__(enabled)
        
        self._reflector = reflector
        self._expertise_store = expertise_store
        self._curator = curator
        self._evolve_on_success = evolve_on_success
        self._evolve_on_failure = evolve_on_failure
        self._auto_curate = auto_curate
        self._min_confidence = min_confidence
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "expertise_evolution"
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Process the turn and evolve expertise based on outcome.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context
        """
        # First, let the rest of the chain process
        result = await next(context)
        
        # Check if we have what we need for evolution
        expertise_id = result.get_metadata(EXPERTISE_ID_KEY)
        if not expertise_id:
            return result
        
        outcome = result.get_metadata(TURN_OUTCOME_KEY)
        if not outcome:
            # No outcome specified, can't evolve
            return result
        
        # Check if we should evolve based on outcome
        if isinstance(outcome, str):
            outcome = TurnOutcome(outcome)
        
        if outcome == TurnOutcome.SUCCESS and not self._evolve_on_success:
            return result
        if outcome == TurnOutcome.FAILURE and not self._evolve_on_failure:
            return result
        
        try:
            await self._evolve_expertise(result, expertise_id, outcome)
        except Exception as e:
            logger.warning(f"Failed to evolve expertise: {e}")
            result.set_metadata(f"{self.name}_error", str(e))
        
        return result
    
    async def _evolve_expertise(
        self,
        context: MiddlewareContext,
        expertise_id: str,
        outcome: TurnOutcome,
    ) -> None:
        """
        Perform expertise evolution based on turn outcome.
        
        Args:
            context: The middleware context with turn information
            expertise_id: ID of the expertise to evolve
            outcome: The turn outcome
        """
        # Load the expertise
        load_fn = getattr(self._expertise_store, "load", None)
        if callable(load_fn):
            expertise = await load_fn(expertise_id)
        else:
            # Backward/compat for older store mocks that used `get`
            get_fn = getattr(self._expertise_store, "get", None)
            expertise = await get_fn(expertise_id) if callable(get_fn) else None
        if not expertise:
            logger.warning(f"Expertise not found: {expertise_id}")
            return
        
        # Get items that were used
        items_used_ids = context.get_metadata(EXPERTISE_ITEMS_USED_KEY) or []
        items_used = [
            expertise.get_item(item_id)
            for item_id in items_used_ids
            if expertise.get_item(item_id)
        ]
        
        # Build the completed turn
        turn = CompletedTurn(
            user_input=context.user_input,
            assistant_response=context.agent_response or "",
            expected_output=context.get_metadata(GROUND_TRUTH_KEY),
        )
        
        # Run reflection
        reflection = await self._reflector.reflect(
            turn=turn,
            items_used=items_used,
            outcome=outcome,
        )
        
        # Store reflection result in context
        context.set_metadata(REFLECTION_RESULT_KEY, reflection)
        
        # Check confidence threshold
        if reflection.confidence < self._min_confidence:
            logger.debug(
                f"Reflection confidence {reflection.confidence} below threshold "
                f"{self._min_confidence}, skipping evolution"
            )
            return
        
        # Update item counts based on feedback
        for item_id, feedback in reflection.item_feedback.items():
            item = expertise.get_item(item_id)
            if item:
                if feedback == UsageFeedback.HELPFUL:
                    item.increment_helpful()
                elif feedback == UsageFeedback.HARMFUL:
                    item.increment_harmful()
        
        # Save updated expertise
        await self._expertise_store.save(expertise)
        
        context.add_flag("expertise_evolved")
        
        # Optionally run curator
        if self._auto_curate and self._curator and reflection.has_suggestions:
            await self._run_curation(context, expertise, reflection)
    
    async def _run_curation(
        self,
        context: MiddlewareContext,
        expertise: Expertise,
        reflection: ReflectionResult,
    ) -> None:
        """
        Run the curator to evolve expertise.
        
        Args:
            context: The middleware context
            expertise: The expertise to curate
            reflection: The reflection result
        """
        try:
            # Build usage stats
            usage_stats = {
                "total_items": expertise.item_count,
                "active_items": expertise.active_item_count,
                "helpful_items": len(reflection.helpful_items),
                "harmful_items": len(reflection.harmful_items),
            }
            
            # Run curation
            updated_expertise, plan = await self._curator.curate(
                expertise=expertise,
                reflection=reflection,
                usage_stats=usage_stats,
            )
            
            # Save updated expertise
            if plan.has_operations:
                await self._expertise_store.save(updated_expertise)
                context.set_metadata(CURATION_PLAN_KEY, plan)
                context.add_flag("expertise_curated")
                
                logger.info(
                    f"Curated expertise {expertise.expertise_id}: "
                    f"{plan.operation_count} operations"
                )
        except Exception as e:
            logger.warning(f"Curation failed: {e}")


class ExpertiseAuditMiddleware(BaseMiddleware):
    """
    Middleware that logs expertise usage and evolution events.
    
    Tracks:
    - Which expertise items were retrieved
    - Which items were used
    - Reflection results
    - Curation operations
    
    Example:
        middleware = ExpertiseAuditMiddleware(
            log_items_retrieved=True,
            log_items_used=True,
            log_reflection=True,
            log_curation=True,
        )
        chain.add(middleware)
    """
    
    def __init__(
        self,
        log_items_retrieved: bool = True,
        log_items_used: bool = True,
        log_reflection: bool = True,
        log_curation: bool = True,
        custom_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            log_items_retrieved: Whether to log retrieved items
            log_items_used: Whether to log used items
            log_reflection: Whether to log reflection results
            log_curation: Whether to log curation plans
            custom_handler: Custom handler for audit events
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled)
        
        self._log_items_retrieved = log_items_retrieved
        self._log_items_used = log_items_used
        self._log_reflection = log_reflection
        self._log_curation = log_curation
        self._custom_handler = custom_handler
        self._events: List[Dict[str, Any]] = []
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "expertise_audit"
    
    @property
    def events(self) -> List[Dict[str, Any]]:
        """Get logged events."""
        return list(self._events)
    
    def clear_events(self) -> None:
        """Clear logged events."""
        self._events.clear()
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Log expertise events after processing.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context
        """
        start_time = time.time()
        
        # Process through the chain
        result = await next(context)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Build audit event
        event = self._build_event(result, duration_ms)
        
        if event:
            self._events.append(event)
            
            if self._custom_handler:
                try:
                    self._custom_handler(event)
                except Exception:
                    pass  # Don't let handler errors break the chain
        
        return result
    
    def _build_event(
        self,
        context: MiddlewareContext,
        duration_ms: float,
    ) -> Optional[Dict[str, Any]]:
        """Build an audit event from context."""
        expertise_id = context.get_metadata(EXPERTISE_ID_KEY)
        if not expertise_id:
            return None
        
        event: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "expertise_id": expertise_id,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "duration_ms": duration_ms,
            "flags": list(context.flags),
        }
        
        # Log retrieved items
        if self._log_items_retrieved:
            items = context.get_metadata(EXPERTISE_ITEMS_KEY) or []
            event["items_retrieved"] = len(items)
            event["item_ids_retrieved"] = [
                item.item_id if hasattr(item, 'item_id') else str(item)
                for item in items[:20]  # Limit to avoid large logs
            ]
        
        # Log used items
        if self._log_items_used:
            items_used = context.get_metadata(EXPERTISE_ITEMS_USED_KEY) or []
            event["items_used"] = items_used
        
        # Log reflection
        if self._log_reflection:
            reflection = context.get_metadata(REFLECTION_RESULT_KEY)
            if reflection:
                event["reflection"] = {
                    "helpful_items": reflection.helpful_items,
                    "harmful_items": reflection.harmful_items,
                    "confidence": reflection.confidence,
                    "has_suggestions": reflection.has_suggestions,
                }
        
        # Log curation
        if self._log_curation:
            plan = context.get_metadata(CURATION_PLAN_KEY)
            if plan:
                event["curation"] = {
                    "operation_count": plan.operation_count,
                    "operations": [
                        {"type": op.type.value, "item_ids": op.item_ids}
                        for op in plan.operations[:10]  # Limit
                    ],
                }
        
        return event

