"""
Audit Middleware implementation.
"""

import time
from typing import Callable, List, Optional, Set

from ctxforge.middleware.audit.store import (
    AuditEvent,
    IAuditStore,
    InMemoryAuditStore,
)
from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction


class AuditMiddleware(BaseMiddleware):
    """
    Audit logging middleware.
    
    Records all operations for observability, compliance, and debugging.
    
    Example:
        store = InMemoryAuditStore()
        middleware = AuditMiddleware(store=store)
        
        # Query logs later
        events = await store.query(user_id="user123")
    """
    
    def __init__(
        self,
        store: Optional[IAuditStore] = None,
        log_input: bool = True,
        log_output: bool = True,
        log_modifications: bool = True,
        redact_pii: bool = True,
        event_types: Optional[Set[str]] = None,
        custom_handler: Optional[Callable[[AuditEvent], None]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            store: Audit store for persisting events
            log_input: Whether to log user input
            log_output: Whether to log agent response
            log_modifications: Whether to log middleware modifications
            redact_pii: Whether to redact input/output if PII was detected
            event_types: Event types to log (all if None)
            custom_handler: Additional handler for audit events
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled)
        
        self._store = store or InMemoryAuditStore()
        self._log_input = log_input
        self._log_output = log_output
        self._log_modifications = log_modifications
        self._redact_pii = redact_pii
        self._event_types = event_types
        self._custom_handler = custom_handler
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "audit"
    
    @property
    def store(self) -> IAuditStore:
        """The audit store."""
        return self._store
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Process and log the operation.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context
        """
        start_time = time.time()
        error_msg: Optional[str] = None
        success = True
        
        try:
            # Process through the chain
            result = await next(context)
            
            # Log after processing (we have all the info now)
            duration_ms = (time.time() - start_time) * 1000
            await self._log_event(result, duration_ms, success, error_msg)
            
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            duration_ms = (time.time() - start_time) * 1000
            await self._log_event(context, duration_ms, success, error_msg)
            raise
    
    async def _log_event(
        self,
        context: MiddlewareContext,
        duration_ms: float,
        success: bool,
        error: Optional[str],
    ) -> None:
        """Create and store an audit event."""
        # Build details
        details = {}
        
        # Input/Output
        if self._log_input:
            if self._redact_pii and context.has_flag("pii_detected_in_input"):
                details["input"] = context.processed_input  # Use redacted version
            else:
                details["input"] = context.user_input
        
        if self._log_output and context.agent_response:
            if self._redact_pii and context.has_flag("pii_detected_in_response"):
                details["output"] = context.processed_response
            else:
                details["output"] = context.agent_response
        
        # Flags and modifications
        details["flags"] = list(context.flags)
        
        if self._log_modifications and context.modifications:
            details["modifications"] = context.modifications
        
        # Metadata
        details["metadata"] = context.metadata
        
        event = AuditEvent(
            user_id=context.user_id,
            session_id=context.session_id,
            event_type="request",
            action="process",
            details=details,
            source="middleware_chain",
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        
        # Check event type filter
        if self._event_types and event.event_type not in self._event_types:
            return
        
        # Store event
        await self._store.log(event)
        
        # Call custom handler if provided
        if self._custom_handler:
            try:
                self._custom_handler(event)
            except Exception:
                pass  # Don't let handler errors break the chain
    
    async def get_events(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Convenience method to query events.
        
        Args:
            user_id: Filter by user
            session_id: Filter by session
            limit: Max events to return
            
        Returns:
            List of audit events
        """
        return await self._store.query(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

