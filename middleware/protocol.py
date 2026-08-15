"""
Middleware Protocol and Context definitions.

Defines the core interfaces for the middleware system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from ctxforge.core.context import ContextSection
from ctxforge.core.session import Session


@dataclass
class MiddlewareContext:
    """
    Context object passed through the middleware chain.
    
    Contains all information about the current request/operation
    and accumulates results from each middleware.
    
    Attributes:
        user_input: The user's input text
        agent_response: The agent's response (if available)
        session: Current session (if available)
        user_id: User identifier
        session_id: Session identifier
        metadata: Arbitrary metadata dict
        flags: Set of flags added by middleware (e.g., "pii_detected")
        modifications: Record of modifications made by middleware
        timestamp: When the context was created
    """
    user_input: str
    agent_response: Optional[str] = None
    session: Optional[Session] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    phase: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    flags: Set[str] = field(default_factory=set)
    modifications: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Processed versions (after middleware)
    processed_input: Optional[str] = None
    processed_response: Optional[str] = None

    # Context sections contributed by middleware (injected during assembly).
    # Middleware should use add_section() to contribute context (skills,
    # scoped memories, etc.) rather than concatenating into processed_input.
    context_sections: List[ContextSection] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize processed versions to original if not set."""
        if self.processed_input is None:
            self.processed_input = self.user_input
        if self.processed_response is None:
            self.processed_response = self.agent_response
    
    def add_section(
        self,
        name: str,
        content: str,
        priority: int = 0,
        is_required: bool = False,
    ) -> None:
        """Add a context section to be injected during assembly.

        Use this instead of modifying ``processed_input`` when the content
        is supplementary context (skills, memories, etc.) rather than a
        transformation of the user's query.
        """
        self.context_sections.append(ContextSection(
            name=name,
            content=content,
            priority=priority,
            is_required=is_required,
        ))

    def add_flag(self, flag: str) -> None:
        """Add a flag to the context."""
        self.flags.add(flag)
    
    def has_flag(self, flag: str) -> bool:
        """Check if a flag is set."""
        return flag in self.flags
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self.metadata.get(key, default)
    
    def record_modification(self, middleware: str, details: Any) -> None:
        """Record a modification made by a middleware."""
        if middleware not in self.modifications:
            self.modifications[middleware] = []
        self.modifications[middleware].append(details)


@dataclass
class MiddlewareResult:
    """
    Result of middleware chain execution.
    
    Attributes:
        context: The final context after all middleware
        success: Whether all middleware succeeded
        stopped_by: Name of middleware that stopped the chain (if any)
        error: Error message if failed
        processing_time_ms: Total processing time in milliseconds
    """
    context: MiddlewareContext
    success: bool = True
    stopped_by: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


# Type alias for the next function in the chain
NextFunction = Callable[[MiddlewareContext], Awaitable[MiddlewareContext]]


@runtime_checkable
class IMiddleware(Protocol):
    """
    Protocol for middleware components.
    
    Middleware follows the chain-of-responsibility pattern.
    Each middleware can:
    - Modify the context
    - Add flags or metadata
    - Call the next middleware
    - Stop the chain (by not calling next)
    
    Example:
        class LoggingMiddleware(IMiddleware):
            @property
            def name(self) -> str:
                return "logging"
            
            async def process(
                self,
                context: MiddlewareContext,
                next: NextFunction,
            ) -> MiddlewareContext:
                print(f"Input: {context.user_input}")
                result = await next(context)
                print(f"Output: {result.processed_input}")
                return result
    """
    
    @property
    def name(self) -> str:
        """
        Unique identifier for this middleware.
        
        Used for logging, debugging, and recording modifications.
        """
        ...
    
    async def process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Process the context and optionally call next middleware.
        
        Args:
            context: The current middleware context
            next: Function to call the next middleware in chain
            
        Returns:
            The (possibly modified) context
        """
        ...

