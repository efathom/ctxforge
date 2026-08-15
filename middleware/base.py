"""
Base Middleware and Chain implementation.

Provides abstract base class for middleware and the chain executor.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional

from ctxforge.middleware.protocol import (
    IMiddleware,
    MiddlewareContext,
    MiddlewareResult,
    NextFunction,
)


class BaseMiddleware(IMiddleware, ABC):
    """
    Abstract base class for middleware implementations.
    
    Provides common functionality:
    - Enabled/disabled state
    - Error handling
    - Logging hooks
    
    Subclasses must implement:
    - name property
    - _do_process method
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the middleware.
        
        Args:
            enabled: Whether this middleware is active
        """
        self._enabled = enabled
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this middleware."""
        ...
    
    @property
    def enabled(self) -> bool:
        """Whether this middleware is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable this middleware."""
        self._enabled = value
    
    async def process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Process the context, delegating to _do_process.
        
        Handles enabled/disabled state and error handling.
        
        Args:
            context: The current middleware context
            next: Function to call the next middleware
            
        Returns:
            The processed context
        """
        if not self._enabled:
            # Skip this middleware
            return await next(context)
        
        try:
            return await self._do_process(context, next)
        except StopChainException:
            # Re-raise StopChainException - it's intentional
            raise
        except Exception as e:
            # Record the error but don't stop the chain by default
            context.set_metadata(f"{self.name}_error", str(e))
            return await next(context)
    
    @abstractmethod
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Perform the actual processing.
        
        Subclasses implement their logic here.
        
        Args:
            context: The current middleware context
            next: Function to call the next middleware
            
        Returns:
            The processed context
        """
        ...


class MiddlewareChain:
    """
    Executes a chain of middleware in sequence.
    
    Middleware are executed in the order they are added.
    Each middleware can modify the context and decide whether
    to continue the chain.
    
    Example:
        chain = MiddlewareChain()
        chain.add(PIIMiddleware())
        chain.add(AuditMiddleware())
        chain.add(RateLimitMiddleware())
        
        context = MiddlewareContext(user_input="Hello")
        result = await chain.execute(context)
        
        if result.success:
            print(result.context.processed_input)
    """
    
    def __init__(self):
        """Initialize an empty middleware chain."""
        self._middleware: List[IMiddleware] = []
    
    def add(self, middleware: IMiddleware) -> "MiddlewareChain":
        """
        Add a middleware to the chain.
        
        Args:
            middleware: The middleware to add
            
        Returns:
            Self for chaining
        """
        self._middleware.append(middleware)
        return self
    
    def insert(self, index: int, middleware: IMiddleware) -> "MiddlewareChain":
        """
        Insert a middleware at a specific position.
        
        Args:
            index: Position to insert at
            middleware: The middleware to insert
            
        Returns:
            Self for chaining
        """
        self._middleware.insert(index, middleware)
        return self
    
    def remove(self, name: str) -> bool:
        """
        Remove a middleware by name.
        
        Args:
            name: Name of the middleware to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, m in enumerate(self._middleware):
            if m.name == name:
                self._middleware.pop(i)
                return True
        return False
    
    def get(self, name: str) -> Optional[IMiddleware]:
        """
        Get a middleware by name.
        
        Args:
            name: Name of the middleware
            
        Returns:
            The middleware or None
        """
        for m in self._middleware:
            if m.name == name:
                return m
        return None
    
    def clear(self) -> None:
        """Remove all middleware from the chain."""
        self._middleware.clear()
    
    @property
    def middleware(self) -> List[IMiddleware]:
        """Get list of middleware in the chain."""
        return list(self._middleware)
    
    async def execute(
        self,
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """
        Execute the middleware chain.
        
        Args:
            context: The initial context
            
        Returns:
            MiddlewareResult with final context and status
        """
        start_time = time.time()
        
        if not self._middleware:
            # No middleware, return as-is
            return MiddlewareResult(
                context=context,
                success=True,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            # Build the chain from the end
            async def terminal(ctx: MiddlewareContext) -> MiddlewareContext:
                return ctx
            
            # Wrap middleware from last to first
            next_fn = terminal
            for middleware in reversed(self._middleware):
                # Capture current middleware and next in closure
                next_fn = self._wrap_middleware(middleware, next_fn)
            
            # Execute the chain
            result_context = await next_fn(context)
            
            return MiddlewareResult(
                context=result_context,
                success=True,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except StopChainException as e:
            return MiddlewareResult(
                context=context,
                success=False,
                stopped_by=e.middleware_name,
                error=e.reason,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return MiddlewareResult(
                context=context,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _wrap_middleware(
        self,
        middleware: IMiddleware,
        next_fn: NextFunction,
    ) -> NextFunction:
        """Wrap a middleware with the next function."""
        async def wrapped(context: MiddlewareContext) -> MiddlewareContext:
            return await middleware.process(context, next_fn)
        return wrapped


class StopChainException(Exception):
    """
    Exception to stop the middleware chain.
    
    Middleware can raise this to prevent further processing.
    
    Example:
        if rate_limit_exceeded:
            raise StopChainException("rate_limit", "Too many requests")
    """
    
    def __init__(self, middleware_name: str, reason: str):
        """
        Initialize the exception.
        
        Args:
            middleware_name: Name of the middleware stopping the chain
            reason: Reason for stopping
        """
        self.middleware_name = middleware_name
        self.reason = reason
        super().__init__(f"{middleware_name}: {reason}")

