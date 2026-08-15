"""
Rate Limit Middleware implementation.
"""

from typing import Callable, Optional

from ctxforge.middleware.base import BaseMiddleware, StopChainException
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction
from ctxforge.middleware.ratelimit.limiter import (
    IRateLimiter,
    TokenBucketLimiter,
)


class RateLimitMiddleware(BaseMiddleware):
    """
    Rate limiting middleware.
    
    Limits the rate of requests per user/session to prevent abuse.
    
    Example:
        # 10 requests per second per user
        middleware = RateLimitMiddleware(
            limiter=TokenBucketLimiter(rate=10.0, capacity=20),
        )
        
        # 100 requests per minute, stop on limit
        from ctxforge.middleware.ratelimit import SlidingWindowLimiter
        middleware = RateLimitMiddleware(
            limiter=SlidingWindowLimiter(limit=100, window_seconds=60),
            stop_on_limit=True,
        )
    """
    
    def __init__(
        self,
        limiter: Optional[IRateLimiter] = None,
        key_func: Optional[Callable[[MiddlewareContext], str]] = None,
        stop_on_limit: bool = True,
        tokens_per_request: int = 1,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            limiter: Rate limiter implementation
            key_func: Function to extract rate limit key from context
            stop_on_limit: Whether to stop the chain when limit is exceeded
            tokens_per_request: Tokens to consume per request
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled)
        
        self._limiter = limiter or TokenBucketLimiter(rate=10.0, capacity=100)
        self._key_func = key_func or self._default_key_func
        self._stop_on_limit = stop_on_limit
        self._tokens_per_request = tokens_per_request
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "rate_limit"
    
    @property
    def limiter(self) -> IRateLimiter:
        """The rate limiter."""
        return self._limiter
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Check rate limit before processing.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context
        """
        key = self._key_func(context)
        result = await self._limiter.consume(key, self._tokens_per_request)
        
        # Record rate limit info
        context.set_metadata("rate_limit_remaining", result.remaining)
        context.set_metadata("rate_limit_reset_at", result.reset_at)
        
        if not result.allowed:
            context.add_flag("rate_limited")
            context.set_metadata("rate_limit_retry_after", result.retry_after)
            
            if self._stop_on_limit:
                raise StopChainException(
                    self.name,
                    f"Rate limit exceeded. Retry after {result.retry_after:.1f}s",
                )
        
        return await next(context)
    
    def _default_key_func(self, context: MiddlewareContext) -> str:
        """
        Default function to extract rate limit key.
        
        Uses user_id if available, otherwise session_id.
        """
        if context.user_id:
            return f"user:{context.user_id}"
        elif context.session_id:
            return f"session:{context.session_id}"
        else:
            return "anonymous"

