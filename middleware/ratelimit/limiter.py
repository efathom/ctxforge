"""
Rate Limiter implementations.

Provides various rate limiting algorithms.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.
    
    Attributes:
        allowed: Whether the request is allowed
        remaining: Number of requests remaining in window
        reset_at: Timestamp when the limit resets
        retry_after: Seconds until retry is allowed (if not allowed)
    """
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


@runtime_checkable
class IRateLimiter(Protocol):
    """Protocol for rate limiter implementations."""
    
    async def check(self, key: str) -> RateLimitResult:
        """
        Check if a request is allowed.
        
        Args:
            key: Identifier for the rate limit bucket (e.g., user_id)
            
        Returns:
            RateLimitResult with decision and metadata
        """
        ...
    
    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """
        Consume tokens and check if allowed.
        
        Args:
            key: Identifier for the rate limit bucket
            tokens: Number of tokens to consume
            
        Returns:
            RateLimitResult with decision
        """
        ...
    
    async def reset(self, key: str) -> None:
        """
        Reset the rate limit for a key.
        
        Args:
            key: Identifier to reset
        """
        ...


class TokenBucketLimiter(IRateLimiter):
    """
    Token bucket rate limiter.
    
    Tokens are added at a fixed rate up to a maximum.
    Each request consumes tokens. When empty, requests are denied.
    
    Good for allowing bursts while maintaining average rate.
    
    Example:
        # 10 requests per second, burst of 20
        limiter = TokenBucketLimiter(
            rate=10.0,       # tokens per second
            capacity=20,     # max tokens
        )
    """
    
    def __init__(
        self,
        rate: float,
        capacity: int,
        initial_tokens: Optional[int] = None,
    ):
        """
        Initialize the limiter.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum token capacity
            initial_tokens: Starting tokens (defaults to capacity)
        """
        self._rate = rate
        self._capacity = capacity
        self._initial_tokens = initial_tokens if initial_tokens is not None else capacity
        
        # State per key
        self._tokens: Dict[str, float] = defaultdict(lambda: float(self._initial_tokens))
        self._last_update: Dict[str, float] = defaultdict(time.time)
        self._lock = asyncio.Lock()
    
    async def check(self, key: str) -> RateLimitResult:
        """Check without consuming tokens."""
        async with self._lock:
            self._refill(key)
            tokens = self._tokens[key]
            
            return RateLimitResult(
                allowed=tokens >= 1,
                remaining=int(tokens),
                reset_at=self._last_update[key] + (self._capacity - tokens) / self._rate,
                retry_after=None if tokens >= 1 else (1 - tokens) / self._rate,
            )
    
    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Consume tokens from the bucket."""
        async with self._lock:
            self._refill(key)
            
            current = self._tokens[key]
            
            if current >= tokens:
                self._tokens[key] = current - tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self._tokens[key]),
                    reset_at=time.time() + (self._capacity - self._tokens[key]) / self._rate,
                )
            else:
                # Not enough tokens
                needed = tokens - current
                retry_after = needed / self._rate
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=time.time() + retry_after,
                    retry_after=retry_after,
                )
    
    async def reset(self, key: str) -> None:
        """Reset bucket to full capacity."""
        async with self._lock:
            self._tokens[key] = float(self._capacity)
            self._last_update[key] = time.time()
    
    def _refill(self, key: str) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update[key]
        self._last_update[key] = now
        
        # Add tokens based on elapsed time
        new_tokens = self._tokens[key] + elapsed * self._rate
        self._tokens[key] = min(new_tokens, float(self._capacity))


class SlidingWindowLimiter(IRateLimiter):
    """
    Sliding window rate limiter.
    
    Tracks requests within a time window and limits the count.
    Uses a sliding window for smoother rate limiting.
    
    Example:
        # 100 requests per minute
        limiter = SlidingWindowLimiter(
            limit=100,
            window_seconds=60,
        )
    """
    
    def __init__(
        self,
        limit: int,
        window_seconds: float,
    ):
        """
        Initialize the limiter.
        
        Args:
            limit: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self._limit = limit
        self._window = window_seconds
        
        # Track request timestamps per key
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def check(self, key: str) -> RateLimitResult:
        """Check without recording a request."""
        async with self._lock:
            self._cleanup(key)
            count = len(self._requests[key])
            
            return RateLimitResult(
                allowed=count < self._limit,
                remaining=max(0, self._limit - count),
                reset_at=self._get_reset_time(key),
                retry_after=self._get_retry_after(key) if count >= self._limit else None,
            )
    
    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Record request(s) and check limit."""
        async with self._lock:
            self._cleanup(key)
            count = len(self._requests[key])
            
            if count + tokens <= self._limit:
                # Record the requests
                now = time.time()
                for _ in range(tokens):
                    self._requests[key].append(now)
                
                return RateLimitResult(
                    allowed=True,
                    remaining=self._limit - len(self._requests[key]),
                    reset_at=self._get_reset_time(key),
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=self._get_reset_time(key),
                    retry_after=self._get_retry_after(key),
                )
    
    async def reset(self, key: str) -> None:
        """Clear all requests for a key."""
        async with self._lock:
            self._requests[key].clear()
    
    def _cleanup(self, key: str) -> None:
        """Remove expired requests from the window."""
        cutoff = time.time() - self._window
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > cutoff
        ]
    
    def _get_reset_time(self, key: str) -> float:
        """Get when the oldest request will expire."""
        if not self._requests[key]:
            return time.time()
        return self._requests[key][0] + self._window
    
    def _get_retry_after(self, key: str) -> float:
        """Get seconds until a slot is available."""
        if not self._requests[key]:
            return 0.0
        oldest = self._requests[key][0]
        retry = oldest + self._window - time.time()
        return max(0.0, retry)


class InMemoryRateLimiter(IRateLimiter):
    """
    Simple in-memory rate limiter.
    
    Fixed window counter - simple but can have edge effects.
    Use TokenBucketLimiter or SlidingWindowLimiter for production.
    """
    
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
    ):
        """
        Initialize the limiter.
        
        Args:
            limit: Max requests per window
            window_seconds: Window size
        """
        self._limit = limit
        self._window = window_seconds
        
        self._counts: Dict[str, int] = defaultdict(int)
        self._reset_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def check(self, key: str) -> RateLimitResult:
        """Check without incrementing."""
        async with self._lock:
            self._maybe_reset(key)
            count = self._counts[key]
            
            return RateLimitResult(
                allowed=count < self._limit,
                remaining=max(0, self._limit - count),
                reset_at=self._reset_times.get(key, time.time() + self._window),
            )
    
    async def consume(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Increment counter and check."""
        async with self._lock:
            self._maybe_reset(key)
            
            if self._counts[key] + tokens <= self._limit:
                self._counts[key] += tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=self._limit - self._counts[key],
                    reset_at=self._reset_times[key],
                )
            else:
                retry_after = self._reset_times[key] - time.time()
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=self._reset_times[key],
                    retry_after=max(0, retry_after),
                )
    
    async def reset(self, key: str) -> None:
        """Reset counter for a key."""
        async with self._lock:
            self._counts[key] = 0
            self._reset_times[key] = time.time() + self._window
    
    def _maybe_reset(self, key: str) -> None:
        """Reset if window has expired."""
        now = time.time()
        
        if key not in self._reset_times:
            self._reset_times[key] = now + self._window
        elif now >= self._reset_times[key]:
            self._counts[key] = 0
            self._reset_times[key] = now + self._window

