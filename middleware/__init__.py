"""
Middleware module for the ctxforge framework.

Provides cross-cutting concerns that can be applied to the context
processing pipeline:

- PIIMiddleware: Detect and redact personally identifiable information
- RateLimitMiddleware: Throttle requests to prevent abuse
- AuditMiddleware: Log all operations for observability
- ContentFilterMiddleware: Filter harmful or inappropriate content

Usage:
    from ctxforge.middleware import MiddlewareChain, PIIMiddleware
    
    # Create middleware chain
    chain = MiddlewareChain()
    chain.add(PIIMiddleware())
    chain.add(AuditMiddleware())
    
    # Process through chain
    context = MiddlewareContext(user_input="My email is test@example.com")
    result = await chain.execute(context)
"""

from ctxforge.middleware.audit import (
    AuditEvent,
    AuditMiddleware,
    InMemoryAuditStore,
)
from ctxforge.middleware.base import (
    BaseMiddleware,
    MiddlewareChain,
)
from ctxforge.middleware.content import (
    ContentFilterMiddleware,
    KeywordFilter,
)
from ctxforge.middleware.expertise import (
    ExpertiseAuditMiddleware,
    ExpertiseEvolutionMiddleware,
    ExpertiseRetrievalMiddleware,
)
from ctxforge.middleware.pii import (
    PIIDetector,
    PIIMiddleware,
    PIIRedactor,
    PIIType,
)
from ctxforge.middleware.protocol import (
    IMiddleware,
    MiddlewareContext,
    MiddlewareResult,
)
from ctxforge.middleware.ratelimit import (
    RateLimitMiddleware,
    SlidingWindowLimiter,
    TokenBucketLimiter,
)

__all__ = [
    # Core
    "IMiddleware",
    "MiddlewareContext",
    "MiddlewareResult",
    "BaseMiddleware",
    "MiddlewareChain",
    # PII
    "PIIDetector",
    "PIIRedactor",
    "PIIMiddleware",
    "PIIType",
    # Rate Limiting
    "RateLimitMiddleware",
    "TokenBucketLimiter",
    "SlidingWindowLimiter",
    # Audit
    "AuditMiddleware",
    "AuditEvent",
    "InMemoryAuditStore",
    # Content Filtering
    "ContentFilterMiddleware",
    "KeywordFilter",
    # Expertise
    "ExpertiseEvolutionMiddleware",
    "ExpertiseRetrievalMiddleware",
    "ExpertiseAuditMiddleware",
]

