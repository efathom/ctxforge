"""
Expertise middleware for the ctxforge framework.

Provides middleware for integrating expertise evolution into the middleware chain:

- ExpertiseEvolutionMiddleware: Handles reflection and curation after turns
- ExpertiseRetrievalMiddleware: Retrieves relevant expertise items before processing
- ExpertiseAuditMiddleware: Logs expertise usage and evolution

Usage:
    from ctxforge.middleware.expertise import (
        ExpertiseEvolutionMiddleware,
        ExpertiseRetrievalMiddleware,
    )
    
    # Create middleware chain with expertise
    chain = MiddlewareChain()
    chain.add(ExpertiseRetrievalMiddleware(retriever=retriever))
    chain.add(ExpertiseEvolutionMiddleware(
        reflector=reflector,
        curator=curator,
        expertise_store=store,
    ))
"""

from ctxforge.middleware.expertise.middleware import (
    ExpertiseAuditMiddleware,
    ExpertiseEvolutionMiddleware,
    ExpertiseRetrievalMiddleware,
)

__all__ = [
    "ExpertiseEvolutionMiddleware",
    "ExpertiseRetrievalMiddleware",
    "ExpertiseAuditMiddleware",
]

