"""
Expertise System for ctxforge.

The Expertise module provides a structured, evolving knowledge base that learns
from interactions. Inspired by ACE framework's "Playbook" concept, it enables:

- Structured knowledge organization by sections
- Usage tracking (helpful/harmful counts)
- Reflection on turn outcomes
- Curation through ADD/UPDATE/MERGE/DELETE operations
- Offline and online learning modes

Example usage:
    from ctxforge.expertise import (
        Expertise,
        ExpertiseItem,
        ExpertiseSection,
        UsageFeedback,
    )
    
    # Create an expertise knowledge base
    expertise = Expertise(
        expertise_id="my-expertise",
        name="Customer Support",
        domain="support",
    )
    
    # Add items
    item = ExpertiseItem(
        item_id="strat-00001",
        section=ExpertiseSection.STRATEGIES,
        content="Always greet the customer by name",
    )
    expertise.items.append(item)
    
    # For indexer and retriever, import from retrieval:
    from ctxforge.retrieval import ExpertiseIndexer, ExpertiseRetriever
"""

from ctxforge.core.expertise import (
    CompletedTurn,
    CurationOp,
    CurationPlan,
    CuratorOperation,
    Expertise,
    # Core models
    ExpertiseItem,
    # Enums
    ExpertiseSection,
    ExpertiseStats,
    ExpertiseUsageLog,
    ReflectionResult,
    SimilarGroup,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.expertise.analyzer import (
    ExpertiseAnalyzer,
    QualityReport,
)
from ctxforge.expertise.consolidator import ExpertiseConsolidator
from ctxforge.expertise.curator import (
    ExpertiseCurator,
    MockCurator,
    RuleBasedCurator,
)
from ctxforge.expertise.operations import (
    apply_curation_plan,
    format_item_line,
    generate_item_id,
    parse_item_line,
    validate_operation,
)
from ctxforge.expertise.reflector import (
    ExpertiseReflector,
    MockReflector,
    RuleBasedReflector,
)

__all__ = [
    # Enums
    "ExpertiseSection",
    "UsageFeedback",
    "TurnOutcome",
    "CuratorOperation",
    # Core models
    "ExpertiseItem",
    "Expertise",
    "CompletedTurn",
    "ExpertiseUsageLog",
    "ReflectionResult",
    "CurationOp",
    "CurationPlan",
    "ExpertiseStats",
    "SimilarGroup",
    # Reflector
    "ExpertiseReflector",
    "MockReflector",
    "RuleBasedReflector",
    # Curator
    "ExpertiseCurator",
    "MockCurator",
    "RuleBasedCurator",
    # Operations
    "generate_item_id",
    "format_item_line",
    "parse_item_line",
    "apply_curation_plan",
    "validate_operation",
    # Consolidator
    "ExpertiseConsolidator",
    # Analyzer
    "ExpertiseAnalyzer",
    "QualityReport",
]


def __getattr__(name: str):
    """
    Lazy import for backward compatibility.
    
    These components have moved to ctxforge.retrieval module.
    Please update your imports to use the new locations.
    """
    if name == "ExpertiseIndexer":
        from ctxforge.retrieval.indexers.expertise import ExpertiseIndexer
        return ExpertiseIndexer
    elif name == "ExpertiseRetriever":
        from ctxforge.retrieval.retrievers.expertise import ExpertiseRetriever
        return ExpertiseRetriever
    elif name == "ExpertiseRetrievalResult":
        from ctxforge.retrieval.retrievers.expertise import ExpertiseRetrievalResult
        return ExpertiseRetrievalResult
    elif name == "ExpertiseRetrievalConfig":
        from ctxforge.retrieval.retrievers.expertise import ExpertiseRetrievalConfig
        return ExpertiseRetrievalConfig
    elif name == "EffectivenessReranker":
        from ctxforge.retrieval.rerankers.expertise import EffectivenessReranker
        return EffectivenessReranker
    elif name == "HybridExpertiseRetriever":
        from ctxforge.retrieval.retrievers.expertise import HybridExpertiseRetriever
        return HybridExpertiseRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
