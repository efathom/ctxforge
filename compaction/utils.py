"""
Shared utilities for compaction implementations.
"""

from typing import Awaitable, Callable, List, Optional

from ctxforge.core.events import Event

# Type aliases
SummarizeFunc = Callable[[str, Optional[str]], Awaitable[str]]
ScoringFunc = Callable[[Event], float]


def estimate_tokens_simple(text: str) -> int:
    """
    Simple token estimation: ~4 chars per token.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4 + 1


def estimate_event_tokens(events: List[Event]) -> int:
    """
    Estimate total tokens for a list of events.
    
    Args:
        events: List of events to estimate
        
    Returns:
        Total estimated token count
    """
    total = 0
    for event in events:
        total += estimate_tokens_simple(event.content)
        if event.metadata:
            total += estimate_tokens_simple(str(event.metadata))
    return total

