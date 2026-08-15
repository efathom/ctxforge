"""
Shared utilities for retrieval implementations.
"""

from typing import Awaitable, Callable, List

from ctxforge.core.memory import MemoryItem
from ctxforge.protocols.retriever import RetrievalConfig

# Type alias for embedding function
EmbeddingFunc = Callable[[str], Awaitable[List[float]]]


def apply_memory_filters(
    memories: List[MemoryItem],
    config: RetrievalConfig,
) -> List[MemoryItem]:
    """
    Apply configuration filters to memories.
    
    Args:
        memories: List of memories to filter
        config: Retrieval configuration with filter settings
        
    Returns:
        Filtered list of memories
    """
    filtered = []
    
    for memory in memories:
        # Filter by active status
        if not config.include_inactive and not memory.is_active:
            continue
        
        # Filter by memory type
        if config.memory_types and memory.type not in config.memory_types:
            continue
        
        # Filter by tags
        if config.tags:
            if not any(tag in memory.tags for tag in config.tags):
                continue
        
        # Filter by confidence
        if memory.confidence_score < config.min_confidence:
            continue
        
        filtered.append(memory)
    
    return filtered
