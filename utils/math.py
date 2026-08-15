"""
Math utilities for the ctxforge framework.

Provides common mathematical operations used across the framework.
"""

import math
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1, where 1 is identical)
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance (0 to infinity, where 0 is identical)
    """
    if len(a) != len(b) or len(a) == 0:
        return float('inf')
    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def dot_product(a: List[float], b: List[float]) -> float:
    """
    Compute dot product of two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product
    """
    if len(a) != len(b):
        return 0.0
    
    return sum(x * y for x, y in zip(a, b, strict=False))


def normalize_vector(v: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        v: Vector to normalize
        
    Returns:
        Normalized vector with magnitude 1
    """
    if not v:
        return []
    
    magnitude = math.sqrt(sum(x * x for x in v))
    
    if magnitude == 0:
        return v
    
    return [x / magnitude for x in v]

