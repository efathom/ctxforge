"""Content hashing utilities for O(1) deduplication."""

import hashlib


def compute_content_hash(content: str, memory_type: str) -> str:
    """Compute a deterministic hash for deduplication.

    Normalises whitespace and lowercases the content before hashing so that
    trivial formatting differences collapse to the same hash.

    Args:
        content: The memory content text.
        memory_type: The memory type value (e.g. ``"semantic"``).

    Returns:
        A 16-character hex digest.
    """
    normalized = " ".join(content.lower().split())
    return hashlib.sha256(f"{memory_type}:{normalized}".encode()).hexdigest()[:16]
