"""Memory categorizer.

Assigns memories to hierarchical categories based on embedding similarity.
"""

from typing import Callable, List, Optional

from ctxforge.core.categories import CategoryAssignment, MemoryCategory
from ctxforge.core.memory import MemoryItem
from ctxforge.utils.math import cosine_similarity


class MemoryCategorizer:
    """Assigns memories to categories using embedding similarity."""

    def __init__(
        self,
        categories: Optional[List[MemoryCategory]] = None,
        embedding_func: Optional[Callable] = None,
        similarity_threshold: float = 0.6,
    ):
        self._categories: List[MemoryCategory] = list(categories or [])
        self._embedding_func = embedding_func
        self._similarity_threshold = similarity_threshold

    def add_category(self, category: MemoryCategory) -> None:
        """Register a new category."""
        self._categories.append(category)

    def categorize(self, memory: MemoryItem) -> List[CategoryAssignment]:
        """Return category assignments for *memory* above the threshold."""
        if memory.embedding is None:
            return []

        assignments: List[CategoryAssignment] = []
        for cat in self._categories:
            if cat.embedding is None:
                continue
            sim = cosine_similarity(memory.embedding, cat.embedding)
            if sim >= self._similarity_threshold:
                assignments.append(CategoryAssignment(
                    memory_id=memory.memory_id,
                    category_id=cat.category_id,
                    confidence=sim,
                ))
        return assignments

    def auto_create_category(
        self,
        memories: List[MemoryItem],
        name: str,
        description: str = "",
    ) -> MemoryCategory:
        """Create a category whose embedding is the centroid of *memories*."""
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        if not embeddings:
            return MemoryCategory(name=name, description=description)

        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += v
        n = len(embeddings)
        centroid = [v / n for v in centroid]

        return MemoryCategory(name=name, description=description, embedding=centroid)
