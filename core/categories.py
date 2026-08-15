"""Hierarchical memory categories.

Provides models for grouping memories into semantic categories
with optional embedding-based assignment.
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryCategory(BaseModel):
    """A category for grouping related memories."""

    category_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CategoryAssignment(BaseModel):
    """Links a memory to a category with a confidence score."""

    memory_id: str
    category_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
