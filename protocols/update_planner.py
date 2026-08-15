from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol

from ctxforge.core.memory import MemoryItem


class MemoryOperationType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


@dataclass
class MemoryOperation:
    op: MemoryOperationType
    new_temp_id: Optional[str] = None
    target_memory_id: Optional[str] = None
    target_scope_id: Optional[str] = None
    content: Optional[str] = None
    confidence: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    rationale: Optional[str] = None


class IMemoryUpdatePlanner(Protocol):
    async def plan(
        self,
        *,
        user_id: str,
        query: str,
        new_items: List[MemoryItem],
        user_candidates: Dict[str, List[MemoryItem]],
        global_candidates: Dict[str, List[MemoryItem]],
        model: Optional[str] = None,
    ) -> List[MemoryOperation]:
        ...


