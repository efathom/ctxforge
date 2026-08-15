from __future__ import annotations

"""
Memory update planning service.

This service encapsulates the LLM-driven update planning flow:
- fetch per-item candidate memories (user + optional global scope)
- ask the update planner to produce operations (ADD/UPDATE/DELETE/NONE)
- apply those operations to the memory store (including indexing updates when configured)
"""

import asyncio
from typing import Dict, List, Optional, Tuple

from ctxforge.config.base import EngineConfig
from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.protocols.update_planner import (
    IMemoryUpdatePlanner,
    MemoryOperation,
    MemoryOperationType,
)
from ctxforge.retrieval.indexers.memory import MemoryIndexer


class MemoryUpdateService:
    """
    Owns update-planning dependencies and applies plans to the underlying memory store.

    This is intentionally engine-independent so it can be tested in isolation.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        memory_store: IMemoryStore,
        update_planner: IMemoryUpdatePlanner,
        memory_indexer: Optional[MemoryIndexer] = None,
    ):
        self._cfg = config
        self._store = memory_store
        self._planner = update_planner
        self._indexer = memory_indexer

    async def plan_and_apply(
        self,
        *,
        user_id: str,
        query: str,
        new_items: List[MemoryItem],
    ) -> None:
        """Plan and apply operations for extracted `new_items`."""
        if not new_items:
            return

        temp_ids = [f"n{i+1}" for i in range(len(new_items))]
        per_item_limit = int(self._cfg.extraction.update_planning_candidates_per_item or 0)

        enable_global = bool(getattr(self._cfg.scopes, "enable_global", False))
        global_scope_id = str(getattr(self._cfg.scopes, "global_scope_id", "global"))
        allow_global_writes = bool(getattr(self._cfg.scopes, "allow_global_writes", False))

        user_candidates: Dict[str, List[MemoryItem]] = {}
        global_candidates: Dict[str, List[MemoryItem]] = {}

        async def fetch_candidates_for_item(
            tid: str,
            item: MemoryItem,
        ) -> Tuple[str, List[MemoryItem], List[MemoryItem]]:
            if per_item_limit <= 0:
                return tid, [], []

            u = await self._store.search(
                MemoryQuery(user_id=user_id, query_text=item.content, limit=per_item_limit)
            )

            g: List[MemoryItem] = []
            if enable_global and user_id != global_scope_id:
                g = await self._store.search(
                    MemoryQuery(
                        user_id=global_scope_id,
                        query_text=item.content,
                        limit=per_item_limit,
                    )
                )
            return tid, u, g

        fetched = await asyncio.gather(
            *[fetch_candidates_for_item(tid, item) for tid, item in zip(temp_ids, new_items, strict=False)],
            return_exceptions=True,
        )

        for r in fetched:
            if isinstance(r, Exception):
                continue
            tid, u, g = r
            user_candidates[tid] = u
            global_candidates[tid] = g

        model = self._cfg.extraction.update_planning_model or self._cfg.llm.model
        operations = await self._planner.plan(
            user_id=user_id,
            query=query,
            new_items=new_items,
            user_candidates=user_candidates,
            global_candidates=global_candidates,
            model=model,
        )

        await self._apply_operations(
            user_id=user_id,
            operations=operations,
            temp_ids=temp_ids,
            new_items=new_items,
            global_scope_id=global_scope_id,
            allow_global_writes=allow_global_writes,
        )

    async def _apply_operations(
        self,
        *,
        user_id: str,
        operations: List[MemoryOperation],
        temp_ids: List[str],
        new_items: List[MemoryItem],
        global_scope_id: str,
        allow_global_writes: bool,
    ) -> None:
        """
        Apply a planner-produced update plan to the memory store.

        This preserves existing engine semantics:
        - DELETE is a soft delete (deactivate + update)
        - UPDATE does not allow cross-scope moves
        - global writes can be disallowed by config
        """
        for op in operations:
            if op.op == MemoryOperationType.NONE:
                continue

            if op.op == MemoryOperationType.ADD:
                if not op.new_temp_id or op.new_temp_id not in temp_ids:
                    continue
                idx = temp_ids.index(op.new_temp_id)
                base = new_items[idx].model_copy(deep=True)

                target_scope_id = op.target_scope_id or user_id
                if target_scope_id == global_scope_id and not allow_global_writes:
                    target_scope_id = user_id

                base.user_id = target_scope_id
                if op.content:
                    base.content = op.content.strip()
                if op.confidence is not None:
                    base.confidence_score = max(0.0, min(1.0, float(op.confidence)))
                if op.tags:
                    base.tags = list(dict.fromkeys(op.tags))

                await self._store.add(base)
                if self._indexer is not None:
                    try:
                        await self._indexer.index_item(base, scope_id=base.user_id)
                    except Exception:
                        pass
                continue

            if op.op == MemoryOperationType.UPDATE:
                if not op.target_memory_id:
                    continue
                target = await self._store.get(op.target_memory_id)
                if target is None:
                    continue

                target_scope_id = op.target_scope_id or target.user_id
                if target_scope_id == global_scope_id and not allow_global_writes:
                    continue
                if target.user_id != target_scope_id:
                    continue

                if op.content:
                    target.update_content(op.content)
                if op.confidence is not None:
                    target.update_confidence(float(op.confidence))
                if op.tags:
                    target.tags = list(dict.fromkeys(list(target.tags) + list(op.tags)))

                ok = await self._store.update(target)
                if ok and self._indexer is not None:
                    try:
                        await self._indexer.index_item(target, scope_id=target.user_id)
                    except Exception:
                        pass
                continue

            if op.op == MemoryOperationType.DELETE:
                if not op.target_memory_id:
                    continue
                target = await self._store.get(op.target_memory_id)
                if target is None:
                    continue

                target_scope_id = op.target_scope_id or target.user_id
                if target_scope_id == global_scope_id and not allow_global_writes:
                    continue
                if target.user_id != target_scope_id:
                    continue

                target.deactivate()
                ok = await self._store.update(target)
                if ok and self._indexer is not None:
                    try:
                        await self._indexer.index_item(target, scope_id=target.user_id)
                    except Exception:
                        pass
                continue


