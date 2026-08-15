from __future__ import annotations

"""
Memory service.

This service owns memory-store interactions and optional indexing side-effects.
It exists to keep `ctxforge` thin and to consolidate memory behaviors
behind a stable API.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from ctxforge.config.base import EngineConfig
from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.core.memory_index import MemoryIndex, MemoryIndexEntry
from ctxforge.core.sufficiency import ProgressiveRetrievalStats, SufficiencyResult
from ctxforge.engine.services.headline_service import HeadlineService
from ctxforge.engine.services.sufficiency_service import SufficiencyService
from ctxforge.protocols.retriever import IRetriever, RetrievalConfig
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.retrieval.indexers.memory import MemoryIndexer
from ctxforge.retrieval.ranking import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


def _format_indexing_exception(e: Exception) -> str:
    """
    Best-effort formatting for index-time exceptions.

    This is especially useful for OpenAI/Azure OpenAI SDK errors where `str(e)`
    can be a generic message like "Connection error." while the request URL is
    available on the exception.
    """
    parts = [f"{type(e).__name__}: {e}"]

    req = getattr(e, "request", None)
    if req is not None:
        try:
            method = getattr(req, "method", None)
            url = getattr(req, "url", None)
            if method or url:
                parts.append(f"request={method} {url}")
        except Exception:
            pass

    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            status = getattr(resp, "status_code", None)
            url = getattr(resp, "url", None)
            if status or url:
                parts.append(f"response={status} {url}")
        except Exception:
            pass

    return " | ".join([p for p in parts if p])


class MemoryService:
    """Owns memory store + optional indexer and provides memory CRUD operations."""

    def __init__(
        self,
        *,
        config: Optional[EngineConfig] = None,
        memory_store: IMemoryStore,
        memory_indexer: Optional[MemoryIndexer] = None,
        memory_retriever_provider: Optional[Callable[[], Optional[IRetriever]]] = None,
        sufficiency_service: Optional[SufficiencyService] = None,
    ):
        self._cfg = config
        self._store = memory_store
        self._indexer = memory_indexer
        self._get_retriever = memory_retriever_provider or (lambda: None)
        self._sufficiency = sufficiency_service

    async def add(self, memory: MemoryItem) -> str:
        memory_id = await self._store.add(memory)
        if self._indexer is not None:
            try:
                await self._indexer.index_item(memory, scope_id=memory.user_id)
            except Exception as e:
                logger.warning(f"Memory indexing failed for {memory_id}: {_format_indexing_exception(e)}")
        return memory_id

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        return await self._store.get(memory_id)

    async def update(self, memory: MemoryItem) -> bool:
        ok = await self._store.update(memory)
        if ok and self._indexer is not None:
            try:
                await self._indexer.index_item(memory, scope_id=memory.user_id)
            except Exception as e:
                logger.warning(f"Memory re-indexing failed for {memory.memory_id}: {_format_indexing_exception(e)}")
        return ok

    async def delete(self, memory_id: str) -> bool:
        mem = await self._store.get(memory_id)
        ok = await self._store.delete(memory_id)
        if ok and mem is not None and self._indexer is not None:
            try:
                await self._indexer.remove_item(memory_id, scope_id=mem.user_id)
            except Exception as e:
                logger.warning(f"Memory de-indexing failed for {memory_id}: {_format_indexing_exception(e)}")
        return ok

    async def delete_all_user_memories(
        self,
        *,
        user_id: str,
        include_inactive: bool = True,
        batch_size: int = 1000,
    ) -> int:
        deleted = 0
        while True:
            items = await self._store.get_by_user(
                user_id,
                limit=batch_size,
                include_inactive=include_inactive,
            )
            if not items:
                break

            for mem in items:
                if mem.memory_id:
                    ok = await self._store.delete(mem.memory_id)
                    if ok:
                        deleted += 1

            if len(items) < batch_size:
                break

        if self._indexer is not None:
            try:
                await self._indexer.remove_all(scope_id=user_id)
            except Exception as e:
                logger.warning(f"Memory index clear failed for user_id={user_id}: {_format_indexing_exception(e)}")

        return deleted

    async def deactivate(self, memory_id: str) -> bool:
        memory = await self._store.get(memory_id)
        if memory is None:
            return False
        memory.deactivate()
        return await self.update(memory)

    async def get_by_user(self, *, user_id: str, limit: int = 100) -> List[MemoryItem]:
        return await self._store.get_by_user(user_id, limit)

    async def search_by_query(self, query: MemoryQuery) -> List[MemoryItem]:
        """
        Search memories using a full MemoryQuery.

        Bypasses the retriever and scopes — goes directly to the store.
        Useful for filtered queries (by tags, types, confidence, etc.)
        that don't need semantic retrieval or global scope merging.
        """
        return await self._store.search(query)

    async def search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """
        Search relevant memories for a user.

        Behavior:
        - If a memory retriever is configured, use it.
        - If scopes enable global retrieval, merge user + global results (dedupe by memory_id).
        - If no config is provided, this falls back to store search (single-scope).
        """
        try:
            retriever = self._get_retriever()
            cfg = self._cfg

            if cfg is None:
                return await self._store.search(MemoryQuery(user_id=user_id, query_text=query, limit=limit))

            scopes = getattr(cfg, "scopes", None)
            enable_global = bool(getattr(scopes, "enable_global", False))
            global_scope_id = str(getattr(scopes, "global_scope_id", "global") or "global")
            global_limit = int(getattr(scopes, "global_retrieval_limit", 0) or 0)
            global_weight = float(getattr(scopes, "global_score_weight", 0.8) or 0.8)

            if user_id == global_scope_id:
                enable_global = False

            if retriever is not None:
                try:
                    user_results = await retriever.retrieve(
                        query=query,
                        user_id=user_id,
                        config=RetrievalConfig(limit=limit),
                    )

                    if not enable_global or global_limit <= 0:
                        return [r.memory for r in user_results]

                    global_results = await retriever.retrieve(
                        query=query,
                        user_id=global_scope_id,
                        config=RetrievalConfig(limit=min(global_limit, limit)),
                    )

                    scored: Dict[str, float] = {}
                    memories: Dict[str, MemoryItem] = {}
                    is_user_scope: Dict[str, bool] = {}

                    for r in user_results:
                        mid = r.memory.memory_id
                        scored[mid] = max(scored.get(mid, 0.0), float(r.score))
                        memories[mid] = r.memory
                        is_user_scope[mid] = True

                    for r in global_results:
                        mid = r.memory.memory_id
                        weighted = float(r.score) * global_weight
                        if mid in scored:
                            continue
                        scored[mid] = max(scored.get(mid, 0.0), weighted)
                        memories[mid] = r.memory
                        is_user_scope[mid] = False

                    merged_ids = sorted(
                        memories.keys(),
                        key=lambda mid: (
                            scored.get(mid, 0.0),
                            1 if is_user_scope.get(mid, False) else 0,
                        ),
                        reverse=True,
                    )
                    return [memories[mid] for mid in merged_ids[:limit]]
                except Exception as e:
                    # Fail open: if embedding/vector retrieval is temporarily down (e.g. Azure 5xx),
                    # fall back to store-based search instead of returning no memories.
                    logger.warning(
                        f"Memory retriever search failed; falling back to store search: {_format_indexing_exception(e)}"
                    )
                    retriever = None

            user_items = await self._store.search(MemoryQuery(user_id=user_id, query_text=query, limit=limit))
            if not enable_global or global_limit <= 0:
                return user_items

            global_items = await self._store.search(
                MemoryQuery(
                    user_id=global_scope_id,
                    query_text=query,
                    limit=min(global_limit, limit),
                )
            )

            merged: List[MemoryItem] = []
            seen = set()
            for m in user_items + global_items:
                if m.memory_id in seen:
                    continue
                seen.add(m.memory_id)
                merged.append(m)
                if len(merged) >= limit:
                    break
            return merged
        except Exception as e:
            logger.warning(f"Failed to search memories: {_format_indexing_exception(e)}")
            return []

    async def search_as_index(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        headline_service: Optional["HeadlineService"] = None,
    ) -> "MemoryIndex":
        """
        Search memories and return as progressive disclosure index.

        Args:
            user_id: User to search for
            query: Search query
            limit: Maximum results
            headline_service: Optional service to generate headlines for
                memories that don't have them

        Returns:
            MemoryIndex with headline entries for progressive disclosure
        """
        memories = await self.search(user_id=user_id, query=query, limit=limit)

        index = MemoryIndex(total_memories=len(memories))

        for memory in memories:
            # Generate headline if service provided and memory lacks one
            if headline_service and not memory.has_headline():
                try:
                    await headline_service.generate_and_update(memory)
                    # Optionally persist the updated memory with headline
                    await self.update(memory)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate headline for {memory.memory_id}: {e}"
                    )

            entry = MemoryIndexEntry.from_memory(memory)
            index.add(entry)

        return index

    async def search_with_sufficiency(
        self,
        *,
        user_id: str,
        query: str,
        initial_limit: int = 5,
        max_limit: int = 20,
    ) -> Tuple[List[MemoryItem], SufficiencyResult, ProgressiveRetrievalStats]:
        """
        Search with sufficiency checking - progressively retrieve until sufficient.

        Uses the sufficiency service to evaluate if retrieved content adequately
        answers the query. If not, fetches more results iteratively.

        Args:
            user_id: User to search for
            query: Search query
            initial_limit: Starting number of results
            max_limit: Maximum results to fetch

        Returns:
            Tuple of (memories, sufficiency_result, retrieval_stats)
        """
        # If no sufficiency service, fall back to normal search
        if self._sufficiency is None:
            memories = await self.search(user_id=user_id, query=query, limit=initial_limit)
            return (
                memories,
                SufficiencyResult.enough("Sufficiency checking not configured"),
                ProgressiveRetrievalStats(
                    total_iterations=1,
                    initial_results=len(memories),
                    final_results=len(memories),
                ),
            )

        # Define retriever function for progressive retrieval
        async def retriever(limit: int) -> List[MemoryItem]:
            return await self.search(user_id=user_id, query=query, limit=limit)

        # Define formatter for memories
        def formatter(memories: List[MemoryItem]) -> str:
            if not memories:
                return ""
            return "\n".join(
                f"{i}. {m.content[:500]}" for i, m in enumerate(memories, 1)
            )

        # Use progressive retrieval
        return await self._sufficiency.progressive_retrieve(
            query=query,
            retriever_func=retriever,
            formatter_func=formatter,
            initial_limit=initial_limit,
            max_limit=max_limit,
        )

    # ------------------------------------------------------------------
    # Multi-view search (keyword + hybrid)
    # ------------------------------------------------------------------

    async def keyword_search(
        self,
        *,
        user_id: str,
        keywords: List[str],
        limit: int = 10,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> List[MemoryItem]:
        """Search memories by keyword overlap.

        Delegates to the underlying store's ``keyword_search`` method.
        """
        return await self._store.keyword_search(
            user_id=user_id,
            keywords=keywords,
            limit=limit,
            filters=filters,
        )

    async def hybrid_search(
        self,
        *,
        user_id: str,
        query: str,
        keywords: Optional[List[str]] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        limit: int = 10,
    ) -> List[MemoryItem]:
        """Merge semantic and keyword search using Reciprocal Rank Fusion.

        When ``keywords`` are provided the method runs both a semantic search
        and a keyword search in parallel, then fuses the two ranked lists.
        When no keywords are given it falls back to a plain semantic search.
        """
        semantic_results = await self.search(
            user_id=user_id, query=query, limit=limit,
        )

        if not keywords:
            return semantic_results

        kw_results = await self.keyword_search(
            user_id=user_id, keywords=keywords, limit=limit, filters=filters,
        )

        if not kw_results:
            return semantic_results

        return reciprocal_rank_fusion(
            ranked_lists=[semantic_results, kw_results],
            key_fn=lambda m: m.memory_id,
            limit=limit,
        )
