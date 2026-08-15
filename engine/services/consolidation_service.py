"""
Memory consolidation service.

Performs background maintenance on the memory store:
- **Decay**: Reduce importance of old memories over time.
- **Merge**: Mark near-duplicate memories as superseded.
- **Prune**: Soft-delete memories whose importance falls below a threshold.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List

from ctxforge.config.base import ConsolidationConfig
from ctxforge.core.memory import MemoryItem
from ctxforge.protocols.llm import IEmbeddingProvider
from ctxforge.protocols.storage import IMemoryStore
from ctxforge.utils.math import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    """Summary of a consolidation run."""

    decayed: int = 0
    merged: int = 0
    pruned: int = 0
    errors: List[str] = field(default_factory=list)


class ConsolidationService:
    """Three-pass memory consolidation: decay, merge, prune."""

    def __init__(
        self,
        memory_store: IMemoryStore,
        embedding_provider: IEmbeddingProvider,
        config: ConsolidationConfig,
    ):
        self._store = memory_store
        self._embedder = embedding_provider
        self._cfg = config

    async def consolidate(self, user_id: str) -> ConsolidationReport:
        """Run the full three-pass consolidation pipeline for a user."""
        report = ConsolidationReport()

        memories = await self._store.get_by_user(user_id, limit=10000, include_inactive=False)
        if not memories:
            return report

        # Pass 1: Decay
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=self._cfg.max_age_days)

        for mem in memories:
            if mem.created_at < cutoff and mem.importance > 0.0:
                new_importance = mem.importance * self._cfg.decay_factor
                if new_importance != mem.importance:
                    mem.importance = new_importance
                    mem.updated_at = now
                    await self._store.update(mem)
                    report.decayed += 1

        # Pass 2: Merge near-duplicates
        embeddings = await self._ensure_embeddings(memories)
        merged_ids: Dict[str, bool] = {}

        for i, mem_a in enumerate(memories):
            if mem_a.memory_id in merged_ids:
                continue
            if mem_a.superseded_by is not None:
                continue
            emb_a = embeddings.get(mem_a.memory_id)
            if emb_a is None:
                continue

            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                if mem_b.memory_id in merged_ids:
                    continue
                if mem_b.superseded_by is not None:
                    continue
                emb_b = embeddings.get(mem_b.memory_id)
                if emb_b is None:
                    continue

                sim = cosine_similarity(emb_a, emb_b)
                if sim >= self._cfg.merge_similarity_threshold:
                    # Keep the higher-importance one, supersede the other.
                    if mem_a.importance >= mem_b.importance:
                        winner, loser = mem_a, mem_b
                    else:
                        winner, loser = mem_b, mem_a

                    loser.superseded_by = winner.memory_id
                    loser.is_active = False
                    loser.updated_at = now
                    await self._store.update(loser)
                    merged_ids[loser.memory_id] = True
                    report.merged += 1

        # Pass 3: Prune low-importance memories
        for mem in memories:
            if mem.memory_id in merged_ids:
                continue
            if mem.superseded_by is not None:
                continue
            if mem.importance < self._cfg.min_importance:
                mem.superseded_by = "__pruned__"
                mem.is_active = False
                mem.updated_at = now
                await self._store.update(mem)
                report.pruned += 1

        return report

    async def _ensure_embeddings(
        self, memories: List[MemoryItem]
    ) -> Dict[str, List[float]]:
        """Build a mapping of memory_id -> embedding, generating as needed."""
        result: Dict[str, List[float]] = {}
        to_embed: List[MemoryItem] = []

        for mem in memories:
            if mem.embedding:
                result[mem.memory_id] = mem.embedding
            else:
                to_embed.append(mem)

        if to_embed:
            texts = [m.restatement or m.content for m in to_embed]
            resp = await self._embedder.embed(texts)
            for mem, emb in zip(to_embed, resp.embeddings, strict=False):
                result[mem.memory_id] = emb
                mem.embedding = emb

        return result
