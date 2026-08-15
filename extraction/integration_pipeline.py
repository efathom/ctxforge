"""
Multi-stage memory integration pipeline.

Implements a 5-stage pipeline (Detect -> Summarize -> Dedup -> Integrate -> Store)
that produces higher-quality memories by filtering non-preference feedback,
summarizing before storage, finding similar existing memories, and merging
new info with existing memories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.extraction.integration_config import IntegrationConfig, IntegrationResult
from ctxforge.protocols.update_planner import MemoryOperationType

if TYPE_CHECKING:
    from ctxforge.engine.services.preference_evolution_service import (
        PreferenceEvolutionService,
    )
    from ctxforge.protocols.extractor import ExtractionCandidate
    from ctxforge.protocols.llm import IEmbeddingProvider, ILLMProvider
    from ctxforge.protocols.storage import IMemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DETECT_PROMPT = """Analyze the following extracted information. Does it contain \
actionable personal preference, fact, or behavioral pattern worth remembering?

Information: {content}
Conversation context: {query}

Answer ONLY 'Yes' or 'No'. Answer 'No' for:
- Simple acknowledgments (ok, thanks, sure)
- Task-only content with no personal information
- Vague or unclear statements"""

SUMMARIZE_PROMPT = """Summarize the following personal information in one clear, \
self-contained sentence. Resolve any pronouns or ambiguous references using the \
conversation context.

Information: {content}
Context: {query}

Output only the summary sentence."""

INTEGRATE_PROMPT = """Create a concise, integrated summary combining these two \
pieces of information:

Existing memory: {existing}
New information: {new_info}

If the new information indicates a preference change, prioritize the NEW \
preference and note the change.
Provide a single coherent summary without redundancy."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class IntegrationStage(str, Enum):
    DETECT = "detect"
    SUMMARIZE = "summarize"
    DEDUP = "dedup"
    INTEGRATE = "integrate"
    STORE = "store"


@dataclass
class IntegrationContext:
    """Tracks state through the pipeline for one candidate."""

    candidate: Any  # ExtractionCandidate
    user_id: str
    query: str
    # Pipeline state
    is_actionable: bool = True
    summarized_content: Optional[str] = None
    similar_memory: Optional[MemoryItem] = None
    similarity_score: float = 0.0
    operation: MemoryOperationType = MemoryOperationType.ADD
    merged_content: Optional[str] = None
    stage_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MemoryIntegrationPipeline:
    """Multi-stage memory integration."""

    def __init__(
        self,
        llm: ILLMProvider,
        memory_store: IMemoryStore,
        embedding_provider: Optional[IEmbeddingProvider] = None,
        config: Optional[IntegrationConfig] = None,
        preference_evolution_service: Optional[PreferenceEvolutionService] = None,
    ):
        self._llm = llm
        self._memory_store = memory_store
        self._embedding_provider = embedding_provider
        self._config = config or IntegrationConfig()
        self._preference_evolution_service = preference_evolution_service

    async def process(
        self,
        candidates: List[ExtractionCandidate],
        user_id: str,
        query: str,
    ) -> List[IntegrationResult]:
        """Run all candidates through the 5-stage pipeline."""
        if not candidates:
            return []

        contexts = [
            IntegrationContext(candidate=c, user_id=user_id, query=query)
            for c in candidates
        ]

        # Stage 1: Detect
        contexts = await self._detect(contexts)
        contexts = [c for c in contexts if c.is_actionable]
        if not contexts:
            return []

        # Stage 2: Summarize
        contexts = await self._summarize(contexts)

        # Stage 3: Dedup
        contexts = await self._dedup(contexts)

        # Stage 4: Integrate
        contexts = await self._integrate(contexts)

        # Stage 5: Store
        return await self._store(contexts)

    # ------------------------------------------------------------------
    # Stage 1: Detect
    # ------------------------------------------------------------------

    async def _detect(
        self,
        contexts: List[IntegrationContext],
    ) -> List[IntegrationContext]:
        """Does each candidate contain actionable preference/personal info?"""
        for ctx in contexts:
            # Skip detection for high-confidence extractions
            if (
                self._config.skip_detect_for_high_confidence
                and getattr(ctx.candidate, "confidence", 0) > 0.9
            ):
                ctx.is_actionable = True
                ctx.stage_metadata["detect_skipped"] = True
                continue

            prompt = DETECT_PROMPT.format(
                content=ctx.candidate.content,
                query=ctx.query,
            )
            try:
                response = await self._llm.generate(
                    prompt,
                    model=self._config.model,
                    max_tokens=10,
                    temperature=0.0,
                )
                answer = response.content.strip().lower()
                ctx.is_actionable = answer.startswith("yes")
                ctx.stage_metadata["detect_answer"] = answer
            except Exception as e:
                logger.warning("Integration detect failed: %s", e)
                ctx.is_actionable = True  # fail-open

        return contexts

    # ------------------------------------------------------------------
    # Stage 2: Summarize
    # ------------------------------------------------------------------

    async def _summarize(
        self,
        contexts: List[IntegrationContext],
    ) -> List[IntegrationContext]:
        """Extract clean preference statement from each candidate."""
        for ctx in contexts:
            prompt = SUMMARIZE_PROMPT.format(
                content=ctx.candidate.content,
                query=ctx.query,
            )
            try:
                response = await self._llm.generate(
                    prompt,
                    model=self._config.model,
                    max_tokens=100,
                    temperature=0.0,
                )
                ctx.summarized_content = response.content.strip()
            except Exception as e:
                logger.warning("Integration summarize failed: %s", e)
                ctx.summarized_content = ctx.candidate.content

        return contexts

    # ------------------------------------------------------------------
    # Stage 3: Dedup
    # ------------------------------------------------------------------

    async def _dedup(
        self,
        contexts: List[IntegrationContext],
    ) -> List[IntegrationContext]:
        """Find similar existing memories for each candidate."""
        for ctx in contexts:
            search_text = ctx.summarized_content or ctx.candidate.content
            try:
                existing = await self._memory_store.search(
                    MemoryQuery(
                        user_id=ctx.user_id,
                        query_text=search_text,
                        limit=self._config.max_candidates_per_search,
                    )
                )
            except Exception as e:
                logger.warning("Integration dedup search failed: %s", e)
                existing = []

            if not existing:
                ctx.operation = MemoryOperationType.ADD
                continue

            # Pick the best match by simple text overlap or embedding similarity
            best_match = existing[0]
            best_score = self._text_similarity(
                search_text, best_match.content
            )

            for mem in existing[1:]:
                score = self._text_similarity(search_text, mem.content)
                if score > best_score:
                    best_match = mem
                    best_score = score

            ctx.similarity_score = best_score

            if best_score >= self._config.similarity_threshold:
                ctx.similar_memory = best_match
                ctx.operation = MemoryOperationType.UPDATE
            else:
                ctx.operation = MemoryOperationType.ADD

        return contexts

    # ------------------------------------------------------------------
    # Stage 4: Integrate
    # ------------------------------------------------------------------

    async def _integrate(
        self,
        contexts: List[IntegrationContext],
    ) -> List[IntegrationContext]:
        """Merge new info with existing memory or pass through for ADD."""
        for ctx in contexts:
            if ctx.operation != MemoryOperationType.UPDATE or ctx.similar_memory is None:
                ctx.merged_content = ctx.summarized_content or ctx.candidate.content
                continue

            # Check for preference evolution before merging
            if self._preference_evolution_service is not None:
                new_text = ctx.summarized_content or ctx.candidate.content
                change = await self._preference_evolution_service.detect_preference_change(
                    new_content=new_text,
                    existing_memory=ctx.similar_memory,
                    query=ctx.query,
                )
                if change is not None:
                    ctx.stage_metadata["preference_change"] = True
                    ctx.stage_metadata["change_type"] = change.change_type

            # Merge via LLM
            new_text = ctx.summarized_content or ctx.candidate.content
            prompt = INTEGRATE_PROMPT.format(
                existing=ctx.similar_memory.content,
                new_info=new_text,
            )
            try:
                response = await self._llm.generate(
                    prompt,
                    model=self._config.model,
                    max_tokens=200,
                    temperature=0.0,
                )
                ctx.merged_content = response.content.strip()
            except Exception as e:
                logger.warning("Integration merge failed: %s", e)
                ctx.merged_content = new_text

        return contexts

    # ------------------------------------------------------------------
    # Stage 5: Store
    # ------------------------------------------------------------------

    async def _store(
        self,
        contexts: List[IntegrationContext],
    ) -> List[IntegrationResult]:
        """Apply ADD/UPDATE operations to memory store."""
        results: List[IntegrationResult] = []

        for ctx in contexts:
            final_content = ctx.merged_content or ctx.summarized_content or ctx.candidate.content
            preference_changed = ctx.stage_metadata.get("preference_change", False)

            if ctx.operation == MemoryOperationType.UPDATE and ctx.similar_memory is not None:
                # Update existing memory
                old_memory = ctx.similar_memory
                old_memory.update_content(final_content)
                if ctx.summarized_content:
                    old_memory.restatement = ctx.summarized_content
                try:
                    await self._memory_store.update(old_memory)

                    # Apply preference evolution if detected
                    if (
                        preference_changed
                        and self._preference_evolution_service is not None
                    ):
                        new_mem = ctx.candidate.to_memory_item(ctx.user_id)
                        new_mem.content = final_content
                        change = ctx.stage_metadata.get("preference_change_obj")
                        if change is not None:
                            await self._preference_evolution_service.apply_preference_change(
                                change=change,
                                new_memory=new_mem,
                                old_memory=old_memory,
                            )
                except Exception as e:
                    logger.warning("Integration store (update) failed: %s", e)

                results.append(IntegrationResult(
                    memory_item=old_memory,
                    operation="update",
                    was_actionable=True,
                    similarity_score=ctx.similarity_score,
                    preference_changed=preference_changed,
                    stage_metadata=ctx.stage_metadata,
                ))
            else:
                # Add new memory
                new_mem = ctx.candidate.to_memory_item(ctx.user_id)
                new_mem.content = final_content
                if ctx.summarized_content:
                    new_mem.restatement = ctx.summarized_content
                try:
                    await self._memory_store.add(new_mem)
                except Exception as e:
                    logger.warning("Integration store (add) failed: %s", e)

                results.append(IntegrationResult(
                    memory_item=new_mem,
                    operation="add",
                    was_actionable=True,
                    similarity_score=ctx.similarity_score,
                    preference_changed=preference_changed,
                    stage_metadata=ctx.stage_metadata,
                ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple word-overlap Jaccard similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
