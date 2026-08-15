from __future__ import annotations

"""
Expertise service.

This service owns expertise subsystem logic:
- CRUD for Expertise knowledge bases
- adding items
- retrieval of items (via expertise retriever, with store fallback)
- reflection/curation flow on feedback
- "prepare context with expertise" helper

It exists to keep `ctxforge` thin and to consolidate expertise behavior
behind a stable API.
"""

import logging
import time
from typing import Any, Awaitable, Callable, List, Optional

from ctxforge.config.base import EngineConfig
from ctxforge.core.context import Context
from ctxforge.core.expertise import (
    CompletedTurn,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.protocols.expertise import ICurator, IExpertiseRetriever, IExpertiseStore, IReflector

logger = logging.getLogger(__name__)


class ExpertiseService:
    """Owns expertise dependencies and provides expertise subsystem entrypoints."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        expertise_store_provider: Callable[[], Optional[IExpertiseStore]],
        expertise_retriever_provider: Callable[[], Optional[IExpertiseRetriever]],
        reflector_provider: Callable[[], Optional[IReflector]],
        curator_provider: Callable[[], Optional[ICurator]],
        memory_service_provider: Callable[[], Any],
        # Kept for backward compatibility but no longer used internally.
        record_turn: Optional[Callable[..., Awaitable[None]]] = None,
        prepare_context: Optional[Callable[..., Awaitable[Context]]] = None,
    ):
        self._cfg = config
        self._get_store = expertise_store_provider
        self._get_retriever = expertise_retriever_provider
        self._get_reflector = reflector_provider
        self._get_curator = curator_provider
        self._get_memory_service = memory_service_provider
        self._record_turn = record_turn
        self._prepare_context = prepare_context

    async def create_expertise(
        self,
        *,
        expertise_id: str,
        name: str,
        domain: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Expertise:
        expertise = Expertise(
            expertise_id=expertise_id,
            name=name,
            domain=domain,
            description=description or "",
        )

        store = self._get_store()
        if store is not None:
            await store.save(expertise)
            logger.info(f"Created expertise: {expertise_id}")
        return expertise

    async def load_expertise(self, *, expertise_id: str) -> Optional[Expertise]:
        store = self._get_store()
        if store is None:
            return None
        return await store.load(expertise_id)

    async def save_expertise(self, *, expertise: Expertise) -> None:
        store = self._get_store()
        if store is None:
            return
        await store.save(expertise)

    async def add_expertise_item(
        self,
        *,
        expertise_id: str,
        section: ExpertiseSection,
        content: str,
        source: Optional[str] = None,
    ) -> Optional[ExpertiseItem]:
        expertise = await self.load_expertise(expertise_id=expertise_id)
        if not expertise:
            return None

        item = expertise.add_item(
            section=section,
            content=content,
            source=source or "user",
        )
        await self.save_expertise(expertise=expertise)
        return item

    async def retrieve_expertise_items(
        self,
        *,
        expertise_id: str,
        query: str,
        limit: int = 5,
    ) -> List[ExpertiseItem]:
        retriever = self._get_retriever()
        store = self._get_store()

        if retriever is None:
            if store is None:
                return []
            expertise = await store.load(expertise_id)
            if not expertise:
                return []
            return expertise.active_items[:limit]

        try:
            return await retriever.retrieve_items(query=query, scope_id=expertise_id, limit=limit)
        except Exception as e:
            logger.warning(f"Expertise retrieval failed (falling back to store): {e}")
            if store is None:
                return []
            expertise = await store.load(expertise_id)
            if not expertise:
                return []
            return expertise.active_items[:limit]

    async def prepare_context_with_expertise(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        expertise_id: str,
        max_expertise_items: int = 5,
        **kwargs,
    ) -> Context:
        context = await self._prepare_context(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            **kwargs,
        )

        expertise_items = await self.retrieve_expertise_items(
            expertise_id=expertise_id,
            query=user_input,
            limit=max_expertise_items,
        )

        context.expertise_items = expertise_items
        context.expertise_id = expertise_id
        context.expertise_items_used = [item.item_id for item in expertise_items]
        context.metadata["expertise_id"] = expertise_id
        context.metadata["expertise_item_count"] = len(expertise_items)
        return context

    async def reflect_and_curate(
        self,
        *,
        session_id: str,
        user_input: str,
        assistant_response: str,
        expertise_items_used: List[str],
        outcome: TurnOutcome,
        expertise_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Optional[Expertise]:
        """
        Run expertise reflection and optional curation (without recording the turn).

        This is the pure expertise-feedback path. The caller is responsible for
        recording the turn separately via ``record_turn``.

        Returns:
            The updated ``Expertise`` if curation was performed, ``None`` otherwise.
        """
        reflector = self._get_reflector()
        if reflector is None:
            return None

        turn = CompletedTurn(
            turn_id=f"{session_id}-{int(time.time())}",
            user_input=user_input,
            assistant_response=assistant_response,
            expected_output=ground_truth,
        )

        items_used: List[ExpertiseItem] = []
        store = self._get_store()
        if expertise_id and store is not None:
            expertise = await store.load(expertise_id)
            if expertise:
                for item_id in expertise_items_used:
                    item = expertise.get_item(item_id)
                    if item:
                        items_used.append(item)

        reflection = await reflector.reflect(
            turn=turn,
            items_used=items_used,
            outcome=outcome,
        )

        if not expertise_id or store is None:
            return None

        expertise = await store.load(expertise_id)
        if not expertise:
            return None

        for item_id, feedback in reflection.item_feedback.items():
            item = expertise.get_item(item_id)
            if item:
                if feedback == UsageFeedback.HELPFUL:
                    item.increment_helpful()
                elif feedback == UsageFeedback.HARMFUL:
                    item.increment_harmful()

        await store.save(expertise)

        curator = self._get_curator()
        if curator is not None and (reflection.suggested_additions or reflection.suggested_removals):
            expertise, plan = await curator.curate(
                expertise=expertise,
                reflection=reflection,
                usage_stats={"session_id": session_id},
            )
            await store.save(expertise)
            logger.debug(f"Curated expertise with {plan.operation_count} operations")
            return expertise

        return expertise

    async def record_turn_with_feedback(
        self,
        *,
        session_id: str,
        user_id: str,
        user_input: str,
        assistant_response: str,
        expertise_items_used: List[str],
        outcome: TurnOutcome,
        expertise_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Optional[Expertise]:
        """
        Record a completed turn and apply expertise feedback/curation (best-effort).

        .. deprecated::
            Use ``engine.record_turn(expertise_items_used=..., outcome=...)`` instead.
            This method is kept for backward compatibility and delegates to
            ``_record_turn`` + ``reflect_and_curate``.
        """
        if self._record_turn is not None:
            await self._record_turn(
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
                assistant_response=assistant_response,
            )

        return await self.reflect_and_curate(
            session_id=session_id,
            user_input=user_input,
            assistant_response=assistant_response,
            expertise_items_used=expertise_items_used,
            outcome=outcome,
            expertise_id=expertise_id,
            ground_truth=ground_truth,
        )


