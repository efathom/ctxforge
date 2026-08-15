"""
LLM-based rerankers.
"""

import json
from typing import Any, Dict, Generic, List, Optional, TypeVar

from ctxforge.engine.registry import registry
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.context import ContextRetrievalResult, IContextItem, IContextReranker
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.protocols.retriever import IReranker, RetrievalResult

DEFAULT_LLM_RERANK_PROMPT = """You are a ranking model. Given a query and a set of candidate memories, assign a relevance score to each candidate.

Return ONLY valid JSON as an array:
[
  {"id": "<memory_id>", "score": 0.0},
  {"id": "<memory_id>", "score": 0.0}
]

Rules:
- score is between 0.0 and 1.0
- higher score = more relevant to the query
- include every candidate id exactly once
"""


TItem = TypeVar("TItem", bound=IContextItem)


class LLMContextReranker(Generic[TItem], IContextReranker[TItem]):
    """
    LLM-based reranker for generic `ContextRetrievalResult[TItem]`.

    Works for any context item that exposes `item_id` and `content` (via IContextItem),
    e.g. MemoryItem and ExpertiseItem.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_LLM_RERANK_PROMPT,
    ):
        self._llm = llm_provider
        self._model = model
        self._system_prompt = system_prompt

    @property
    def name(self) -> str:
        return "llm"

    async def rerank(
        self,
        query: str,
        results: List[ContextRetrievalResult[TItem]],
        top_k: Optional[int] = None,
    ) -> List[ContextRetrievalResult[TItem]]:
        if not results:
            return results

        candidates = results[:top_k] if top_k else list(results)

        user_content_lines = [
            f"Query: {query}",
            "",
            "Candidates:",
        ]
        for r in candidates:
            user_content_lines.append(f"- id: {r.item.item_id}")
            user_content_lines.append(f"  text: {r.item.content}")

        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content="\n".join(user_content_lines)),
        ]

        resp = await self._llm.chat(messages=messages, model=self._model, temperature=0.0, max_tokens=800)
        json_str = extract_json_from_text(resp.content or "")
        if not json_str:
            return candidates

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return candidates

        if not isinstance(data, list):
            return candidates

        score_by_id: Dict[str, float] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            score = item.get("score")
            if isinstance(mid, str):
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                score_by_id[mid] = max(0.0, min(1.0, score_f))

        updated: List[ContextRetrievalResult[TItem]] = []
        for r in candidates:
            new_score = score_by_id.get(r.item.item_id, r.score)
            updated.append(
                ContextRetrievalResult(
                    item=r.item,
                    score=new_score,
                    retrieval_method=f"{r.retrieval_method}+llm",
                    metadata={
                        **(r.metadata or {}),
                        "llm_rerank_model": self._model or self._llm.default_model,
                    },
                )
            )

        updated.sort(key=lambda x: x.score, reverse=True)
        return updated[:top_k] if top_k else updated


@registry.register_reranker("llm")
class LLMReranker(IReranker):
    def __init__(
        self,
        llm_provider: ILLMProvider,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_LLM_RERANK_PROMPT,
    ):
        self._llm = llm_provider
        self._model = model
        self._system_prompt = system_prompt

    @property
    def name(self) -> str:
        return "llm"

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if not results:
            return results

        ctx_reranker: LLMContextReranker = LLMContextReranker(
            llm_provider=self._llm,
            model=self._model,
            system_prompt=self._system_prompt,
        )
        ctx_results = [
            ContextRetrievalResult(
                item=r.memory,
                score=r.score,
                retrieval_method=r.retrieval_method,
                metadata=r.metadata or {},
            )
            for r in results
        ]
        reranked_ctx = await ctx_reranker.rerank(query=query, results=ctx_results, top_k=top_k)
        return [
            RetrievalResult(
                memory=r.item,
                score=r.score,
                retrieval_method=r.retrieval_method,
                metadata=r.metadata or {},
            )
            for r in reranked_ctx
        ]


class LLMExpertiseReranker:
    """
    LLM-based reranker for expertise retrieval results.

    Note: Expertise retrieval results are not `RetrievalResult` (memory) objects,
    but the reranker interface is compatible by duck-typing (rerank(query, results, top_k)).
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_LLM_RERANK_PROMPT,
    ):
        self._llm = llm_provider
        self._model = model
        self._system_prompt = system_prompt

    @property
    def name(self) -> str:
        return "llm"

    async def rerank(self, query: str, results: List[Any], top_k: Optional[int] = None) -> List[Any]:
        # Keep this class for backward compatibility with older wiring that expected a
        # duck-typed expertise reranker. Delegate to the unified reranker.
        ctx_reranker: LLMContextReranker = LLMContextReranker(
            llm_provider=self._llm,
            model=self._model,
            system_prompt=self._system_prompt,
        )
        return await ctx_reranker.rerank(query=query, results=results, top_k=top_k)


