import pytest

from ctxforge.core.expertise import ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.protocols.context import ContextRetrievalResult, IContextReranker


class ReverseScoreReranker(IContextReranker):
    @property
    def name(self) -> str:
        return "reverse_score"

    async def rerank(self, query: str, results, top_k=None):
        reranked = sorted(results, key=lambda r: r.score)  # ascending
        return reranked[:top_k] if top_k else reranked


@pytest.mark.asyncio
async def test_same_reranker_reorders_memory_and_expertise_results():
    reranker = ReverseScoreReranker()

    # Memory results
    m1 = MemoryItem(user_id="u1", content="A", type=MemoryType.SEMANTIC, source=MemorySource.USER_EXPLICIT)
    m2 = MemoryItem(user_id="u1", content="B", type=MemoryType.SEMANTIC, source=MemorySource.USER_EXPLICIT)
    mem_results = [
        ContextRetrievalResult(item=m1, score=0.9, retrieval_method="semantic", metadata={}),
        ContextRetrievalResult(item=m2, score=0.1, retrieval_method="semantic", metadata={}),
    ]
    mem_reranked = await reranker.rerank(query="q", results=mem_results)
    assert mem_reranked[0].item.content == "B"

    # Expertise results
    e1 = ExpertiseItem(
        item_id="e1",
        section=ExpertiseSection.STRATEGIES,
        content="X",
    )
    e2 = ExpertiseItem(
        item_id="e2",
        section=ExpertiseSection.STRATEGIES,
        content="Y",
    )
    exp_results = [
        ContextRetrievalResult(item=e1, score=0.8, retrieval_method="semantic", metadata={}),
        ContextRetrievalResult(item=e2, score=0.2, retrieval_method="semantic", metadata={}),
    ]
    exp_reranked = await reranker.rerank(query="q", results=exp_results)
    assert exp_reranked[0].item.content == "Y"

