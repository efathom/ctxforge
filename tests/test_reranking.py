import pytest

from ctxforge.core.memory import MemoryFactory
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.protocols.retriever import IRetriever, RetrievalConfig, RetrievalResult
from ctxforge.retrieval.rerankers.llm import LLMReranker
from ctxforge.retrieval.retrievers.reranking import RerankingRetriever


class _FakeRetriever(IRetriever):
    def __init__(self, results):
        self._results = results

    @property
    def name(self) -> str:
        return "fake"

    async def retrieve(self, query: str, user_id: str, config=None):
        return list(self._results)

    async def retrieve_by_embedding(self, embedding, user_id: str, config=None):
        return list(self._results)

    async def retrieve_related(self, memory_id: str, user_id: str, limit: int = 5):
        return list(self._results)[:limit]


@pytest.mark.asyncio
async def test_llm_reranker_reorders_results():
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(['[{"id": "m2", "score": 0.99}, {"id": "m1", "score": 0.1}]'])

    m1 = MemoryFactory.semantic_memory(user_id="u", content="User likes apples")
    m1.memory_id = "m1"
    m2 = MemoryFactory.semantic_memory(user_id="u", content="User likes bananas")
    m2.memory_id = "m2"

    results = [
        RetrievalResult(memory=m1, score=0.6, retrieval_method="semantic"),
        RetrievalResult(memory=m2, score=0.5, retrieval_method="semantic"),
    ]

    reranker = LLMReranker(llm_provider=llm, model="mock-model")
    out = await reranker.rerank("bananas", results)

    assert [r.memory.memory_id for r in out] == ["m2", "m1"]


@pytest.mark.asyncio
async def test_reranking_retriever_wraps_base():
    llm = MockLLMProvider(latency_ms=0)
    llm.set_responses(['[{"id": "m2", "score": 0.9}, {"id": "m1", "score": 0.1}]'])

    m1 = MemoryFactory.semantic_memory(user_id="u", content="alpha")
    m1.memory_id = "m1"
    m2 = MemoryFactory.semantic_memory(user_id="u", content="beta")
    m2.memory_id = "m2"

    base_results = [
        RetrievalResult(memory=m1, score=0.6, retrieval_method="semantic"),
        RetrievalResult(memory=m2, score=0.5, retrieval_method="semantic"),
    ]

    base = _FakeRetriever(base_results)
    reranker = LLMReranker(llm_provider=llm, model="mock-model")
    wrapped = RerankingRetriever(base=base, reranker=reranker)

    out = await wrapped.retrieve("beta", "u", config=RetrievalConfig(limit=2))
    assert [r.memory.memory_id for r in out] == ["m2", "m1"]


