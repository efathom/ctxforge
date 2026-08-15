import pytest

from ctxforge.config.base import EngineConfig
from ctxforge.core.memory import MemoryItem, MemoryQuery, MemoryType
from ctxforge.engine.services.memory_service import MemoryService


class _FakeStore:
    def __init__(self, items):
        self._items = list(items)

    async def search(self, query: MemoryQuery):
        return self._items[: query.limit]

    async def add(self, memory: MemoryItem) -> str:  # pragma: no cover
        return memory.memory_id

    async def get(self, memory_id: str):  # pragma: no cover
        return None

    async def update(self, memory: MemoryItem) -> bool:  # pragma: no cover
        return True

    async def delete(self, memory_id: str) -> bool:  # pragma: no cover
        return True

    async def get_by_user(self, user_id: str, limit: int = 100, include_inactive: bool = False):  # pragma: no cover
        return []


class _FailingRetriever:
    @property
    def name(self) -> str:  # pragma: no cover
        return "failing"

    async def retrieve(self, *args, **kwargs):
        class InternalServerError(Exception):
            pass

        raise InternalServerError("Error code: 500 - internal server error")

    async def retrieve_by_embedding(self, *args, **kwargs):  # pragma: no cover
        return []

    async def retrieve_related(self, *args, **kwargs):  # pragma: no cover
        return []


@pytest.mark.asyncio
async def test_memory_service_falls_back_to_store_search_when_retriever_fails():
    mem = MemoryItem(user_id="u1", content="hello", type=MemoryType.SEMANTIC)
    store = _FakeStore([mem])

    cfg = EngineConfig()
    svc = MemoryService(
        config=cfg,
        memory_store=store,
        memory_retriever_provider=lambda: _FailingRetriever(),
    )

    out = await svc.search(user_id="u1", query="hello", limit=5)
    assert len(out) == 1
    assert out[0].content == "hello"

