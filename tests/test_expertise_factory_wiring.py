"""
Tests that EngineFactory can wire expertise retrieval via config.expertise.*

Uses fake vector store + fake embedding provider (no external services).
"""

import pytest

from ctxforge.config.defaults import TESTING_CONFIG
from ctxforge.engine.factory import EngineFactory
from ctxforge.storage.memory.expertise import InMemoryExpertiseStore
from ctxforge.storage.memory.memory import InMemoryMemoryStore
from ctxforge.storage.memory.session import InMemorySessionStore
from ctxforge.tests.test_vectorstore_memory_retrieval import FakeEmbeddingProvider, FakeVectorStore


@pytest.mark.asyncio
async def test_factory_wires_expertise_retriever_when_enabled():
    config = TESTING_CONFIG.merge_with(
        {
            "expertise": {
                "enabled": True,
                "store": {"backend": "memory"},
                "vectorstore": {"backend": "chromadb"},
            }
        }
    )

    factory = EngineFactory()
    engine = await factory.create(
        config=config,
        session_store=InMemorySessionStore(),
        memory_store=InMemoryMemoryStore(),
        expertise_store=InMemoryExpertiseStore(),
        expertise_embedding_provider=FakeEmbeddingProvider(),
        expertise_vector_store=FakeVectorStore(),
    )

    assert engine.expertise_store is not None
    assert engine.expertise_retriever is not None


