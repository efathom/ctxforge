"""
Engine dependency container.

This dataclass is used for dependency-aware component construction, especially middleware.
It allows middleware factories (and eventually other component factories) to access a stable
set of engine wiring dependencies without coupling to EngineFactory internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ctxforge.config.base import EngineConfig
from ctxforge.protocols.expertise import IExpertiseRetriever, IExpertiseStore
from ctxforge.protocols.llm import IEmbeddingProvider, ILLMProvider
from ctxforge.protocols.storage import IMemoryStore, ISessionStore
from ctxforge.protocols.tokenizer import ITokenizerProvider
from ctxforge.protocols.vectorstore import IVectorStore
from ctxforge.retrieval.indexers.memory import MemoryIndexer


@dataclass(frozen=True)
class EngineDeps:
    config: EngineConfig

    # Core stores
    session_store: ISessionStore
    memory_store: IMemoryStore

    # LLM-related
    llm_provider: Optional[ILLMProvider] = None
    embedding_provider: Optional[IEmbeddingProvider] = None
    tokenizer_provider: Optional[ITokenizerProvider] = None

    # Vector/indexing
    vector_store: Optional[IVectorStore] = None
    memory_indexer: Optional[MemoryIndexer] = None

    # Expertise
    expertise_store: Optional[IExpertiseStore] = None
    expertise_retriever: Optional[IExpertiseRetriever] = None

    # Optional subsystems/services (kept as Any for now)
    reflector: Optional[Any] = None
    curator: Optional[Any] = None
    scoped_memory_service: Optional[Any] = None
    skill_service: Optional[Any] = None
    graph_service: Optional[Any] = None

