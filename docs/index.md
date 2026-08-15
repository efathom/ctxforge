# ctxforge

**A highly extensible, configurable context-engine framework for LLM agents.**

ctxforge manages the full lifecycle of context *around* LLM calls — session
management, long-term memory, knowledge graphs, expertise, retrieval,
compaction, and middleware — **without owning LLM inference**. It produces a
`Context` object you feed to any LLM or agent framework (OpenAI, Anthropic,
LangChain, LangGraph, or a custom loop).

## Design principles

- **Inference-agnostic** — the engine never calls an LLM itself; you inject an `ILLMProvider`.
- **Protocol-based extensibility** — any duck-typed component can be plugged in via `protocols/`.
- **Configuration-driven** — everything is wired from a Pydantic `EngineConfig` (YAML/JSON/env).
- **Modular** — memory, graph, expertise, retrieval, compaction, and middleware are independent subsystems.

## Documentation

- [Getting Started](getting-started.md) — install and run your first turn
- [Configuration](configuration.md) — the `EngineConfig` schema, YAML/JSON/env loading
- [Core Concepts](core-concepts.md) — `Session`, `Event`, `MemoryItem`, `Expertise`, `Context`
- [The Engine](engine.md) — `CtxForge`: `prepare_context` / `record_turn` lifecycle
- [Memory & Retrieval](memory-and-retrieval.md) — memory types, retrieval strategies, reranking
- [Compaction](compaction.md) — keeping context within the token budget
- [Middleware](middleware.md) — PII, rate limiting, audit, and more
- [Knowledge Graph](knowledge-graph.md) — entity/relation memory with Neo4j/in-memory stores
- [Expertise](expertise.md) — evolving, ACE-style playbooks
- [Providers](providers.md) — LLM & embedding providers (OpenAI, Azure, OpenRouter, local)
- [Storage](storage.md) — session/memory stores and vector stores
- [Plugins](plugins.md) — registering custom components
- [CLI](cli.md) — `validate-config`, `list-components`, `rebuild-indexes`

## Repository layout

| Directory | Purpose |
|-----------|---------|
| `engine/` | `CtxForge` engine, `EngineFactory`, `registry`, services |
| `core/` | data models: `Session`, `Event`, `MemoryItem`, `Expertise`, `Context` |
| `config/` | Pydantic `EngineConfig` schema and loaders |
| `protocols/` | duck-typed contracts for pluggable components |
| `retrieval/` | retrievers, rerankers, indexers |
| `compaction/` | condensers and context assembly |
| `middleware/` | cross-cutting pipeline stages |
| `graph/` | knowledge-graph extraction, stores, retrieval |
| `expertise/` | expertise store, reflector, curator |
| `extraction/` | memory extraction (pattern, LLM, hybrid) |
| `storage/` | in-memory / Redis / Postgres / MySQL stores |
| `vectorstores/` | ChromaDB / Pinecone / Weaviate integrations |
| `llm/` | LLM & embedding providers |
| `examples/` | runnable demos |
| `tests/` | test suite (pytest + pytest-asyncio) |

## Quick example

```python
import asyncio
from ctxforge.engine.factory import EngineFactory
from ctxforge.config.loader import load_config

async def main():
    engine = await EngineFactory.create(load_config())

    context = await engine.prepare_context(
        session_id="sess_123", user_id="user_456",
        user_input="What restaurants did I like?",
    )
    messages = context.to_openai_messages()  # send to any LLM

    await engine.record_turn(
        session_id="sess_123", user_id="user_456",
        user_input="What restaurants did I like?",
        assistant_response="You liked the Italian place downtown.",
    )
    await engine.close()

asyncio.run(main())
```
