# ctxforge

A highly extensible, configurable **context engine framework for LLM agents**.

ctxforge manages the full lifecycle of context *around* LLM calls — session management,
long-term memory, knowledge graphs, expertise, retrieval, compaction, and middleware —
**without owning LLM inference**. It produces a `Context` object you feed to any LLM
or agent framework (OpenAI, Anthropic, LangChain, LangGraph, or a custom loop).

## Design principles

- **Inference-agnostic** — the engine never calls an LLM itself; you inject an `ILLMProvider`.
- **Protocol-based extensibility** — any duck-typed component can be plugged in via `protocols/`.
- **Configuration-driven** — everything is wired from a Pydantic `EngineConfig` (YAML/JSON/env).
- **Modular** — memory, graph, expertise, retrieval, compaction, and middleware are independent subsystems.

## Installation

```bash
pip install ctxforge
# with optional backends
pip install "ctxforge[openai,postgres,chromadb]"
```

## Quick start

```python
import asyncio
from ctxforge.engine.factory import EngineFactory
from ctxforge.config.loader import load_config

async def main():
    config = load_config()  # defaults to mock providers + in-memory stores
    engine = await EngineFactory.create(config)

    # 1. Prepare context (before your LLM call)
    context = await engine.prepare_context(
        session_id="sess_123",
        user_id="user_456",
        user_input="What restaurants did I like?",
    )

    # 2. Send to any LLM
    messages = context.to_openai_messages()

    # 3. Record the turn (after the LLM response)
    await engine.record_turn(
        session_id="sess_123",
        user_id="user_456",
        user_input="What restaurants did I like?",
        assistant_response="You liked the Italian place downtown.",
    )

    await engine.close()

asyncio.run(main())
```

## CLI

```bash
python -m ctxforge validate-config path/to/config.yaml
python -m ctxforge list-components
python -m ctxforge rebuild-indexes path/to/config.yaml --memory --memory-user-id u1
```

## Configuration

Configuration is a Pydantic model (`EngineConfig`) loaded from YAML, JSON, or env vars
(`CTXFORGE_*`). See `config/defaults.py` for all defaults and `config/base.py` for the schema.

```yaml
llm:
  provider: openai
  model: gpt-4o
storage:
  session:
    backend: redis
  memory:
    store_backend: postgres
    vector:
      backend: chromadb
      embedding:
        provider: openai
        model: text-embedding-3-small
```

## Key subsystems

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| Engine | `engine/context_engine.py` | `prepare_context()` / `record_turn()` lifecycle |
| Factory | `engine/factory.py` | dependency-injection wiring from config |
| Registry | `engine/registry.py` | pluggable component registry |
| Core models | `core/` | `Event`, `Session`, `MemoryItem`, `Expertise`, `Context` |
| Protocols | `protocols/` | duck-typed contracts for extensibility |
| Memory | `core/memory.py`, `storage/`, `retrieval/` | semantic/episodic/procedural memory + vector search |
| Knowledge graph | `graph/` | ontology-constrained extraction, Neo4j/in-memory stores |
| Expertise | `expertise/` | ACE-style evolving playbooks with reflection/curation |
| Middleware | `middleware/` | PII redaction, rate limiting, audit, skills, and more |

## Development

```bash
pip install -e ".[dev,all]"
pytest
ruff check .
mypy ctxforge
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
