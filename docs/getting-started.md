# Getting Started

## Installation

```bash
pip install ctxforge
```

Optional backends are installed as extras:

```bash
pip install "ctxforge[openai,postgres,chromadb]"
pip install "ctxforge[huggingface]"   # local sentence-transformers embeddings
pip install "ctxforge[all]"           # everything
```

For development:

```bash
pip install -e ".[dev,all]"
```

## Requirements

- Python 3.10+
- `pydantic>=2`, `pyyaml`, `python-dotenv`, `aiofiles`

## First turn

ctxforge is inference-agnostic: you bring your own LLM and pass it to the
engine's helpers, or you just use the engine to build the prompt and call the
LLM yourself.

```python
import asyncio
from ctxforge.engine.factory import EngineFactory
from ctxforge.config.loader import load_config

async def main():
    # Defaults to in-memory stores + mock LLM. Point at real services via config.
    engine = await EngineFactory.create(load_config())

    # 1. Prepare context (before the LLM call)
    context = await engine.prepare_context(
        session_id="sess_123",
        user_id="user_456",
        user_input="What restaurants did I like?",
    )

    # 2. Use with any LLM (OpenAI here, but any framework works)
    messages = context.to_openai_messages()
    # response = openai.chat.completions.create(model="gpt-4", messages=messages)

    # 3. Record the turn (after the LLM response) — triggers memory extraction,
    #    compaction, and (optionally) expertise reflection in the background.
    await engine.record_turn(
        session_id="sess_123",
        user_id="user_456",
        user_input="What restaurants did I like?",
        assistant_response="You liked the Italian place downtown.",
    )

    await engine.close()

asyncio.run(main())
```

## Bring-your-own-LLM helpers

For a complete turn in one call, use `ctxforge.helpers`:

```python
from ctxforge.helpers import answer_two_step, answer_with_controller

answer = await answer_two_step(
    engine, llm,
    session_id="sess_123", user_id="user_456",
    user_input="What restaurants did I like?",
)
```

## Configuration in 30 seconds

```yaml
# config.yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}        # env-var interpolation is supported
storage:
  session:
    backend: memory                 # or redis / postgres / mysql
  memory:
    store_backend: memory
    vector:
      backend: chromadb             # or pinecone / weaviate
      embedding:
        provider: openai
        model: text-embedding-3-small
```

```python
engine = await EngineFactory.create(load_config("config.yaml"))
```

See [Configuration](configuration.md) for the full schema.
