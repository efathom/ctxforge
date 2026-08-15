# The Engine

`CtxForge` (`ctxforge/engine/context_engine.py`) is the single entry point. It
does **not** own LLM inference, agent loops, or tool execution — it manages the
context around them.

## Creating an engine

```python
from ctxforge.engine.factory import EngineFactory
from ctxforge.config.loader import load_config

# Recommended: factory wires everything from config
engine = await EngineFactory.create(load_config("config.yaml"))

# Or provide custom components directly
engine = await EngineFactory.create(
    config,
    session_store=MySessionStore(),
    retriever=MyRetriever(),
)
```

## The lifecycle

```
prepare_context()  ──►  your LLM call  ──►  record_turn()
```

### 1. `prepare_context` (before the LLM call)

```python
context = await engine.prepare_context(
    session_id="sess_123",
    user_id="user_456",
    user_input="What restaurants did I like?",
    system_instructions=None,     # optional override
    include_memories=True,
    include_history=True,
    include_graph=True,
    max_history_events=None,
    max_memories=None,
)
```

Returns a `Context`. Optional keyword flags compose additional behavior:

| Flag | Effect |
|------|--------|
| `expertise_id="..."` | attach expertise items to the context |
| `use_controller=True, llm=...` | iterative retrieval controller (bounded evidence gathering) |
| `return_session=True` | return `(Context, Session)` |
| `return_two_step_inputs=True` | return `TwoStepInputs` for graph+memory fusion |

### 2. `record_turn` (after the LLM call)

```python
updated_expertise = await engine.record_turn(
    session_id="sess_123",
    user_id="user_456",
    user_input="What restaurants did I like?",
    assistant_response="You liked the Italian place downtown.",
    # Optional expertise feedback:
    expertise_items_used=["item-1"],
    outcome=TurnOutcome.SUCCESS,
    expertise_id="my-expertise",
)
```

This appends user + assistant events, saves the session, and triggers (per
config) background memory extraction, compaction, graph ingestion, and
expertise reflection/curation.

## Partial / streaming turns

```python
await engine.record_user_message(session_id, user_id, "hi")
await engine.record_assistant_message(session_id, user_id, "hello!")
await engine.record_tool_use(session_id, user_id, "search", {"q": "x"}, "results")
```

## Session management

```python
session = await engine.get_session("sess_123", "user_456")
await engine.update_session_state("sess_123", "user_456", cart=["a"])
await engine.delete_session("sess_123")
await engine.list_sessions("user_456", limit=10)
```

## Memory management

```python
await engine.add_memory(memory)
await engine.get_memory(memory_id)
await engine.update_memory(memory)
await engine.search_memories("user_456", "coffee", limit=5)
await engine.search_memories_by_query(MemoryQuery(user_id="u", tags=["__suggested"]))
await engine.get_user_memories("user_456", limit=100)
await engine.deactivate_memory(memory_id)      # soft delete
await engine.delete_memory(memory_id)          # hard delete
await engine.delete_all_user_memories("user_456")
```

## Shutdown

```python
await engine.close()   # waits for background tasks and closes owned resources
```

## Deprecated aliases

`prepare_two_step_inputs`, `prepare_context_with_controller`,
`prepare_context_with_session`, `answer_two_step`, `answer_with_controller`
are deprecated in favor of `prepare_context(..., flag=...)` and
`ctxforge.helpers.*`.
