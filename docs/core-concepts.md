# Core Concepts

The data models in `ctxforge.core` are the vocabulary the engine operates on.

## Session — the "workbench"

A `Session` is the active working state of a conversation.

```python
from ctxforge.core.session import Session, SessionState

session = Session(user_id="user_456", session_id="sess_123")
session.add_user_message("Hello")
session.add_agent_message("Hi there!")
session.state.set("cart", ["item-1"])   # mutable working memory
session.set_summary("User greeted the agent.")
```

- `events: List[Event]` — immutable log of what happened
- `state: SessionState` — mutable scratchpad (shopping cart, verification status, …)
- `summary: Optional[str]` — rolled-up history of older events
- `version: int` — optimistic-locking counter

## Event — immutable audit log

Events are frozen (immutable) for audit integrity.

```python
from ctxforge.core.events import Event, EventType, EventFactory

event = EventFactory.user_message("I love coffee")
event = EventFactory.agent_message("Noted!", model="gpt-4")
event = EventFactory.tool_call("search", {"q": "coffee"})
```

`EventType`: `USER`, `AGENT`, `TOOL_CALL`, `TOOL_OUTPUT`, `SYSTEM`, `SUMMARY`,
`ERROR`, `METADATA`.

## MemoryItem — long-term memory

A unit of persistent knowledge with provenance and trust.

```python
from ctxforge.core.memory import MemoryItem, MemoryType, MemorySource, MemoryFactory

mem = MemoryFactory.semantic_memory(
    user_id="user_456", content="User is vegetarian",
    source=MemorySource.USER_EXPLICIT, confidence=0.9,
)
```

Key fields:

- `type` — `SEMANTIC` (facts), `EPISODIC` (events), `PROCEDURAL` (how-to), `PREFERENCE`, `TOOL`
- `confidence_score` — 0.0–1.0 trust
- `importance` / `superseded_by` — decay + soft-delete chaining
- `restatement` — disambiguated, self-contained version of `content`
- `headline` / `subtitle` — progressive-disclosure tiers
- `keywords`, `persons`, `locations`, `topics` — multi-view lexical indexing
- `embedding` — optional vector for semantic search

`MemoryQuery` supports filtering by tags, types, confidence, active status, etc.

## Expertise — evolving playbooks

An `Expertise` is a versioned collection of `ExpertiseItem`s organized into
`ExpertiseSection`s (strategies, formulas, code snippets, mistakes, heuristics,
context clues). Items track `helpful_count` / `harmful_count` for
effectiveness scoring. See [Expertise](expertise.md).

## Context — the assembled prompt

The `Context` object is the engine's output: everything needed to invoke an LLM.

```python
context = await engine.prepare_context(...)

context.system_instructions
context.memories          # retrieved MemoryItems
context.events            # recent session history
context.sections          # priority-ordered context sections
context.current_query

# Convert to provider message formats
context.to_messages()             # generic [{role, content}]
context.to_openai_messages()
context.to_anthropic_messages()   # (system, messages) tuple
context.to_langchain_messages()
```

Sections support priority ordering and greedy token packing via
`context.priority_pack_sections(budget, token_fn=...)`.

## Protocols

Every pluggable component is defined as a duck-typed `Protocol` in `ctxforge.protocols`:

- `ISessionStore`, `IMemoryStore` — storage
- `ILLMProvider`, `IEmbeddingProvider` — model access
- `IRetriever`, `IReranker` — retrieval
- `IMemoryExtractor` — memory extraction
- `IExpertiseStore`, `IExpertiseRetriever`, `IReflector`, `ICurator` — expertise
- `IVectorStore` — vector indexes
- `ICondenser` / `IContextAssembler` — compaction

Any class implementing the required methods works, regardless of inheritance.
