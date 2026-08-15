# Expertise

The expertise system models evolving, ACE-inspired "playbooks" — structured,
curated knowledge that improves with use. It's opt-in:

```yaml
expertise:
  enabled: true
  store:
    backend: memory          # or redis / postgres / mysql
  vectorstore:
    backend: memory          # chromadb / pinecone / weaviate for semantic retrieval
    embedding: { provider: openai, model: text-embedding-3-small }
  retrieval:
    enabled: true
    default_limit: 10
    rerank_enabled: false
    reranker: effectiveness  # effectiveness | usage_recency | diversity | llm
```

## Concepts

- **Expertise** — versioned collection of items.
- **ExpertiseItem** — a single piece of domain knowledge, tagged with
  `helpful_count` / `harmful_count` and an `effectiveness_score`.
- **ExpertiseSection** — `STRATEGIES`, `FORMULAS`, `CODE_SNIPPETS`,
  `COMMON_MISTAKES`, `HEURISTICS`, `CONTEXT_CLUES`, `CUSTOM`.

## Reflection + curation loop

After a turn, if you report which items were used and the outcome, the engine
runs a reflect→curate cycle:

```python
updated = await engine.record_turn(
    session_id="sess_123", user_id="user_456",
    user_input="...", assistant_response="...",
    expertise_items_used=["strat-00001"],
    outcome=TurnOutcome.SUCCESS,
    expertise_id="my-expertise",
)
```

- **Reflector** tags each used item `helpful` / `harmful` / `neutral`.
- **Curator** proposes `ADD` / `UPDATE` / `MERGE` / `DELETE` operations.

## Retrieval

Attach expertise to a turn:

```python
context = await engine.prepare_context(
    session_id="sess_123", user_id="user_456",
    user_input="How should I approach this refactor?",
    expertise_id="my-expertise",
    max_expertise_items=5,
)
```

## Snapshots & semantic models

`dynamic_context.snapshots.enabled` versions expertise over time;
`dynamic_context.semantic_model.enabled` injects a compact semantic-model
anchor into the context.
