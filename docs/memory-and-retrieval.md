# Memory & Retrieval

## Memory types

`MemoryType`: `SEMANTIC`, `EPISODIC`, `PROCEDURAL`, `PREFERENCE`, `TOOL`.

Memories carry provenance (`MemorySource`), `confidence_score`, `importance`
(decays over time), `superseded_by` (soft-delete chains), and multi-view
indexing fields (`keywords`, `persons`, `locations`, `topics`).

## Extraction

Memories are extracted from conversations (after `record_turn`) by the
configured `IMemoryExtractor`:

- **PatternExtractor** — regex rules for common patterns (fast, no LLM)
- **LLMExtractor** — LLM-driven, multi-pass, with source alignment
- **EntityExtractor** — named-entity recognition (optional spaCy)
- **HybridExtractor** — combines the above with dedup/merge

Config lives under `extraction:` — `enabled`, `async_processing`,
`extract_semantic`, `extract_episodic`, `min_confidence`, `use_llm`, etc.

## Retrieval strategies

`retrieval.strategy` selects the retriever:

| Strategy | Description |
|----------|-------------|
| `keyword` | lexical/BM25-style matching (no embeddings required) |
| `semantic` | embedding similarity via a vector store |
| `hybrid` | semantic + keyword with weights |
| `temporal` | recency-weighted |
| `salience` | salience-scored |

```yaml
retrieval:
  strategy: hybrid
  default_limit: 5
  semantic_weight: 0.7
  keyword_weight: 0.3
```

## Reranking

A two-stage retriever can wrap the base strategy with a reranker:

```yaml
retrieval:
  rerank_enabled: true
  reranker: llm            # llm | effectiveness | usage_recency | diversity
  rerank_model: null       # defaults to provider default
  rerank_top_k: 10
```

## Vector search

Semantic/hybrid retrieval needs a vector store (`storage.memory.vector.backend`)
and an embedding provider (`storage.memory.vector.embedding`). See
[Storage](storage.md) and [Providers](providers.md).

If no vector store is configured, retrieval falls back to keyword and the
engine still works.

## Fast-path retrieval

`memory_quality.retrieval_fast_path` enables an O(1) cache for common query
shapes (count/list/relation queries), skipping full retrieval on a hit.

## Unified cross-store retrieval

`dynamic_context.unified_retrieval.enabled` merges results across expertise,
memory, and graph sources with configurable weights and a merge strategy
(`interleave`, `score_only`, `source_first`).
