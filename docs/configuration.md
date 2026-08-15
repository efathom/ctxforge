# Configuration

All ctxforge configuration is expressed through the Pydantic `EngineConfig`
model (`ctxforge/config/base.py`) and loaded from YAML, JSON, environment
variables, or dictionaries.

## Loading

```python
from ctxforge.config.loader import load_config

config = load_config()                          # defaults
config = load_config("config.yaml")             # from a file
config = load_config("config.json", use_env=True)
config = load_config(overrides={"llm": {"model": "gpt-4o"}})
```

Priority (later overrides earlier):

1. `DEFAULT_CONFIG` built-ins
2. config file (YAML/JSON)
3. environment variables (`CTXFORGE_*`)
4. programmatic overrides

## Presets

`ctxforge.config.defaults` ships three presets:

```python
from ctxforge.config.defaults import DEFAULT_CONFIG, DEVELOPMENT_CONFIG, TESTING_CONFIG
```

- `DEFAULT_CONFIG` — mock LLM, in-memory stores (works with zero setup)
- `TESTING_CONFIG` — mock + in-memory, synchronous extraction/compaction
- `DEVELOPMENT_CONFIG` — verbose debug logging

## Top-level structure

```yaml
name: ctxforge
version: "0.1.0"
debug: false

llm: { provider, model, api_key, api_base, temperature, max_tokens, extra_params }
storage:
  session: { backend, ttl_seconds, max_sessions_per_user, extra_params }
  memory:
    store_backend: memory|redis|postgres|mysql
    vector: { backend, index_name, embedding: {...}, extra_params }
expertise: { enabled, store, vectorstore, retrieval }
retrieval: { strategy, default_limit, semantic_weight, keyword_weight, rerank_enabled, reranker }
compaction: { strategy, event_threshold, token_threshold, keep_recent, ... }
pipelines: { prepare: {chain: [...]}, record: {chain: [...]} }
extraction: { enabled, async_processing, extract_semantic, use_llm, ... }
observability: { log_level, log_to_file, tracing_enabled, metrics_enabled }
prompts: { system_template, memory_section_name, history_section_name, max_history_events }
graph: { enabled, store, ontology, extraction, embeddings, retrieval, ... }
skills: { enabled, store, ... }
scoped_memory: { enabled, ... }
plugins: { modules: [], registrations: [] }
dynamic_context: { approval, semantic_model, snapshots, unified_retrieval, ... }
memory_quality: { model_routing, entropy_gate, consolidation, retrieval_fast_path, ... }
extensions: {}
```

Most subsystems are **off by default** and enabled with an `enabled: true`
flag (`graph`, `expertise`, `scoped_memory`, `retrieval_controller`, etc.).

## Environment variables

Any value can be overridden with `CTXFORGE_<SECTION>_<KEY>`:

```bash
export CTXFORGE_LLM_PROVIDER=openai
export CTXFORGE_STORAGE_SESSION_BACKEND=redis
export CTXFORGE_DEBUG=true
```

Strings support `${VAR}` / `${VAR:-default}` interpolation inside config files.

> **Note:** env-var overrides split on `_`, so a field such as `api_key` maps to
> a nested `api.key` and will not be applied. Prefer setting underscore-containing
> fields (like `api_key`, `extra_params`, `base_url`) directly in the config file
> or via `extra_params` JSON.

## Programmatic

```python
from ctxforge.config.base import EngineConfig

config = EngineConfig.from_dict({"llm": {"provider": "openai"}})
config = config.merge_with({"retrieval": {"default_limit": 10}})
data = config.to_dict()
```

## Validating

The CLI validates cross-field constraints (e.g. Pinecone requires an API key):

```bash
python -m ctxforge validate-config config.yaml
```

See [CLI](cli.md).
