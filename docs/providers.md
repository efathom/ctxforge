# Providers

Providers implement the `ILLMProvider` / `IEmbeddingProvider` protocols. The
factory selects them from `llm.provider` and the embedding config's `provider`.

## LLM providers

| Provider | `llm.provider` | Notes |
|----------|----------------|-------|
| OpenAI | `openai` | `api_key`, `model` |
| Azure OpenAI | `azure` / `azure_openai` | endpoint, `api_version`, deployments via `extra_params` |
| OpenRouter | `openrouter` | OpenAI-compatible; set `api_key`, `model` (a slug like `openai/gpt-4o-mini`) |
| Mock | `mock` | deterministic, for tests |

### OpenRouter

```yaml
llm:
  provider: openrouter
  api_key: ${OPENROUTER_API_KEY}
  model: openai/gpt-4o-mini
  extra_params:
    http_referer: https://example.com   # optional attribution
    site_title: My App                  # optional attribution
```

## Embedding providers

| Provider | config | Notes |
|----------|--------|-------|
| OpenAI | `openai` | `model`, `api_key` |
| Azure OpenAI | `azure` | deployment + endpoint |
| Local (sentence-transformers) | `local` | in-process HuggingFace model |
| Mock | `mock` | deterministic |

### Local / self-hosted embeddings

Two options:

**In-process (sentence-transformers):**

```yaml
storage:
  memory:
    vector:
      embedding:
        provider: local
        model: BAAI/bge-small-en-v1.5
```

The factory auto-derives the vector dimension from the loaded model
(`pip install 'ctxforge[huggingface]'`).

**OpenAI-compatible server (TEI / Ollama / vLLM):**

```yaml
storage:
  memory:
    vector:
      backend: chromadb
      embedding:
        provider: openai                 # any OpenAI-compatible server
        model: BAAI/bge-small-en-v1.5
        base_url: http://localhost:8080/v1
        api_key: ""                      # usually empty for local servers
        dimension: 384                   # must match the model
```

Example servers:

```bash
# Text Embeddings Inference (recommended)
docker run -d -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id BAAI/bge-small-en-v1.5

# Ollama
ollama pull nomic-embed-text   # serves http://localhost:11434/v1
```

Common dimensions: `bge-small` = 384, `bge-large` = 1024, `all-MiniLM-L6-v2` = 384, `e5-large` = 1024.

## Custom providers

Write a class implementing `ILLMProvider` (or `IEmbeddingProvider`), register
it, and reference it by name — see [Plugins](plugins.md).
