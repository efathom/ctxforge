# Storage

Storage is split into **stores** (records) and **vector stores** (embeddings).

## Session & memory stores

`storage.session.backend` and `storage.memory.store_backend`:

| Backend | Package |
|---------|---------|
| `memory` | built-in (no deps) |
| `redis` | `redis` |
| `postgres` | `asyncpg` |
| `mysql` | `aiomysql` |

```yaml
storage:
  session:
    backend: redis
    ttl_seconds: 86400
  memory:
    store_backend: postgres
    store_connection_string: postgres://user:pass@localhost/ctxforge
    vector:
      backend: chromadb
      embedding: { provider: openai, model: text-embedding-3-small }
```

> Note the split: `store_backend` is the persistence backend, `vector.backend`
> is the index. Legacy flat keys (`storage.memory.backend`, etc.) are accepted
> and translated automatically.

## Vector stores

`storage.memory.vector.backend`:

| Backend | Package |
|---------|---------|
| `memory` | no index (keyword retrieval only) |
| `chromadb` | `chromadb` |
| `pinecone` | `pinecone-client` |
| `weaviate` | `weaviate-client` |

```yaml
storage:
  memory:
    vector:
      backend: pinecone
      index_name: agent_memories
      extra_params: { api_key: ${PINECONE_API_KEY}, environment: us-west1-gcp }
```

Each vector store has its own config dataclass (`ChromaConfig`,
`PineconeConfig`, `WeaviateConfig`).

## Expertise / skills / scoped-memory stores

`expertise.store.backend`, `skills.store.backend`, and
`scoped_memory.store.backend` each support `memory`, `redis`, `postgres`, and
`mysql` with the same pattern.
