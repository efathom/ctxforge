# CLI

```bash
python -m ctxforge --help
```

## `validate-config`

Load and validate an `EngineConfig`, including cross-field checks that Pydantic
doesn't catch (e.g. Pinecone requires an API key).

```bash
python -m ctxforge validate-config config.yaml
python -m ctxforge validate-config config.json --json   # machine-readable
```

## `list-components`

List registered component types (optionally after loading plugins).

```bash
python -m ctxforge list-components
python -m ctxforge list-components --json
python -m ctxforge list-components --config config.yaml   # load plugins first
```

## `rebuild-indexes`

Rebuild memory/expertise vector indexes from their stores.

```bash
# Rebuild memory index for a user
python -m ctxforge rebuild-indexes config.yaml --memory --memory-user-id u1

# Rebuild all expertise indexes
python -m ctxforge rebuild-indexes config.yaml --expertise
```

Requires the relevant vector store + indexer to be configured
(`storage.memory.vector.backend != memory`, `expertise.vectorstore.backend != memory`).
