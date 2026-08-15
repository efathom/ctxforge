# Knowledge Graph

The graph subsystem (`ctxforge.graph`) maintains entity/relation memory
alongside flat `MemoryItem`s. It's opt-in:

```yaml
graph:
  enabled: true
  store:
    backend: memory          # or neo4j
  extraction:
    enabled: true            # LLM entity/relation extraction on ingest
  embeddings:
    enabled: false           # node semantic search
  retrieval:
    enabled: true
    methods: [keyword, bfs, semantic]
    max_facts: 20
    max_entities: 20
  section_name: "Graph Memory"
```

## Pipeline

1. **Extraction** — an `IGraphExtractor` pulls entities/edges from episodes,
   constrained by a `GraphOntology` (valid entity/edge types + attributes).
2. **Maintenance** — optional contradiction detection, temporal enrichment
   (`valid_at`/`invalid_at`), and KNN entity linking (`SAME_AS`).
3. **Communities** — optional clustering of nodes into communities.
4. **Retrieval** — hybrid seeding (keyword/BFS/semantic) + optional PPR
   reranking, path mining, and topology-aware serialization into the context.

## Stores

- `InMemoryGraphStore` — tests / lightweight local use
- `Neo4jGraphStore` — production, with full-text + vector indexes

```yaml
graph:
  store:
    backend: neo4j
    neo4j:
      url: bolt://localhost:7687
      username: neo4j
      password: neo4j
```

## Custom ontology

```python
# my_ontology.py
from ctxforge.graph.ontology import GraphOntology

GRAPH_ONTOLOGY = GraphOntology(
    entity_types={"Person": PersonModel},
    edge_types={"KNOWS": EdgeModel},
    allowed_edges={"KNOWS": [("Person", "Person")]},
)
```

```yaml
graph:
  ontology:
    module: my_ontology
    attr_name: GRAPH_ONTOLOGY
```

## Two-step fusion

Combine graph facts with memory in a two-step synthesis:

```python
from ctxforge.helpers import answer_two_step
answer = await answer_two_step(engine, llm, session_id=..., user_id=..., user_input=...)
```
