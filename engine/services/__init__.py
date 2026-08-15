"""
Engine services.

These modules encapsulate subsystem logic (graph, retrieval, memory update planning, etc.)
so `ctxforge` can stay focused on orchestration and public APIs.

Service map (roughly):
- `SessionService`: session persistence + lifecycle
- `TurnRecordingService`: event recording + background tasks (extraction/compaction/graph ingestion)
- `MemoryService`: memory CRUD + optional indexing + retrieval adapter
- `MemoryUpdateService`: LLM-driven memory update planning + applying operations
- `ExpertiseService`: expertise KB CRUD + retrieval + reflection/curation workflows
- `GraphService`: graph ingestion/retrieval + periodic community rebuild hooks
- `AssemblyService`: create/invoke configured context assembler + post-assembly inserts (e.g., graph section)
- `CompactionService`: compaction policy wiring + persistence side-effects
- `ContextWindowService`: token counting + context “overview” breakdown for observability
- `FusionService`: fuse KG-grounded and memory-grounded answers into a final response
"""


