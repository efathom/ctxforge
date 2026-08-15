# Compaction

Compaction keeps the conversation within the model's context window by rolling
up older history into summaries.

```yaml
compaction:
  strategy: summarize          # summarize | prune | sliding_window | importance | structured | pipeline
  event_threshold: 10          # trigger after N events
  token_threshold: 4000        # trigger after N tokens
  keep_recent: 5               # always keep the last N events
  max_summary_tokens: 500
  async_compaction: true
  include_tool_calls: true
```

## Strategies

| Strategy | Behavior |
|----------|----------|
| `summarize` | replace old events with an LLM summary |
| `prune` | drop old events, keep `keep_recent` |
| `sliding_window` | fixed window of recent events |
| `importance` | keep highest-importance events |
| `structured` | structured summaries (LLM function calling) |
| `pipeline` | ordered list of condenser steps |

## Pipeline

```yaml
compaction:
  strategy: pipeline
  pipeline:
    - type: sliding_window
      config: { max_events: 20 }
    - type: summarizing
      config: { max_summary_tokens: 300 }
```

## Context health monitoring

```yaml
compaction:
  health:
    enabled: true
    warning_threshold: 0.7     # % of window used
    critical_threshold: 0.85
    inject_warnings: false
```

## Assembly & token budgeting

The `Context` supports priority-ordered sections and greedy token packing:

```python
context.add_section(name="Rules", content="...", priority=10, is_required=True)
sections = context.priority_pack_sections(budget=3000, token_fn=my_counter)
```

The configured `tokenizer_provider` (derived from the LLM provider) is used for
accurate budgeting when available.
