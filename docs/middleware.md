# Middleware

Middleware runs cross-cutting concerns at specific phases of the `prepare` and
`record` pipelines. It's configured in `pipelines.prepare.chain` and
`pipelines.record.chain`; each entry has a `type`, `priority`, optional
`phases`, and `config`.

```yaml
pipelines:
  prepare:
    chain:
      - type: pii
        enabled: true
        priority: 100
        phases: [prepare_input]
        config: { redact: true }
      - type: query_rewriter
        enabled: true
        priority: 800
        phases: [prepare_input]
  record:
    chain:
      - type: pii
        enabled: true
        priority: 100
        phases: [record_input_output]
```

Higher `priority` runs first.

## Built-in middleware

| Type | Purpose |
|------|---------|
| `pii` | detect & redact personally identifiable information |
| `ratelimit` | token-bucket / sliding-window throttling |
| `audit` | log operations to an audit store |
| `content` | keyword/content filtering |
| `query_rewriter` | resolve pronouns/references before retrieval |
| `expertise` | expertise retrieval / evolution / audit |
| `skills` | skill injection & activation |
| `scoped_memory` | hierarchical scoped-memory injection |
| `intent_notes` | persist structured intent notes on events |
| `search_before_respond` | trigger a search when the query matches knowledge-domain keywords |
| `tool_compression` | compress large tool outputs |
| `approval` | human-in-the-loop approval gating |

## Writing custom middleware

```python
from ctxforge.middleware import BaseMiddleware, MiddlewareContext

class MyMiddleware(BaseMiddleware):
    @property
    def name(self) -> str:
        return "my_middleware"

    async def _do_process(self, context, next):
        context.set_metadata("custom", True)
        return await next(context)
```

Register it (see [Plugins](plugins.md)) and reference it by `type` in a chain.
