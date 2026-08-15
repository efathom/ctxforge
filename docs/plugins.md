# Plugins

ctxforge is designed for extension without core-code changes. There are two
mechanisms, both driven by the `ComponentRegistry` (`ctxforge.engine.registry`).

## 1. Module with a `register()` function

A module that exposes `register(registry)`:

```python
# my_plugin.py
from ctxforge.protocols.llm import ILLMProvider

class MyLLM(ILLMProvider):
    ...

def register(registry):
    registry.register_llm("my_llm")(MyLLM)
    registry.register_middleware("my_middleware")(MyMiddleware)
```

```yaml
plugins:
  modules: [my_plugin]
```

## 2. Class-path registrations

```yaml
plugins:
  registrations:
    - component_type: llm
      name: my_llm
      class_path: my_pkg.my_module:MyLLM
    - component_type: reranker
      name: my_reranker
      class_path: my_pkg.rerankers:MyReranker
```

## Component types

`llm`, `embedding`, `session_store`, `memory_store`, `expertise_store`,
`scoped_memory_store`, `skill_store`, `retriever`, `condenser` (alias
`compactor`), `extractor`, `middleware`, `reranker`, `assembler`.

## Decorator registration

```python
from ctxforge.engine.registry import registry

@registry.register_llm("my_llm")
class MyLLM(ILLMProvider):
    ...
```

## Middleware factories

Dependency-aware middleware can be registered with a factory that receives
`config` and `EngineDeps`:

```python
@registry.register_middleware_factory("my_middleware")
def create(config: dict, deps: EngineDeps):
    return MyMiddleware(deps.memory_store)
```

## Listing components

```bash
python -m ctxforge list-components [--config config.yaml]
```
