# Contributing to ctxforge

Thanks for your interest in contributing! This guide covers how to set up a development environment, run the test suite, and submit a change.

## Development environment

- **Python 3.10+**
- Optional: Docker (for Redis/Postgres/MySQL/Neo4j-backed demos and local embedding servers)

```bash
git clone https://github.com/efathom/ctxforge.git
cd ctxforge
pip install -e ".[dev,all]"
```

## Build & test

```bash
pytest              # unit + contract tests (mock providers, no network)
ruff check .        # lint
mypy -p ctxforge    # type check
uv build            # build sdist + wheel
```

## Code style

- Follow PEP 8; `ruff` (with the repo's `[tool.ruff]` config) is the source of truth.
- Use `logging` (never `print`) in library code.
- Keep component contracts in `ctxforge/protocols/`.
- Return errors up the call stack rather than swallowing them silently.

## Conventions

- Pluggable components (stores, providers, retrievers, middleware) are defined as `Protocol`s and registered in `ctxforge.engine.registry`.
- The `EngineFactory` wires components from `EngineConfig`; add new config under `ctxforge/config/base.py`.
- Middleware runs in the `prepare`/`record` pipelines configured under `pipelines.*`.
- New providers self-register via `registry.register_llm()` / `registry.register_embedding()`.

## Pull request process

1. Open an issue (or comment on an existing one) to discuss larger changes before implementing.
2. Fork the repo and create a feature branch.
3. Make your change, adding tests where practical.
4. Run `pytest`, `ruff check .`, and `mypy -p ctxforge`.
5. Open a PR using the pull request template.

All contributions are made under the [Apache 2.0](LICENSE) license.

## Getting help

- Ask questions in [Discussions](https://github.com/efathom/ctxforge/discussions).
- Report bugs via [Issues](https://github.com/efathom/ctxforge/issues).
