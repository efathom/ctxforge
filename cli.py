"""
ctxforge CLI (M3).

Commands:
- validate-config: load + validate EngineConfig
- list-components: print registered component types (optionally after loading plugins)
- rebuild-indexes: rebuild memory/expertise vector indexes
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Optional, Sequence

from ctxforge.config.loader import load_config
from ctxforge.engine.factory import EngineFactory
from ctxforge.engine.registry import registry


def build_parser() -> argparse.ArgumentParser:
    # CLI name (kept short for ergonomics).
    parser = argparse.ArgumentParser(prog="ctxforge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate-config", help="Validate a config file (YAML/JSON)")
    p_validate.add_argument("path", help="Path to config file")
    p_validate.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    p_list = sub.add_parser("list-components", help="List registered components in the registry")
    p_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    p_list.add_argument(
        "--config",
        help="Optional config file; loads plugins before listing components",
    )

    p_rebuild = sub.add_parser("rebuild-indexes", help="Rebuild vector indexes from stores")
    p_rebuild.add_argument("path", help="Path to config file")
    p_rebuild.add_argument("--memory-user-id", help="User ID to rebuild memory index for")
    p_rebuild.add_argument("--expertise", action="store_true", help="Rebuild expertise indexes (all expertise)")
    p_rebuild.add_argument("--memory", action="store_true", help="Rebuild memory index (requires --memory-user-id)")

    return parser


def _registry_snapshot() -> dict:
    return {
        "llm_providers": registry.list_llm_providers(),
        "embedding_providers": registry.list_embedding_providers(),
        "session_stores": registry.list_session_stores(),
        "memory_stores": registry.list_memory_stores(),
        "expertise_stores": registry.list_expertise_stores(),
        "retrievers": registry.list_retrievers(),
        "condensers": registry.list_condensers(),
        # Backward-compatible alias.
        "compactors": registry.list_compactors(),
        "extractors": registry.list_extractors(),
        "middleware": registry.list_middleware(),
        "rerankers": registry.list_rerankers(),
        "assemblers": registry.list_assemblers(),
    }


def cmd_list_components(as_json: bool = False, config_path: Optional[str] = None) -> int:
    # Optional: load plugins so registry includes user extensions.
    if config_path:
        cfg = load_config(file_path=config_path)
        # Load plugins into this CLI module's registry (can be monkeypatched in tests).
        factory = EngineFactory(component_registry=registry)
        factory._load_plugins(cfg)

    data = _registry_snapshot()
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    for k, v in data.items():
        print(f"{k}:")
        for item in sorted(v):
            print(f"  - {item}")
    return 0


def cmd_validate_config(path: str, as_json: bool = False) -> int:
    cfg = load_config(file_path=path)
    factory = EngineFactory()
    result = factory.validate_config(cfg)

    if as_json:
        print(json.dumps({"ok": result.ok, "errors": result.errors, "warnings": result.warnings}, indent=2))
        return 0 if result.ok else 1

    if result.ok:
        print("Config OK")
    else:
        print("Config INVALID:")
        for e in result.errors:
            print(f"- {e}")

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"- {w}")

    return 0 if result.ok else 1


async def cmd_rebuild_indexes(
    path: str,
    memory_user_id: Optional[str],
    rebuild_expertise: bool,
    rebuild_memory: bool,
) -> int:
    cfg = load_config(file_path=path)
    factory = EngineFactory()
    engine = await factory.build(cfg)

    try:
        if rebuild_memory:
            if not memory_user_id:
                raise ValueError("--memory requires --memory-user-id")
            if engine.memory_indexer is None:
                raise ValueError("Memory indexer not configured (enable storage.memory.backend != memory)")
            items = await engine.memory_store.get_by_user(memory_user_id, limit=100000, include_inactive=True)
            await engine.memory_indexer.index_all(items, scope_id=memory_user_id)
            print(f"Rebuilt memory index for user_id={memory_user_id} (items={len(items)})")

        if rebuild_expertise:
            if engine.expertise_store is None or engine.expertise_indexer is None:
                raise ValueError("Expertise store/indexer not configured (enable expertise + expertise.vectorstore.backend != memory)")
            expertise_list = await engine.expertise_store.list_expertise(limit=100000)
            total = 0
            for exp in expertise_list:
                total += await engine.expertise_indexer.index_all(exp, only_active=False)
            print(f"Rebuilt expertise index (expertise={len(expertise_list)}, items={total})")

        if not rebuild_memory and not rebuild_expertise:
            print("Nothing to do. Use --memory and/or --expertise.")
            return 2

        return 0
    finally:
        await engine.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-components":
        return cmd_list_components(as_json=bool(args.json), config_path=getattr(args, "config", None))

    if args.command == "validate-config":
        return cmd_validate_config(path=args.path, as_json=bool(args.json))

    if args.command == "rebuild-indexes":
        return asyncio.run(
            cmd_rebuild_indexes(
                path=args.path,
                memory_user_id=args.memory_user_id,
                rebuild_expertise=bool(args.expertise),
                rebuild_memory=bool(args.memory),
            )
        )

    parser.print_help()
    return 2


