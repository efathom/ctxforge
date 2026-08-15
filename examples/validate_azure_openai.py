#!/usr/bin/env python3
"""
Validate Azure OpenAI connectivity for BOTH chat (LLM) and embeddings.

This script is intended to catch the common failure modes quickly:
- SSL trust issues (corporate/internal CAs)
- wrong endpoint
- wrong API version
- wrong deployment names (DeploymentNotFound)
- missing/invalid API key

It uses the same provider implementation as ctxforge:
  - ctxforge.llm.azure_openai_provider.AzureOpenAILLMProvider
  - ctxforge.llm.azure_openai_provider.AzureOpenAIEmbeddingProvider

By default, it will attempt to load environment variables from:
  - ctxforge/examples/.env (if python-dotenv is installed)

Required env vars (recommended):
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_VERSION
  AZURE_OPENAI_CHAT_DEPLOYMENT
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT

TLS / CA bundle:
  CA_BUNDLE_TRUST_CA_FILE=/etc/lipki/public-ca.crt   (or any PEM/CRT bundle)

Usage:
  python -m ctxforge.examples.validate_azure_openai
  python ctxforge/examples/validate_azure_openai.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

# Ensure `ctxforge` is importable when running as a script (not `python -m ...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxforge.llm.azure_openai_provider import (
    AzureOpenAIConfig,
    AzureOpenAIEmbeddingProvider,
    AzureOpenAILLMProvider,
)
from ctxforge.protocols.llm import ChatMessage


def _try_load_dotenv(env_file: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return v


def _pick_ca_bundle_path() -> Tuple[Optional[str], bool]:
    """
    Returns (path, exists).
    The provider uses this same env var, but we print it here for clarity.
    """
    ca_path = _get_env("CA_BUNDLE_TRUST_CA_FILE") or "/etc/lipki/public-ca.crt"
    exists = bool(ca_path and Path(ca_path).exists())
    return ca_path, exists


def _format_exception(e: Exception) -> str:
    parts = [f"{type(e).__name__}: {e}"]

    req = getattr(e, "request", None)
    if req is not None:
        try:
            method = getattr(req, "method", None)
            url = getattr(req, "url", None)
            if method or url:
                parts.append(f"request={method} {url}")
        except Exception:
            pass

    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            status = getattr(resp, "status_code", None)
            url = getattr(resp, "url", None)
            if status or url:
                parts.append(f"response={status} {url}")
        except Exception:
            pass

    body = getattr(e, "body", None)
    if body is not None:
        parts.append(f"body={body}")

    return " | ".join([p for p in parts if p])


def _is_http_5xx(e: Exception) -> bool:
    """
    Best-effort: detect server-side errors from the OpenAI/Azure OpenAI SDK.
    """
    name = type(e).__name__
    if name in {"InternalServerError", "ServiceUnavailableError"}:
        return True
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and 500 <= status <= 599:
        return True
    # Some SDK exceptions keep a response object.
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            code = getattr(resp, "status_code", None)
            if isinstance(code, int) and 500 <= code <= 599:
                return True
        except Exception:
            pass
    return False


def _build_demo_like_messages(prompt: str) -> list[ChatMessage]:
    """
    Build a slightly heavier request shape that resembles the demo:
    - non-trivial system prompt
    - user prompt
    - some dummy "retrieved memories" embedded in the user message
    """
    system = (
        "You are a helpful AI assistant with memory capabilities.\n"
        "When relevant memories are provided, use them to personalize your responses.\n"
        "Be concise.\n"
    )
    fake_memories = "\n".join(
        [
            "- User likes spicy food",
            "- User prefers concise answers",
            "- Global tip: tofu stir-fry is a great spicy dinner option",
        ]
    )
    user = (
        f"{prompt}\n\n"
        "Retrieved Memories:\n"
        f"{fake_memories}\n"
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


def _resolve_settings(args: argparse.Namespace) -> dict:
    api_key = args.api_key or _get_env("AZURE_OPENAI_API_KEY")
    endpoint = args.endpoint or _get_env("AZURE_OPENAI_ENDPOINT")
    api_version = args.api_version or _get_env("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
    chat_deployment = args.chat_deployment or _get_env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    embedding_deployment = args.embedding_deployment or _get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    missing = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not chat_deployment and not args.skip_llm:
        missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not embedding_deployment and not args.skip_embedding:
        missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if missing:
        raise ValueError("Missing required env vars: " + ", ".join(missing))

    return {
        "api_key": api_key or "",
        "endpoint": endpoint or "",
        "api_version": api_version,
        "chat_deployment": chat_deployment or "",
        "embedding_deployment": embedding_deployment or "",
    }

def _load_replay_payload(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Replay file not found: {path}")
    with open(p, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Replay file must contain a JSON object.")
    req = data.get("request")
    if not isinstance(req, dict):
        raise ValueError("Replay file missing 'request' object.")
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("Replay file missing 'request.messages' list.")
    return data


async def _validate_llm(cfg: AzureOpenAIConfig, prompt: str) -> None:
    provider = AzureOpenAILLMProvider(cfg)
    messages = (
        _build_demo_like_messages(prompt)
        if cfg and getattr(cfg, "deployment", None) and getattr(cfg, "azure_endpoint", None)
        else [ChatMessage(role="user", content=prompt)]
    )
    resp = await provider.chat(messages=messages)
    text = (resp.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned empty response content.")
    print(f"LLM OK: model={resp.model} tokens={resp.total_tokens} sample={text[:80]!r}")


async def _validate_embeddings(cfg: AzureOpenAIConfig, text: str) -> None:
    provider = AzureOpenAIEmbeddingProvider(cfg)
    vec = await provider.embed_single(text)
    if not vec:
        raise RuntimeError("Embedding returned an empty vector.")
    print(f"EMBEDDINGS OK: deployment={cfg.embedding_deployment} dim={len(vec)} sample={vec[:5]}")


async def _run(args: argparse.Namespace) -> int:
    settings = _resolve_settings(args)

    ca_path, ca_exists = _pick_ca_bundle_path()
    print("Azure OpenAI validation")
    print(f"- endpoint: {settings['endpoint']}")
    print(f"- api_version: {settings['api_version']}")
    print(f"- chat_deployment: {settings['chat_deployment'] if not args.skip_llm else '(skipped)'}")
    print(f"- embedding_deployment: {settings['embedding_deployment'] if not args.skip_embedding else '(skipped)'}")
    print(f"- CA_BUNDLE_TRUST_CA_FILE: {ca_path} (exists={ca_exists})")
    print(f"- has_api_key: {bool(settings['api_key'])}")

    # Build a single config object used for both providers.
    cfg = AzureOpenAIConfig(
        api_key=settings["api_key"],
        azure_endpoint=settings["endpoint"],
        api_version=settings["api_version"],
        deployment=settings["chat_deployment"] or "unused",
        embedding_deployment=settings["embedding_deployment"] or "unused",
    )

    llm_ok = True
    emb_ok = True
    saw_5xx = False

    if not args.skip_llm:
        replay_messages = None
        replay_kwargs = None
        if args.replay_file:
            replay = _load_replay_payload(args.replay_file)
            req = replay.get("request", {})
            # Use the captured messages verbatim.
            replay_messages = [
                ChatMessage(role=m.get("role", "user"), content=m.get("content") or "")
                for m in req.get("messages", [])
                if isinstance(m, dict)
            ]
            replay_kwargs = {
                "model": req.get("model"),
                "temperature": req.get("temperature"),
                "max_tokens": req.get("max_tokens"),
                "stop": req.get("stop"),
                "functions": req.get("functions"),
            }
            # If the replay file has endpoint/version/deployment, prefer those unless user explicitly overrides.
            if not args.endpoint and isinstance(replay.get("endpoint"), str):
                cfg.azure_endpoint = replay["endpoint"]
            if not args.api_version and isinstance(replay.get("api_version"), str):
                cfg.api_version = replay["api_version"]
            if not args.chat_deployment and isinstance(replay.get("deployment"), str):
                cfg.deployment = replay["deployment"]

        # Optional: build the *actual* engine context the demo uses, and send those messages.
        # This helps reproduce 500s that only happen with the demo's real context formatting.
        engine_ctx_messages = None
        if args.engine_context and replay_messages is None:
            try:
                from ctxforge.examples.config import (
                    load_config,  # local import to avoid overhead when unused
                )
                from ctxforge.examples.run_demo import ContextEngineDemo

                demo_cfg = load_config(None)
                demo = ContextEngineDemo(demo_cfg, use_postgres=False, use_expertise=False)
                await demo.setup(
                    reset_chroma=bool(args.reset_chroma),
                    reset_memories=False,
                    exercise_plan_features=False,
                    graph_backend="memory",
                )
                context = await demo.engine.prepare_context(
                    session_id=demo.session_id,
                    user_id=demo.user_id,
                    user_input=args.prompt,
                    include_history=True,
                    include_memories=True,
                )
                engine_ctx_messages = [
                    ChatMessage(role=m["role"], content=m.get("content") or "")
                    for m in context.to_openai_messages()
                ]
                if args.verbose:
                    roles = [m.role for m in engine_ctx_messages]
                    total_chars = sum(len(m.content or "") for m in engine_ctx_messages)
                    print(f"engine_context: messages={len(engine_ctx_messages)} roles={roles} total_chars={total_chars}")
            except Exception as e:
                print(f"ENGINE CONTEXT SETUP FAILED: {_format_exception(e)}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
                llm_ok = False

        for i in range(max(1, int(args.llm_repeat))):
            try:
                provider = AzureOpenAILLMProvider(cfg)
                if replay_messages is not None:
                    messages = replay_messages
                elif engine_ctx_messages is not None:
                    messages = engine_ctx_messages
                else:
                    messages = (
                        _build_demo_like_messages(args.prompt)
                        if args.demo_like
                        else [
                            ChatMessage(role="system", content="You are a test harness. Reply with a short acknowledgement."),
                            ChatMessage(role="user", content=args.prompt),
                        ]
                    )
                if replay_messages is not None and replay_kwargs is not None:
                    resp = await provider.chat(messages=messages, **{k: v for k, v in replay_kwargs.items() if v is not None})
                else:
                    resp = await provider.chat(messages=messages)
                text = (resp.content or "").strip()
                if not text:
                    raise RuntimeError("LLM returned empty response content.")
                print(f"LLM OK [{i+1}/{int(args.llm_repeat)}]: model={resp.model} tokens={resp.total_tokens} sample={text[:80]!r}")
            except Exception as e:
                llm_ok = False
                if _is_http_5xx(e):
                    saw_5xx = True
                print(f"LLM FAILED [{i+1}/{int(args.llm_repeat)}]: {_format_exception(e)}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
                if int(args.llm_repeat) <= 1:
                    break

    if not args.skip_embedding:
        try:
            await _validate_embeddings(cfg, text=args.embedding_text)
        except Exception as e:
            emb_ok = False
            print(f"EMBEDDINGS FAILED: {_format_exception(e)}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()

    if saw_5xx:
        print("WARNING: Observed at least one HTTP 5xx from the chat endpoint during validation.", file=sys.stderr)

    if llm_ok and emb_ok:
        return 0
    if not llm_ok and not emb_ok:
        return 10
    if saw_5xx:
        return 20
    if not llm_ok:
        return 11
    return 12


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Azure OpenAI LLM + embedding deployments")
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).parent / ".env"),
        help="Path to a dotenv file to load (default: ctxforge/examples/.env)",
    )
    parser.add_argument("--endpoint", default=None, help="Azure endpoint (overrides AZURE_OPENAI_ENDPOINT)")
    parser.add_argument("--api-version", default=None, help="Azure API version (overrides AZURE_OPENAI_API_VERSION)")
    parser.add_argument("--api-key", default=None, help="Azure API key (overrides AZURE_OPENAI_API_KEY)")
    parser.add_argument("--chat-deployment", default=None, help="Chat deployment name (overrides AZURE_OPENAI_CHAT_DEPLOYMENT)")
    parser.add_argument("--embedding-deployment", default=None, help="Embedding deployment name (overrides AZURE_OPENAI_EMBEDDING_DEPLOYMENT)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip chat/LLM validation")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding validation")
    parser.add_argument("--prompt", default="Say 'ok' and nothing else.", help="Prompt used for LLM validation")
    parser.add_argument(
        "--demo-like",
        action="store_true",
        help="Use a heavier, demo-like chat payload (system prompt + fake retrieved memories) to surface flaky 5xx errors.",
    )
    parser.add_argument(
        "--engine-context",
        action="store_true",
        help="Build the real ctxforge engine context (like run_demo) and validate chat using context.to_openai_messages().",
    )
    parser.add_argument(
        "--replay-file",
        default=None,
        help="Path to a captured Azure chat payload JSON (written by CTXFORGE_CAPTURE_LLM_FAILURES=1) to replay.",
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="When using --engine-context, delete the Chroma persist directory before building the engine (same behavior as run_demo --reset-chroma).",
    )
    parser.add_argument(
        "--llm-repeat",
        type=int,
        default=1,
        help="Repeat the LLM validation N times to detect flaky 5xx behavior (default: 1).",
    )
    parser.add_argument("--embedding-text", default="hello embedding", help="Text used for embedding validation")
    parser.add_argument("--verbose", action="store_true", help="Print full tracebacks on failures")
    args = parser.parse_args()

    _try_load_dotenv(Path(args.env_file))

    try:
        return asyncio.run(_run(args))
    except ValueError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


