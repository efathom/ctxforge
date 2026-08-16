"""
Azure OpenAI Provider implementations.

Provides LLM and Embedding providers using Azure OpenAI via the OpenAI Python SDK.

Notes:
- Azure OpenAI uses *deployment names* in the `model=` parameter of the OpenAI client.
- This module is intentionally lightweight and mirrors `openai_provider.py` behavior.
"""

import asyncio
import json
import logging
import os
import random
import ssl
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from ctxforge.engine.registry import registry
from ctxforge.llm._openai_wire import normalize_tools, serialize_openai_message
from ctxforge.protocols.llm import (
    ChatMessage,
    EmbeddingResponse,
    IEmbeddingProvider,
    ILLMProvider,
    LLMResponse,
)

logger = logging.getLogger(__name__)


def _retry_after_seconds_from_exception(e: Exception) -> Optional[float]:
    """
    Best-effort extraction of server-provided retry delay.

    Azure gateways sometimes include retry hints even when returning 5xx.
    """
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if not headers:
        return None

    def _get(h: str) -> Optional[str]:
        try:
            return headers.get(h)  # httpx.Headers-like
        except Exception:
            try:
                return headers.get(h.lower())
            except Exception:
                return None

    # Prefer millisecond hints when present.
    for key in ("retry-after-ms", "x-ms-retry-after-ms"):
        v = _get(key)
        if v:
            try:
                return float(v) / 1000.0
            except Exception:
                pass

    v = _get("retry-after")
    if v:
        try:
            return float(v)
        except Exception:
            return None

    return None


@dataclass
class AzureOpenAIConfig:
    """
    Azure OpenAI configuration.

    Attributes:
        api_key: Azure OpenAI API key (resource key)
        azure_endpoint: Azure endpoint, e.g. https://<resource>.openai.azure.com
        api_version: Azure OpenAI API version, e.g. 2024-02-15-preview
        deployment: Chat/completions deployment name
        embedding_deployment: Embedding deployment name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        ca_bundle_path: Optional path to a CA bundle (.crt/.pem) to trust for TLS.
                        If unset, we also check env var CA_BUNDLE_TRUST_CA_FILE, and finally
                        fall back to /etc/lipki/public-ca.crt (if present).
        timeout: Request timeout in seconds (passed to the OpenAI SDK).
        max_retries: Maximum retry attempts for transient failures (in addition to SDK defaults).
    """

    api_key: str
    azure_endpoint: str
    api_version: str = "2024-02-15-preview"
    deployment: str = "gpt-4"
    embedding_deployment: str = "text-embedding-3-small"
    max_tokens: int = 1000
    temperature: float = 0.7
    ca_bundle_path: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


def _create_async_http_client(cfg: AzureOpenAIConfig):
    """
    Create an httpx.AsyncClient with optional custom CA bundle for TLS verification.

    This mirrors how some internal environments (e.g. corporate proxies) require
    a non-default trust store.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for Azure OpenAI client customization.") from None

    ca_path = (
        os.getenv("CA_BUNDLE_TRUST_CA_FILE") or cfg.ca_bundle_path or "/etc/lipki/public-ca.crt"
    )

    if ca_path and os.path.exists(ca_path):
        ssl_context = ssl.create_default_context(cafile=ca_path)
        return httpx.AsyncClient(verify=ssl_context)

    # Default SSL verification (system trust store)
    return httpx.AsyncClient()


class AzureOpenAILLMProvider(ILLMProvider):
    """Azure OpenAI LLM provider for chat completions."""

    def __init__(self, config: AzureOpenAIConfig):
        self._config = config
        self._client = None

    @property
    def name(self) -> str:
        return "azure"

    @property
    def default_model(self) -> str:
        # In Azure OpenAI, the "model" is the deployment name.
        return self._config.deployment

    async def _get_client(self):
        """Lazy initialize the Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from None

            http_client = _create_async_http_client(self._config)
            self._client = AsyncAzureOpenAI(
                api_key=self._config.api_key,
                azure_endpoint=self._config.azure_endpoint,
                api_version=self._config.api_version,
                http_client=http_client,
                timeout=self._config.timeout,
                max_retries=0,  # we implement explicit retries below for better visibility/control
            )
        return self._client

    def _is_retryable(self, e: Exception) -> bool:
        """
        Best-effort classification of transient errors worth retrying.
        We use name-based checks so tests can simulate these without constructing SDK exceptions.
        """
        name = type(e).__name__
        if name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "ServiceUnavailableError",
        }:
            return True
        # Also retry on httpx transport issues when they bubble up.
        if name in {"ConnectError", "ReadTimeout", "RemoteProtocolError"}:
            return True
        return False

    async def _with_retries(self, fn, *, op_name: str):
        attempts = max(0, int(self._config.max_retries or 0))
        base_delay = 0.5
        max_delay = 8.0

        last_err: Optional[Exception] = None
        for attempt in range(attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last_err = e
                if attempt >= attempts or not self._is_retryable(e):
                    raise

                # Exponential backoff with jitter.
                delay = min(max_delay, base_delay * (2**attempt))
                delay = delay * (0.75 + 0.5 * random.random())
                retry_after = _retry_after_seconds_from_exception(e)
                if retry_after is not None:
                    delay = max(delay, float(retry_after))

                # Best-effort details for debugging transient gateway issues.
                req = getattr(e, "request", None)
                req_url = None
                try:
                    req_url = getattr(req, "url", None) if req is not None else None
                except Exception:
                    req_url = None
                body = getattr(e, "body", None)
                activity_id = None
                try:
                    if isinstance(body, dict):
                        activity_id = body.get("activityId") or body.get("activity_id")
                except Exception:
                    activity_id = None

                logger.warning(
                    f"{op_name} transient failure; retrying "
                    f"(attempt {attempt + 1}/{attempts}, sleep={delay:.2f}s): "
                    f"{type(e).__name__}: {e}"
                    + (f" | url={req_url}" if req_url else "")
                    + (f" | activityId={activity_id}" if activity_id else "")
                )
                await asyncio.sleep(delay)

        # Should be unreachable, but keep mypy happy.
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"{op_name} failed with unknown error")

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.time()
        client = await self._get_client()

        openai_messages = [serialize_openai_message(m) for m in messages]
        tools = normalize_tools(functions)

        async def _do_request():
            resolved_max = max_tokens or self._config.max_tokens
            request_kwargs: Dict[str, Any] = {
                # Azure expects deployment name here.
                "model": model or self._config.deployment,
                "messages": openai_messages,
                "max_completion_tokens": resolved_max,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "stop": stop,
            }
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "auto"
            request_kwargs.update(kwargs)
            return await client.chat.completions.create(**request_kwargs)

        try:
            resp = await self._with_retries(
                _do_request, op_name="azure_openai.chat.completions.create"
            )
        except Exception as e:
            # Optional: capture the full request payload for postmortem/replay when 5xx happens.
            try:
                capture_enabled = str(os.getenv("CTXFORGE_CAPTURE_LLM_FAILURES", "")).lower() in {
                    "1",
                    "true",
                    "yes",
                }
                status_code = getattr(e, "status_code", None)
                is_5xx = type(e).__name__ in {"InternalServerError", "ServiceUnavailableError"} or (
                    isinstance(status_code, int) and 500 <= status_code <= 599
                )
                if capture_enabled and is_5xx:
                    out_dir = os.getenv("CTXFORGE_LLM_FAILURE_DIR") or str(
                        (
                            Path(__file__).resolve().parents[1] / "examples" / "llm_failures"
                        ).resolve()
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    req = getattr(e, "request", None)
                    req_url = None
                    try:
                        req_url = str(getattr(req, "url", None)) if req is not None else None
                    except Exception:
                        req_url = None

                    payload = {
                        "provider": "azure",
                        "endpoint": self._config.azure_endpoint,
                        "api_version": self._config.api_version,
                        "deployment": self._config.deployment,
                        "request_url": req_url,
                        "error_type": type(e).__name__,
                        "error_str": str(e),
                        "error_body": getattr(e, "body", None),
                        "captured_at": time.time(),
                        "request": {
                            "model": model or self._config.deployment,
                            "messages": openai_messages,
                            "max_tokens": max_tokens or self._config.max_tokens,
                            "temperature": (
                                temperature if temperature is not None else self._config.temperature
                            ),
                            "stop": stop,
                            "tools": tools,
                            "tool_choice": "auto" if tools else None,
                        },
                    }

                    fname = f"azure_chat_5xx_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
                    out_path = os.path.join(out_dir, fname)
                    with open(out_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    logger.warning(f"Captured Azure chat 5xx payload to: {out_path}")
            except Exception:
                # Never mask the original exception.
                pass
            raise

        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = (
            int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
            if usage
            else (input_tokens + output_tokens)
        )

        return LLMResponse(
            content=content,
            model=model or self._config.deployment,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            finish_reason=getattr(resp.choices[0], "finish_reason", None),
            latency_ms=(time.time() - start) * 1000,
            raw_response=resp.model_dump(),
        )

    async def stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Azure OpenAI streaming not implemented in this provider yet.")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        # Best-effort heuristic until tokenizer_provider is wired in engine.
        return int(len(text.split()) * 1.3)

    def count_message_tokens(self, messages: List[ChatMessage], model: Optional[str] = None) -> int:
        total = 0
        for m in messages:
            total += 4  # role overhead heuristic
            total += self.count_tokens(m.content, model=model)
        return total


class AzureOpenAIEmbeddingProvider(IEmbeddingProvider):
    """Azure OpenAI Embedding provider."""

    def __init__(self, config: AzureOpenAIConfig):
        self._config = config
        self._client = None
        self._dimensions: Optional[int] = None

    @property
    def name(self) -> str:
        return "azure"

    @property
    def default_model(self) -> str:
        # Azure expects deployment name
        return self._config.embedding_deployment

    @property
    def embedding_dimension(self) -> int:
        return self._dimensions or self._default_dimensions_for_model(
            self._config.embedding_deployment
        )

    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from None

            http_client = _create_async_http_client(self._config)
            self._client = AsyncAzureOpenAI(
                api_key=self._config.api_key,
                azure_endpoint=self._config.azure_endpoint,
                api_version=self._config.api_version,
                http_client=http_client,
                timeout=self._config.timeout,
                max_retries=0,  # explicit retries below for better visibility/control
            )
        return self._client

    def _is_retryable(self, e: Exception) -> bool:
        """
        Best-effort classification of transient errors worth retrying.
        We use name-based checks so tests can simulate these without constructing SDK exceptions.
        """
        name = type(e).__name__
        if name in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "ServiceUnavailableError",
        }:
            return True
        if name in {"ConnectError", "ReadTimeout", "RemoteProtocolError"}:
            return True
        return False

    async def _with_retries(self, fn, *, op_name: str):
        attempts = max(0, int(self._config.max_retries or 0))
        base_delay = 0.5
        max_delay = 8.0

        last_err: Optional[Exception] = None
        for attempt in range(attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last_err = e
                if attempt >= attempts or not self._is_retryable(e):
                    raise

                delay = min(max_delay, base_delay * (2**attempt))
                delay = delay * (0.75 + 0.5 * random.random())
                retry_after = _retry_after_seconds_from_exception(e)
                if retry_after is not None:
                    delay = max(delay, float(retry_after))

                req = getattr(e, "request", None)
                req_url = None
                try:
                    req_url = getattr(req, "url", None) if req is not None else None
                except Exception:
                    req_url = None
                body = getattr(e, "body", None)
                activity_id = None
                try:
                    if isinstance(body, dict):
                        activity_id = body.get("activityId") or body.get("activity_id")
                except Exception:
                    activity_id = None

                logger.warning(
                    f"{op_name} transient failure; retrying "
                    f"(attempt {attempt + 1}/{attempts}, sleep={delay:.2f}s): "
                    f"{type(e).__name__}: {e}"
                    + (f" | url={req_url}" if req_url else "")
                    + (f" | activityId={activity_id}" if activity_id else "")
                )
                await asyncio.sleep(delay)

        if last_err is not None:
            raise last_err
        raise RuntimeError(f"{op_name} failed with unknown error")

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(
                embeddings=[], model=model or self.default_model, total_tokens=0, latency_ms=0.0
            )

        start = time.time()
        client = await self._get_client()

        async def _do_request():
            return await client.embeddings.create(
                model=model or self._config.embedding_deployment,
                input=texts,
                **kwargs,
            )

        response = await self._with_retries(_do_request, op_name="azure_openai.embeddings.create")

        embeddings = sorted(response.data, key=lambda x: x.index)
        if embeddings:
            self._dimensions = len(embeddings[0].embedding)

        usage = getattr(response, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

        return EmbeddingResponse(
            embeddings=[e.embedding for e in embeddings],
            model=model or self._config.embedding_deployment,
            total_tokens=total_tokens,
            latency_ms=(time.time() - start) * 1000,
        )

    async def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        resp = await self.embed([text], model=model, **kwargs)
        return resp.embeddings[0] if resp.embeddings else []

    def _default_dimensions_for_model(self, model: str) -> int:
        # Best-effort mapping; in Azure the deployment name may not equal the underlying model name.
        # Allow users to override via EmbeddingConfig.dimension elsewhere when needed.
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(model, 1536)


# Register providers for config-driven wiring
registry.register_llm("azure")(AzureOpenAILLMProvider)
registry.register_embedding("azure")(AzureOpenAIEmbeddingProvider)
# Common alias
registry.register_llm("azure_openai")(AzureOpenAILLMProvider)
registry.register_embedding("azure_openai")(AzureOpenAIEmbeddingProvider)
