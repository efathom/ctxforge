"""
OpenAI Provider implementations.

Provides LLM and Embedding providers using OpenAI's API.
"""

import time
from dataclasses import dataclass
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


@dataclass
class OpenAIConfig:
    """
    OpenAI API configuration.

    Attributes:
        api_key: OpenAI API key
        model: Chat model to use (default: gpt-4)
        embedding_model: Embedding model (default: text-embedding-3-small)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """

    api_key: str
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 1000
    temperature: float = 0.7
    # Optional override of the OpenAI-compatible endpoint (e.g. a local
    # Text Embeddings Inference / Ollama / vLLM server).
    base_url: Optional[str] = None


class OpenAILLMProvider(ILLMProvider):
    """
    OpenAI LLM provider for chat completions.

    Example:
        from ctxforge.llm import OpenAILLMProvider, OpenAIConfig

        config = OpenAIConfig(api_key="sk-...")
        provider = OpenAILLMProvider(config)

        response = await provider.generate(
            prompt="What is Python?",
            system_prompt="You are a helpful assistant.",
        )
    """

    def __init__(self, config: OpenAIConfig):
        """
        Initialize the provider.

        Args:
            config: OpenAI configuration
        """
        self._config = config
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._config.model

    async def _get_client(self):
        """Lazy initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from None
            self._client = AsyncOpenAI(
                api_key=self._config.api_key, base_url=self._config.base_url or None
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI's chat API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to the API

        Returns:
            LLMResponse with content and metadata
        """
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

        request_kwargs: Dict[str, Any] = {
            "model": model or self._config.model,
            "messages": openai_messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "stop": stop,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        request_kwargs.update(kwargs)

        resp = await client.chat.completions.create(**request_kwargs)

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
            model=model or self._config.model,
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
        raise NotImplementedError("OpenAI streaming not implemented in this provider yet.")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        # Best-effort heuristic until tokenizer_provider is wired in engine.
        return int(len(text.split()) * 1.3)

    def count_message_tokens(self, messages: List[ChatMessage], model: Optional[str] = None) -> int:
        total = 0
        for m in messages:
            total += 4  # role overhead heuristic
            total += self.count_tokens(m.content, model=model)
        return total


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """
    OpenAI Embedding provider.

    Example:
        from ctxforge.llm import OpenAIEmbeddingProvider, OpenAIConfig

        config = OpenAIConfig(api_key="sk-...")
        provider = OpenAIEmbeddingProvider(config)

        embedding = await provider.embed("Hello world")
        embeddings = await provider.embed_batch(["Hello", "World"])
    """

    def __init__(self, config: OpenAIConfig):
        """
        Initialize the provider.

        Args:
            config: OpenAI configuration
        """
        self._config = config
        self._client = None
        self._dimensions: Optional[int] = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._config.embedding_model

    @property
    def embedding_dimension(self) -> int:
        return self._dimensions or self._default_dimensions_for_model(self._config.embedding_model)

    async def _get_client(self):
        """Lazy initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from None
            self._client = AsyncOpenAI(
                api_key=self._config.api_key, base_url=self._config.base_url or None
            )
        return self._client

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.

        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        if not texts:
            return EmbeddingResponse(
                embeddings=[], model=model or self.default_model, total_tokens=0, latency_ms=0.0
            )

        start = time.time()
        client = await self._get_client()

        response = await client.embeddings.create(
            model=model or self._config.embedding_model,
            input=texts,
            **kwargs,
        )

        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)

        if embeddings:
            self._dimensions = len(embeddings[0].embedding)

        usage = getattr(response, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

        return EmbeddingResponse(
            embeddings=[e.embedding for e in embeddings],
            model=model or self._config.embedding_model,
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
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(model, 1536)


# Register providers for config-driven wiring
registry.register_llm("openai")(OpenAILLMProvider)
registry.register_embedding("openai")(OpenAIEmbeddingProvider)
