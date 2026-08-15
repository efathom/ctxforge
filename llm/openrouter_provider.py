"""
OpenRouter provider implementation.

OpenRouter exposes an OpenAI-compatible API, so this provider reuses
``OpenAILLMProvider`` and only changes the base URL (and optional
attribution headers). No embeddings provider is provided because OpenRouter
does not currently offer an embeddings endpoint.
"""

from dataclasses import dataclass
from typing import Optional

from ctxforge.engine.registry import registry
from ctxforge.llm.openai_provider import OpenAIConfig, OpenAILLMProvider

# Default OpenRouter API endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class OpenRouterConfig(OpenAIConfig):
    """
    OpenRouter API configuration.

    Attributes:
        api_key: OpenRouter API key (e.g. ``sk-or-v1-...``)
        model: OpenRouter model slug (e.g. ``openai/gpt-4o-mini``)
        base_url: API base URL (defaults to OpenRouter's endpoint)
        http_referer: Optional site URL for OpenRouter attribution
        site_title: Optional site/app title for OpenRouter attribution
    """

    base_url: str = OPENROUTER_BASE_URL
    http_referer: Optional[str] = None
    site_title: Optional[str] = None


class OpenRouterLLMProvider(OpenAILLMProvider):
    """
    LLM provider for the OpenRouter API.

    Example:
        from ctxforge.llm.openrouter_provider import OpenRouterConfig, OpenRouterLLMProvider

        config = OpenRouterConfig(api_key="sk-or-v1-...", model="openai/gpt-4o-mini")
        provider = OpenRouterLLMProvider(config)

        response = await provider.chat([...])
    """

    def __init__(self, config: OpenRouterConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "openrouter"

    async def _get_client(self):
        """Lazy-initialize the OpenAI client pointed at OpenRouter."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from None

            headers: dict = {}
            if self._config.http_referer:
                headers["HTTP-Referer"] = self._config.http_referer
            if self._config.site_title:
                headers["X-Title"] = self._config.site_title

            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                default_headers=headers or None,
            )
        return self._client


# Register for config-driven wiring (``llm.provider: openrouter``).
registry.register_llm("openrouter")(OpenRouterLLMProvider)
