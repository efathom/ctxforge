"""
LLM providers for the ctxforge framework.

Provides implementations of LLM and embedding providers:
- Mock provider (testing)
- OpenAI provider
- Anthropic provider (coming soon)
- Azure OpenAI provider
- Local providers (Ollama, vLLM) (coming soon)
"""

from ctxforge.llm.azure_openai_provider import (
    AzureOpenAIConfig,
    AzureOpenAIEmbeddingProvider,
    AzureOpenAILLMProvider,
)
from ctxforge.llm.local_embedding_provider import (
    LocalEmbeddingConfig,
    LocalEmbeddingProvider,
)
from ctxforge.llm.mock_provider import MockEmbeddingProvider, MockLLMProvider
from ctxforge.llm.openai_provider import (
    OpenAIConfig,
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
)
from ctxforge.llm.openrouter_provider import OpenRouterConfig, OpenRouterLLMProvider
from ctxforge.llm.task_model_resolver import TaskModelResolver, TaskType

__all__ = [
    # Mock (testing)
    "MockLLMProvider",
    "MockEmbeddingProvider",
    # OpenAI
    "OpenAIConfig",
    "OpenAILLMProvider",
    "OpenAIEmbeddingProvider",
    # OpenRouter
    "OpenRouterConfig",
    "OpenRouterLLMProvider",
    # Local / HuggingFace embeddings
    "LocalEmbeddingConfig",
    "LocalEmbeddingProvider",
    # Azure OpenAI
    "AzureOpenAIConfig",
    "AzureOpenAILLMProvider",
    "AzureOpenAIEmbeddingProvider",
    # Task model routing
    "TaskModelResolver",
    "TaskType",
]

