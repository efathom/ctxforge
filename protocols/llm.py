"""
LLM Provider Protocol Interfaces.

Defines the contracts for LLM and embedding providers.
Supports multiple providers (OpenAI, Anthropic, Azure, local models)
with a unified interface.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Response from an LLM generation call."""
    
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on token counts (placeholder)."""
        # This would be provider-specific in real implementations
        return (self.input_tokens * 0.00001) + (self.output_tokens * 0.00003)


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    
    role: str  # system, user, assistant, function
    content: str
    name: Optional[str] = None  # For function messages
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """Response from an embedding generation call."""
    
    embeddings: List[List[float]]
    model: str
    total_tokens: int = 0
    latency_ms: float = 0.0


@runtime_checkable
class ILLMProvider(Protocol):
    """
    Protocol for LLM providers.
    
    Implementations provide text/chat generation capabilities.
    Should handle:
    - Both completion and chat APIs
    - Streaming responses
    - Token counting
    - Error handling and retries
    
    Example implementations:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Azure OpenAI
    - Local (Ollama, vLLM)
    """
    
    @property
    def name(self) -> str:
        """The name of this provider."""
        ...
    
    @property
    def default_model(self) -> str:
        """The default model for this provider."""
        ...
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion for a prompt.
        
        Args:
            prompt: The input prompt
            model: The model to use (defaults to provider's default)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with the generated content and metadata
            
        Raises:
            LLMError: If generation fails
            RateLimitError: If rate limited
            TokenLimitError: If token limit exceeded
        """
        ...
    
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
        """
        Generate a chat completion.
        
        Args:
            messages: The conversation history
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stop: Stop sequences
            functions: Function definitions for function calling
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with the generated content and metadata
        """
        ...
    
    async def stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion.
        
        Args:
            messages: The conversation history
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of the generated response
        """
        ...
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            model: The model to use for tokenization
            
        Returns:
            The number of tokens
        """
        ...
    
    def count_message_tokens(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
    ) -> int:
        """
        Count tokens in a list of chat messages.
        
        Args:
            messages: The messages to count tokens for
            model: The model to use for tokenization
            
        Returns:
            The total number of tokens
        """
        ...


@runtime_checkable
class IEmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.
    
    Implementations provide text embedding capabilities for
    semantic search and similarity matching.
    
    Example implementations:
    - OpenAI embeddings
    - Cohere embeddings
    - Sentence Transformers (local)
    """
    
    @property
    def name(self) -> str:
        """The name of this provider."""
        ...
    
    @property
    def default_model(self) -> str:
        """The default embedding model."""
        ...
    
    @property
    def embedding_dimension(self) -> int:
        """The dimension of embeddings produced."""
        ...
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts.
        
        Args:
            texts: The texts to embed
            model: The model to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            EmbeddingResponse with the embeddings and metadata
        """
        ...
    
    async def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Convenience method for single text embedding.
        
        Args:
            text: The text to embed
            model: The model to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The embedding vector
        """
        ...

