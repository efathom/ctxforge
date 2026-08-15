"""
Mock LLM and Embedding providers for testing.

These providers simulate LLM behavior without making actual API calls.
Useful for testing, development, and demonstrations.
"""

import asyncio
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from ctxforge.engine.registry import registry
from ctxforge.protocols.llm import (
    ChatMessage,
    EmbeddingResponse,
    IEmbeddingProvider,
    ILLMProvider,
    LLMResponse,
)


@registry.register_llm("mock")
class MockLLMProvider(ILLMProvider):
    """
    Mock LLM provider for testing.
    
    Returns configurable responses with simulated latency.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        default_response: str = "I have noted that. How else can I help?",
        latency_ms: float = 100.0,
    ):
        """
        Initialize the mock provider.
        
        Args:
            config: Optional configuration dictionary
            default_response: Default response text
            latency_ms: Simulated latency in milliseconds
        """
        self._config = config or {}
        self._default_response = default_response
        self._latency_ms = latency_ms
        self._responses: List[str] = []
        self._call_count = 0
        self._last_messages: Optional[List[ChatMessage]] = None
    
    @property
    def name(self) -> str:
        """The name of this provider."""
        return "mock"
    
    @property
    def default_model(self) -> str:
        """The default model for this provider."""
        return "mock-model"
    
    def set_responses(self, responses: List[str]) -> None:
        """
        Set a sequence of responses to return.
        
        Responses are returned in order, then cycles back.
        
        Args:
            responses: List of response strings
        """
        self._responses = responses
    
    def get_next_response(self) -> str:
        """Get the next response in the sequence."""
        if not self._responses:
            return self._default_response
        
        response = self._responses[self._call_count % len(self._responses)]
        return response
    
    @property
    def call_count(self) -> int:
        """Number of times generate/chat was called."""
        return self._call_count
    
    @property
    def last_messages(self) -> Optional[List[ChatMessage]]:
        """The last messages passed to chat()."""
        return self._last_messages
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion."""
        start_time = time.time()
        
        # Simulate latency
        await asyncio.sleep(self._latency_ms / 1000)
        
        response_text = self.get_next_response()
        self._call_count += 1
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_text,
            model=model or self.default_model,
            input_tokens=len(prompt.split()),
            output_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split()),
            finish_reason="stop",
            latency_ms=latency,
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
        """Generate a chat completion."""
        start_time = time.time()
        
        # Store messages for testing
        self._last_messages = messages
        
        # Simulate latency
        await asyncio.sleep(self._latency_ms / 1000)
        
        response_text = self.get_next_response()
        self._call_count += 1
        
        # Calculate tokens
        input_tokens = sum(len(m.content.split()) for m in messages)
        output_tokens = len(response_text.split())
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_text,
            model=model or self.default_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            finish_reason="stop",
            latency_ms=latency,
        )
    
    async def stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion."""
        response_text = self.get_next_response()
        self._call_count += 1
        
        # Stream word by word
        words = response_text.split()
        for i, word in enumerate(words):
            await asyncio.sleep(self._latency_ms / 1000 / len(words))
            yield word + (" " if i < len(words) - 1 else "")
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text (simple word-based estimation)."""
        # Simple approximation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)
    
    def count_message_tokens(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            # Add overhead for role tokens
            total += 4  # Role overhead
            total += self.count_tokens(msg.content)
        return total


@registry.register_embedding("mock")
class MockEmbeddingProvider(IEmbeddingProvider):
    """
    Mock embedding provider for testing.
    
    Returns random embeddings with consistent dimensions.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dimension: int = 1536,
        latency_ms: float = 50.0,
    ):
        """
        Initialize the mock provider.
        
        Args:
            config: Optional configuration dictionary
            dimension: Embedding dimension
            latency_ms: Simulated latency in milliseconds
        """
        self._config = config or {}
        self._dimension = dimension
        self._latency_ms = latency_ms
        self._cache: Dict[str, List[float]] = {}
    
    @property
    def name(self) -> str:
        """The name of this provider."""
        return "mock"
    
    @property
    def default_model(self) -> str:
        """The default embedding model."""
        return "mock-embedding"
    
    @property
    def embedding_dimension(self) -> int:
        """The dimension of embeddings produced."""
        return self._dimension
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a deterministic embedding for text."""
        # Use hash for reproducibility
        random.seed(hash(text) % (2**32))
        embedding = [random.gauss(0, 1) for _ in range(self._dimension)]
        
        # Normalize
        magnitude = sum(x**2 for x in embedding) ** 0.5
        embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        start_time = time.time()
        
        # Simulate latency
        await asyncio.sleep(self._latency_ms / 1000)
        
        embeddings = []
        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                emb = self._generate_embedding(text)
                self._cache[text] = emb
                embeddings.append(emb)
        
        latency = (time.time() - start_time) * 1000
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model or self.default_model,
            total_tokens=sum(len(t.split()) for t in texts),
            latency_ms=latency,
        )
    
    async def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Generate embedding for a single text."""
        response = await self.embed([text], model, **kwargs)
        return response.embeddings[0]

