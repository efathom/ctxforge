"""
Tests for LLM providers.
"""

import pytest

from ctxforge.llm.mock_provider import MockEmbeddingProvider, MockLLMProvider
from ctxforge.protocols.llm import ChatMessage


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a fresh provider for each test."""
        return MockLLMProvider(latency_ms=10)  # Fast for testing
    
    def test_properties(self, provider):
        """Test provider properties."""
        assert provider.name == "mock"
        assert provider.default_model == "mock-model"
    
    @pytest.mark.asyncio
    async def test_generate(self, provider):
        """Test basic generation."""
        response = await provider.generate("Hello, world!")
        
        assert response.content is not None
        assert response.model == "mock-model"
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_model(self, provider):
        """Test generation with custom model."""
        response = await provider.generate(
            "Hello",
            model="custom-model",
        )
        
        assert response.model == "custom-model"
    
    @pytest.mark.asyncio
    async def test_chat(self, provider):
        """Test chat completion."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!"),
        ]
        
        response = await provider.chat(messages)
        
        assert response.content is not None
        assert response.input_tokens > 0
    
    @pytest.mark.asyncio
    async def test_chat_stores_messages(self, provider):
        """Test that chat stores last messages."""
        messages = [
            ChatMessage(role="user", content="Test message"),
        ]
        
        await provider.chat(messages)
        
        assert provider.last_messages == messages
    
    @pytest.mark.asyncio
    async def test_set_responses(self, provider):
        """Test setting custom responses."""
        provider.set_responses([
            "Response 1",
            "Response 2",
            "Response 3",
        ])
        
        r1 = await provider.generate("Q1")
        assert r1.content == "Response 1"
        
        r2 = await provider.generate("Q2")
        assert r2.content == "Response 2"
        
        r3 = await provider.generate("Q3")
        assert r3.content == "Response 3"
        
        # Cycles back
        r4 = await provider.generate("Q4")
        assert r4.content == "Response 1"
    
    @pytest.mark.asyncio
    async def test_call_count(self, provider):
        """Test call counting."""
        assert provider.call_count == 0
        
        await provider.generate("Q1")
        assert provider.call_count == 1
        
        await provider.chat([ChatMessage(role="user", content="Q2")])
        assert provider.call_count == 2
    
    @pytest.mark.asyncio
    async def test_stream(self, provider):
        """Test streaming generation."""
        provider.set_responses(["Hello world"])
        
        chunks = []
        async for chunk in provider.stream([
            ChatMessage(role="user", content="Test"),
        ]):
            chunks.append(chunk)
        
        assert len(chunks) == 2  # "Hello " and "world"
        assert "".join(chunks) == "Hello world"
    
    def test_count_tokens(self, provider):
        """Test token counting."""
        text = "Hello world this is a test"
        tokens = provider.count_tokens(text)
        
        # Approximately 1.3 tokens per word
        assert 6 <= tokens <= 10
    
    def test_count_message_tokens(self, provider):
        """Test counting message tokens."""
        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello world"),
        ]
        
        tokens = provider.count_message_tokens(messages)
        
        # Should include overhead + content
        assert tokens > 0


class TestMockEmbeddingProvider:
    """Tests for MockEmbeddingProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a fresh provider for each test."""
        return MockEmbeddingProvider(dimension=128, latency_ms=5)
    
    def test_properties(self, provider):
        """Test provider properties."""
        assert provider.name == "mock"
        assert provider.default_model == "mock-embedding"
        assert provider.embedding_dimension == 128
    
    @pytest.mark.asyncio
    async def test_embed(self, provider):
        """Test embedding generation."""
        response = await provider.embed(["Hello", "World"])
        
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 128
        assert len(response.embeddings[1]) == 128
        assert response.model == "mock-embedding"
    
    @pytest.mark.asyncio
    async def test_embed_single(self, provider):
        """Test single text embedding."""
        embedding = await provider.embed_single("Hello world")
        
        assert len(embedding) == 128
    
    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self, provider):
        """Test that embeddings are normalized."""
        embedding = await provider.embed_single("Test")
        
        # Calculate magnitude
        magnitude = sum(x**2 for x in embedding) ** 0.5
        
        # Should be approximately 1.0
        assert abs(magnitude - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_embeddings_are_deterministic(self, provider):
        """Test that same text produces same embedding."""
        emb1 = await provider.embed_single("Hello world")
        emb2 = await provider.embed_single("Hello world")
        
        assert emb1 == emb2
    
    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, provider):
        """Test that different texts produce different embeddings."""
        emb1 = await provider.embed_single("Hello")
        emb2 = await provider.embed_single("World")
        
        assert emb1 != emb2
    
    @pytest.mark.asyncio
    async def test_embedding_cached(self, provider):
        """Test that embeddings are cached."""
        await provider.embed_single("Test")
        await provider.embed_single("Test")
        
        # Should use cached value
        assert "Test" in provider._cache

