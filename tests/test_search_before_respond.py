"""
Tests for Search-Before-Respond Middleware.

Tests the search-first directive injection for knowledge-seeking queries.
"""

from unittest.mock import AsyncMock

import pytest

from ctxforge.middleware.protocol import MiddlewareContext
from ctxforge.middleware.search_before_respond import (
    SearchBeforeRespondMiddleware,
    SearchIntentClassifier,
)


class TestSearchBeforeRespondMiddleware:
    """Tests for SearchBeforeRespondMiddleware."""
    
    @pytest.fixture
    def middleware(self):
        """Create middleware with default settings."""
        return SearchBeforeRespondMiddleware()
    
    def test_middleware_name(self, middleware):
        """Test middleware name."""
        assert middleware.name == "search_before_respond"
    
    def test_should_inject_for_questions(self, middleware):
        """Test that questions trigger directive injection."""
        assert middleware.should_inject_directive("How do I query the database?")
        assert middleware.should_inject_directive("What is the schema for users?")
        assert middleware.should_inject_directive("Why did the query fail?")
        assert middleware.should_inject_directive("Where can I find this data?")
    
    def test_should_inject_for_question_marks(self, middleware):
        """Test that question marks trigger injection."""
        assert middleware.should_inject_directive("Can you help me?")
        assert middleware.should_inject_directive("Is this correct?")
    
    def test_should_not_inject_for_greetings(self, middleware):
        """Test that greetings don't trigger injection."""
        assert not middleware.should_inject_directive("Hello")
        assert not middleware.should_inject_directive("Hi!")
        assert not middleware.should_inject_directive("Thanks")
        assert not middleware.should_inject_directive("Bye")
    
    def test_should_not_inject_for_empty(self, middleware):
        """Test that empty input doesn't trigger injection."""
        assert not middleware.should_inject_directive("")
        assert not middleware.should_inject_directive(None)
    
    def test_should_inject_for_commands(self, middleware):
        """Test that command-like queries trigger injection."""
        assert middleware.should_inject_directive("Show me all users")
        assert middleware.should_inject_directive("List the recent orders")
        assert middleware.should_inject_directive("Find customers in Seattle")
    
    def test_generate_directive(self, middleware):
        """Test directive generation."""
        directive = middleware.generate_directive()
        
        assert "Search Requirement" in directive
        assert "search the knowledge base" in directive
        assert "search_expertise" in directive or "search" in directive.lower()
    
    def test_custom_domains(self):
        """Test middleware with custom domains."""
        middleware = SearchBeforeRespondMiddleware(
            knowledge_domains=["SQL queries", "business rules"],
        )
        
        directive = middleware.generate_directive()
        
        assert "SQL queries" in directive
        assert "business rules" in directive
    
    def test_custom_keywords(self):
        """Test middleware with custom trigger keywords."""
        middleware = SearchBeforeRespondMiddleware(
            trigger_keywords=["fetch", "retrieve"],
        )
        
        assert middleware.should_inject_directive("Fetch user data")
        assert middleware.should_inject_directive("Can you retrieve orders?")
        # Default keywords should not apply
        assert not middleware.should_inject_directive("Hello there")
    
    @pytest.mark.asyncio
    async def test_process_injects_directive(self, middleware):
        """Test that processing injects directive into context."""
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="How do I query users?",
        )
        
        async def mock_next(ctx):
            return ctx
        
        result = await middleware.process(context, mock_next)
        
        assert result.get_metadata("search_before_respond_active") is True
        
        directives = result.get_metadata("system_directives")
        assert directives is not None
        assert len(directives) > 0
    
    @pytest.mark.asyncio
    async def test_process_skips_non_questions(self, middleware):
        """Test that processing skips non-questions."""
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="Hello",
        )
        
        async def mock_next(ctx):
            return ctx
        
        result = await middleware.process(context, mock_next)
        
        assert result.get_metadata("search_before_respond_active") is None
        assert result.get_metadata("system_directives") is None
    
    @pytest.mark.asyncio
    async def test_disabled_middleware(self):
        """Test that disabled middleware passes through."""
        middleware = SearchBeforeRespondMiddleware(enabled=False)
        
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="How do I query users?",
        )
        
        async def mock_next(ctx):
            return ctx
        
        result = await middleware.process(context, mock_next)
        
        assert result.get_metadata("search_before_respond_active") is None
    
    @pytest.mark.asyncio
    async def test_auto_search(self):
        """Test automatic search when enabled."""
        mock_searcher = AsyncMock(return_value=[
            {"content": "Result 1"},
            {"content": "Result 2"},
        ])
        
        middleware = SearchBeforeRespondMiddleware(
            auto_search=True,
            searcher=mock_searcher,
            max_results=5,
        )
        
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="How do I query users?",
        )
        
        async def mock_next(ctx):
            return ctx
        
        result = await middleware.process(context, mock_next)
        
        mock_searcher.assert_called_once()
        results = result.get_metadata("auto_search_results")
        assert results is not None
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_auto_search_failure_handled(self):
        """Test that auto search failures are handled gracefully."""
        mock_searcher = AsyncMock(side_effect=Exception("Search failed"))
        
        middleware = SearchBeforeRespondMiddleware(
            auto_search=True,
            searcher=mock_searcher,
        )
        
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="How do I query users?",
        )
        
        async def mock_next(ctx):
            return ctx
        
        # Should not raise
        result = await middleware.process(context, mock_next)
        
        # Directive should still be injected
        assert result.get_metadata("search_before_respond_active") is True


class TestSearchIntentClassifier:
    """Tests for SearchIntentClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier."""
        return SearchIntentClassifier()

    @pytest.mark.asyncio
    async def test_classify_query_intent(self, classifier):
        """Test classifying query intent."""
        result = await classifier.classify("How do I query the users table?")
        assert "query" in result.intents or "procedure" in result.intents

        result = await classifier.classify("Write a SQL query for this")
        assert "query" in result.intents

    @pytest.mark.asyncio
    async def test_classify_lookup_intent(self, classifier):
        """Test classifying lookup intent."""
        result = await classifier.classify("What is the schema for users?")
        assert "lookup" in result.intents

        result = await classifier.classify("Tell me about the order status field")
        assert "lookup" in result.intents

    @pytest.mark.asyncio
    async def test_classify_procedure_intent(self, classifier):
        """Test classifying procedure intent."""
        result = await classifier.classify("How should I handle this case?")
        assert "procedure" in result.intents

        result = await classifier.classify("What are the steps to process a refund?")
        assert "procedure" in result.intents

    @pytest.mark.asyncio
    async def test_classify_validation_intent(self, classifier):
        """Test classifying validation intent."""
        result = await classifier.classify("Is this correct?")
        assert "validation" in result.intents

        result = await classifier.classify("Does this look right?")
        assert "validation" in result.intents

    @pytest.mark.asyncio
    async def test_classify_comparison_intent(self, classifier):
        """Test classifying comparison intent."""
        result = await classifier.classify("What is the difference between A and B?")
        assert "comparison" in result.intents

        result = await classifier.classify("Compare inner join vs outer join")
        assert "comparison" in result.intents

    @pytest.mark.asyncio
    async def test_classify_multiple_intents(self, classifier):
        """Test that multiple intents can be detected."""
        # This query might have both lookup and comparison intents
        result = await classifier.classify("What is the difference between varchar and text?")
        assert len(result.intents) >= 1  # At least one intent

    @pytest.mark.asyncio
    async def test_classify_no_intent(self, classifier):
        """Test classifying input with no clear intent."""
        result = await classifier.classify("Hello there")
        assert len(result.intents) == 0

    @pytest.mark.asyncio
    async def test_classify_returns_confidence(self, classifier):
        """Test that classify returns confidence scores."""
        result = await classifier.classify("How do I write a SQL query?")
        assert result.confidence > 0
        assert result.method == "pattern"
        assert len(result.intent_scores) > 0
    
    def test_get_search_domains_query(self, classifier):
        """Test getting search domains for query intent."""
        domains = classifier.get_search_domains({"query"})
        
        assert "query_patterns" in domains or "sql_examples" in domains
    
    def test_get_search_domains_validation(self, classifier):
        """Test getting search domains for validation intent."""
        domains = classifier.get_search_domains({"validation"})
        
        assert "rules" in domains or "gotchas" in domains
    
    def test_get_search_domains_multiple(self, classifier):
        """Test getting search domains for multiple intents."""
        domains = classifier.get_search_domains({"query", "validation"})
        
        assert len(domains) > 0
    
    def test_get_search_domains_empty(self, classifier):
        """Test getting search domains for no intents."""
        domains = classifier.get_search_domains(set())

        assert domains == ["general"]


class TestIntentClassificationResult:
    """Tests for IntentClassificationResult dataclass."""

    def test_primary_intent_from_scores(self):
        """Test getting primary intent from scores."""
        from ctxforge.middleware.search_before_respond import IntentClassificationResult

        result = IntentClassificationResult(
            intents={"query", "procedure"},
            confidence=0.9,
            intent_scores={"query": 0.9, "procedure": 0.7},
            method="test",
        )

        assert result.primary_intent == "query"

    def test_primary_intent_from_intents_set(self):
        """Test getting primary intent when no scores."""
        from ctxforge.middleware.search_before_respond import IntentClassificationResult

        result = IntentClassificationResult(
            intents={"lookup"},
            confidence=0.8,
            intent_scores={},
            method="test",
        )

        assert result.primary_intent == "lookup"

    def test_primary_intent_empty(self):
        """Test primary intent when empty."""
        from ctxforge.middleware.search_before_respond import IntentClassificationResult

        result = IntentClassificationResult(
            intents=set(),
            confidence=0.0,
            method="test",
        )

        assert result.primary_intent is None


class TestEmbeddingIntentClassifier:
    """Tests for EmbeddingIntentClassifier."""

    @pytest.mark.asyncio
    async def test_uninitialized_returns_empty(self):
        """Test that uninitialized classifier returns empty result."""
        from ctxforge.middleware.search_before_respond import EmbeddingIntentClassifier

        classifier = EmbeddingIntentClassifier(embedding_provider=None)

        result = await classifier.classify("How do I query users?")

        assert len(result.intents) == 0
        assert result.confidence == 0.0
        assert "uninitialized" in result.method

    @pytest.mark.asyncio
    async def test_with_mock_embedder(self):
        """Test classifier with mock embedding provider."""
        from ctxforge.middleware.search_before_respond import EmbeddingIntentClassifier

        # Create a simple mock embedder
        class MockEmbedder:
            async def embed(self, text: str) -> list:
                # Return a simple hash-based embedding
                import hashlib
                h = hashlib.md5(text.lower().encode()).hexdigest()
                return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

            async def embed_batch(self, texts: list) -> list:
                return [await self.embed(t) for t in texts]

        classifier = EmbeddingIntentClassifier(
            embedding_provider=MockEmbedder(),
            similarity_threshold=0.5,
        )
        await classifier.initialize()

        result = await classifier.classify("How do I query users?")

        # With mock embedder, results will vary but should work
        assert result.method == "embedding"


class TestLLMIntentClassifier:
    """Tests for LLMIntentClassifier."""

    @pytest.mark.asyncio
    async def test_no_llm_with_fallback(self):
        """Test fallback to pattern classifier when no LLM."""
        from ctxforge.middleware.search_before_respond import LLMIntentClassifier

        classifier = LLMIntentClassifier(
            llm_provider=None,
            fallback_to_pattern=True,
        )

        result = await classifier.classify("How do I write a SQL query?")

        assert result.method == "pattern_fallback"
        assert len(result.intents) > 0

    @pytest.mark.asyncio
    async def test_no_llm_no_fallback(self):
        """Test empty result when no LLM and no fallback."""
        from ctxforge.middleware.search_before_respond import LLMIntentClassifier

        classifier = LLMIntentClassifier(
            llm_provider=None,
            fallback_to_pattern=False,
        )

        result = await classifier.classify("How do I write a SQL query?")

        assert result.method == "llm_unavailable"
        assert len(result.intents) == 0

    @pytest.mark.asyncio
    async def test_with_mock_llm(self):
        """Test classifier with mock LLM provider."""
        from ctxforge.middleware.search_before_respond import LLMIntentClassifier

        class MockLLM:
            async def complete(self, prompt: str) -> str:
                # Return a mock JSON response
                return '{"intents": ["query", "procedure"], "confidence": 0.95, "reasoning": "test"}'

        classifier = LLMIntentClassifier(llm_provider=MockLLM())

        result = await classifier.classify("How do I write a SQL query?")

        assert result.method == "llm"
        assert "query" in result.intents
        assert "procedure" in result.intents
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_handles_malformed_json(self):
        """Test handling of malformed LLM response."""
        from ctxforge.middleware.search_before_respond import LLMIntentClassifier

        class BadLLM:
            async def complete(self, prompt: str) -> str:
                return "This is not valid JSON"

        classifier = LLMIntentClassifier(
            llm_provider=BadLLM(),
            fallback_to_pattern=True,
        )

        result = await classifier.classify("How do I write a query?")

        # Should fall back to pattern classifier
        assert result.method == "pattern_fallback"

    @pytest.mark.asyncio
    async def test_handles_json_in_markdown(self):
        """Test handling of JSON wrapped in markdown."""
        from ctxforge.middleware.search_before_respond import LLMIntentClassifier

        class MarkdownLLM:
            async def complete(self, prompt: str) -> str:
                return '''```json
{"intents": ["lookup"], "confidence": 0.85, "reasoning": "User wants info"}
```'''

        classifier = LLMIntentClassifier(llm_provider=MarkdownLLM())

        result = await classifier.classify("What is a foreign key?")

        assert result.method == "llm"
        assert "lookup" in result.intents

    @pytest.mark.asyncio
    async def test_filters_invalid_intents(self):
        """Test that invalid intents are filtered out."""
        from ctxforge.middleware.search_before_respond import LLMIntentClassifier

        class InvalidIntentLLM:
            async def complete(self, prompt: str) -> str:
                return '{"intents": ["query", "invalid_intent", "made_up"], "confidence": 0.9}'

        classifier = LLMIntentClassifier(llm_provider=InvalidIntentLLM())

        result = await classifier.classify("Write a query")

        assert "query" in result.intents
        assert "invalid_intent" not in result.intents
        assert "made_up" not in result.intents


class TestHybridIntentClassifier:
    """Tests for HybridIntentClassifier."""

    @pytest.mark.asyncio
    async def test_uses_pattern_when_confident(self):
        """Test that high-confidence patterns are used."""
        from ctxforge.middleware.search_before_respond import HybridIntentClassifier

        classifier = HybridIntentClassifier(
            embedding_provider=None,
            pattern_confidence_threshold=0.8,
            combine_results=False,
        )

        result = await classifier.classify("How do I query users by email?")

        assert "pattern" in result.method
        assert len(result.intents) > 0

    @pytest.mark.asyncio
    async def test_combines_results(self):
        """Test that results can be combined."""
        from ctxforge.middleware.search_before_respond import HybridIntentClassifier

        classifier = HybridIntentClassifier(
            embedding_provider=None,
            combine_results=True,
        )

        result = await classifier.classify("What is the difference between A and B?")

        # Should have results from pattern classifier
        assert result.method == "hybrid_combined"

    @pytest.mark.asyncio
    async def test_get_search_domains(self):
        """Test domain mapping works."""
        from ctxforge.middleware.search_before_respond import HybridIntentClassifier

        classifier = HybridIntentClassifier()

        result = await classifier.classify("Write a SQL query for this")
        domains = classifier.get_search_domains(result.intents)

        assert len(domains) > 0


class TestClassifyIntentFunction:
    """Tests for the classify_intent convenience function."""

    @pytest.mark.asyncio
    async def test_default_classifier(self):
        """Test using default classifier."""
        from ctxforge.middleware.search_before_respond import classify_intent

        result = await classify_intent("How do I write a query?")

        assert result.confidence >= 0
        assert isinstance(result.intents, set)

    @pytest.mark.asyncio
    async def test_custom_classifier(self):
        """Test using custom classifier."""
        from ctxforge.middleware.search_before_respond import (
            SearchIntentClassifier,
            classify_intent,
        )

        classifier = SearchIntentClassifier()
        result = await classify_intent("Explain this concept", classifier=classifier)

        assert "lookup" in result.intents
