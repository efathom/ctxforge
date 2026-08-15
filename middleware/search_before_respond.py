"""
Search-Before-Respond Middleware.

A middleware that ensures the agent searches the knowledge base
before generating a response. This prevents hallucination by
forcing knowledge retrieval for eligible queries.

If a question requires domain knowledge, first perform an expertise
search. This ensures you get the right context before responding.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction
from ctxforge.utils.math import cosine_similarity

logger = logging.getLogger(__name__)


# =============================================================================
# Classification Result
# =============================================================================

@dataclass
class IntentClassificationResult:
    """Result of intent classification with confidence scores."""
    intents: Set[str]
    confidence: float  # Overall confidence (0.0 to 1.0)
    intent_scores: Dict[str, float] = field(default_factory=dict)  # Per-intent scores
    method: str = "unknown"  # Classification method used

    @property
    def primary_intent(self) -> Optional[str]:
        """Get the highest-scoring intent."""
        if not self.intent_scores:
            return list(self.intents)[0] if self.intents else None
        return max(self.intent_scores, key=self.intent_scores.get)


# =============================================================================
# Embedding Provider Protocol
# =============================================================================

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


class SearchBeforeRespondMiddleware(BaseMiddleware):
    """
    Middleware that injects a search-first directive into context.
    
    When enabled, this middleware:
    1. Analyzes the user query for knowledge-seeking intent
    2. Injects a directive to search before responding
    3. Optionally triggers automatic retrieval
    
    Example usage in agent setup:
    ```python
    middleware = SearchBeforeRespondMiddleware(
        knowledge_domains=["sql_queries", "business_rules", "gotchas"],
        trigger_keywords=["how", "what", "why", "query", "find"],
    )
    engine.add_middleware(middleware)
    ```
    
    The injected directive appears as a system instruction:
    "Before answering questions about [domain], search the knowledge base
     using search_expertise or search_memories to retrieve relevant context."
    """
    
    def __init__(
        self,
        knowledge_domains: Optional[List[str]] = None,
        trigger_keywords: Optional[List[str]] = None,
        auto_search: bool = False,
        searcher: Optional[Callable] = None,
        max_results: int = 5,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            knowledge_domains: Domains to search (for directive text)
            trigger_keywords: Keywords that trigger the directive
            auto_search: If True, automatically perform search
            searcher: Optional search function for auto_search
            max_results: Maximum results for auto search
            enabled: Whether this middleware is active
        """
        super().__init__(enabled=enabled)
        self._domains = knowledge_domains or [
            "domain knowledge",
            "past experiences",
            "documented patterns",
        ]
        self._keywords = set(kw.lower() for kw in (trigger_keywords or [
            "how", "what", "why", "where", "when", "which",
            "explain", "describe", "tell me", "find", "get",
            "query", "show", "list", "calculate", "determine",
        ]))
        self._auto_search = auto_search
        self._searcher = searcher
        self._max_results = max_results
    
    @property
    def name(self) -> str:
        return "search_before_respond"
    
    def should_inject_directive(self, user_input: str) -> bool:
        """
        Check if the user input warrants a search-first directive.
        
        Args:
            user_input: The user's query
            
        Returns:
            True if directive should be injected
        """
        if not user_input:
            return False
        
        input_lower = user_input.lower()
        
        # Check for question-like patterns
        is_question = (
            "?" in user_input or
            any(input_lower.startswith(kw) for kw in self._keywords) or
            any(f" {kw} " in f" {input_lower} " for kw in self._keywords)
        )
        
        # Exclude simple greetings
        greetings = {"hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"}
        is_greeting = input_lower.strip().rstrip("!.,") in greetings
        
        return is_question and not is_greeting
    
    def generate_directive(self) -> str:
        """
        Generate the search-before-respond directive.
        
        Returns:
            Directive text to inject into context
        """
        domains_str = ", ".join(self._domains)
        
        return (
            f"**Search Requirement**: Before answering questions about {domains_str}, "
            f"you must first search the knowledge base using available search tools "
            f"(search_expertise, search_memories, or semantic_search). "
            f"Only provide answers based on retrieved knowledge. "
            f"If no relevant knowledge is found, state that clearly before proceeding."
        )
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """Process the context, optionally injecting search directive."""
        
        user_input = context.user_input or ""
        
        if self.should_inject_directive(user_input):
            # Inject the directive
            directive = self.generate_directive()
            
            # Add to context metadata for system prompt assembly
            existing_directives = context.get_metadata("system_directives") or []
            existing_directives.append(directive)
            context.set_metadata("system_directives", existing_directives)
            
            # Mark that search-first was triggered
            context.set_metadata("search_before_respond_active", True)
            
            logger.debug(f"Injected search-before-respond directive for: {user_input[:50]}...")
            
            # Optionally perform automatic search
            if self._auto_search and self._searcher:
                try:
                    results = await self._searcher(user_input, limit=self._max_results)
                    context.set_metadata("auto_search_results", results)
                    logger.debug(f"Auto-search returned {len(results)} results")
                except Exception as e:
                    logger.warning(f"Auto-search failed: {e}")
        
        return await next(context)


# =============================================================================
# Base Intent Classifier
# =============================================================================

class BaseIntentClassifier(ABC):
    """Abstract base class for intent classifiers."""

    # Standard intent to domain mapping
    DOMAIN_MAPPING = {
        "query": ["query_patterns", "sql_examples", "schema"],
        "lookup": ["definitions", "facts", "entities"],
        "procedure": ["procedures", "workflows", "steps"],
        "validation": ["rules", "constraints", "gotchas"],
        "comparison": ["definitions", "comparisons", "facts"],
        "troubleshooting": ["errors", "debugging", "solutions"],
        "example": ["examples", "samples", "templates"],
    }

    @abstractmethod
    async def classify(self, user_input: str) -> IntentClassificationResult:
        """
        Classify the intent(s) of a user query.

        Args:
            user_input: The user's query

        Returns:
            IntentClassificationResult with intents and confidence
        """
        pass

    def get_search_domains(self, intents: Set[str]) -> List[str]:
        """
        Map intents to relevant search domains.

        Args:
            intents: Set of detected intents

        Returns:
            List of knowledge domains to search
        """
        domains = set()
        for intent in intents:
            domains.update(self.DOMAIN_MAPPING.get(intent, []))

        return list(domains) if domains else ["general"]


# =============================================================================
# Pattern-Based Classifier (Original)
# =============================================================================

class SearchIntentClassifier(BaseIntentClassifier):
    """
    Pattern-based intent classifier using regex.

    Fast but limited to predefined patterns.
    Best for: Quick classification with known query patterns.
    """

    # Intent patterns with weights
    PATTERNS = {
        "query": [
            (r"how\s+do\s+i\s+(?:query|get|find|retrieve)", 1.0),
            (r"(?:sql|query)\s+(?:for|to)", 0.9),
            (r"write\s+(?:a\s+)?(?:sql|query)", 0.9),
            (r"select\s+.+\s+from", 0.8),
        ],
        "lookup": [
            (r"what\s+is\s+(?:the|a)", 0.9),
            (r"tell\s+me\s+about", 0.8),
            (r"explain\s+", 0.85),
            (r"describe\s+", 0.8),
            (r"define\s+", 0.9),
        ],
        "procedure": [
            (r"how\s+(?:do|can|should)\s+i", 0.9),
            (r"what\s+(?:are\s+)?the\s+steps", 0.95),
            (r"walk\s+me\s+through", 0.9),
            (r"step\s+by\s+step", 0.85),
            (r"guide\s+me", 0.8),
        ],
        "validation": [
            (r"is\s+(?:this|it)\s+(?:correct|right|valid)", 0.95),
            (r"does\s+this\s+look\s+(?:right|correct)", 0.9),
            (r"verify\s+", 0.85),
            (r"check\s+(?:if|whether)", 0.85),
            (r"validate\s+", 0.9),
        ],
        "comparison": [
            (r"(?:what\s+is\s+)?the\s+difference\s+between", 0.95),
            (r"compare\s+", 0.9),
            (r"\bvs\.?\b", 0.7),
            (r"versus\b", 0.8),
            (r"which\s+(?:is|one)\s+better", 0.85),
        ],
        "troubleshooting": [
            (r"(?:why|how)\s+(?:is|does)\s+.+\s+(?:not|fail|error)", 0.9),
            (r"fix\s+(?:this|the)", 0.85),
            (r"debug\s+", 0.9),
            (r"(?:getting|got)\s+(?:an?\s+)?error", 0.85),
            (r"doesn't\s+work", 0.8),
        ],
        "example": [
            (r"(?:show|give)\s+(?:me\s+)?(?:an?\s+)?example", 0.95),
            (r"for\s+example", 0.7),
            (r"sample\s+(?:code|query)", 0.9),
            (r"can\s+you\s+demonstrate", 0.85),
        ],
    }

    def __init__(self):
        self._compiled_patterns: Dict[str, List[Tuple[Any, float]]] = {}
        for intent, patterns in self.PATTERNS.items():
            self._compiled_patterns[intent] = [
                (re.compile(p, re.IGNORECASE), weight)
                for p, weight in patterns
            ]

    async def classify(self, user_input: str) -> IntentClassificationResult:
        """Classify using regex patterns with confidence scores."""
        intent_scores: Dict[str, float] = {}

        for intent, patterns in self._compiled_patterns.items():
            max_score = 0.0
            for pattern, weight in patterns:
                if pattern.search(user_input):
                    max_score = max(max_score, weight)
            if max_score > 0:
                intent_scores[intent] = max_score

        intents = set(intent_scores.keys())
        confidence = max(intent_scores.values()) if intent_scores else 0.0

        return IntentClassificationResult(
            intents=intents,
            confidence=confidence,
            intent_scores=intent_scores,
            method="pattern",
        )

    # Backward compatibility
    def classify_sync(self, user_input: str) -> Set[str]:
        """Synchronous classification (returns just intents)."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(self.classify(user_input))
        return result.intents


# =============================================================================
# Embedding-Based Classifier
# =============================================================================

class EmbeddingIntentClassifier(BaseIntentClassifier):
    """
    Embedding-based intent classifier using semantic similarity.

    Compares query embeddings against intent exemplar embeddings.
    Best for: Semantic understanding of varied query phrasings.

    Example:
        ```python
        classifier = EmbeddingIntentClassifier(embedding_provider=my_embedder)
        await classifier.initialize()  # Pre-compute intent embeddings
        result = await classifier.classify("How can I optimize my database?")
        ```
    """

    # Intent exemplars - representative queries for each intent
    INTENT_EXEMPLARS = {
        "query": [
            "How do I query the database?",
            "Write a SQL query to find users",
            "Get data from the orders table",
            "Retrieve records matching this condition",
            "Search for items in the database",
        ],
        "lookup": [
            "What is a foreign key?",
            "Explain the concept of normalization",
            "Tell me about database indexes",
            "Describe what a transaction is",
            "Define the term 'primary key'",
        ],
        "procedure": [
            "How do I create a new table?",
            "What are the steps to set up replication?",
            "Walk me through the migration process",
            "Guide me on configuring the connection pool",
            "Step by step instructions for backup",
        ],
        "validation": [
            "Is this query correct?",
            "Does this schema look right?",
            "Verify if my approach is valid",
            "Check whether this will work",
            "Am I doing this correctly?",
        ],
        "comparison": [
            "What is the difference between INNER and OUTER join?",
            "Compare PostgreSQL and MySQL",
            "Which is better, NoSQL or SQL?",
            "Pros and cons of each approach",
            "How does A differ from B?",
        ],
        "troubleshooting": [
            "Why is my query not working?",
            "Getting an error when connecting",
            "Fix the performance issue",
            "Debug this slow query",
            "Query fails with timeout",
        ],
        "example": [
            "Show me an example of a JOIN",
            "Give me a sample INSERT statement",
            "Can you demonstrate with code?",
            "Example of using transactions",
            "Sample query for aggregation",
        ],
    }

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.7,
        top_k_intents: int = 3,
    ):
        """
        Initialize the embedding-based classifier.

        Args:
            embedding_provider: Provider for generating embeddings
            similarity_threshold: Minimum similarity to consider a match
            top_k_intents: Maximum number of intents to return
        """
        self._embedder = embedding_provider
        self._threshold = similarity_threshold
        self._top_k = top_k_intents
        self._intent_embeddings: Dict[str, List[List[float]]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-compute embeddings for intent exemplars."""
        if self._initialized or self._embedder is None:
            return

        logger.info("Initializing intent embeddings...")

        for intent, exemplars in self.INTENT_EXEMPLARS.items():
            embeddings = await self._embedder.embed_batch(exemplars)
            self._intent_embeddings[intent] = embeddings

        self._initialized = True
        logger.info(f"Initialized embeddings for {len(self._intent_embeddings)} intents")

    async def classify(self, user_input: str) -> IntentClassificationResult:
        """Classify using embedding similarity."""
        if not self._initialized or self._embedder is None:
            # Fall back to empty result if not initialized
            return IntentClassificationResult(
                intents=set(),
                confidence=0.0,
                intent_scores={},
                method="embedding_uninitialized",
            )

        # Get query embedding
        query_embedding = await self._embedder.embed(user_input)

        # Compare against each intent's exemplars
        intent_scores: Dict[str, float] = {}

        for intent, exemplar_embeddings in self._intent_embeddings.items():
            # Use max similarity across all exemplars for this intent
            similarities = [
                cosine_similarity(query_embedding, exemplar)
                for exemplar in exemplar_embeddings
            ]
            max_similarity = max(similarities) if similarities else 0.0
            if max_similarity >= self._threshold:
                intent_scores[intent] = max_similarity

        # Sort and take top k
        sorted_intents = sorted(
            intent_scores.items(), key=lambda x: x[1], reverse=True
        )[:self._top_k]

        intents = set(intent for intent, _ in sorted_intents)
        intent_scores = dict(sorted_intents)
        confidence = max(intent_scores.values()) if intent_scores else 0.0

        return IntentClassificationResult(
            intents=intents,
            confidence=confidence,
            intent_scores=intent_scores,
            method="embedding",
        )


# =============================================================================
# LLM-Based Classifier
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str) -> str:
        """Generate completion for prompt."""
        ...


class LLMIntentClassifier(BaseIntentClassifier):
    """
    LLM-based intent classifier using natural language understanding.

    Uses an LLM to classify query intents with high accuracy.
    Best for: Complex queries, nuanced understanding, edge cases.

    Example:
        ```python
        classifier = LLMIntentClassifier(llm_provider=my_llm)
        result = await classifier.classify("How can I optimize this slow query?")
        print(f"Intents: {result.intents}, Confidence: {result.confidence:.0%}")
        ```
    """

    # Available intents for classification
    AVAILABLE_INTENTS = [
        "query",          # Writing/executing database queries
        "lookup",         # Looking up definitions, facts, information
        "procedure",      # Step-by-step instructions, how-to guides
        "validation",     # Checking if something is correct/valid
        "comparison",     # Comparing two or more things
        "troubleshooting",  # Debugging, fixing errors, solving problems
        "example",        # Requesting examples, samples, demonstrations
    ]

    CLASSIFICATION_PROMPT = '''Classify the user's query into one or more intent categories.

Available categories:
- query: Writing or executing database queries, SQL statements
- lookup: Looking up definitions, facts, concepts, information
- procedure: Step-by-step instructions, how-to guides, processes
- validation: Checking if something is correct, valid, or right
- comparison: Comparing two or more things, differences, trade-offs
- troubleshooting: Debugging, fixing errors, solving problems
- example: Requesting examples, samples, code demonstrations

User query: "{query}"

Respond with a JSON object containing:
- "intents": list of matching intent names (from the categories above)
- "confidence": overall confidence score from 0.0 to 1.0
- "reasoning": brief explanation of why these intents were chosen

Example response:
{{"intents": ["query", "procedure"], "confidence": 0.9, "reasoning": "Asking how to write"}}

Your response (JSON only):'''

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        temperature: float = 0.0,
        fallback_to_pattern: bool = True,
    ):
        """
        Initialize the LLM-based classifier.

        Args:
            llm_provider: Provider for LLM completions
            temperature: LLM temperature (0.0 for deterministic)
            fallback_to_pattern: If True, fall back to patterns on LLM failure
        """
        self._llm = llm_provider
        self._temperature = temperature
        self._fallback = fallback_to_pattern
        self._pattern_classifier = SearchIntentClassifier() if fallback_to_pattern else None

    async def classify(self, user_input: str) -> IntentClassificationResult:
        """Classify using LLM."""
        if self._llm is None:
            # No LLM, use fallback or return empty
            if self._fallback and self._pattern_classifier:
                result = await self._pattern_classifier.classify(user_input)
                return IntentClassificationResult(
                    intents=result.intents,
                    confidence=result.confidence,
                    intent_scores=result.intent_scores,
                    method="pattern_fallback",
                )
            return IntentClassificationResult(
                intents=set(),
                confidence=0.0,
                intent_scores={},
                method="llm_unavailable",
            )

        try:
            # Build prompt
            prompt = self.CLASSIFICATION_PROMPT.format(query=user_input)

            # Get LLM response
            response = await self._llm.complete(prompt)

            # Parse JSON response
            result = self._parse_response(response)

            return result

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

            # Fallback to pattern classifier
            if self._fallback and self._pattern_classifier:
                result = await self._pattern_classifier.classify(user_input)
                return IntentClassificationResult(
                    intents=result.intents,
                    confidence=result.confidence,
                    intent_scores=result.intent_scores,
                    method="pattern_fallback",
                )

            return IntentClassificationResult(
                intents=set(),
                confidence=0.0,
                intent_scores={},
                method="llm_error",
            )

    def _parse_response(self, response: str) -> IntentClassificationResult:
        """Parse LLM JSON response into classification result."""
        import json

        # Clean response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {response[:100]}") from None

        # Validate and extract intents
        raw_intents = data.get("intents", [])
        valid_intents = set(
            i for i in raw_intents
            if i in self.AVAILABLE_INTENTS
        )

        confidence = float(data.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Create intent scores (equal weight for LLM-classified intents)
        intent_scores = {intent: confidence for intent in valid_intents}

        return IntentClassificationResult(
            intents=valid_intents,
            confidence=confidence,
            intent_scores=intent_scores,
            method="llm",
        )


# =============================================================================
# Hybrid Classifier
# =============================================================================

class HybridIntentClassifier(BaseIntentClassifier):
    """
    Hybrid intent classifier combining pattern and embedding approaches.

    Uses fast pattern matching first, falls back to embedding similarity
    when confidence is low.

    Example:
        ```python
        classifier = HybridIntentClassifier(
            embedding_provider=my_embedder,
            pattern_confidence_threshold=0.8,
        )
        await classifier.initialize()
        result = await classifier.classify("How do I optimize queries?")
        print(f"Intents: {result.intents}, Method: {result.method}")
        ```
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        pattern_confidence_threshold: float = 0.8,
        embedding_similarity_threshold: float = 0.7,
        combine_results: bool = True,
    ):
        """
        Initialize the hybrid classifier.

        Args:
            embedding_provider: Provider for generating embeddings
            pattern_confidence_threshold: Confidence above which patterns are trusted
            embedding_similarity_threshold: Minimum similarity for embedding matches
            combine_results: If True, combine pattern + embedding results
        """
        self._pattern_classifier = SearchIntentClassifier()
        self._embedding_classifier = EmbeddingIntentClassifier(
            embedding_provider=embedding_provider,
            similarity_threshold=embedding_similarity_threshold,
        )
        self._pattern_threshold = pattern_confidence_threshold
        self._combine = combine_results

    async def initialize(self) -> None:
        """Initialize the embedding classifier."""
        await self._embedding_classifier.initialize()

    async def classify(self, user_input: str) -> IntentClassificationResult:
        """
        Classify using hybrid approach.

        Strategy:
        1. First, try pattern-based classification
        2. If confidence >= threshold, use pattern results
        3. If confidence < threshold, use embedding-based classification
        4. If combine_results=True, merge results from both
        """
        # Step 1: Pattern-based classification
        pattern_result = await self._pattern_classifier.classify(user_input)

        # Step 2: Check if patterns are confident enough
        if pattern_result.confidence >= self._pattern_threshold and not self._combine:
            return IntentClassificationResult(
                intents=pattern_result.intents,
                confidence=pattern_result.confidence,
                intent_scores=pattern_result.intent_scores,
                method="pattern",
            )

        # Step 3: Try embedding-based classification
        embedding_result = await self._embedding_classifier.classify(user_input)

        # Step 4: Combine or select best
        if self._combine:
            # Merge results, taking max score per intent
            combined_scores: Dict[str, float] = {}

            for intent, score in pattern_result.intent_scores.items():
                combined_scores[intent] = score

            for intent, score in embedding_result.intent_scores.items():
                if intent in combined_scores:
                    combined_scores[intent] = max(combined_scores[intent], score)
                else:
                    combined_scores[intent] = score

            intents = set(combined_scores.keys())
            confidence = max(combined_scores.values()) if combined_scores else 0.0

            return IntentClassificationResult(
                intents=intents,
                confidence=confidence,
                intent_scores=combined_scores,
                method="hybrid_combined",
            )
        else:
            # Use embedding result if pattern confidence was low
            if embedding_result.confidence > pattern_result.confidence:
                return IntentClassificationResult(
                    intents=embedding_result.intents,
                    confidence=embedding_result.confidence,
                    intent_scores=embedding_result.intent_scores,
                    method="embedding_fallback",
                )
            else:
                return IntentClassificationResult(
                    intents=pattern_result.intents,
                    confidence=pattern_result.confidence,
                    intent_scores=pattern_result.intent_scores,
                    method="pattern",
                )


# =============================================================================
# Convenience function
# =============================================================================

async def classify_intent(
    user_input: str,
    classifier: Optional[BaseIntentClassifier] = None,
) -> IntentClassificationResult:
    """
    Classify user intent using the provided or default classifier.

    Args:
        user_input: The user's query
        classifier: Optional classifier instance (defaults to pattern-based)

    Returns:
        IntentClassificationResult with intents and confidence
    """
    if classifier is None:
        classifier = SearchIntentClassifier()

    return await classifier.classify(user_input)
