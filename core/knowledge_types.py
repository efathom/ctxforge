"""
Structured Knowledge Types.

Provides explicit classification for knowledge items to improve
retrieval and context assembly.

Knowledge types help the agent understand:
- What KIND of knowledge this is (rule vs pattern vs example)
- When it should be retrieved (always vs on-demand)
- How it should be rendered in context
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class KnowledgeType(str, Enum):
    """
    Classification of knowledge items.
    
    Each type has different retrieval and rendering behavior:
    - RULE: Always-applicable constraints (render in system context)
    - PATTERN: Reusable templates/queries (retrieve on-demand)
    - GOTCHA: Common mistakes to avoid (high-priority retrieval)
    - EXAMPLE: Reference implementations (retrieve for similar queries)
    - DEFINITION: Domain term definitions (retrieve on-demand)
    - PROCEDURE: Step-by-step workflows (retrieve for how-to queries)
    """
    RULE = "rule"                  # Must be followed, always applicable
    PATTERN = "pattern"            # Reusable template (SQL, code, etc.)
    GOTCHA = "gotcha"              # Common mistake to avoid
    EXAMPLE = "example"            # Reference implementation
    DEFINITION = "definition"      # Domain term definition
    PROCEDURE = "procedure"        # Step-by-step workflow
    INSIGHT = "insight"            # General learning/observation
    CONSTRAINT = "constraint"      # Limit or boundary condition


class KnowledgeScope(str, Enum):
    """
    Scope/applicability of knowledge.
    """
    GLOBAL = "global"              # Applies everywhere
    ENTITY = "entity"              # Applies to specific entity/table
    OPERATION = "operation"        # Applies to specific operation
    USER = "user"                  # User-specific


class StructuredKnowledge(BaseModel):
    """
    A structured knowledge item with explicit typing.
    
    This extends the base ExpertiseItem/MemoryItem with:
    - Explicit knowledge_type classification
    - Scope (what it applies to)
    - Conditions (when it applies)
    - Priority (retrieval ranking)
    
    Example - SQL rule:
    ```python
    StructuredKnowledge(
        knowledge_type=KnowledgeType.RULE,
        content="Always use TO_DATE when filtering by date columns",
        scope=KnowledgeScope.ENTITY,
        applies_to=["race_results.date", "race_wins.date"],
        priority=10,  # High priority
    )
    ```
    
    Example - Query pattern:
    ```python
    StructuredKnowledge(
        knowledge_type=KnowledgeType.PATTERN,
        name="Championship Winners by Year",
        content="SELECT year, name FROM drivers_championship WHERE position = 1",
        scope=KnowledgeScope.OPERATION,
        applies_to=["championship queries"],
        source_question="Who won the championship in [year]?",
    )
    ```
    """
    # Classification
    knowledge_type: KnowledgeType
    scope: KnowledgeScope = KnowledgeScope.GLOBAL
    
    # Content
    name: Optional[str] = None
    content: str
    summary: Optional[str] = None
    
    # Applicability
    applies_to: List[str] = Field(default_factory=list)  # Entities, operations, etc.
    conditions: List[str] = Field(default_factory=list)  # When this applies
    
    # Provenance
    source_question: Optional[str] = None  # Question that led to this
    source_answer: Optional[str] = None
    validated_by: Optional[str] = None
    
    # Retrieval hints
    priority: int = Field(default=5, ge=1, le=10)  # 1-10, higher = more important
    tags: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def matches_entity(self, entity: str) -> bool:
        """Check if this knowledge applies to an entity."""
        if self.scope == KnowledgeScope.GLOBAL:
            return True
        return entity in self.applies_to or any(
            entity.startswith(a) for a in self.applies_to
        )
    
    def to_prompt_format(self) -> str:
        """Convert to format for LLM prompt."""
        type_prefix = {
            KnowledgeType.RULE: "📋 RULE",
            KnowledgeType.PATTERN: "📝 PATTERN",
            KnowledgeType.GOTCHA: "⚠️ GOTCHA",
            KnowledgeType.EXAMPLE: "💡 EXAMPLE",
            KnowledgeType.DEFINITION: "📖 DEFINITION",
            KnowledgeType.PROCEDURE: "📌 PROCEDURE",
            KnowledgeType.INSIGHT: "💭 INSIGHT",
            KnowledgeType.CONSTRAINT: "🚫 CONSTRAINT",
        }
        prefix = type_prefix.get(self.knowledge_type, "")
        
        if self.name:
            return f"{prefix} - {self.name}: {self.content}"
        return f"{prefix}: {self.content}"


class KnowledgeClassifier(ABC):
    """Abstract base class for knowledge classifiers."""
    
    @abstractmethod
    async def classify(
        self, content: str, context: Optional[str] = None
    ) -> Tuple[KnowledgeType, float]:
        """
        Classify knowledge content.
        
        Args:
            content: The knowledge content to classify
            context: Optional additional context
            
        Returns:
            Tuple of (KnowledgeType, confidence score 0-1)
        """
        ...


class HeuristicKnowledgeClassifier(KnowledgeClassifier):
    """
    Improved heuristic classifier using regex patterns and weighted scoring.
    
    Uses multiple signals:
    - Keyword presence (weighted)
    - Structural patterns (SQL, numbered steps, etc.)
    - Sentence structure analysis
    """
    
    # Weighted patterns: (pattern, weight)
    RULE_PATTERNS = [
        (r'\b(must|shall|always|never|required|mandatory)\b', 1.0),
        (r'\b(ensure|verify|validate)\s+that\b', 0.7),
        (r'^(do not|don\'t)\b', 0.6),
        (r'\bevery\s+(request|call|operation)\b', 0.5),
    ]
    
    GOTCHA_PATTERNS = [
        (r'\b(don\'t forget|watch out|be careful|beware)\b', 1.0),
        (r'\b(common mistake|pitfall|trap|gotcha)\b', 1.0),
        (r'\b(avoid|prevent)\s+\w+ing\b', 0.7),
        (r'\b(can cause|may lead to|will result in)\s+\w*(error|issue|problem|bug)', 0.8),
    ]
    
    PATTERN_PATTERNS = [
        (r'\bSELECT\b.*\bFROM\b', 1.0),  # SQL
        (r'\b(INSERT|UPDATE|DELETE)\s+INTO?\b', 1.0),  # SQL DML
        (r'^(def|function|class|async def)\s+\w+', 1.0),  # Code
        (r'```\w*\n', 0.9),  # Code block
        (r'\{\{.*\}\}', 0.7),  # Template
        (r'\$\{?\w+\}?', 0.5),  # Variable interpolation
    ]
    
    PROCEDURE_PATTERNS = [
        (r'^\s*(step\s*\d+|1\.|first,?)\s*', 1.0),
        (r'\b(then|next|after that|finally)\b', 0.6),
        (r'\b(follow these steps|process|workflow|procedure)\b', 0.8),
        (r'^\s*[-*]\s+\w', 0.4),  # Bullet list
        (r'\d+\)\s+\w', 0.7),  # Numbered list
    ]
    
    DEFINITION_PATTERNS = [
        (r'\b(is defined as|means|refers to|is when)\b', 1.0),
        (r'^[A-Z][a-z]+\s+(is|are)\s+(a|an|the)\b', 0.8),
        (r'\b(definition|meaning|concept)\s*:', 0.9),
        (r'\baka\b|\bi\.e\.\b', 0.5),
    ]
    
    CONSTRAINT_PATTERNS = [
        (r'\b(limit|maximum|minimum|at most|at least)\b', 1.0),
        (r'\b(cannot|must not|should not)\s+exceed\b', 1.0),
        (r'\b(up to|no more than|between)\s+\d+', 0.8),
        (r'\b(restricted|bounded|capped)\b', 0.6),
    ]
    
    EXAMPLE_PATTERNS = [
        (r'\b(for example|e\.g\.|such as|like this)\b', 1.0),
        (r'\b(consider|imagine|suppose)\s+(the following|this)\b', 0.7),
        (r'^example:', 1.0),
        (r'\b(here\'s|here is)\s+(an|a)\s+example\b', 0.9),
    ]
    
    TYPE_PATTERNS = {
        KnowledgeType.RULE: RULE_PATTERNS,
        KnowledgeType.GOTCHA: GOTCHA_PATTERNS,
        KnowledgeType.PATTERN: PATTERN_PATTERNS,
        KnowledgeType.PROCEDURE: PROCEDURE_PATTERNS,
        KnowledgeType.DEFINITION: DEFINITION_PATTERNS,
        KnowledgeType.CONSTRAINT: CONSTRAINT_PATTERNS,
        KnowledgeType.EXAMPLE: EXAMPLE_PATTERNS,
    }
    
    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            min_confidence: Minimum confidence to return a type (else INSIGHT)
        """
        self._min_confidence = min_confidence
        self._compiled_patterns: Dict[KnowledgeType, List[Tuple[re.Pattern, float]]] = {}
        
        for kt, patterns in self.TYPE_PATTERNS.items():
            self._compiled_patterns[kt] = [
                (re.compile(p, re.IGNORECASE | re.MULTILINE), w)
                for p, w in patterns
            ]
    
    async def classify(
        self, content: str, context: Optional[str] = None
    ) -> Tuple[KnowledgeType, float]:
        """Classify using weighted pattern matching."""
        return self._classify_sync(content, context)
    
    def _classify_sync(
        self, content: str, context: Optional[str] = None
    ) -> Tuple[KnowledgeType, float]:
        """Synchronous classification."""
        scores: Dict[KnowledgeType, float] = {}
        
        for kt, patterns in self._compiled_patterns.items():
            max_weight = 0.0
            total_weight = 0.0
            matches = 0
            for pattern, weight in patterns:
                if pattern.search(content):
                    max_weight = max(max_weight, weight)
                    total_weight += weight
                    matches += 1
            
            if matches > 0:
                # Use max weight as base, add bonus for multiple matches
                # This ensures a single high-weight match gives high confidence
                confidence = min(1.0, max_weight + (matches - 1) * 0.15)
                scores[kt] = confidence
        
        if not scores:
            return (KnowledgeType.INSIGHT, 0.5)
        
        # Get best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score < self._min_confidence:
            return (KnowledgeType.INSIGHT, 0.5)
        
        return (best_type, best_score)


class LLMKnowledgeClassifier(KnowledgeClassifier):
    """
    LLM-based knowledge classifier for accurate classification.
    
    Uses an LLM to understand context and nuance that heuristics miss.
    
    Example usage:
    ```python
    classifier = LLMKnowledgeClassifier(llm_provider)
    knowledge_type, confidence = await classifier.classify(
        "Always validate input before saving to database"
    )
    # => (KnowledgeType.RULE, 0.95)
    ```
    """
    
    CLASSIFICATION_PROMPT = '''Classify the following knowledge content into one of these types:
- RULE: Must-follow constraints (contains "must", "always", "never", etc.)
- PATTERN: Reusable templates, SQL queries, code snippets
- GOTCHA: Common mistakes to avoid, pitfalls, warnings
- PROCEDURE: Step-by-step instructions, workflows
- DEFINITION: Explanations of terms, concepts
- CONSTRAINT: Limits, bounds, capacity restrictions
- EXAMPLE: Demonstrations, sample implementations
- INSIGHT: General observations, learnings (default)

Content to classify:
"""
{content}
"""

Respond with ONLY a JSON object:
{{"type": "<TYPE>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}}'''
    
    def __init__(self, llm_provider, model: Optional[str] = None):
        """
        Initialize the LLM classifier.
        
        Args:
            llm_provider: LLM provider implementing ILLMProvider
            model: Optional model override
        """
        self._llm = llm_provider
        self._model = model
    
    async def classify(
        self, content: str, context: Optional[str] = None
    ) -> Tuple[KnowledgeType, float]:
        """Classify using LLM."""
        import json
        
        prompt = self.CLASSIFICATION_PROMPT.format(content=content[:2000])
        
        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model,
                temperature=0.0,
                max_tokens=100,
            )
            
            # Parse JSON response
            result = json.loads(response.text.strip())
            type_str = result.get("type", "INSIGHT").upper()
            confidence = float(result.get("confidence", 0.7))
            
            try:
                knowledge_type = KnowledgeType(type_str.lower())
            except ValueError:
                knowledge_type = KnowledgeType.INSIGHT
            
            return (knowledge_type, confidence)
            
        except Exception:
            # Fallback to heuristic
            heuristic = HeuristicKnowledgeClassifier()
            return await heuristic.classify(content, context)


class HybridKnowledgeClassifier(KnowledgeClassifier):
    """
    Hybrid classifier: fast heuristics first, LLM for uncertain cases.
    
    This provides a good balance of speed and accuracy:
    - High-confidence heuristic matches are used directly
    - Low-confidence cases are sent to LLM for better classification
    """
    
    def __init__(
        self,
        llm_provider=None,
        heuristic_threshold: float = 0.7,
        model: Optional[str] = None,
    ):
        """
        Initialize the hybrid classifier.
        
        Args:
            llm_provider: Optional LLM provider (heuristic-only if not provided)
            heuristic_threshold: Confidence above which to skip LLM
            model: Optional model override for LLM
        """
        self._heuristic = HeuristicKnowledgeClassifier()
        self._llm_classifier = (
            LLMKnowledgeClassifier(llm_provider, model) if llm_provider else None
        )
        self._threshold = heuristic_threshold
    
    async def classify(
        self, content: str, context: Optional[str] = None
    ) -> Tuple[KnowledgeType, float]:
        """Classify using hybrid approach."""
        # Try heuristic first
        kt, confidence = await self._heuristic.classify(content, context)
        
        # If confident enough or no LLM available, return heuristic result
        if confidence >= self._threshold or self._llm_classifier is None:
            return (kt, confidence)
        
        # Use LLM for uncertain cases
        return await self._llm_classifier.classify(content, context)


# Default classifier instance
_default_classifier = HeuristicKnowledgeClassifier()


def classify_knowledge(content: str, context: Optional[str] = None) -> KnowledgeType:
    """
    Classify knowledge content using the default heuristic classifier.
    
    For async usage or LLM-based classification, use the classifier classes directly:
    ```python
    classifier = HybridKnowledgeClassifier(llm_provider)
    knowledge_type, confidence = await classifier.classify(content)
    ```
    
    Args:
        content: The knowledge content to classify
        context: Optional additional context
        
    Returns:
        The classified KnowledgeType
    """
    kt, _ = _default_classifier._classify_sync(content, context)
    return kt


async def classify_knowledge_async(
    content: str,
    context: Optional[str] = None,
    classifier: Optional[KnowledgeClassifier] = None,
) -> Tuple[KnowledgeType, float]:
    """
    Async knowledge classification with confidence score.
    
    Args:
        content: The knowledge content to classify
        context: Optional additional context
        classifier: Optional custom classifier (uses default heuristic if not provided)
        
    Returns:
        Tuple of (KnowledgeType, confidence score)
    """
    if classifier is None:
        classifier = _default_classifier
    return await classifier.classify(content, context)
