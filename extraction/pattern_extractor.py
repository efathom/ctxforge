"""
Pattern-based Memory Extractor.

Uses regex patterns to identify and extract memories from text.
Fast and predictable, good for common patterns like preferences
and personal facts.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ctxforge.core.memory import MemoryType
from ctxforge.extraction.base import BaseExtractor
from ctxforge.extraction.utils import clean_extraction
from ctxforge.protocols.extractor import (
    ExtractionCandidate,
    ExtractionConfig,
)
from ctxforge.utils.similarity import ISimilarityCalculator

logger = logging.getLogger(__name__)


@dataclass
class PatternRule:
    """A pattern rule for extraction."""
    
    pattern: str
    memory_type: MemoryType
    confidence: float
    tags: List[str] = field(default_factory=list)
    template: Optional[str] = None  # Format string for extracted content
    flags: int = re.IGNORECASE
    
    def compile(self) -> re.Pattern:
        """Compile the regex pattern."""
        return re.compile(self.pattern, self.flags)


# Default extraction patterns organized by category
DEFAULT_PATTERNS: Dict[str, List[PatternRule]] = {
    "preferences": [
        PatternRule(
            pattern=r"\bi\s+(?:really\s+)?(?:love|adore)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            tags=["preference", "positive"],
            template="User loves {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:really\s+)?(?:like|enjoy|prefer)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.80,
            tags=["preference", "positive"],
            template="User likes {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:really\s+)?(?:hate|dislike|can't stand|detest)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            tags=["preference", "negative"],
            template="User dislikes {0}",
        ),
        PatternRule(
            pattern=r"\bmy\s+favorite\s+(.+?)\s+is\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.90,
            tags=["preference", "favorite"],
            template="User's favorite {0} is {1}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:always|usually|typically)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.75,
            tags=["preference", "habit"],
            template="User typically {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:never|rarely|seldom)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.75,
            tags=["preference", "avoidance"],
            template="User rarely/never {0}",
        ),
    ],
    
    "personal_facts": [
        PatternRule(
            pattern=r"\bi\s+am\s+(?:a|an)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            tags=["identity", "profession"],
            template="User is a {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+work\s+(?:at|for)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.90,
            tags=["work", "employer"],
            template="User works at {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+work\s+as\s+(?:a|an)?\s*(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.90,
            tags=["work", "profession"],
            template="User works as {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+live\s+in\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.90,
            tags=["location", "residence"],
            template="User lives in {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:have|own)\s+(?:a|an)?\s*(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.75,
            tags=["possession"],
            template="User has {0}",
        ),
        PatternRule(
            pattern=r"\bmy\s+name\s+is\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.95,
            tags=["identity", "name"],
            template="User's name is {0}",
        ),
        PatternRule(
            pattern=r"\bi'm\s+(\d+)\s+years?\s+old",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.90,
            tags=["identity", "age"],
            template="User is {0} years old",
        ),
        PatternRule(
            pattern=r"\bi\s+speak\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            tags=["skill", "language"],
            template="User speaks {0}",
        ),
    ],
    
    "experiences": [
        PatternRule(
            pattern=r"\bi\s+(?:went|traveled|visited)\s+(?:to\s+)?(.+?)(?:\s+last|\s+yesterday|\s+recently|\.|,|!|$)",
            memory_type=MemoryType.EPISODIC,
            confidence=0.80,
            tags=["travel", "experience"],
            template="User visited {0}",
        ),
        PatternRule(
            pattern=r"\blast\s+(?:week|month|year)\s+i\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.EPISODIC,
            confidence=0.75,
            tags=["recent", "experience"],
            template="Recently, user {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:just|recently)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.EPISODIC,
            confidence=0.75,
            tags=["recent", "experience"],
            template="User recently {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+used\s+to\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.EPISODIC,
            confidence=0.70,
            tags=["past", "experience"],
            template="User used to {0}",
        ),
    ],
    
    "goals_interests": [
        PatternRule(
            pattern=r"\bi\s+(?:want|need)\s+to\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.70,
            tags=["goal", "desire"],
            template="User wants to {0}",
        ),
        PatternRule(
            pattern=r"\bi'm\s+(?:interested|curious)\s+(?:in|about)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.80,
            tags=["interest"],
            template="User is interested in {0}",
        ),
        PatternRule(
            pattern=r"\bi'm\s+(?:learning|studying)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.80,
            tags=["learning", "goal"],
            template="User is learning {0}",
        ),
        PatternRule(
            pattern=r"\bi'm\s+(?:trying|working)\s+(?:to|on)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.75,
            tags=["goal", "current"],
            template="User is working on {0}",
        ),
    ],
    
    "skills_expertise": [
        PatternRule(
            pattern=r"\bi\s+(?:know|understand)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.70,
            tags=["knowledge", "skill"],
            template="User knows {0}",
        ),
        PatternRule(
            pattern=r"\bi'm\s+(?:good|great|excellent)\s+at\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            tags=["skill", "expertise"],
            template="User is skilled at {0}",
        ),
        PatternRule(
            pattern=r"\bi\s+(?:can|am able to)\s+(.+?)(?:\.|,|!|$)",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.70,
            tags=["ability", "skill"],
            template="User can {0}",
        ),
    ],
}


class PatternExtractor(BaseExtractor):
    """
    Pattern-based memory extractor.
    
    Uses regex patterns to identify common expressions of preferences,
    facts, and experiences. Fast and predictable.
    
    Example:
        extractor = PatternExtractor()
        result = await extractor.extract(
            user_input="I love Italian food and I live in New York",
            agent_response="That's great!"
        )
        # Returns candidates for "loves Italian food" and "lives in New York"
    """
    
    def __init__(
        self,
        patterns: Optional[Dict[str, List[PatternRule]]] = None,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the pattern extractor.
        
        Args:
            patterns: Custom patterns (uses defaults if not provided)
            default_config: Default extraction configuration
            similarity_calculator: Calculator for text similarity
        """
        super().__init__(default_config, similarity_calculator)
        
        self._patterns = patterns or DEFAULT_PATTERNS
        self._compiled_patterns: List[Tuple[re.Pattern, PatternRule]] = []
        self._compile_patterns()
    
    @property
    def name(self) -> str:
        """The name of this extractor."""
        return "pattern"
    
    def _compile_patterns(self) -> None:
        """Compile all regex patterns for efficiency."""
        self._compiled_patterns = []
        
        for category, rules in self._patterns.items():
            for rule in rules:
                try:
                    compiled = rule.compile()
                    self._compiled_patterns.append((compiled, rule))
                except re.error as e:
                    # Log but don't fail on bad patterns
                    logger.warning("Invalid pattern in %s: %s", category, e)
    
    def add_pattern(
        self,
        pattern: str,
        memory_type: MemoryType,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        template: Optional[str] = None,
        category: str = "custom",
    ) -> None:
        """
        Add a custom extraction pattern.
        
        Args:
            pattern: Regex pattern with capture groups
            memory_type: Type of memory to create
            confidence: Confidence score for matches
            tags: Tags to add to extracted memories
            template: Optional format string for content
            category: Category name for the pattern
        """
        rule = PatternRule(
            pattern=pattern,
            memory_type=memory_type,
            confidence=confidence,
            tags=tags or [],
            template=template,
        )
        
        if category not in self._patterns:
            self._patterns[category] = []
        
        self._patterns[category].append(rule)
        
        # Add to compiled patterns
        try:
            compiled = rule.compile()
            self._compiled_patterns.append((compiled, rule))
        except re.error as e:
            raise ValueError(f"Invalid pattern: {e}") from e
    
    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """
        Extract memories using pattern matching.
        
        Args:
            text: The text to extract from
            config: Extraction configuration
            context: Optional additional context
            
        Returns:
            List of extraction candidates
        """
        if not text:
            return []
        
        candidates = []
        
        # Also use custom patterns from config if provided
        if config.custom_patterns:
            for pattern_str, template in config.custom_patterns.items():
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    matches = pattern.findall(text)
                    for match in matches:
                        content = self._format_match(match, template)
                        if content:
                            candidates.append(ExtractionCandidate(
                                content=clean_extraction(content),
                                memory_type=MemoryType.SEMANTIC,
                                confidence=0.75,
                                source_text=text,
                                tags=["custom"],
                            ))
                except re.error:
                    continue
        
        # Apply all compiled patterns
        for compiled_pattern, rule in self._compiled_patterns:
            matches = compiled_pattern.findall(text)
            
            for match in matches:
                content = self._format_match(match, rule.template)
                
                if content and len(content) > 3:  # Minimum content length
                    candidates.append(ExtractionCandidate(
                        content=clean_extraction(content),
                        memory_type=rule.memory_type,
                        confidence=rule.confidence,
                        source_text=text,
                        tags=list(rule.tags),
                        metadata={"pattern_category": self._get_pattern_category(rule)},
                    ))
        
        # Deduplicate similar candidates
        return self._deduplicate_candidates(candidates)
    
    def _format_match(
        self,
        match: Any,
        template: Optional[str],
    ) -> Optional[str]:
        """
        Format a regex match into content.
        
        Args:
            match: The regex match (string or tuple)
            template: Optional format template
            
        Returns:
            Formatted content string
        """
        if not match:
            return None
        
        # Handle both single group and multiple group matches
        if isinstance(match, str):
            groups = [match]
        else:
            groups = list(match)
        
        # Clean up groups
        groups = [g.strip() if isinstance(g, str) else str(g) for g in groups]
        groups = [g for g in groups if g]  # Remove empty
        
        if not groups:
            return None
        
        if template:
            try:
                return template.format(*groups)
            except (IndexError, KeyError):
                return groups[0]
        else:
            return groups[0]
    
    def _get_pattern_category(self, rule: PatternRule) -> str:
        """Get the category a rule belongs to."""
        for category, rules in self._patterns.items():
            if rule in rules:
                return category
        return "unknown"
    
    def get_patterns_summary(self) -> Dict[str, int]:
        """
        Get a summary of available patterns.
        
        Returns:
            Dictionary mapping category to pattern count
        """
        return {
            category: len(rules)
            for category, rules in self._patterns.items()
        }

