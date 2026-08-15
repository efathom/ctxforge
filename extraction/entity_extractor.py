"""
Entity-based Memory Extractor.

Extracts named entities (people, places, dates, organizations, etc.)
from text using pattern-based rules. Optionally integrates with
NLP libraries like spaCy for more sophisticated extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

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
class Entity:
    """A detected named entity."""
    
    text: str
    entity_type: str  # PERSON, LOCATION, DATE, ORGANIZATION, etc.
    start: int
    end: int
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityPatterns:
    """
    Pattern-based entity detection.
    
    Provides regex patterns for common entity types.
    More reliable for structured data, less flexible than NLP.
    """
    
    # Date patterns
    DATE_PATTERNS = [
        # ISO format
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # Written dates
        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})\b',
        r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s*\d{4})\b',
        # Relative dates
        r'\b(last\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b',
        r'\b(next\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b',
        r'\b(yesterday|today|tomorrow)\b',
        # Month/Year
        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
    ]
    
    # Time patterns
    TIME_PATTERNS = [
        r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b',
        r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b',
        r'\b((?:morning|afternoon|evening|night))\b',
    ]
    
    # Email patterns
    EMAIL_PATTERNS = [
        r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
    ]
    
    # Phone patterns
    PHONE_PATTERNS = [
        r'\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b',
        r'\b(\d{3}[-.\s]\d{3}[-.\s]\d{4})\b',
    ]
    
    # URL patterns
    URL_PATTERNS = [
        r'\b(https?://[^\s<>"{}|\\^`\[\]]+)\b',
        r'\b(www\.[^\s<>"{}|\\^`\[\]]+)\b',
    ]
    
    # Money patterns
    MONEY_PATTERNS = [
        r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b',
        r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD)\b',
        r'€(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b',
        r'£(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b',
    ]
    
    # Percentage patterns
    PERCENTAGE_PATTERNS = [
        r'\b(\d+(?:\.\d+)?)\s*%',
        r'\b(\d+(?:\.\d+)?)\s*percent\b',
    ]
    
    # Common location indicators (cities, countries)
    LOCATION_INDICATORS = [
        'in', 'at', 'from', 'to', 'near', 'around',
        'live', 'living', 'lived', 'moved', 'moving',
        'visit', 'visiting', 'visited', 'travel', 'traveling', 'traveled',
    ]
    
    # Common organization suffixes
    ORG_SUFFIXES = [
        'Inc', 'LLC', 'Ltd', 'Corp', 'Corporation', 'Company', 'Co',
        'University', 'College', 'Institute', 'School',
        'Foundation', 'Organization', 'Association',
        'Bank', 'Hospital', 'Clinic',
    ]


class EntityExtractor(BaseExtractor):
    """
    Named entity extractor.
    
    Detects entities like dates, locations, organizations, and
    personal information from text using pattern matching.
    
    Example:
        extractor = EntityExtractor()
        result = await extractor.extract(
            user_input="I moved to San Francisco in January 2023",
            agent_response="Nice! How do you like it?"
        )
        # Returns candidates for location (San Francisco) and date (January 2023)
    """
    
    def __init__(
        self,
        entity_types: Optional[Set[str]] = None,
        use_nlp: bool = False,
        nlp_model: Optional[str] = None,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the entity extractor.
        
        Args:
            entity_types: Types of entities to extract (None = all)
            use_nlp: Whether to use spaCy NLP (requires spacy)
            nlp_model: spaCy model name if using NLP
            default_config: Default extraction configuration
            similarity_calculator: Calculator for text similarity
        """
        super().__init__(default_config, similarity_calculator)
        
        self._entity_types = entity_types or {
            "DATE", "TIME", "LOCATION", "ORGANIZATION",
            "EMAIL", "PHONE", "URL", "MONEY", "PERCENTAGE", "PERSON",
        }
        self._use_nlp = use_nlp
        self._nlp = None
        
        if use_nlp:
            self._init_nlp(nlp_model)
    
    @property
    def name(self) -> str:
        """The name of this extractor."""
        if self._use_nlp and self._nlp:
            return "entity:nlp"
        return "entity:pattern"
    
    def _init_nlp(self, model: Optional[str]) -> None:
        """Initialize spaCy NLP model if available."""
        try:
            import spacy
            model_name = model or "en_core_web_sm"
            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                # Model not downloaded, skip NLP
                logger.warning("spaCy model '%s' not found. Using pattern-only extraction.", model_name)
                self._use_nlp = False
        except ImportError:
            # spaCy not installed
            self._use_nlp = False
    
    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """
        Extract entities from text.
        
        Args:
            text: The text to extract from
            config: Extraction configuration
            context: Optional additional context
            
        Returns:
            List of extraction candidates
        """
        if not text:
            return []
        
        entities = []
        
        # Pattern-based extraction
        entities.extend(self._extract_with_patterns(text))
        
        # NLP-based extraction if available
        if self._use_nlp and self._nlp:
            entities.extend(self._extract_with_nlp(text))
        
        # Convert entities to candidates
        candidates = self._entities_to_candidates(entities, text)
        
        return self._deduplicate_candidates(candidates)
    
    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Date extraction
        if "DATE" in self._entity_types:
            for pattern in EntityPatterns.DATE_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(1),
                        entity_type="DATE",
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.85,
                    ))
        
        # Time extraction
        if "TIME" in self._entity_types:
            for pattern in EntityPatterns.TIME_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(1),
                        entity_type="TIME",
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.85,
                    ))
        
        # Email extraction
        if "EMAIL" in self._entity_types:
            for pattern in EntityPatterns.EMAIL_PATTERNS:
                for match in re.finditer(pattern, text):
                    entities.append(Entity(
                        text=match.group(1),
                        entity_type="EMAIL",
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.95,
                    ))
        
        # Phone extraction
        if "PHONE" in self._entity_types:
            for pattern in EntityPatterns.PHONE_PATTERNS:
                for match in re.finditer(pattern, text):
                    entities.append(Entity(
                        text=match.group(0),
                        entity_type="PHONE",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90,
                    ))
        
        # URL extraction
        if "URL" in self._entity_types:
            for pattern in EntityPatterns.URL_PATTERNS:
                for match in re.finditer(pattern, text):
                    entities.append(Entity(
                        text=match.group(0),
                        entity_type="URL",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    ))
        
        # Money extraction
        if "MONEY" in self._entity_types:
            for pattern in EntityPatterns.MONEY_PATTERNS:
                for match in re.finditer(pattern, text):
                    entities.append(Entity(
                        text=match.group(0),
                        entity_type="MONEY",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90,
                    ))
        
        # Percentage extraction
        if "PERCENTAGE" in self._entity_types:
            for pattern in EntityPatterns.PERCENTAGE_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(0),
                        entity_type="PERCENTAGE",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90,
                    ))
        
        # Location extraction (heuristic-based)
        if "LOCATION" in self._entity_types:
            entities.extend(self._extract_locations(text))
        
        # Organization extraction (heuristic-based)
        if "ORGANIZATION" in self._entity_types:
            entities.extend(self._extract_organizations(text))
        
        # Person name extraction (heuristic-based)
        if "PERSON" in self._entity_types:
            entities.extend(self._extract_persons(text))
        
        return entities
    
    def _extract_locations(self, text: str) -> List[Entity]:
        """Extract potential location entities."""
        entities = []
        
        # Look for capitalized words after location indicators
        for indicator in EntityPatterns.LOCATION_INDICATORS:
            pattern = rf'\b{indicator}\s+(?:to\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
            for match in re.finditer(pattern, text):
                location = match.group(1)
                # Filter out common false positives
                if location.lower() not in ['i', 'the', 'a', 'an', 'my', 'your']:
                    entities.append(Entity(
                        text=location,
                        entity_type="LOCATION",
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.70,
                    ))
        
        return entities
    
    def _extract_organizations(self, text: str) -> List[Entity]:
        """Extract potential organization entities."""
        entities = []
        
        # Look for words followed by organization suffixes
        for suffix in EntityPatterns.ORG_SUFFIXES:
            pattern = rf'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+{suffix})\b'
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    entity_type="ORGANIZATION",
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.80,
                ))
        
        # Also check for "work at/for" patterns
        pattern = r'\bwork(?:s|ing|ed)?\s+(?:at|for)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
        for match in re.finditer(pattern, text):
            entities.append(Entity(
                text=match.group(1),
                entity_type="ORGANIZATION",
                start=match.start(1),
                end=match.end(1),
                confidence=0.75,
            ))
        
        return entities
    
    def _extract_persons(self, text: str) -> List[Entity]:
        """Extract potential person name entities."""
        entities = []
        
        # Look for names after common patterns
        name_patterns = [
            r'\bmy\s+(?:name\s+is|friend|colleague|wife|husband|partner|brother|sister|mom|dad|mother|father)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b(?:this\s+is|meet)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b(?:called|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        ]
        
        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    entity_type="PERSON",
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.75,
                ))
        
        return entities
    
    def _extract_with_nlp(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NLP."""
        entities = []
        
        if not self._nlp:
            return entities
        
        doc = self._nlp(text)
        
        # Map spaCy entity types to our types
        type_map = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",  # Geopolitical entity
            "LOC": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENTAGE",
        }
        
        for ent in doc.ents:
            entity_type = type_map.get(ent.label_)
            
            if entity_type and entity_type in self._entity_types:
                entities.append(Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,
                    metadata={"spacy_label": ent.label_},
                ))
        
        return entities
    
    def _entities_to_candidates(
        self,
        entities: List[Entity],
        source_text: str,
    ) -> List[ExtractionCandidate]:
        """Convert entities to extraction candidates."""
        candidates = []
        
        for entity in entities:
            # Create memory content based on entity type
            content = self._entity_to_content(entity, source_text)
            
            if content:
                # Determine memory type
                memory_type = self._entity_to_memory_type(entity)
                
                candidates.append(ExtractionCandidate(
                    content=clean_extraction(content),
                    memory_type=memory_type,
                    confidence=entity.confidence,
                    source_text=source_text,
                    tags=[entity.entity_type.lower(), "entity"],
                    metadata={
                        "entity_type": entity.entity_type,
                        "entity_text": entity.text,
                    },
                ))
        
        return candidates
    
    def _entity_to_content(self, entity: Entity, source_text: str) -> Optional[str]:
        """Create memory content from an entity."""
        # Context around entity (for potential future use)
        start = max(0, entity.start - 50)
        end = min(len(source_text), entity.end + 50)
        _context = source_text[start:end]  # noqa: F841
        
        # Create content based on entity type
        templates = {
            "DATE": f"User mentioned date: {entity.text}",
            "TIME": f"User mentioned time: {entity.text}",
            "LOCATION": f"User mentioned location: {entity.text}",
            "ORGANIZATION": f"User mentioned organization: {entity.text}",
            "PERSON": f"User mentioned person: {entity.text}",
            "EMAIL": f"User's email: {entity.text}",
            "PHONE": f"User's phone: {entity.text}",
            "URL": f"User mentioned URL: {entity.text}",
            "MONEY": f"User mentioned amount: {entity.text}",
            "PERCENTAGE": f"User mentioned: {entity.text}",
        }
        
        return templates.get(entity.entity_type, f"User mentioned: {entity.text}")
    
    def _entity_to_memory_type(self, entity: Entity) -> MemoryType:
        """Determine memory type from entity type."""
        # Dates and times are usually episodic (about events)
        if entity.entity_type in ["DATE", "TIME"]:
            return MemoryType.EPISODIC
        
        # Most other entities are semantic facts
        return MemoryType.SEMANTIC

