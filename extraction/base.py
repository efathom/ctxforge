"""
Base Extractor abstract class.

Provides common functionality for all memory extractors.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session
from ctxforge.extraction.utils import normalize_text
from ctxforge.protocols.extractor import (
    ExtractionCandidate,
    ExtractionConfig,
    ExtractionResult,
    IMemoryExtractor,
)
from ctxforge.utils.similarity import (
    ISimilarityCalculator,
    TextSimilarityCalculator,
)


class BaseExtractor(IMemoryExtractor, ABC):
    """
    Abstract base class for memory extractors.
    
    Provides common functionality for:
    - Configuration management
    - Result filtering
    - Candidate validation
    - Timing and metadata
    
    Subclasses must implement:
    - _do_extract(): Core extraction logic
    - name property
    """
    
    def __init__(
        self,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the base extractor.
        
        Args:
            default_config: Default configuration for extractions
            similarity_calculator: Calculator for text similarity (uses TextSimilarityCalculator if not provided)
        """
        self._default_config = default_config or ExtractionConfig()
        self._similarity_calculator = similarity_calculator or TextSimilarityCalculator()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this extractor."""
        ...
    
    @property
    def similarity_calculator(self) -> ISimilarityCalculator:
        """The similarity calculator being used."""
        return self._similarity_calculator
    
    @similarity_calculator.setter
    def similarity_calculator(self, calculator: ISimilarityCalculator) -> None:
        """Set a new similarity calculator."""
        self._similarity_calculator = calculator
    
    @abstractmethod
    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """
        Perform the actual extraction.
        
        Subclasses implement their extraction logic here.
        
        Args:
            text: The text to extract from
            config: Extraction configuration
            context: Optional additional context
            
        Returns:
            List of extraction candidates
        """
        ...
    
    def _get_config(self, config: Optional[ExtractionConfig]) -> ExtractionConfig:
        """Get configuration, using defaults if not provided."""
        if config is None:
            return self._default_config
        return config
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using the configured calculator.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return self._similarity_calculator.calculate(text1, text2)
    
    async def extract(
        self,
        user_input: str,
        agent_response: str,
        session: Optional[Session] = None,
        config: Optional[ExtractionConfig] = None,
    ) -> ExtractionResult:
        """
        Extract memories from a conversation turn.
        
        Combines user input and agent response for context-aware extraction.
        
        Args:
            user_input: The user's input
            agent_response: The agent's response
            session: Optional session for additional context
            config: Optional extraction configuration
            
        Returns:
            ExtractionResult with candidates
        """
        start_time = time.time()
        config = self._get_config(config)
        
        # Combine for context, but focus extraction on user input
        # as it contains the user's information
        context = {
            "user_input": user_input,
            "agent_response": agent_response,
            "session": session,
        }
        
        # Primary extraction from user input
        candidates = await self._do_extract(
            normalize_text(user_input),
            config,
            context,
        )
        
        # Apply filtering
        candidates = self._filter_candidates(candidates, config)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            candidates=candidates,
            processing_time_ms=processing_time,
            metadata={
                "extractor": self.name,
                "input_length": len(user_input),
                "response_length": len(agent_response),
            },
        )
    
    async def extract_from_text(
        self,
        text: str,
        config: Optional[ExtractionConfig] = None,
    ) -> ExtractionResult:
        """
        Extract memories from raw text.
        
        Args:
            text: The text to extract from
            config: Optional extraction configuration
            
        Returns:
            ExtractionResult with candidates
        """
        start_time = time.time()
        config = self._get_config(config)
        
        candidates = await self._do_extract(
            normalize_text(text),
            config,
            context=None,
        )
        
        # Apply filtering
        candidates = self._filter_candidates(candidates, config)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            candidates=candidates,
            processing_time_ms=processing_time,
            metadata={
                "extractor": self.name,
                "text_length": len(text),
            },
        )
    
    def _filter_candidates(
        self,
        candidates: List[ExtractionCandidate],
        config: ExtractionConfig,
    ) -> List[ExtractionCandidate]:
        """
        Filter candidates based on configuration.
        
        Args:
            candidates: The candidates to filter
            config: Extraction configuration
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for candidate in candidates:
            # Check confidence threshold
            if candidate.confidence < config.min_confidence:
                continue
            
            # Check memory type filters
            if candidate.memory_type == MemoryType.SEMANTIC and not config.extract_semantic:
                continue
            if candidate.memory_type == MemoryType.EPISODIC and not config.extract_episodic:
                continue
            if candidate.memory_type == MemoryType.PROCEDURAL and not config.extract_procedural:
                continue
            if candidate.memory_type == MemoryType.PREFERENCE and not getattr(config, 'extract_preference', True):
                continue
            if candidate.memory_type == MemoryType.TOOL and not getattr(config, 'extract_tool', False):
                continue

            filtered.append(candidate)
        
        # Sort by confidence (highest first) and limit
        filtered.sort(key=lambda c: c.confidence, reverse=True)
        
        if config.max_candidates and len(filtered) > config.max_candidates:
            filtered = filtered[:config.max_candidates]
        
        return filtered
    
    async def validate_candidate(
        self,
        candidate: ExtractionCandidate,
        existing_memories: List[MemoryItem],
    ) -> bool:
        """
        Validate a candidate against existing memories.
        
        Checks for duplicates and obvious conflicts.
        
        Args:
            candidate: The candidate to validate
            existing_memories: Existing memories for the user
            
        Returns:
            True if the candidate should be stored
        """
        if not candidate.content or not candidate.content.strip():
            return False
        
        # Check for duplicates
        for memory in existing_memories:
            similarity = self._calculate_similarity(
                candidate.content,
                memory.content,
            )
            
            # High similarity indicates duplicate
            if similarity > 0.85:
                return False
        
        return True
    
    def _deduplicate_candidates(
        self,
        candidates: List[ExtractionCandidate],
        threshold: float = 0.8,
    ) -> List[ExtractionCandidate]:
        """
        Remove duplicate candidates.
        
        Keeps the candidate with higher confidence when duplicates found.
        
        Args:
            candidates: The candidates to deduplicate
            threshold: Similarity threshold for duplicate detection
            
        Returns:
            Deduplicated list
        """
        if not candidates:
            return []
        
        # Sort by confidence (keep highest)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        )
        
        unique = []
        for candidate in sorted_candidates:
            is_duplicate = False
            for existing in unique:
                similarity = self._calculate_similarity(
                    candidate.content,
                    existing.content,
                )
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(candidate)
        
        return unique
