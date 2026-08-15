"""
Hybrid Memory Extractor.

Combines multiple extraction strategies (pattern, entity, LLM) for
comprehensive memory extraction. Deduplicates and merges results
from different extractors.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.core.session import Session
from ctxforge.extraction.base import BaseExtractor
from ctxforge.extraction.entity_extractor import EntityExtractor
from ctxforge.extraction.pattern_extractor import PatternExtractor
from ctxforge.protocols.extractor import (
    ExtractionCandidate,
    ExtractionConfig,
    ExtractionResult,
    IMemoryExtractor,
)
from ctxforge.utils.similarity import ISimilarityCalculator

logger = logging.getLogger(__name__)


class HybridExtractor(BaseExtractor):
    """
    Hybrid memory extractor combining multiple strategies.
    
    Runs multiple extractors in parallel and combines their results,
    deduplicating and keeping the highest-confidence version of
    similar candidates.
    
    Example:
        # Pattern + Entity extraction (no LLM required)
        extractor = HybridExtractor()
        
        # With LLM for additional intelligence
        from ctxforge.extraction import LLMExtractor
        llm_extractor = LLMExtractor(llm_provider=my_llm)
        extractor = HybridExtractor(extractors=[llm_extractor])
        
        result = await extractor.extract(
            user_input="I'm a developer at Google in San Francisco",
            agent_response="That's interesting!"
        )
        # Combines pattern matches, entity detection, and LLM insights
        
        # With custom similarity calculator
        from ctxforge.extraction.similarity import LevenshteinSimilarityCalculator
        calculator = LevenshteinSimilarityCalculator()
        extractor = HybridExtractor(similarity_calculator=calculator)
    """
    
    def __init__(
        self,
        extractors: Optional[List[IMemoryExtractor]] = None,
        include_patterns: bool = True,
        include_entities: bool = True,
        dedup_threshold: float = 0.75,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize the hybrid extractor.
        
        Args:
            extractors: Additional extractors to include
            include_patterns: Include built-in pattern extractor
            include_entities: Include built-in entity extractor
            dedup_threshold: Similarity threshold for deduplication
            default_config: Default extraction configuration
            similarity_calculator: Calculator for text similarity
        """
        super().__init__(default_config, similarity_calculator)
        
        self._extractors: List[IMemoryExtractor] = []
        self._dedup_threshold = dedup_threshold
        
        # Add built-in extractors if requested (pass similarity calculator)
        if include_patterns:
            self._extractors.append(
                PatternExtractor(
                    default_config=default_config,
                    similarity_calculator=self._similarity_calculator,
                )
            )
        
        if include_entities:
            self._extractors.append(
                EntityExtractor(
                    default_config=default_config,
                    similarity_calculator=self._similarity_calculator,
                )
            )
        
        # Add custom extractors
        if extractors:
            self._extractors.extend(extractors)
    
    @property
    def name(self) -> str:
        """The name of this extractor."""
        extractor_names = [e.name for e in self._extractors]
        return f"hybrid:[{', '.join(extractor_names)}]"
    
    @property
    def extractors(self) -> List[IMemoryExtractor]:
        """The extractors being used."""
        return list(self._extractors)
    
    def add_extractor(self, extractor: IMemoryExtractor) -> None:
        """
        Add an extractor to the hybrid.
        
        Args:
            extractor: The extractor to add
        """
        self._extractors.append(extractor)
    
    def remove_extractor(self, name: str) -> bool:
        """
        Remove an extractor by name.
        
        Args:
            name: The name of the extractor to remove
            
        Returns:
            True if an extractor was removed
        """
        for i, extractor in enumerate(self._extractors):
            if extractor.name == name:
                self._extractors.pop(i)
                return True
        return False
    
    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """
        Extract using all configured extractors.
        
        Runs extractors in parallel and combines results.
        
        Args:
            text: The text to extract from
            config: Extraction configuration
            context: Optional additional context
            
        Returns:
            Combined and deduplicated candidates
        """
        if not text or not self._extractors:
            return []
        
        # Run all extractors in parallel
        tasks = []
        for extractor in self._extractors:
            # Use extract_from_text for raw text, extract for conversation
            if context and "user_input" in context:
                task = extractor.extract(
                    user_input=context["user_input"],
                    agent_response=context.get("agent_response", ""),
                    session=context.get("session"),
                    config=config,
                )
            else:
                task = extractor.extract_from_text(text, config)
            tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all candidates
        all_candidates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log but don't fail
                logger.warning("Extractor %s failed: %s", self._extractors[i].name, result)
                continue
            
            if isinstance(result, ExtractionResult):
                # Tag candidates with their source extractor
                for candidate in result.candidates:
                    candidate.metadata["source_extractor"] = self._extractors[i].name
                    all_candidates.append(candidate)
        
        # Deduplicate and merge
        merged = self._merge_candidates(all_candidates)
        
        return merged
    
    def _merge_candidates(
        self,
        candidates: List[ExtractionCandidate],
    ) -> List[ExtractionCandidate]:
        """
        Merge similar candidates from different extractors.
        
        Keeps the highest confidence version and combines metadata/tags.
        
        Args:
            candidates: All candidates from all extractors
            
        Returns:
            Merged and deduplicated candidates
        """
        if not candidates:
            return []
        
        # Sort by confidence (highest first)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        )
        
        merged = []
        
        for candidate in sorted_candidates:
            # Check if similar candidate already exists
            similar_idx = self._find_similar(candidate, merged)
            
            if similar_idx is not None:
                # Merge with existing
                merged[similar_idx] = self._merge_two_candidates(
                    merged[similar_idx],
                    candidate,
                )
            else:
                # Add as new
                merged.append(candidate)
        
        return merged
    
    def _find_similar(
        self,
        candidate: ExtractionCandidate,
        existing: List[ExtractionCandidate],
    ) -> Optional[int]:
        """
        Find a similar candidate in the existing list.
        
        Args:
            candidate: The candidate to match
            existing: Existing candidates
            
        Returns:
            Index of similar candidate or None
        """
        for i, other in enumerate(existing):
            similarity = self._calculate_similarity(
                candidate.content,
                other.content,
            )
            
            if similarity >= self._dedup_threshold:
                return i
        
        return None
    
    def _merge_two_candidates(
        self,
        primary: ExtractionCandidate,
        secondary: ExtractionCandidate,
    ) -> ExtractionCandidate:
        """
        Merge two similar candidates.
        
        Keeps primary content (higher confidence) but enriches
        with secondary's tags and metadata.
        
        Args:
            primary: Higher confidence candidate
            secondary: Lower confidence candidate
            
        Returns:
            Merged candidate
        """
        # Combine tags
        combined_tags = list(set(primary.tags + secondary.tags))
        
        # Merge metadata
        merged_metadata = {**secondary.metadata, **primary.metadata}
        
        # Track sources
        sources = merged_metadata.get("source_extractors", [])
        if primary.metadata.get("source_extractor"):
            sources.append(primary.metadata["source_extractor"])
        if secondary.metadata.get("source_extractor"):
            sources.append(secondary.metadata["source_extractor"])
        merged_metadata["source_extractors"] = list(set(sources))
        
        return ExtractionCandidate(
            content=primary.content,  # Keep primary content
            memory_type=primary.memory_type,
            confidence=primary.confidence,  # Keep primary confidence
            source_text=primary.source_text,
            tags=combined_tags,
            metadata=merged_metadata,
        )
    
    async def extract(
        self,
        user_input: str,
        agent_response: str,
        session: Optional[Session] = None,
        config: Optional[ExtractionConfig] = None,
    ) -> ExtractionResult:
        """
        Extract memories from a conversation turn.
        
        Overrides base to pass context properly.
        
        Args:
            user_input: The user's input
            agent_response: The agent's response
            session: Optional session for additional context
            config: Optional extraction configuration
            
        Returns:
            ExtractionResult with combined candidates
        """
        import time
        start_time = time.time()
        config = self._get_config(config)
        
        context = {
            "user_input": user_input,
            "agent_response": agent_response,
            "session": session,
        }
        
        candidates = await self._do_extract(
            user_input,
            config,
            context,
        )
        
        # Apply filtering from base class
        candidates = self._filter_candidates(candidates, config)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            candidates=candidates,
            processing_time_ms=processing_time,
            metadata={
                "extractor": self.name,
                "extractors_used": [e.name for e in self._extractors],
                "input_length": len(user_input),
                "response_length": len(agent_response),
            },
        )
    
    async def validate_candidate(
        self,
        candidate: ExtractionCandidate,
        existing_memories: List[MemoryItem],
    ) -> bool:
        """
        Validate using all extractors that support validation.
        
        Returns True only if all validators agree.
        
        Args:
            candidate: The candidate to validate
            existing_memories: Existing memories for the user
            
        Returns:
            True if the candidate should be stored
        """
        # Use base validation first
        if not await super().validate_candidate(candidate, existing_memories):
            return False
        
        # Additional validation from child extractors
        for extractor in self._extractors:
            try:
                if not await extractor.validate_candidate(candidate, existing_memories):
                    return False
            except NotImplementedError:
                # Extractor doesn't implement validation
                continue
        
        return True
