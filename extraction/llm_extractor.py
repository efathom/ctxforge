"""
LLM-based Memory Extractor.

Uses an LLM to intelligently extract memories from conversations.
More flexible and context-aware than pattern matching, but slower
and requires an LLM provider.

Enhanced features:
- Multi-pass extraction for improved recall
- Source text alignment for provenance tracking
- Chunking support for long documents
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ctxforge.core.memory import MemoryType
from ctxforge.extraction.alignment import (
    AlignmentStatus,
    CharSpan,
    WordAligner,
    merge_non_overlapping_spans,
)
from ctxforge.extraction.base import BaseExtractor
from ctxforge.extraction.chunking import ChunkIterator, TextChunk, make_batches, sliding_window
from ctxforge.extraction.typed_prompts import get_typed_prompt
from ctxforge.extraction.utils import extract_json_from_text, parse_confidence
from ctxforge.protocols.extractor import (
    ExtractionCandidate,
    ExtractionConfig,
)
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.utils.similarity import ISimilarityCalculator

logger = logging.getLogger(__name__)


# System prompt for memory extraction
EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system. Your job is to analyze conversations and identify facts worth remembering about the user.

For each piece of information you extract, categorize it as:
- SEMANTIC: General facts, preferences, beliefs, identity (e.g., "likes coffee", "is a software engineer")
- EPISODIC: Specific events or experiences with temporal context (e.g., "visited Paris last summer", "started new job in March")
- PROCEDURAL: Workflows, processes, or how-to knowledge the user follows (e.g., "always reviews code before merging", "prefers to outline before writing")

Guidelines:
1. Focus on information about the USER, not general facts
2. Extract specific, actionable facts (not vague statements)
3. Assign confidence based on how explicitly the user stated the information
4. Higher confidence (0.8-1.0) for direct statements like "I am...", "I love..."
5. Lower confidence (0.5-0.7) for implied or inferred information
6. Add relevant tags for categorization
7. Skip greetings, thanks, and meta-conversation
8. IMPORTANT: Use exact phrases from the source text when possible for better source grounding
9. CRITICAL: For the "restatement" field, produce a self-contained version of the fact:
   - Resolve ALL pronouns to proper nouns (e.g., "he" -> "Bob", "there" -> "Seattle")
   - Convert ALL relative time references to absolute dates when possible (e.g., "tomorrow" -> "2026-02-16", "last summer" -> "summer 2025")
   - The restatement must be understandable WITHOUT any surrounding conversation context
10. Extract structured entities: persons mentioned, locations, and timestamps

Return your extractions as a JSON array. Each item should have:
{
  "content": "Clear, concise statement of the fact (original phrasing)",
  "restatement": "Self-contained version with resolved pronouns and absolute dates",
  "type": "SEMANTIC" | "EPISODIC" | "PROCEDURAL",
  "confidence": 0.0 to 1.0,
  "tags": ["relevant", "tags"],
  "keywords": ["important", "search", "terms"],
  "topics": ["travel", "career"],
  "entities": {
    "persons": ["Alice", "Bob"],
    "locations": ["Seattle", "Paris"],
    "timestamps": ["2026-02-16"]
  }
}

If no meaningful information to extract, return an empty array: []"""


USER_PROMPT_TEMPLATE = """Analyze this conversation turn and extract any facts worth remembering about the user.

User said: "{user_input}"
{context_section}

Extract memorable facts about the user (or empty array if none).
Remember: produce a "restatement" that resolves all pronouns and relative dates into a self-contained fact."""


_BATCH_USER_PROMPT = """Analyze the following conversation turns and extract any facts worth remembering about the user.

{turns_section}
{dedup_section}
Extract memorable facts about the user (or empty array if none).
Remember: produce a "restatement" that resolves all pronouns and relative dates into a self-contained fact."""


class LLMExtractor(BaseExtractor):
    """
    LLM-based memory extractor.
    
    Uses a language model to intelligently identify and extract
    memories from conversations. More flexible than pattern matching,
    can understand context and nuance.
    
    Enhanced features:
    - Multi-pass extraction for improved recall
    - Source text alignment for provenance tracking
    - Chunking support for long documents
    
    Example:
        from ctxforge.llm import OpenAIProvider
        
        llm = OpenAIProvider(api_key="...")
        extractor = LLMExtractor(llm_provider=llm)
        
        result = await extractor.extract(
            user_input="I've been learning Python for about 6 months now",
            agent_response="That's great progress!"
        )
        # Returns candidates like "User has been learning Python for 6 months"
    """
    
    def __init__(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        llm_func: Optional[Callable[[str], Awaitable[str]]] = None,
        system_prompt: Optional[str] = None,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
        aligner: Optional[WordAligner] = None,
        use_typed_prompts: bool = False,
    ):
        """
        Initialize the LLM extractor.

        Args:
            llm_provider: An ILLMProvider implementation
            llm_func: Alternative: a simple async function(prompt) -> response
            system_prompt: Custom system prompt (uses default if not provided)
            default_config: Default extraction configuration
            similarity_calculator: Calculator for text similarity
            aligner: Word aligner for source grounding (optional)
            use_typed_prompts: Use per-type prompts for targeted extraction
        """
        super().__init__(default_config, similarity_calculator)

        self._llm_provider = llm_provider
        self._llm_func = llm_func
        self._system_prompt = system_prompt or EXTRACTION_SYSTEM_PROMPT
        self._aligner = aligner  # Lazy initialization
        self._use_typed_prompts = use_typed_prompts

        if not llm_provider and not llm_func:
            raise ValueError("Either llm_provider or llm_func must be provided")
    
    def _get_aligner(self, config: ExtractionConfig) -> WordAligner:
        """Get or create word aligner with config settings."""
        if self._aligner is not None:
            return self._aligner
        return WordAligner(
            fuzzy_threshold=config.fuzzy_alignment_threshold,
            accept_partial=config.accept_partial_matches,
        )
    
    @property
    def name(self) -> str:
        """The name of this extractor."""
        if self._llm_provider:
            return f"llm:{self._llm_provider.name}"
        return "llm:custom"
    
    async def _call_llm(
        self,
        user_prompt: str,
        config: ExtractionConfig,
    ) -> str:
        """
        Call the LLM with the extraction prompt.
        
        Args:
            user_prompt: The user prompt
            config: Extraction configuration
            
        Returns:
            LLM response text
        """
        if self._llm_provider:
            messages = [
                ChatMessage(role="system", content=self._system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
            
            response = await self._llm_provider.chat(
                messages=messages,
                model=config.model,
                temperature=config.temperature,
                max_tokens=1000,
            )
            
            return response.content
        else:
            # Use simple function
            full_prompt = f"{self._system_prompt}\n\n{user_prompt}"
            return await self._llm_func(full_prompt)
    
    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """
        Extract memories using LLM with multi-pass and alignment support.
        
        Args:
            text: The text to extract from
            config: Extraction configuration
            context: Optional additional context
            
        Returns:
            List of extraction candidates
        """
        if not text or len(text.strip()) < 5:
            return []

        if self._use_typed_prompts:
            return await self._typed_extract(text, config, context)

        # Check if we need chunking for long text
        if len(text) > config.max_chunk_size:
            return await self._extract_chunked(text, config, context)

        # Multi-pass extraction
        all_candidates_by_pass: List[List[ExtractionCandidate]] = []
        
        for pass_num in range(1, config.extraction_passes + 1):
            pass_candidates = await self._single_pass_extract(
                text, config, context, pass_num
            )
            all_candidates_by_pass.append(pass_candidates)
        
        # Merge non-overlapping candidates from multiple passes
        merged = self._merge_passes(all_candidates_by_pass, config)
        
        return merged
    
    async def _typed_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """Run per-type extraction using specialised prompts."""
        active_types: List[MemoryType] = []
        if getattr(config, "extract_semantic", True):
            active_types.append(MemoryType.SEMANTIC)
        if getattr(config, "extract_episodic", True):
            active_types.append(MemoryType.EPISODIC)
        if getattr(config, "extract_procedural", False):
            active_types.append(MemoryType.PROCEDURAL)
        if getattr(config, "extract_preference", True):
            active_types.append(MemoryType.PREFERENCE)
        if getattr(config, "extract_tool", False):
            active_types.append(MemoryType.TOOL)

        original_prompt = self._system_prompt
        all_candidates: List[ExtractionCandidate] = []

        for mtype in active_types:
            self._system_prompt = get_typed_prompt(mtype)
            try:
                candidates = await self._single_pass_extract(
                    text, config, context, pass_num=1
                )
                all_candidates.extend(candidates)
            finally:
                self._system_prompt = original_prompt

        return all_candidates

    async def _single_pass_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]],
        pass_num: int,
    ) -> List[ExtractionCandidate]:
        """Single extraction pass with alignment."""
        # Build context section if available
        context_section = ""
        if context:
            if context.get("agent_response"):
                context_section = f'\nAssistant responded: "{context["agent_response"]}"'
            if context.get("gist_context"):
                context_section += (
                    "\n\nPreviously extracted atomic facts for this text:\n"
                    + context["gist_context"]
                )
        
        # Build the prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            user_input=text,
            context_section=context_section,
        )
        
        # Add pass context for multi-pass
        if pass_num > 1:
            user_prompt += f"\n\n(Pass {pass_num}: Look for any facts you may have missed.)"
        
        try:
            # Call LLM
            response = await self._call_llm(user_prompt, config)
            
            # Parse response
            candidates = self._parse_llm_response(response, text)
            
            # Align candidates to source text
            if config.enable_alignment:
                candidates = self._align_candidates(candidates, text, config)
            
            # Tag pass number
            for c in candidates:
                c.extraction_pass = pass_num
            
            return candidates
            
        except Exception as e:
            # Log error but don't fail
            logger.warning("LLM extraction error (pass %s): %s", pass_num, e)
            return []
    
    async def extract_batch(
        self,
        turns: List[Tuple[str, str]],
        window_size: int = 10,
        overlap_context: Optional[List[str]] = None,
        max_concurrency: int = 1,
    ) -> List[ExtractionCandidate]:
        """Extract memories from multiple conversation turns using sliding windows.

        Groups *turns* into windows of *window_size*.  Each window is processed
        with a single LLM call.

        When ``max_concurrency`` is 1 (default), windows are processed
        sequentially and each window receives the previously extracted facts
        as deduplication context.  When ``max_concurrency > 1``, independent
        windows are processed in parallel (bounded by a semaphore) without
        inter-window dedup context; deduplication is performed at the end.

        Args:
            turns: List of ``(user_input, agent_response)`` pairs.
            window_size: Maximum turns per extraction window.
            overlap_context: Optional list of already-known facts to seed
                deduplication context for the first window.
            max_concurrency: Maximum number of windows to process in
                parallel.  ``1`` means fully sequential with dedup context
                forwarding.

        Returns:
            Merged list of extraction candidates across all windows.
        """
        if not turns:
            return []

        stride = max(1, window_size)
        windows = sliding_window(turns, window_size, stride)

        all_candidates: List[ExtractionCandidate] = []

        if max_concurrency <= 1:
            # Sequential: forward extracted facts as dedup context.
            dedup_facts: List[str] = list(overlap_context or [])
            for window in windows:
                candidates = await self._extract_window(window, dedup_facts)
                all_candidates.extend(candidates)
                dedup_facts.extend(c.content for c in candidates)
        else:
            # Parallel: process independent windows concurrently.
            sem = asyncio.Semaphore(max_concurrency)
            seed_context = list(overlap_context or [])

            async def _run(w: List[Tuple[str, str]]) -> List[ExtractionCandidate]:
                async with sem:
                    return await self._extract_window(w, seed_context)

            results = await asyncio.gather(*[_run(w) for w in windows])
            for batch in results:
                all_candidates.extend(batch)

        # Deduplicate across windows by normalised content
        seen: Dict[str, bool] = {}
        deduped: List[ExtractionCandidate] = []
        for c in all_candidates:
            key = c.content.lower().strip()
            if key not in seen:
                seen[key] = True
                deduped.append(c)

        return deduped

    async def _extract_window(
        self,
        turns: List[Tuple[str, str]],
        dedup_facts: List[str],
    ) -> List[ExtractionCandidate]:
        """Process a single window of turns with one LLM call."""
        lines: List[str] = []
        for i, (user_input, agent_response) in enumerate(turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f'  User: "{user_input}"')
            if agent_response:
                lines.append(f'  Assistant: "{agent_response}"')
        turns_section = "\n".join(lines)

        dedup_section = ""
        if dedup_facts:
            facts_str = "\n".join(f"- {f}" for f in dedup_facts)
            dedup_section = (
                f"\nAlready extracted facts (do NOT re-extract these):\n{facts_str}\n"
            )

        user_prompt = _BATCH_USER_PROMPT.format(
            turns_section=turns_section,
            dedup_section=dedup_section,
        )

        config = self._default_config or ExtractionConfig()

        try:
            response = await self._call_llm(user_prompt, config)
            source_text = "\n".join(
                f"{u} {a}" for u, a in turns
            )
            return self._parse_llm_response(response, source_text)
        except Exception:
            return []

    async def _extract_chunked(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractionCandidate]:
        """Extract from text using chunking for long documents."""
        chunks = list(ChunkIterator(text, max_char_buffer=config.max_chunk_size))
        
        all_candidates: List[ExtractionCandidate] = []
        
        # Process chunks in batches
        for batch in make_batches(iter(chunks), config.parallel_chunks):
            batch_results = await asyncio.gather(*[
                self._extract_chunk(chunk, config, context)
                for chunk in batch
            ])
            
            for chunk, candidates in zip(batch, batch_results, strict=False):
                # Adjust spans for chunk offset (source_span is CharSpan)
                for c in candidates:
                    if c.source_span:
                        c.source_span = CharSpan(
                            start_pos=c.source_span.start_pos + chunk.char_span.start_pos,
                            end_pos=c.source_span.end_pos + chunk.char_span.start_pos,
                        )
                all_candidates.extend(candidates)
        
        return all_candidates
    
    async def _extract_chunk(
        self,
        chunk: TextChunk,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractionCandidate]:
        """Extract from a single chunk (single pass)."""
        return await self._single_pass_extract(chunk.text, config, context, 1)
    
    def _align_candidates(
        self,
        candidates: List[ExtractionCandidate],
        source_text: str,
        config: ExtractionConfig,
    ) -> List[ExtractionCandidate]:
        """Align candidates to their source positions."""
        aligner = self._get_aligner(config)
        aligned = []
        
        for candidate in candidates:
            result = aligner.align(
                extraction_text=candidate.content,
                source_text=source_text,
            )
            
            # Use proper types directly (no circular dependency now)
            candidate.alignment_status = result.status
            candidate.source_span = result.char_span
            candidate.matched_text = result.matched_text
            
            # Adjust confidence based on alignment quality
            if result.status == AlignmentStatus.MATCH_EXACT:
                candidate.confidence = min(1.0, candidate.confidence + 0.1)
            elif result.status == AlignmentStatus.MATCH_FUZZY:
                candidate.confidence = candidate.confidence * result.confidence
            elif result.status == AlignmentStatus.UNALIGNED:
                candidate.confidence = candidate.confidence * 0.8
            
            aligned.append(candidate)
        
        return aligned
    
    def _merge_passes(
        self,
        candidates_by_pass: List[List[ExtractionCandidate]],
        config: ExtractionConfig,
    ) -> List[ExtractionCandidate]:
        """Merge candidates from multiple passes, keeping first-pass wins."""
        if len(candidates_by_pass) == 1:
            return candidates_by_pass[0]
        
        # If alignment is not enabled, just concatenate and dedupe by content
        if not config.enable_alignment:
            seen_content = set()
            merged = []
            for pass_candidates in candidates_by_pass:
                for c in pass_candidates:
                    content_key = c.content.lower().strip()
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        merged.append(c)
            return merged
        
        # Use span-based merging for aligned candidates
        # source_span is now CharSpan directly
        spans_by_pass: List[List[tuple]] = []
        for pass_candidates in candidates_by_pass:
            spans = []
            for c in pass_candidates:
                if c.source_span:
                    # source_span is already a CharSpan
                    spans.append((c.source_span, c))
                else:
                    # No span - create synthetic one based on content
                    # Use a non-overlapping range
                    fake_span = CharSpan(-len(c.content) - 1, -1)
                    spans.append((fake_span, c))
            spans_by_pass.append(spans)
        
        # Merge
        merged_spans = merge_non_overlapping_spans(spans_by_pass)
        
        return [c for _, c in merged_spans]
    
    def _parse_llm_response(
        self,
        response: str,
        source_text: str,
    ) -> List[ExtractionCandidate]:
        """
        Parse the LLM response into extraction candidates.
        
        Args:
            response: The LLM response text
            source_text: The original source text
            
        Returns:
            List of extraction candidates
        """
        candidates = []
        
        # Try to extract JSON from response
        json_str = self._extract_json(response)
        
        if not json_str:
            return []
        
        try:
            data = json.loads(json_str)
            
            # Handle both array and single object
            if isinstance(data, dict):
                data = [data]
            
            if not isinstance(data, list):
                return []
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                content = item.get("content", "").strip()
                if not content:
                    continue
                
                # Parse memory type
                type_str = item.get("type", "SEMANTIC").upper()
                memory_type = self._parse_memory_type(type_str)
                
                # Parse confidence
                confidence = parse_confidence(item.get("confidence", 0.7))
                
                # Parse tags
                tags = item.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",")]
                tags = [str(t).lower().strip() for t in tags if t]

                # Parse restatement (disambiguated self-contained fact)
                restatement = (item.get("restatement") or "").strip() or None

                # Parse structured entities
                raw_entities = item.get("entities", {})
                extracted_entities: Dict[str, Any] = {}
                if isinstance(raw_entities, dict):
                    for key in ("persons", "locations", "timestamps"):
                        val = raw_entities.get(key, [])
                        if isinstance(val, list):
                            extracted_entities[key] = [
                                str(v) for v in val if v
                            ]

                # Parse keywords and topics for multi-view indexing
                raw_keywords = item.get("keywords", [])
                if isinstance(raw_keywords, str):
                    raw_keywords = [k.strip() for k in raw_keywords.split(",")]
                keywords = [str(k).lower().strip() for k in raw_keywords if k]

                raw_topics = item.get("topics", [])
                if isinstance(raw_topics, str):
                    raw_topics = [t.strip() for t in raw_topics.split(",")]
                topics = [str(t).lower().strip() for t in raw_topics if t]

                # Build metadata with multi-view fields
                candidate_metadata: Dict[str, Any] = {"extractor": "llm"}
                if keywords:
                    candidate_metadata["keywords"] = keywords
                if topics:
                    candidate_metadata["topics"] = topics

                candidates.append(ExtractionCandidate(
                    content=content,
                    memory_type=memory_type,
                    confidence=confidence,
                    source_text=source_text,
                    tags=tags,
                    metadata=candidate_metadata,
                    restatement=restatement,
                    extracted_entities=extracted_entities,
                ))
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract any useful info
            pass
        
        return candidates
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from LLM response.
        
        Delegates to shared utility function.
        
        Args:
            text: The response text
            
        Returns:
            JSON string if found, None otherwise
        """
        return extract_json_from_text(text)
    
    def _parse_memory_type(self, type_str: str) -> MemoryType:
        """
        Parse memory type from string.
        
        Args:
            type_str: Type string from LLM
            
        Returns:
            MemoryType enum value
        """
        type_map = {
            "SEMANTIC": MemoryType.SEMANTIC,
            "EPISODIC": MemoryType.EPISODIC,
            "PROCEDURAL": MemoryType.PROCEDURAL,
            "FACT": MemoryType.SEMANTIC,
            "PREFERENCE": MemoryType.PREFERENCE,
            "EVENT": MemoryType.EPISODIC,
            "EXPERIENCE": MemoryType.EPISODIC,
            "SKILL": MemoryType.PROCEDURAL,
            "PROCESS": MemoryType.PROCEDURAL,
            "TOOL": MemoryType.TOOL,
        }
        
        return type_map.get(type_str.upper(), MemoryType.SEMANTIC)


class MockLLMExtractor(LLMExtractor):
    """
    A mock LLM extractor for testing.
    
    Returns predefined responses based on keywords in input.
    Useful for testing extraction pipeline without actual LLM calls.
    """
    
    def __init__(
        self,
        responses: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        """
        Initialize mock extractor.
        
        Args:
            responses: Keyword -> response mapping
            default_config: Default extraction configuration
            similarity_calculator: Calculator for text similarity
        """
        # Use a dummy function to satisfy parent init
        super().__init__(
            llm_func=self._mock_llm,
            default_config=default_config,
            similarity_calculator=similarity_calculator,
        )
        
        self._mock_responses = responses or {}
    
    @property
    def name(self) -> str:
        """The name of this extractor."""
        return "llm:mock"
    
    async def _mock_llm(self, prompt: str) -> str:
        """Mock LLM function."""
        return "[]"
    
    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        """
        Extract using mock responses.
        
        Checks for keywords in text and returns matching responses.
        """
        candidates = []
        text_lower = text.lower()
        
        for keyword, response_list in self._mock_responses.items():
            if keyword.lower() in text_lower:
                for resp in response_list:
                    candidates.append(ExtractionCandidate(
                        content=resp.get("content", keyword),
                        memory_type=self._parse_memory_type(resp.get("type", "SEMANTIC")),
                        confidence=resp.get("confidence", 0.8),
                        source_text=text,
                        tags=resp.get("tags", []),
                    ))
        
        return candidates

