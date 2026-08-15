"""
Headline Generation Service.

Generates compact headlines from memory content using LLM
with structured AI-generated titles for progressive disclosure.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ctxforge.core.memory import MemoryItem
from ctxforge.core.memory_index import MemoryIndex, MemoryIndexEntry
from ctxforge.protocols.llm import ILLMProvider

logger = logging.getLogger(__name__)

# LLM prompt for structured headline generation
HEADLINE_PROMPT = """Generate a headline and subtitle for the following memory content.

<memory_content>
{content}
</memory_content>

<memory_type>{memory_type}</memory_type>

Respond in this exact XML format:
<headline_response>
  <title>[Short title capturing the core topic, max 10 words]</title>
  <subtitle>[One sentence explanation, max 24 words]</subtitle>
</headline_response>

Rules:
- Title should be action-oriented or descriptive
  (e.g., "Prefers dark mode", "Uses pytest for testing")
- Subtitle expands on the title with key details
- Do not include the memory type in the title
- Be concise but informative
"""


class HeadlineService:
    """
    Generates headlines for memories using LLM.

    Generates structured AI titles that are stored persistently
    with the memory for efficient retrieval.
    """

    def __init__(
        self,
        llm_provider: "ILLMProvider",
        max_headline_chars: int = 80,
        max_subtitle_chars: int = 150,
    ):
        self._llm = llm_provider
        self._max_headline = max_headline_chars
        self._max_subtitle = max_subtitle_chars

    async def generate_headline(
        self,
        memory: "MemoryItem",
    ) -> Tuple[str, str]:
        """
        Generate headline and subtitle for a memory using LLM.

        Returns:
            Tuple of (headline, subtitle)
        """
        prompt = HEADLINE_PROMPT.format(
            content=memory.content[:1000],  # Limit content sent to LLM
            memory_type=memory.type.value,
        )

        try:
            response = await self._llm.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.3,  # Low temperature for consistency
            )

            headline, subtitle = self._parse_response(response.content)

            # Enforce length limits
            if len(headline) > self._max_headline:
                headline = headline[:self._max_headline - 3] + "..."
            if subtitle and len(subtitle) > self._max_subtitle:
                subtitle = subtitle[:self._max_subtitle - 3] + "..."

            return headline, subtitle

        except Exception as e:
            logger.warning(f"LLM headline generation failed: {e}")
            # Fallback to simple extraction
            return self._fallback_headline(memory.content), ""

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response XML to extract headline and subtitle."""
        title_match = re.search(r'<title>(.*?)</title>', response, re.DOTALL)
        subtitle_match = re.search(
            r'<subtitle>(.*?)</subtitle>',
            response,
            re.DOTALL
        )

        headline = title_match.group(1).strip() if title_match else ""
        subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

        if not headline:
            raise ValueError("No headline found in LLM response")

        return headline, subtitle

    def _fallback_headline(self, content: str) -> str:
        """Simple fallback when LLM fails - take first sentence."""
        sentences = re.split(r'[.!?]\s+', content.strip())
        if sentences:
            first = sentences[0].strip()
            if len(first) > self._max_headline:
                return first[:self._max_headline - 3] + "..."
            return first
        return content[:self._max_headline]

    async def generate_and_update(
        self,
        memory: "MemoryItem",
    ) -> "MemoryItem":
        """
        Generate headline/subtitle and update memory in-place.

        Returns the memory for convenience.
        """
        headline, subtitle = await self.generate_headline(memory)
        memory.headline = headline
        memory.subtitle = subtitle
        return memory

    async def generate_batch(
        self,
        memories: List["MemoryItem"],
        skip_existing: bool = True,
    ) -> List["MemoryItem"]:
        """
        Generate headlines for a batch of memories.

        Args:
            memories: List of memories to process
            skip_existing: Skip memories that already have headlines

        Returns:
            Updated memories with headlines
        """
        for memory in memories:
            if skip_existing and memory.has_headline():
                continue
            await self.generate_and_update(memory)

        return memories

    async def build_index(
        self,
        memories: List["MemoryItem"],
    ) -> "MemoryIndex":
        """Build a MemoryIndex from memories, generating headlines if needed."""
        index = MemoryIndex(total_memories=len(memories))

        for memory in memories:
            # Use stored headline or generate new one
            if not memory.has_headline():
                await self.generate_and_update(memory)

            entry = MemoryIndexEntry.from_memory(memory)
            index.add(entry)

        return index


class HeadlineServiceFactory:
    """Factory for creating HeadlineService instances."""

    @staticmethod
    def create(
        llm_provider: Optional["ILLMProvider"] = None,
        max_headline_chars: int = 80,
        max_subtitle_chars: int = 150,
    ) -> Optional[HeadlineService]:
        """
        Create a HeadlineService if LLM provider is available.

        Returns None if no LLM provider is given.
        """
        if llm_provider is None:
            return None

        return HeadlineService(
            llm_provider=llm_provider,
            max_headline_chars=max_headline_chars,
            max_subtitle_chars=max_subtitle_chars,
        )
