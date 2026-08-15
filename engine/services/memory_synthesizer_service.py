"""
Memory synthesis service.

Synthesizes retrieved memories into a coherent narrative context before
LLM injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.extraction.integration_config import SynthesizerConfig

if TYPE_CHECKING:
    from ctxforge.protocols.llm import ILLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """Summarize the following retrieved memories that are \
relevant to the current task. Create a comprehensive, coherent narrative that \
preserves ALL preferences, facts, and constraints.

Current query: {query}

Retrieved memories:
{memories}

CRITICAL REQUIREMENTS:
1. Include EVERY preference, like, dislike, and constraint that is RELEVANT \
to the query
2. Preserve specific values and details
3. If memories contain both likes AND dislikes, include BOTH
4. If no memories are relevant, output 'No relevant personalized information \
found.'

Summary:"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MemorySynthesizerService:
    """Synthesizes retrieved memories into coherent narrative context."""

    def __init__(
        self,
        llm: ILLMProvider,
        config: Optional[SynthesizerConfig] = None,
    ):
        self._llm = llm
        self._config = config or SynthesizerConfig()

    async def synthesize(
        self,
        memories: List[MemoryItem],
        query: str,
        max_tokens: int = 300,
    ) -> Optional[str]:
        """Synthesize memories into a coherent narrative.

        Returns None if no relevant memories found, allowing the
        assembler to skip the memory section entirely.
        """
        if not memories:
            return None

        formatted = "\n".join(
            f"- {m.display_content}" for m in memories
        )

        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            memories=formatted,
        )

        effective_max_tokens = max_tokens or self._config.max_synthesis_tokens

        try:
            response = await self._llm.generate(
                prompt,
                model=self._config.model,
                max_tokens=effective_max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("Memory synthesis failed: %s", e)
            return None

        content = response.content.strip()

        # Check for "no relevant info" response
        if "no relevant" in content.lower():
            return None

        return content
