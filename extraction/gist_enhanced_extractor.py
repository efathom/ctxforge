"""
Two-phase gist-enhanced extractor.

Phase 1: Extract atomic gists from the source text.
Phase 2: Extract structured facts with the gists injected as context,
         producing more grounded and complete memory candidates.
"""

import logging
from typing import Any, Dict, List, Optional

from ctxforge.extraction.base import BaseExtractor
from ctxforge.extraction.gist_extractor import GistExtractor
from ctxforge.protocols.extractor import ExtractionCandidate, ExtractionConfig
from ctxforge.utils.similarity import ISimilarityCalculator

logger = logging.getLogger(__name__)


class GistEnhancedExtractor(BaseExtractor):
    """Two-phase extraction: gists first, then facts with gist context.

    The gists help the fact extractor produce more grounded, complete
    extractions by providing a decomposed view of the source material.
    Both gist candidates and fact candidates are returned.
    """

    def __init__(
        self,
        gist_extractor: GistExtractor,
        fact_extractor: "BaseExtractor",
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
    ):
        super().__init__(default_config, similarity_calculator)
        self._gist_extractor = gist_extractor
        self._fact_extractor = fact_extractor

    @property
    def name(self) -> str:
        return "gist_enhanced"

    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        if not text or not text.strip():
            return []

        # Phase 1: extract atomic gists
        gists = await self._gist_extractor._do_extract(text, config, context)

        # Phase 2: extract facts, injecting gists as additional context
        enhanced_context: Dict[str, Any] = dict(context) if context else {}
        if gists:
            gist_context = "\n".join(f"- {g.content}" for g in gists)
            enhanced_context["gist_context"] = gist_context

        facts = await self._fact_extractor._do_extract(
            text, config, enhanced_context
        )

        # Return both gists and facts
        return gists + facts
