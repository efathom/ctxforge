"""
Atomic gist extractor.

Decomposes conversation turns into individually-retrievable, timestamped,
self-contained statements (gists).  Each gist has resolved pronouns and
absolute dates so it can be understood and retrieved in isolation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ctxforge.core.memory import MemoryType
from ctxforge.extraction.base import BaseExtractor
from ctxforge.extraction.gist_prompts import build_gist_prompt
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.extractor import ExtractionCandidate, ExtractionConfig
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.utils.similarity import ISimilarityCalculator

logger = logging.getLogger(__name__)


class GistExtractor(BaseExtractor):
    """Extracts atomic, timestamped gist statements from text.

    Each gist is a single self-contained fact with:
    - Resolved pronouns (no "he/she/they" — uses actual names)
    - Resolved temporal references ("yesterday" → absolute date)
    - One atomic claim per gist
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        default_config: Optional[ExtractionConfig] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
        model: Optional[str] = None,
    ):
        super().__init__(default_config, similarity_calculator)
        self._llm = llm_provider
        self._model = model

    @property
    def name(self) -> str:
        return "gist"

    async def _do_extract(
        self,
        text: str,
        config: ExtractionConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractionCandidate]:
        if not text or not text.strip():
            return []

        messages = build_gist_prompt(text)
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"]) for m in messages
        ]

        model = self._model or config.model
        try:
            response = await self._llm.chat(
                messages=chat_messages,
                model=model,
                temperature=config.temperature,
                max_tokens=1000,
            )
        except Exception:
            logger.warning("Gist extraction LLM call failed", exc_info=True)
            return []

        return self._parse_response(response.content, text)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, source_text: str
    ) -> List[ExtractionCandidate]:
        """Parse the LLM JSON response into ``ExtractionCandidate`` objects."""
        data = self._safe_parse_json(raw)
        if data is None:
            return []

        gists = data.get("gists")
        if not isinstance(gists, list):
            return []

        candidates: List[ExtractionCandidate] = []
        for item in gists:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue

            confidence = self._clamp_confidence(item.get("confidence", 0.7))
            timestamp = (item.get("timestamp") or "").strip()

            metadata: Dict[str, Any] = {"source_type": "gist_extraction"}
            if timestamp:
                metadata["resolved_timestamp"] = timestamp

            candidates.append(
                ExtractionCandidate(
                    content=content,
                    memory_type=MemoryType.EPISODIC,
                    confidence=confidence,
                    source_text=source_text,
                    tags=["gist", "atomic"],
                    metadata=metadata,
                    restatement=content,  # already self-contained
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        extracted = extract_json_from_text(raw)
        if isinstance(extracted, dict):
            return extracted
        if isinstance(extracted, list) and extracted:
            return {"gists": extracted}
        return None

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.7
        return max(0.0, min(1.0, f))
