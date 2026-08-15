from __future__ import annotations

"""
Graph edge temporal enrichment.

This module refines `valid_at` / `invalid_at` for extracted edges. It is intended to be
run after edge extraction and before persistence.

Notes:
- The extractor should be conservative: return nulls if no explicit or resolvable time is present.
- The store layer may still apply backend defaults for missing timestamps.
"""

import json
from typing import Any, Dict, Optional

from dateutil.parser import isoparse

from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.graph import GraphEdge, GraphEpisode
from ctxforge.protocols.graph_maintenance import EdgeTemporalInfo, IGraphEdgeTemporalExtractor
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

_PROMPT = """You extract temporal bounds for a graph edge based on episode text.

Rules:
- Be conservative. Only emit timestamps that are explicitly stated or clearly resolvable.
- If the fact is ongoing, set valid_at and leave invalid_at null (unless an end is stated).
- Use the episode's created_at as REFERENCE_TIME to resolve relative expressions (e.g., "last year").
- Output valid JSON only.

Output JSON schema:
{
  "valid_at": "ISO8601|null",
  "invalid_at": "ISO8601|null"
}
"""


class LLMEdgeTemporalExtractor(IGraphEdgeTemporalExtractor):
    """LLM-backed temporal extractor that returns ISO8601 strings (or nulls)."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        *,
        default_model: Optional[str] = None,
        prompt: str = _PROMPT,
        max_tokens: int = 400,
    ):
        self._llm = llm_provider
        self._default_model = default_model
        self._prompt = prompt
        self._max_tokens = max_tokens

    async def extract_temporal_info(
        self,
        *,
        scope_id: str,
        edge: GraphEdge,
        episodes: list[GraphEpisode],
        model: str | None = None,
    ) -> EdgeTemporalInfo:
        """
        Extract `valid_at` / `invalid_at` bounds for a newly extracted edge.

        This is a *refinement* step: the graph extractor may already provide timestamps,
        but if it did not (or they are coarse), this method asks the LLM to infer time
        bounds from the episode text.

        Guardrails / safety:
        - If there is no episode context, we do nothing.
        - We require strict JSON output from the model; otherwise we do nothing.
        - Even when the model returns strings, we validate they parse as ISO8601; invalid
          timestamps are dropped rather than propagated.

        Reference time:
        - We provide `reference_time` (= the latest episode's `created_at`) to help resolve
          relative expressions like "last year" or "two months ago".
        """
        if not episodes:
            return EdgeTemporalInfo()

        # Use the most recent episode time as a reference point for relative date phrases.
        ref = episodes[-1].created_at.isoformat()

        # Build a compact, structured payload to maximize parseability and reduce token bloat.
        # We include only the last few episodes to avoid pulling in unrelated context.
        payload = {
            "scope_id": scope_id,
            "reference_time": ref,
            "edge": {
                "edge_type": edge.edge_type,
                "fact": edge.fact,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
            },
            "episodes": [
                {
                    "created_at": ep.created_at.isoformat(),
                    "content_type": ep.content_type,
                    "content": ep.content,
                }
                for ep in episodes[-3:]
            ],
        }

        # Ask the LLM for a strict JSON object per `_PROMPT`. Temperature is 0.0 to reduce
        # format variance (we want predictable JSON more than creativity).
        resp = await self._llm.chat(
            messages=[
                ChatMessage(role="system", content=self._prompt),
                ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
            model=model or self._default_model,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )

        # Parse the model output as JSON. If parsing fails, treat it as "no temporal info".
        parsed = self._parse(resp.content or "")
        if parsed is None:
            return EdgeTemporalInfo()

        valid_at = parsed.get("valid_at")
        invalid_at = parsed.get("invalid_at")

        # Validate parseability; if invalid, drop the field. The engine/store layer will
        # decide defaults when the field is absent.
        if isinstance(valid_at, str):
            try:
                isoparse(valid_at)
            except Exception:
                valid_at = None
        else:
            valid_at = None

        if isinstance(invalid_at, str):
            try:
                isoparse(invalid_at)
            except Exception:
                invalid_at = None
        else:
            invalid_at = None

        return EdgeTemporalInfo(valid_at=valid_at, invalid_at=invalid_at)

    def _parse(self, text: str) -> Optional[Dict[str, Any]]:
        json_str = extract_json_from_text(text)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except Exception:
            return None
        return data if isinstance(data, dict) else None


