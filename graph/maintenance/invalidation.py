"""
Graph edge invalidation (contradiction handling).

This module provides an LLM-backed contradiction detector that decides which existing
edges should be marked invalid when a new edge is introduced.

Design goals:
- Conservative invalidation: only invalidate when there is a clear contradiction.
- Strict output parsing: accept only JSON; ignore invalid outputs.
- Safety: only invalidate edges that were provided as candidates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode
from ctxforge.protocols.graph_maintenance import EdgeInvalidationPlan, IGraphContradictionDetector
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

_DEFAULT_PROMPT = """You decide whether any existing graph facts should be invalidated given a new fact.

Rules:
- Be conservative. Only invalidate when the new fact clearly contradicts an existing fact.
- Do NOT invalidate if both facts can be true at the same time (e.g., likes multiple foods).
- Prefer invalidating older facts that represent a single-valued relationship (e.g., WORKS_FOR, LIVES_IN) when the new fact explicitly replaces it.
- Only output valid JSON.

Output JSON schema:
{
  "invalidate_edge_ids": ["edge_id", "..."],
  "rationale": "optional short explanation"
}
"""


class LLMGraphContradictionDetector(IGraphContradictionDetector):
    """LLM-backed contradiction detector that returns an edge-id invalidation plan."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        *,
        default_model: Optional[str] = None,
        prompt: str = _DEFAULT_PROMPT,
        max_tokens: int = 800,
    ):
        self._llm = llm_provider
        self._default_model = default_model
        self._prompt = prompt
        self._max_tokens = max_tokens

    async def detect_contradictions(
        self,
        *,
        scope_id: str,
        new_edge: GraphEdge,
        candidate_edges: List[GraphEdge],
        nodes: List[GraphNode],
        episodes: List[GraphEpisode],
        model: str | None = None,
    ) -> EdgeInvalidationPlan:
        """
        Decide which *existing* edges should be invalidated given a new edge.

        This method is intentionally conservative:
        - If there are no candidates, we do nothing.
        - If the LLM response is not parseable as JSON, we do nothing.
        - Even if the LLM suggests edge ids, we only honor ids that were actually
          provided in `candidate_edges` (prevents the model from "inventing" ids).

        Inputs:
        - `new_edge`: the newly extracted fact (edge) that may replace/contradict older facts.
        - `candidate_edges`: a pre-filtered set of existing edges near the same entities/types.
          The caller controls the candidate selection; this detector only chooses among them.
        - `nodes`/`episodes`: optional extra context to help the LLM disambiguate entity names
          and interpret the new fact in recent conversational context.
        """
        if not candidate_edges:
            return EdgeInvalidationPlan(invalidate_edge_ids=[])

        # Build a structured payload for the LLM. Keeping it JSON makes it easier to parse,
        # and encourages the model to focus on the specific decision: which ids to invalidate.
        candidates_payload = [
            {
                "edge_id": e.edge_id,
                "edge_type": e.edge_type,
                "fact": e.fact,
                "source_node_id": e.source_node_id,
                "target_node_id": e.target_node_id,
                "valid_at": e.valid_at.isoformat() if e.valid_at else None,
                "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None,
            }
            for e in candidate_edges
        ]

        # Minimal representation of the new edge: enough to compare against candidates.
        new_payload = {
            "edge_type": new_edge.edge_type,
            "fact": new_edge.fact,
            "source_node_id": new_edge.source_node_id,
            "target_node_id": new_edge.target_node_id,
        }

        # Helpful context: map ids to human names, and include the last few episode texts.
        # We keep this small to control token usage and reduce unrelated distractions.
        node_map = {n.node_id: n.name for n in nodes if n.node_id and n.name}
        episodes_text = [ep.content for ep in episodes[-3:]] if episodes else []

        user_content = json.dumps(
            {
                "scope_id": scope_id,
                "new_edge": new_payload,
                "candidate_edges": candidates_payload,
                "node_id_to_name": node_map,
                "recent_episodes": episodes_text,
            },
            ensure_ascii=False,
        )

        # Ask the LLM to output a strict JSON object per `_DEFAULT_PROMPT` schema.
        # Temperature is set to 0.0 to reduce variance in the decision format.
        resp = await self._llm.chat(
            messages=[
                ChatMessage(role="system", content=self._prompt),
                ChatMessage(role="user", content=user_content),
            ],
            model=model or self._default_model,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )

        # Parse JSON from the LLM output. If it fails, treat as "no invalidations".
        parsed = self._parse(resp.content or "")
        if parsed is None:
            return EdgeInvalidationPlan(invalidate_edge_ids=[])

        # Normalize and validate ids from the response.
        ids = [x for x in parsed.get("invalidate_edge_ids", []) if isinstance(x, str) and x.strip()]
        rationale = parsed.get("rationale") if isinstance(parsed.get("rationale"), str) else None
        # Safety guardrail: only invalidate edges that were actually provided as candidates.
        allow = {e.edge_id for e in candidate_edges}
        ids = [i for i in ids if i in allow]
        return EdgeInvalidationPlan(invalidate_edge_ids=ids, rationale=rationale)

    def _parse(self, text: str) -> Optional[Dict[str, Any]]:
        json_str = extract_json_from_text(text)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except Exception:
            return None
        return data if isinstance(data, dict) else None


