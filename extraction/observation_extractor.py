"""
LLM-based observation extractor.

Extracts structured observations (decisions, bug fixes, discoveries)
from a list of session events and produces session summary reports.
"""

import json
from typing import Any, Dict, List

from ctxforge.core.events import Event
from ctxforge.core.observation import Observation, ObservationType, SessionSummaryReport
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.llm import ChatMessage, ILLMProvider

_OBSERVATION_SYSTEM_PROMPT = """You are a session analysis system. Given a list of events from a coding session, extract structured observations.

For each observation, classify it as one of:
- decision: A design or implementation decision made during the session.
- bugfix: A bug that was found and fixed.
- discovery: A new insight or understanding gained.
- feature: A feature that was implemented.
- refactor: Refactoring or code improvement work.
- change: A general change that doesn't fit other categories.

Return a JSON array of observations:
[
  {
    "type": "decision",
    "summary": "Short one-line summary",
    "detail": "Optional longer explanation",
    "confidence": 0.8
  }
]

If no meaningful observations, return an empty array: []"""

_SESSION_SUMMARY_PROMPT = """Summarize this coding session into a structured report.

Return a JSON object with these fields:
{
  "request": "What the user originally asked for",
  "investigated": "What was explored or researched",
  "learned": "Key insights or discoveries",
  "completed": "What was actually accomplished",
  "next_steps": "Suggested follow-up actions"
}"""


class ObservationExtractor:
    """Extracts structured observations from session events."""

    def __init__(self, llm_provider: ILLMProvider):
        self._llm = llm_provider

    async def extract(self, events: List[Event]) -> List[Observation]:
        """Extract observations from session events."""
        if not events:
            return []

        event_text = self._format_events(events)
        messages = [
            ChatMessage(role="system", content=_OBSERVATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=event_text),
        ]

        try:
            resp = await self._llm.chat(messages, temperature=0.3, max_tokens=1000)
            return self._parse_observations(resp.content)
        except Exception:
            return []

    async def summarize_session(self, events: List[Event]) -> SessionSummaryReport:
        """Produce a structured session summary."""
        if not events:
            return SessionSummaryReport()

        event_text = self._format_events(events)
        messages = [
            ChatMessage(role="system", content=_SESSION_SUMMARY_PROMPT),
            ChatMessage(role="user", content=event_text),
        ]

        try:
            resp = await self._llm.chat(messages, temperature=0.3, max_tokens=500)
            return self._parse_summary(resp.content)
        except Exception:
            return SessionSummaryReport()

    def _format_events(self, events: List[Event]) -> str:
        lines = []
        for e in events[:50]:
            role = e.type.value if hasattr(e.type, "value") else str(e.type)
            content = e.content[:500] if e.content else ""
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _parse_observations(self, text: str) -> List[Observation]:
        json_str = extract_json_from_text(text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return []
        except json.JSONDecodeError:
            return []

        observations: List[Observation] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue

            obs_type = self._parse_type(item.get("type", "change"))
            confidence = 0.8
            try:
                confidence = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                pass

            observations.append(Observation(
                type=obs_type,
                summary=summary,
                detail=(item.get("detail") or "").strip() or None,
                confidence=min(1.0, max(0.0, confidence)),
            ))

        return observations

    def _parse_type(self, raw: str) -> ObservationType:
        mapping: Dict[str, ObservationType] = {
            "decision": ObservationType.DECISION,
            "bugfix": ObservationType.BUGFIX,
            "discovery": ObservationType.DISCOVERY,
            "feature": ObservationType.FEATURE,
            "refactor": ObservationType.REFACTOR,
            "change": ObservationType.CHANGE,
        }
        return mapping.get(str(raw).lower().strip(), ObservationType.CHANGE)

    def _parse_summary(self, text: str) -> SessionSummaryReport:
        json_str = extract_json_from_text(text)
        if not json_str:
            return SessionSummaryReport()

        try:
            data: Dict[str, Any] = json.loads(json_str)
            if not isinstance(data, dict):
                return SessionSummaryReport()
        except json.JSONDecodeError:
            return SessionSummaryReport()

        return SessionSummaryReport(
            request=str(data.get("request", "")).strip(),
            investigated=str(data.get("investigated", "")).strip(),
            learned=str(data.get("learned", "")).strip(),
            completed=str(data.get("completed", "")).strip(),
            next_steps=str(data.get("next_steps", "")).strip(),
        )
