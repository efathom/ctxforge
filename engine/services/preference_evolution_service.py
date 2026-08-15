"""
Preference evolution tracking service.

Detects when user preferences change over time, maintains a preference
changelog on MemoryItem metadata, and automatically supersedes old
preferences when contradictions are detected.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from ctxforge.core.memory import MemoryItem, MemoryQuery, MemoryType
from ctxforge.extraction.integration_config import PreferenceEvolutionConfig

if TYPE_CHECKING:
    from ctxforge.protocols.llm import ILLMProvider
    from ctxforge.protocols.storage import IMemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHANGE_DETECTION_PROMPT = """Does this new information CHANGE a previously \
stated preference? Look for:
- "but now..." / "now I prefer..."
- "used to like X, but..."
- "changed my mind about..."
- "instead of X, I prefer Y"
- "no longer like..."

If it's just adding NEW information or conditional preferences, answer 'No'.

Existing preference: {existing}
New information: {new_info}

Answer 'Yes' or 'No'."""

EXTRACT_OLD_PREFERENCE_PROMPT = """Based on this new information that indicates \
a preference change, summarize what the PREVIOUS preference was in one brief \
sentence.

New information: {new_info}

Previous preference:"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreferenceChange:
    """Records a single preference change event."""

    old_memory_id: str
    old_content: str
    new_content: str
    changed_at: datetime.datetime = field(
        default_factory=datetime.datetime.now,
    )
    change_type: str = "supersede"  # "supersede" | "refine" | "contradict"
    rationale: Optional[str] = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PreferenceEvolutionService:
    """Tracks and manages preference changes over time."""

    def __init__(
        self,
        llm: ILLMProvider,
        memory_store: IMemoryStore,
        config: Optional[PreferenceEvolutionConfig] = None,
    ):
        self._llm = llm
        self._memory_store = memory_store
        self._config = config or PreferenceEvolutionConfig()

    async def detect_preference_change(
        self,
        new_content: str,
        existing_memory: MemoryItem,
        query: str,
    ) -> Optional[PreferenceChange]:
        """Detect whether new_content changes an existing preference.

        Uses LLM to detect transition phrases. Returns PreferenceChange
        if a change is detected, None otherwise.
        """
        prompt = CHANGE_DETECTION_PROMPT.format(
            existing=existing_memory.content,
            new_info=new_content,
        )
        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=10,
                temperature=0.0,
            )
            answer = response.content.strip().lower()
            if not answer.startswith("yes"):
                return None
        except Exception as e:
            logger.warning("Preference change detection failed: %s", e)
            return None

        return PreferenceChange(
            old_memory_id=existing_memory.memory_id,
            old_content=existing_memory.content,
            new_content=new_content,
            change_type="supersede",
        )

    async def apply_preference_change(
        self,
        change: PreferenceChange,
        new_memory: MemoryItem,
        old_memory: MemoryItem,
    ) -> MemoryItem:
        """Apply a preference change:
        1. Mark old memory as superseded
        2. Store change record in new memory's metadata
        3. Decay old memory's importance
        4. Link old -> new via related_memory_ids
        """
        # Mark old memory as superseded
        old_memory.superseded_by = new_memory.memory_id
        if self._config.auto_supersede:
            old_memory.importance = max(
                0.0,
                old_memory.importance * self._config.importance_decay_on_supersede,
            )

        # Store change record in new memory's metadata
        if self._config.track_history:
            changes_list = new_memory.metadata.get("preference_changes", [])
            changes_list.append({
                "from": change.old_content,
                "to": change.new_content,
                "changed_at": change.changed_at.isoformat(),
                "old_memory_id": change.old_memory_id,
                "change_type": change.change_type,
            })
            new_memory.metadata["preference_changes"] = changes_list
            version = new_memory.metadata.get("preference_version", 0)
            new_memory.metadata["preference_version"] = version + 1

        # Link memories
        if old_memory.memory_id not in new_memory.related_memory_ids:
            new_memory.related_memory_ids.append(old_memory.memory_id)
        if new_memory.memory_id not in old_memory.related_memory_ids:
            old_memory.related_memory_ids.append(new_memory.memory_id)

        # Persist old memory changes
        try:
            await self._memory_store.update(old_memory)
        except Exception as e:
            logger.warning("Failed to update superseded memory: %s", e)

        return new_memory

    async def get_preference_history(
        self,
        user_id: str,
        topic: Optional[str] = None,
    ) -> List[PreferenceChange]:
        """Retrieve preference change history for a user."""
        memories = await self._memory_store.get_by_user(user_id)
        changes: List[PreferenceChange] = []

        for mem in memories:
            pref_changes = mem.metadata.get("preference_changes", [])
            for pc in pref_changes:
                change = PreferenceChange(
                    old_memory_id=pc.get("old_memory_id", ""),
                    old_content=pc.get("from", ""),
                    new_content=pc.get("to", ""),
                    changed_at=datetime.datetime.fromisoformat(
                        pc["changed_at"]
                    ) if "changed_at" in pc else datetime.datetime.now(),
                    change_type=pc.get("change_type", "supersede"),
                )
                if topic is None or topic.lower() in change.old_content.lower() or topic.lower() in change.new_content.lower():
                    changes.append(change)

        return changes

    async def detect_contradictions(
        self,
        user_id: str,
        new_items: List[MemoryItem],
    ) -> List[Tuple[MemoryItem, MemoryItem, str]]:
        """Scan new items against existing PREFERENCE memories for contradictions.

        Returns list of (new_item, conflicting_existing, explanation).
        """
        contradictions: List[Tuple[MemoryItem, MemoryItem, str]] = []

        for new_item in new_items:
            if new_item.type != MemoryType.PREFERENCE:
                continue

            try:
                existing = await self._memory_store.search(
                    MemoryQuery(
                        user_id=user_id,
                        query_text=new_item.content,
                        limit=5,
                    )
                )
            except Exception:
                continue

            for mem in existing:
                if mem.type != MemoryType.PREFERENCE:
                    continue
                if mem.memory_id == new_item.memory_id:
                    continue

                change = await self.detect_preference_change(
                    new_content=new_item.content,
                    existing_memory=mem,
                    query="",
                )
                if change is not None:
                    contradictions.append(
                        (new_item, mem, f"Preference change detected: {change.change_type}")
                    )

        return contradictions
