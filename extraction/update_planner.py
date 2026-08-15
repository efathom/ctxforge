from __future__ import annotations

import json
from typing import Dict, List, Optional

from ctxforge.core.memory import MemoryItem
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.protocols.update_planner import (
    IMemoryUpdatePlanner,
    MemoryOperation,
    MemoryOperationType,
)

DEFAULT_UPDATE_PLANNER_PROMPT = """You are a memory update planning system.

Given:
- a user query (what the user just said)
- newly extracted candidate memories (each has a temp id)
- existing candidate memories from the user scope and (optionally) a global scope

Decide which operations to apply:
- ADD: store a new memory
- UPDATE: modify an existing memory
- DELETE: deactivate an existing memory (soft delete)
- NONE: do nothing for that new item

Rules:
- Prefer updating an existing USER-scoped memory when it's clearly the same fact with better wording or more recent info.
- If a new item contradicts an existing memory, prefer UPDATE (overwrite) or DELETE (deactivate) the old memory.
- If the closest match is in GLOBAL scope, prefer ADD as a user-scoped override or NONE.
- Output MUST be valid JSON and MUST include every new_temp_id exactly once as either ADD, UPDATE, or NONE.
- DELETE may appear without a new_temp_id, but only if it is clearly obsolete/contradictory.

Output JSON schema:
{
  "operations": [
    {
      "op": "ADD" | "UPDATE" | "DELETE" | "NONE",
      "new_temp_id": "n1",
      "target_memory_id": "mem_123",
      "target_scope_id": "user_id_or_global",
      "content": "string",
      "confidence": 0.0,
      "tags": ["tag1"],
      "rationale": "string"
    }
  ]
}

Return ONLY JSON."""


class LLMMemoryUpdatePlanner(IMemoryUpdatePlanner):
    def __init__(
        self,
        llm_provider: ILLMProvider,
        *,
        system_prompt: str = DEFAULT_UPDATE_PLANNER_PROMPT,
        default_model: Optional[str] = None,
        max_tokens: int = 1200,
    ):
        self._llm = llm_provider
        self._system_prompt = system_prompt
        self._default_model = default_model
        self._max_tokens = max_tokens

    async def plan(
        self,
        *,
        user_id: str,
        query: str,
        new_items: List[MemoryItem],
        user_candidates: Dict[str, List[MemoryItem]],
        global_candidates: Dict[str, List[MemoryItem]],
        model: Optional[str] = None,
    ) -> List[MemoryOperation]:
        if not new_items:
            return []

        # Assign stable temp ids if caller didn't attach them.
        temp_ids = [f"n{i+1}" for i in range(len(new_items))]

        prompt_lines: List[str] = [
            f"User id: {user_id}",
            f"Query: {query}",
            "",
            "New items:",
        ]

        for tid, item in zip(temp_ids, new_items, strict=False):
            tags = ", ".join(item.tags) if item.tags else ""
            prompt_lines.append(
                f'- new_temp_id: {tid} | type: {item.type.value} | confidence: {item.confidence_score:.2f} | tags: [{tags}]'
            )
            prompt_lines.append(f"  content: {item.content}")

            uc = user_candidates.get(tid, [])
            gc = global_candidates.get(tid, [])

            prompt_lines.append("  user_candidates:")
            if not uc:
                prompt_lines.append("    - (none)")
            else:
                for m in uc:
                    mtags = ", ".join(m.tags) if m.tags else ""
                    prompt_lines.append(
                        f"    - id: {m.memory_id} | type: {m.type.value} | active: {m.is_active} | confidence: {m.confidence_score:.2f} | tags: [{mtags}]"
                    )
                    prompt_lines.append(f"      content: {m.content}")

            prompt_lines.append("  global_candidates:")
            if not gc:
                prompt_lines.append("    - (none)")
            else:
                for m in gc:
                    mtags = ", ".join(m.tags) if m.tags else ""
                    prompt_lines.append(
                        f"    - id: {m.memory_id} | type: {m.type.value} | active: {m.is_active} | confidence: {m.confidence_score:.2f} | tags: [{mtags}]"
                    )
                    prompt_lines.append(f"      content: {m.content}")

        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content="\n".join(prompt_lines)),
        ]

        resp = await self._llm.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )

        ops = self._parse_operations(resp.content or "", temp_ids=temp_ids)
        if ops is None:
            # Safe fallback: add everything as-is.
            return [
                MemoryOperation(
                    op=MemoryOperationType.ADD,
                    new_temp_id=tid,
                    content=item.content,
                    confidence=item.confidence_score,
                    tags=list(item.tags),
                    target_scope_id=user_id,
                )
                for tid, item in zip(temp_ids, new_items, strict=False)
            ]

        # Ensure every new item is covered once.
        seen = set()
        normalized: List[MemoryOperation] = []
        for op in ops:
            if op.new_temp_id:
                if op.new_temp_id not in temp_ids:
                    continue
                if op.new_temp_id in seen:
                    continue
                seen.add(op.new_temp_id)
            normalized.append(op)

        for tid, item in zip(temp_ids, new_items, strict=False):
            if tid not in seen:
                normalized.append(
                    MemoryOperation(
                        op=MemoryOperationType.ADD,
                        new_temp_id=tid,
                        content=item.content,
                        confidence=item.confidence_score,
                        tags=list(item.tags),
                        target_scope_id=user_id,
                    )
                )

        return normalized

    def _parse_operations(
        self,
        text: str,
        *,
        temp_ids: List[str],
    ) -> Optional[List[MemoryOperation]]:
        json_str = extract_json_from_text(text or "")
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        raw_ops = data.get("operations")
        if not isinstance(raw_ops, list):
            return None

        parsed: List[MemoryOperation] = []
        for raw in raw_ops:
            if not isinstance(raw, dict):
                continue

            op_raw = raw.get("op")
            if not isinstance(op_raw, str):
                continue
            try:
                op_t = MemoryOperationType(op_raw.upper())
            except ValueError:
                continue

            new_temp_id = raw.get("new_temp_id")
            if new_temp_id is not None and not isinstance(new_temp_id, str):
                new_temp_id = None

            # Normalize keys that may appear as strings.
            target_memory_id = raw.get("target_memory_id")
            if target_memory_id is not None and not isinstance(target_memory_id, str):
                target_memory_id = None

            target_scope_id = raw.get("target_scope_id")
            if target_scope_id is not None and not isinstance(target_scope_id, str):
                target_scope_id = None

            content = raw.get("content")
            if content is not None and not isinstance(content, str):
                content = None

            confidence = raw.get("confidence")
            if confidence is not None and not isinstance(confidence, (int, float)):
                confidence = None

            tags = raw.get("tags", [])
            if isinstance(tags, str):
                tags = [tags] if tags else []
            if not isinstance(tags, list):
                tags = []
            tags = [t for t in tags if isinstance(t, str)]

            rationale = raw.get("rationale")
            if rationale is not None and not isinstance(rationale, str):
                rationale = None

            # Basic shape checks.
            if op_t in (MemoryOperationType.UPDATE, MemoryOperationType.DELETE) and not target_memory_id:
                continue
            if op_t in (MemoryOperationType.ADD, MemoryOperationType.UPDATE, MemoryOperationType.NONE):
                if not new_temp_id or new_temp_id not in temp_ids:
                    continue

            parsed.append(
                MemoryOperation(
                    op=op_t,
                    new_temp_id=new_temp_id,
                    target_memory_id=target_memory_id,
                    target_scope_id=target_scope_id,
                    content=content,
                    confidence=float(confidence) if confidence is not None else None,
                    tags=list(tags),
                    rationale=rationale,
                )
            )

        return parsed


