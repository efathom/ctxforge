"""Inline memory reference utilities.

Supports ``[ref:ID]`` and ``[ref:ID1,ID2]`` markers embedded in text.
"""

import re
from typing import Dict, List

_REF_PATTERN = re.compile(r"\[ref:([a-zA-Z0-9_,-]+)\]")


def extract_references(text: str) -> List[str]:
    """Extract unique reference IDs from all ``[ref:...]`` markers in *text*."""
    ids: List[str] = []
    seen: set = set()
    for match in _REF_PATTERN.finditer(text):
        for ref_id in match.group(1).split(","):
            ref_id = ref_id.strip()
            if ref_id and ref_id not in seen:
                ids.append(ref_id)
                seen.add(ref_id)
    return ids


def strip_references(text: str) -> str:
    """Remove all ``[ref:...]`` markers and clean up leftover whitespace."""
    cleaned = _REF_PATTERN.sub("", text)
    # Collapse multiple spaces into one
    cleaned = re.sub(r"  +", " ", cleaned)
    # Remove space before punctuation
    cleaned = re.sub(r" ([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def format_as_citations(
    memory_map: Dict[str, str],
    style: str = "numbered",
) -> str:
    """Format a map of ``{id: content}`` as numbered citations.

    Args:
        memory_map: Mapping from memory ID to its content text.
        style: Citation style (currently only ``"numbered"`` is supported).

    Returns:
        Formatted citation string.
    """
    lines: List[str] = []
    for idx, (mid, content) in enumerate(memory_map.items(), 1):
        lines.append(f"[{idx}] ({mid}): {content}")
    return "\n".join(lines)


def build_reference_map(
    text: str,
    memory_lookup: Dict[str, str],
) -> Dict[str, str]:
    """Extract reference IDs from *text* and return the subset of *memory_lookup*.

    IDs present in the text but missing from *memory_lookup* are silently skipped.
    """
    ref_ids = extract_references(text)
    return {rid: memory_lookup[rid] for rid in ref_ids if rid in memory_lookup}
