"""Per-type extraction prompts for targeted memory extraction.

Each memory type has a specialised system prompt that guides the LLM to
extract the right kind of information with appropriate formatting.
"""

from typing import Dict

from ctxforge.core.memory import MemoryType

TYPED_EXTRACTION_PROMPTS: Dict[MemoryType, str] = {
    MemoryType.SEMANTIC: (
        "You are a semantic fact extraction system. Extract factual statements "
        "about the user. Each fact should be concise (under 30 words), contain "
        "no timestamps, and represent a stable truth. Assign confidence 0.8-1.0 "
        "for direct statements and 0.5-0.7 for inferred facts. Return a JSON "
        'array with fields: content, restatement, type="SEMANTIC", confidence, tags.'
    ),
    MemoryType.EPISODIC: (
        "You are an episodic event extraction system. Extract events and "
        "experiences with temporal anchors (dates, times, durations). Each "
        "event should be under 50 words and resolve relative dates to absolute "
        "dates when possible (e.g. 'yesterday' -> '2026-03-05'). Return a JSON "
        'array with fields: content, restatement, type="EPISODIC", confidence, tags.'
    ),
    MemoryType.PROCEDURAL: (
        "You are a procedural knowledge extraction system. Extract workflows, "
        "processes, step-by-step procedures, and how-to knowledge the user "
        "follows or describes. Focus on actionable sequences. Return a JSON "
        'array with fields: content, restatement, type="PROCEDURAL", confidence, tags.'
    ),
    MemoryType.PREFERENCE: (
        "You are a preference extraction system. Extract likes, dislikes, "
        "habits, and preferences. Note the context in which the preference "
        "applies and estimate habit strength from language cues ('always', "
        "'sometimes', 'usually'). Return a JSON array with fields: content, "
        'restatement, type="PREFERENCE", confidence, tags.'
    ),
    MemoryType.TOOL: (
        "You are a tool usage pattern extraction system. Extract patterns "
        "about how and when the user employs specific tools or commands. "
        "Include when_to_use hints and any parameter preferences. Return a "
        'JSON array with fields: content, restatement, type="TOOL", confidence, tags.'
    ),
}


def get_typed_prompt(memory_type: MemoryType) -> str:
    """Return the extraction prompt for a memory type, with SEMANTIC fallback."""
    return TYPED_EXTRACTION_PROMPTS.get(
        memory_type, TYPED_EXTRACTION_PROMPTS[MemoryType.SEMANTIC]
    )


def get_all_typed_prompts() -> Dict[MemoryType, str]:
    """Return a copy of all typed extraction prompts."""
    return dict(TYPED_EXTRACTION_PROMPTS)
