"""
Intent Note Prompt Template.

LLM prompt for extracting a structured intent note from a conversation event.
"""

INTENT_NOTE_PROMPT = """# Task Objective
Extract a structured intent note for the CURRENT EVENT.

The intent note must be compact, explicit, and useful for later retrieval.

# Definitions
- act: the pragmatic action of the speaker
  (ask, propose, confirm, decide, request, explain, troubleshoot, etc.)
- target: the entity/topic/object the act is about (optional)
- note_text: one concise sentence capturing intent + key detail; resolve pronouns when possible.
- context_scope: optional thread label describing which subtopic this belongs to (optional)
- event_types: optional coarse labels for the nature of this event
  (e.g., configuration, decision, bugfix)
- functional_types: optional labels describing what details are doing
  (choose only from seeds if provided)

# Rules
- Do NOT invent facts. Only use the provided context.
- Keep `note_text` to one sentence.
- If target cannot be determined, set target to null.
- If a field is unknown, use null or an empty list as appropriate.

# Output Format
Return ONLY valid JSON (no markdown), exactly this shape:
{{
  "act": "string",
  "target": "string or null",
  "note_text": "string",
  "context_scope": "string or null",
  "event_types": ["string", "..."],
  "functional_types": ["string", "..."],
  "confidence": 0.0,
  "source": "llm"
}}

# Inputs

## Recent Context (for reference resolution)
{recent_context}

## Functional Type Seeds (optional)
{functional_type_seeds}

## Current Event
role: {role}
content: {content}
"""
