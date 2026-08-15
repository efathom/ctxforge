"""
Prompt templates for atomic gist extraction.

Gists are self-contained, timestamped, atomic statements decomposed
from conversation turns.  Each gist resolves all pronouns and relative
temporal references so it can be understood (and retrieved) in isolation.
"""

from datetime import datetime, timezone
from typing import Dict, List

GIST_SYSTEM_PROMPT = """\
You are a meticulous information extractor.  Your purpose is to distill \
atomic, self-contained facts from messages into a structured JSON format.

## Core Task
For the given message(s), identify every individual fact, event, or claim. \
Restate each one as a concise, self-contained English sentence.

## Input Format
The user will provide the current time and the message text.  You MUST use \
the `current_time` to resolve any relative temporal expressions \
(e.g., "yesterday", "last week").

## Output Format
- Your output MUST be a single, valid JSON object.
- The JSON object must contain one key: `"gists"`.
- `"gists"` is a list of objects, each with:
    - `"content"` (string): The atomic gist statement.
    - `"timestamp"` (string): The resolved ISO-8601 date/datetime for the gist \
(use the message timestamp when no explicit date is mentioned).
    - `"confidence"` (float): 0.0–1.0 indicating extraction certainty.
- Do not add any explanations, comments, or trailing commas.

### Rules
1. **Decomposition**: Decompose complex sentences into multiple gists. \
Each gist must represent a single atomic fact or event.
2. **Timestamp Prefix**: Begin every gist content with the message's \
timestamp in square brackets, e.g., `[2026-03-07T14:28:00]`.
3. **Temporal Resolution**: After any relative temporal reference, add the \
fully-resolved absolute date or date range in parentheses.
   - Time-point example: `...last Thursday (2026-03-05).`
   - Duration example: `...last week (2026-02-28 to 2026-03-06).`
4. **Pronoun Resolution**: Replace ALL pronouns (he, she, they, it, there, \
etc.) with the actual entity names from context.
5. **Completeness**: Capture ALL details for each fact: participants, \
actions, objects, quantities, locations, intentions.
6. If no meaningful information can be extracted, return `{"gists": []}`.

### Example 1
Input:
```
Current time: 2026-01-20T15:57:00
Alice: I fixed the fence last Monday, then bought 3 cows from Peter on Jan 15th
```
Output:
```json
{
  "gists": [
    {
      "content": "[2026-01-20T15:57:00] Alice fixed the fence last Monday (2026-01-13).",
      "timestamp": "2026-01-13",
      "confidence": 0.95
    },
    {
      "content": "[2026-01-20T15:57:00] Alice bought 3 cows from Peter on Jan 15th (2026-01-15).",
      "timestamp": "2026-01-15",
      "confidence": 0.95
    }
  ]
}
```

### Example 2
Input:
```
Current time: 2026-01-20T14:28:00
Bob: I met with my advisor last Thursday morning and submitted the proposal two days later.
```
Output:
```json
{
  "gists": [
    {
      "content": "[2026-01-20T14:28:00] Bob met with Bob's advisor last Thursday morning (2026-01-16).",
      "timestamp": "2026-01-16",
      "confidence": 0.9
    },
    {
      "content": "[2026-01-20T14:28:00] Bob submitted the proposal two days after Thursday (2026-01-18).",
      "timestamp": "2026-01-18",
      "confidence": 0.9
    }
  ]
}
```"""


GIST_USER_TEMPLATE = """\
Current time: {current_time}

{text}

Extract all atomic gists from the above text as a JSON object."""


def build_gist_prompt(
    text: str,
    current_time: str | None = None,
) -> List[Dict[str, str]]:
    """Assemble the full chat-style prompt for gist extraction.

    Args:
        text: The source text to extract gists from.
        current_time: ISO-8601 string for temporal resolution.
            Defaults to ``datetime.now(timezone.utc)``.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts ready
        for an LLM chat call.
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc).isoformat(timespec="seconds")

    user_content = GIST_USER_TEMPLATE.format(
        current_time=current_time,
        text=text,
    )

    return [
        {"role": "system", "content": GIST_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
