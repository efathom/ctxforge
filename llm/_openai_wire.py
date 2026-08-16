"""
Shared OpenAI-compatible wire-format helpers.

``OpenAILLMProvider`` and ``AzureOpenAILLMProvider`` both speak the OpenAI
chat-completions wire format. These helpers translate the framework's
``functions=`` kwarg and ``ChatMessage.function_call`` extension hook into
the modern ``tools`` / ``tool_calls`` / ``role="tool"`` shapes that current
endpoints require (the legacy ``functions`` / ``function_call`` request
params are deprecated and rejected with a 400).
"""

import json
from typing import Any, Dict, List, Optional

from ctxforge.protocols.llm import ChatMessage


def normalize_tools(functions: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize a ``functions=`` list into the modern ``tools`` wire shape.

    Entries already wrapped in the ``{"type": "function", "function": {...}}``
    envelope pass through unchanged; flat legacy dicts
    (``{"name", "description", "parameters"}``) are wrapped. Returns ``None``
    when there are no tools so callers can omit ``tools`` / ``tool_choice``.
    """
    if not functions:
        return None
    tools: List[Dict[str, Any]] = []
    for fn in functions:
        if (
            isinstance(fn, dict)
            and fn.get("type") == "function"
            and isinstance(fn.get("function"), dict)
        ):
            tools.append(fn)
        else:
            tools.append({"type": "function", "function": fn})
    return tools


def _encode_arguments(arguments: Any) -> str:
    """Encode tool-call arguments as a JSON string (the OpenAI wire shape)."""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {})


def serialize_openai_message(m: ChatMessage) -> Dict[str, Any]:
    """
    Serialize a ``ChatMessage`` into the modern OpenAI chat wire shape.

    Translates the two tool-calling shapes the framework expresses through
    the ``function_call`` extension hook:

    * ``ChatMessage(role="assistant", content=..., function_call={"tool_calls":
      [{"id", "name", "arguments": <dict>}]})`` -> assistant ``tool_calls``
      entries with JSON-string arguments.
    * ``ChatMessage(role="function", content=..., function_call={"tool_call_id":
      <id>})`` -> ``role="tool"`` with ``tool_call_id`` (and ``name`` dropped).

    Bare ``role="function"`` messages without a ``tool_call_id`` (a legacy
    compatibility shim) keep their ``name`` so existing callers continue to
    work. Other roles pass through with ``name`` preserved.
    """
    payload: Dict[str, Any] = {"role": m.role, "content": m.content}
    fc = m.function_call or {}
    if m.role == "assistant" and "tool_calls" in fc:
        tool_calls: List[Dict[str, Any]] = []
        for tc in fc["tool_calls"] or []:
            tool_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _encode_arguments(tc.get("arguments")),
                    },
                }
            )
        payload["tool_calls"] = tool_calls
    elif m.role == "function":
        tool_call_id = fc.get("tool_call_id")
        if tool_call_id:
            payload["role"] = "tool"
            payload["tool_call_id"] = tool_call_id
            payload.pop("name", None)
        elif m.name:
            payload["name"] = m.name
    elif m.name:
        payload["name"] = m.name
    return payload
