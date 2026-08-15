"""
Prompt Templates for ctxforge.

This module contains LLM prompt templates used by various services.
"""

from ctxforge.prompts.intent_note import INTENT_NOTE_PROMPT
from ctxforge.prompts.query_rewrite import QUERY_REWRITE_PROMPT
from ctxforge.prompts.sufficiency_judge import SUFFICIENCY_JUDGE_PROMPT

__all__ = [
    "QUERY_REWRITE_PROMPT",
    "SUFFICIENCY_JUDGE_PROMPT",
    "INTENT_NOTE_PROMPT",
]
