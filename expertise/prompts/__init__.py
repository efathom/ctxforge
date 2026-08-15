"""
Prompts for the Expertise system.

Contains prompt templates for reflector and curator components.
"""

from ctxforge.expertise.prompts.curator import (
    CURATOR_ADD_ONLY_PROMPT,
    CURATOR_MAINTENANCE_PROMPT,
    CURATOR_SYSTEM_PROMPT,
    CURATOR_USER_PROMPT,
)
from ctxforge.expertise.prompts.reflector import (
    REFLECTOR_SUCCESS_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
    REFLECTOR_USER_PROMPT_NO_GT,
)

__all__ = [
    # Reflector prompts
    "REFLECTOR_SYSTEM_PROMPT",
    "REFLECTOR_USER_PROMPT",
    "REFLECTOR_USER_PROMPT_NO_GT",
    "REFLECTOR_SUCCESS_PROMPT",
    # Curator prompts
    "CURATOR_SYSTEM_PROMPT",
    "CURATOR_USER_PROMPT",
    "CURATOR_ADD_ONLY_PROMPT",
    "CURATOR_MAINTENANCE_PROMPT",
]

