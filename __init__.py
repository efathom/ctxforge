"""
ctxforge - A highly extensible and configurable context engine framework for LLM agents.

This framework provides:
- Modular architecture with pluggable components
- Protocol-based extensibility for custom implementations
- Configuration-driven setup via YAML/JSON
- Comprehensive middleware pipeline
- Multi-provider LLM support
- Flexible storage backends
- Advanced retrieval strategies
"""

__version__ = "0.1.0"
__author__ = "ctxforge Team"

# Attach a NullHandler so importing ctxforge never emits "no handlers found"
# warnings when the application has not configured logging. Applications should
# call ctxforge.utils.logging.setup_logging() to enable output.
import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from ctxforge.core.context import Context
from ctxforge.core.events import Event, EventType
from ctxforge.core.exceptions import (
    ConfigurationError,
    ContextEngineError,
    LLMError,
    StorageError,
    ValidationError,
)
from ctxforge.core.memory import MemoryItem, MemoryType
from ctxforge.core.session import Session, SessionState

__all__ = [
    # Core data structures
    "Event",
    "EventType",
    "Session",
    "SessionState",
    "MemoryItem",
    "MemoryType",
    "Context",
    # Exceptions
    "ContextEngineError",
    "ConfigurationError",
    "StorageError",
    "LLMError",
    "ValidationError",
]

