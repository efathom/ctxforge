"""
Configuration system for the ctxforge framework.

Provides:
- Pydantic-based configuration schemas
- YAML/JSON/environment configuration loading
- Sensible defaults for all components
- Validation and type safety
"""

from ctxforge.config.base import (
    CompactionConfig,
    EngineConfig,
    ExpertiseConfig,
    ExpertiseRetrievalConfig,
    ExpertiseStoreConfig,
    ExpertiseVectorStoreConfig,
    ExtractionConfig,
    LLMConfig,
    MemoryStoreConfig,
    MemoryVectorStoreConfig,
    MiddlewareItemConfig,
    ObservabilityConfig,
    PipelineConfig,
    PipelinesConfig,
    RetrievalConfig,
    SessionStoreConfig,
    StorageConfig,
)
from ctxforge.config.defaults import DEFAULT_CONFIG
from ctxforge.config.loader import ConfigLoader

__all__ = [
    # Config schemas
    "EngineConfig",
    "LLMConfig",
    "StorageConfig",
    "SessionStoreConfig",
    "MemoryStoreConfig",
    "MemoryVectorStoreConfig",
    "ExpertiseConfig",
    "ExpertiseStoreConfig",
    "ExpertiseVectorStoreConfig",
    "ExpertiseRetrievalConfig",
    "RetrievalConfig",
    "CompactionConfig",
    "PipelinesConfig",
    "PipelineConfig",
    "MiddlewareItemConfig",
    "ExtractionConfig",
    "ObservabilityConfig",
    # Loader
    "ConfigLoader",
    # Defaults
    "DEFAULT_CONFIG",
]

