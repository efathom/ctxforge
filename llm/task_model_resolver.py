"""
Task-based model resolver for ctxforge.

Provides centralized resolution of model names by task type, with fallback
to provider defaults. This enables cost/quality optimization by routing
different tasks to different models.
"""

from enum import Enum
from typing import Optional

from ctxforge.config.base import TaskModelRoutingConfig
from ctxforge.protocols.llm import ILLMProvider


class TaskType(str, Enum):
    """Supported task types for model routing."""

    EXTRACTION = "extraction"
    PLANNING = "planning"
    SYNTHESIS = "synthesis"
    JUDGE = "judge"


class TaskModelResolver:
    """
    Resolves model names by task type using TaskModelRoutingConfig.

    Resolution order:
    1. Task-specific model from config (e.g., extraction_model)
    2. Default model from routing config
    3. Provider's default model (fallback)

    Example:
        resolver = TaskModelResolver(config.memory_quality.model_routing)
        model = resolver.resolve(TaskType.EXTRACTION, llm_provider)
    """

    def __init__(self, routing_config: Optional[TaskModelRoutingConfig] = None):
        """
        Initialize the resolver.

        Args:
            routing_config: Task model routing configuration. If None or disabled,
                            all resolutions fall back to provider default.
        """
        self._config = routing_config
        self._enabled = routing_config is not None and routing_config.enabled

    @property
    def enabled(self) -> bool:
        """Whether task-based routing is enabled."""
        return self._enabled

    def resolve(
        self,
        task: TaskType,
        llm_provider: Optional[ILLMProvider] = None,
        fallback_model: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve the model name for a given task.

        Args:
            task: The task type to resolve a model for.
            llm_provider: Optional LLM provider for fallback to default_model.
            fallback_model: Optional explicit fallback if provider is not available.

        Returns:
            The resolved model name, or None if no model could be resolved.
        """
        if not self._enabled or self._config is None:
            return self._get_fallback(llm_provider, fallback_model)

        task_model = self._get_task_model(task)
        if task_model:
            return task_model

        if self._config.default_model:
            return self._config.default_model

        return self._get_fallback(llm_provider, fallback_model)

    def _get_task_model(self, task: TaskType) -> Optional[str]:
        """Get the configured model for a specific task."""
        if self._config is None:
            return None

        task_to_attr = {
            TaskType.EXTRACTION: "extraction_model",
            TaskType.PLANNING: "planning_model",
            TaskType.SYNTHESIS: "synthesis_model",
            TaskType.JUDGE: "judge_model",
        }

        attr = task_to_attr.get(task)
        if attr:
            return getattr(self._config, attr, None)
        return None

    def _get_fallback(
        self,
        llm_provider: Optional[ILLMProvider],
        fallback_model: Optional[str],
    ) -> Optional[str]:
        """Get the fallback model from provider or explicit fallback."""
        if llm_provider is not None:
            return llm_provider.default_model
        return fallback_model

    def resolve_extraction(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        fallback_model: Optional[str] = None,
    ) -> Optional[str]:
        """Convenience method for extraction task."""
        return self.resolve(TaskType.EXTRACTION, llm_provider, fallback_model)

    def resolve_planning(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        fallback_model: Optional[str] = None,
    ) -> Optional[str]:
        """Convenience method for planning task."""
        return self.resolve(TaskType.PLANNING, llm_provider, fallback_model)

    def resolve_synthesis(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        fallback_model: Optional[str] = None,
    ) -> Optional[str]:
        """Convenience method for synthesis task."""
        return self.resolve(TaskType.SYNTHESIS, llm_provider, fallback_model)

    def resolve_judge(
        self,
        llm_provider: Optional[ILLMProvider] = None,
        fallback_model: Optional[str] = None,
    ) -> Optional[str]:
        """Convenience method for judge task."""
        return self.resolve(TaskType.JUDGE, llm_provider, fallback_model)
