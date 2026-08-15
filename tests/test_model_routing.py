"""
Tests for task-based model routing.
"""

from ctxforge.config.base import TaskModelRoutingConfig
from ctxforge.llm.mock_provider import MockLLMProvider
from ctxforge.llm.task_model_resolver import TaskModelResolver, TaskType


class TestTaskModelResolver:
    """Tests for TaskModelResolver."""

    def test_disabled_returns_provider_default(self):
        """When routing is disabled, resolver returns provider default."""
        config = TaskModelRoutingConfig(enabled=False)
        resolver = TaskModelResolver(config)
        provider = MockLLMProvider()

        result = resolver.resolve(TaskType.EXTRACTION, provider)

        assert result == provider.default_model

    def test_disabled_returns_fallback_when_no_provider(self):
        """When routing is disabled and no provider, returns explicit fallback."""
        config = TaskModelRoutingConfig(enabled=False)
        resolver = TaskModelResolver(config)

        result = resolver.resolve(TaskType.EXTRACTION, fallback_model="fallback-model")

        assert result == "fallback-model"

    def test_none_config_returns_provider_default(self):
        """When config is None, resolver returns provider default."""
        resolver = TaskModelResolver(None)
        provider = MockLLMProvider()

        result = resolver.resolve(TaskType.SYNTHESIS, provider)

        assert result == provider.default_model

    def test_enabled_uses_task_specific_model(self):
        """When enabled, resolver uses task-specific model."""
        config = TaskModelRoutingConfig(
            enabled=True,
            extraction_model="gpt-4o-mini",
            synthesis_model="gpt-4o",
        )
        resolver = TaskModelResolver(config)
        provider = MockLLMProvider()

        extraction_model = resolver.resolve(TaskType.EXTRACTION, provider)
        synthesis_model = resolver.resolve(TaskType.SYNTHESIS, provider)

        assert extraction_model == "gpt-4o-mini"
        assert synthesis_model == "gpt-4o"

    def test_enabled_falls_back_to_routing_default(self):
        """When task model not set, falls back to routing default_model."""
        config = TaskModelRoutingConfig(
            enabled=True,
            default_model="gpt-4-turbo",
            extraction_model="gpt-4o-mini",
        )
        resolver = TaskModelResolver(config)
        provider = MockLLMProvider()

        # Extraction has specific model
        extraction_model = resolver.resolve(TaskType.EXTRACTION, provider)
        # Planning has no specific model, falls back to routing default
        planning_model = resolver.resolve(TaskType.PLANNING, provider)

        assert extraction_model == "gpt-4o-mini"
        assert planning_model == "gpt-4-turbo"

    def test_enabled_falls_back_to_provider_default(self):
        """When no task or routing default, falls back to provider default."""
        config = TaskModelRoutingConfig(
            enabled=True,
            extraction_model="gpt-4o-mini",
        )
        resolver = TaskModelResolver(config)
        provider = MockLLMProvider()

        # Judge has no specific model and no routing default
        judge_model = resolver.resolve(TaskType.JUDGE, provider)

        assert judge_model == provider.default_model

    def test_enabled_property(self):
        """Test enabled property reflects config state."""
        disabled_resolver = TaskModelResolver(TaskModelRoutingConfig(enabled=False))
        enabled_resolver = TaskModelResolver(TaskModelRoutingConfig(enabled=True))
        none_resolver = TaskModelResolver(None)

        assert disabled_resolver.enabled is False
        assert enabled_resolver.enabled is True
        assert none_resolver.enabled is False

    def test_convenience_methods(self):
        """Test convenience methods for each task type."""
        config = TaskModelRoutingConfig(
            enabled=True,
            extraction_model="extraction-model",
            planning_model="planning-model",
            synthesis_model="synthesis-model",
            judge_model="judge-model",
        )
        resolver = TaskModelResolver(config)

        assert resolver.resolve_extraction() == "extraction-model"
        assert resolver.resolve_planning() == "planning-model"
        assert resolver.resolve_synthesis() == "synthesis-model"
        assert resolver.resolve_judge() == "judge-model"

    def test_all_task_types_supported(self):
        """Verify all TaskType enum values are handled."""
        config = TaskModelRoutingConfig(enabled=True, default_model="default")
        resolver = TaskModelResolver(config)

        for task in TaskType:
            result = resolver.resolve(task)
            assert result is not None


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_type_values(self):
        """Verify expected task type values."""
        assert TaskType.EXTRACTION.value == "extraction"
        assert TaskType.PLANNING.value == "planning"
        assert TaskType.SYNTHESIS.value == "synthesis"
        assert TaskType.JUDGE.value == "judge"

    def test_task_type_is_string_enum(self):
        """TaskType value should be usable as string."""
        assert TaskType.EXTRACTION.value == "extraction"
        assert TaskType.PLANNING.value == "planning"
        # Can be compared to string
        assert TaskType.EXTRACTION == "extraction"
        assert TaskType.PLANNING == "planning"


class TestModelRoutingIntegration:
    """Integration tests for model routing with config."""

    def test_routing_config_from_engine_config(self):
        """Test routing config can be extracted from EngineConfig."""
        from ctxforge.config.base import EngineConfig

        config = EngineConfig.model_validate({
            "memory_quality": {
                "model_routing": {
                    "enabled": True,
                    "extraction_model": "gpt-4o-mini",
                    "synthesis_model": "gpt-4o",
                }
            }
        })

        resolver = TaskModelResolver(config.memory_quality.model_routing)

        assert resolver.enabled is True
        assert resolver.resolve_extraction() == "gpt-4o-mini"
        assert resolver.resolve_synthesis() == "gpt-4o"

    def test_routing_disabled_by_default(self):
        """Model routing is disabled by default in EngineConfig."""
        from ctxforge.config.base import EngineConfig

        config = EngineConfig()
        resolver = TaskModelResolver(config.memory_quality.model_routing)

        assert resolver.enabled is False
