"""
Tests for configuration system.
"""

import json
import os
import tempfile

import pytest

from ctxforge.config.base import (
    CompactionConfig,
    CompactionStrategyType,
    EngineConfig,
    LLMConfig,
    LLMProviderType,
    LogLevel,
    MemoryQualityConfig,
    MemoryStoreConfig,
    MemoryVectorStoreConfig,
    ObservabilityConfig,
    RetrievalConfig,
    RetrievalStrategyType,
    SessionStoreConfig,
    StorageBackendType,
    StorageConfig,
    VectorStoreType,
)
from ctxforge.config.defaults import DEFAULT_CONFIG, TESTING_CONFIG
from ctxforge.config.loader import ConfigLoader, load_config
from ctxforge.core.exceptions import ConfigurationError


class TestLLMConfig:
    """Tests for LLMConfig."""
    
    def test_default_values(self):
        """Test default LLM config values."""
        config = LLMConfig()
        
        assert config.provider == LLMProviderType.MOCK
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 30.0
        assert config.max_retries == 3
    
    def test_temperature_bounds(self):
        """Test temperature validation."""
        # Valid
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0
        
        # Invalid
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        config = LLMConfig(max_tokens=1)
        assert config.max_tokens == 1
        
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)


class TestStorageConfig:
    """Tests for StorageConfig."""
    
    def test_default_values(self):
        """Test default storage config values."""
        config = StorageConfig()
        
        assert config.session.backend == StorageBackendType.MEMORY
        assert config.memory.store_backend == StorageBackendType.MEMORY
        assert config.memory.vector.backend == VectorStoreType.MEMORY
    
    def test_session_store_config(self):
        """Test session store config."""
        config = SessionStoreConfig(
            backend=StorageBackendType.REDIS,
            connection_string="redis://localhost:6379",
            ttl_seconds=3600,
        )
        
        assert config.backend == StorageBackendType.REDIS
        assert config.connection_string == "redis://localhost:6379"
        assert config.ttl_seconds == 3600
    
    def test_memory_store_config(self):
        """Test memory store config."""
        config = MemoryStoreConfig(
            store_backend=StorageBackendType.MEMORY,
            vector=MemoryVectorStoreConfig(
                backend=VectorStoreType.PINECONE,
                index_name="my-index",
            ),
        )
        
        assert config.vector.backend == VectorStoreType.PINECONE
        assert config.vector.index_name == "my-index"


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""
    
    def test_default_values(self):
        """Test default retrieval config."""
        config = RetrievalConfig()
        
        assert config.strategy == RetrievalStrategyType.SEMANTIC
        assert config.default_limit == 5
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.rerank_enabled is False
    
    def test_weight_validation(self):
        """Test weight validation."""
        config = RetrievalConfig(
            semantic_weight=0.0,
            keyword_weight=1.0,
        )
        
        assert config.semantic_weight == 0.0
        assert config.keyword_weight == 1.0


class TestCompactionConfig:
    """Tests for CompactionConfig."""
    
    def test_default_values(self):
        """Test default compaction config."""
        config = CompactionConfig()
        
        assert config.strategy == CompactionStrategyType.SUMMARIZE
        assert config.event_threshold == 10
        assert config.token_threshold == 4000
        assert config.keep_recent == 5
        assert config.async_compaction is True


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""
    
    def test_default_values(self):
        """Test default observability config."""
        config = ObservabilityConfig()
        
        assert config.log_level == LogLevel.INFO
        assert config.tracing_enabled is False
        assert config.metrics_enabled is False


class TestEngineConfig:
    """Tests for EngineConfig."""
    
    def test_default_values(self):
        """Test default engine config."""
        config = EngineConfig()
        
        assert config.name == "ctxforge"
        assert config.version == "0.1.0"
        assert config.debug is False
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.memory_quality, MemoryQualityConfig)
    
    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            "name": "my-engine",
            "debug": True,
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
            },
        }
        
        config = EngineConfig.from_dict(data)
        
        assert config.name == "my-engine"
        assert config.debug is True
        assert config.llm.provider == LLMProviderType.OPENAI
        assert config.llm.model == "gpt-3.5-turbo"
    
    def test_to_dict(self):
        """Test converting config to dict."""
        config = EngineConfig(name="test-engine")
        data = config.to_dict()
        
        assert data["name"] == "test-engine"
        assert "llm" in data
        assert "storage" in data
    
    def test_merge_with(self):
        """Test merging configs."""
        base = EngineConfig(
            name="base",
            llm=LLMConfig(model="gpt-4"),
        )
        
        merged = base.merge_with({
            "name": "merged",
            "llm": {
                "temperature": 0.5,
            },
        })
        
        assert merged.name == "merged"
        assert merged.llm.model == "gpt-4"  # Preserved
        assert merged.llm.temperature == 0.5  # Updated


class TestConfigLoader:
    """Tests for ConfigLoader."""
    
    def test_load_from_json(self):
        """Test loading config from JSON file."""
        config_data = {
            "name": "json-config",
            "debug": True,
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(config_data, f)
            f.flush()
            
            try:
                loader = ConfigLoader()
                config = loader.load_from_file(f.name)
                
                assert config.name == "json-config"
                assert config.debug is True
            finally:
                os.unlink(f.name)
    
    def test_load_from_dict(self):
        """Test loading config from dict."""
        loader = ConfigLoader()
        config = loader.load_from_dict({
            "name": "dict-config",
        })
        
        assert config.name == "dict-config"
    
    def test_file_not_found(self):
        """Test error when file not found."""
        loader = ConfigLoader()
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load_from_file("/nonexistent/config.yaml")
        
        assert "not found" in str(exc_info.value)
    
    def test_unsupported_format(self):
        """Test error for unsupported format."""
        with tempfile.NamedTemporaryFile(
            suffix=".xml",
            delete=False,
        ) as f:
            try:
                loader = ConfigLoader()
                
                with pytest.raises(ConfigurationError) as exc_info:
                    loader.load_from_file(f.name)
                
                assert "Unsupported" in str(exc_info.value)
            finally:
                os.unlink(f.name)
    
    def test_env_overrides(self):
        """Test environment variable overrides."""
        os.environ["CTXFORGE_DEBUG"] = "true"
        os.environ["CTXFORGE_LLM_MODEL"] = "gpt-3.5-turbo"
        
        try:
            loader = ConfigLoader()
            config = EngineConfig()
            config = loader.with_env_overrides(config)
            
            assert config.debug is True
            assert config.llm.model == "gpt-3.5-turbo"
        finally:
            del os.environ["CTXFORGE_DEBUG"]
            del os.environ["CTXFORGE_LLM_MODEL"]
    
    def test_parse_env_value_types(self):
        """Test parsing different value types."""
        loader = ConfigLoader()
        
        assert loader._parse_env_value("true") is True
        assert loader._parse_env_value("false") is False
        assert loader._parse_env_value("123") == 123
        assert loader._parse_env_value("3.14") == 3.14
        assert loader._parse_env_value("hello") == "hello"
    
    def test_save_to_json(self):
        """Test saving config to JSON."""
        config = EngineConfig(name="save-test")
        
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            delete=False,
        ) as f:
            try:
                loader = ConfigLoader()
                loader.save_to_file(config, f.name)
                
                # Reload and verify
                reloaded = loader.load_from_file(f.name)
                assert reloaded.name == "save-test"
            finally:
                os.unlink(f.name)


class TestLoadConfigFunction:
    """Tests for load_config convenience function."""
    
    def test_load_defaults(self):
        """Test loading default config."""
        config = load_config()
        
        assert config.name == "ctxforge"
    
    def test_load_with_overrides(self):
        """Test loading with overrides."""
        config = load_config(
            overrides={"name": "custom", "debug": True},
            use_env=False,
        )
        
        assert config.name == "custom"
        assert config.debug is True
    
    def test_load_from_file_with_overrides(self):
        """Test loading from file with overrides."""
        config_data = {"name": "file-config"}
        
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(config_data, f)
            f.flush()
            
            try:
                config = load_config(
                    file_path=f.name,
                    overrides={"debug": True},
                    use_env=False,
                )
                
                assert config.name == "file-config"
                assert config.debug is True
            finally:
                os.unlink(f.name)


class TestDefaultConfigs:
    """Tests for default configuration presets."""
    
    def test_default_config(self):
        """Test DEFAULT_CONFIG."""
        assert DEFAULT_CONFIG.name == "ctxforge"
        assert DEFAULT_CONFIG.llm.provider == LLMProviderType.MOCK
        assert DEFAULT_CONFIG.storage.session.backend == StorageBackendType.MEMORY
    
    def test_testing_config(self):
        """Test TESTING_CONFIG."""
        assert TESTING_CONFIG.debug is True
        assert TESTING_CONFIG.llm.provider == LLMProviderType.MOCK
        assert TESTING_CONFIG.extraction.async_processing is False
        assert TESTING_CONFIG.compaction.async_compaction is False


class TestMemoryQualityConfig:
    """Tests for memory quality configuration."""

    def test_memory_quality_default_values(self):
        config = EngineConfig()

        assert config.memory_quality.model_routing.enabled is False
        assert config.memory_quality.entropy_gate.enabled is False
        assert config.memory_quality.consolidation.enabled is False
        assert config.memory_quality.retrieval_fast_path.enabled is False
        assert config.memory_quality.graph_path_mining.enabled is False
        assert config.memory_quality.answer_normalization.enabled is False

    def test_memory_quality_overrides(self):
        config = EngineConfig.model_validate(
            {
                "memory_quality": {
                    "model_routing": {
                        "enabled": True,
                        "extraction_model": "gpt-4o-mini",
                        "synthesis_model": "gpt-4o",
                    },
                    "entropy_gate": {
                        "enabled": True,
                        "similarity_threshold": 0.95,
                    },
                }
            }
        )

        assert config.memory_quality.model_routing.enabled is True
        assert config.memory_quality.model_routing.extraction_model == "gpt-4o-mini"
        assert config.memory_quality.model_routing.synthesis_model == "gpt-4o"
        assert config.memory_quality.entropy_gate.enabled is True
        assert config.memory_quality.entropy_gate.similarity_threshold == 0.95

