"""
Configuration Loader for the ctxforge framework.

Supports loading configuration from multiple sources:
- YAML files
- JSON files
- Environment variables
- Python dictionaries
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctxforge.config.base import EngineConfig
from ctxforge.core.exceptions import ConfigurationError


class ConfigLoader:
    """
    Loads and merges configuration from multiple sources.
    
    Configuration is loaded in the following priority order
    (later sources override earlier ones):
    1. Default configuration
    2. Configuration file (YAML/JSON)
    3. Environment variables
    4. Programmatic overrides
    
    Example:
        loader = ConfigLoader()
        config = loader.load_from_file("config.yaml")
        config = loader.with_env_overrides(config)
    """
    
    # Environment variable prefix used for config overrides.
    ENV_PREFIX = "CTXFORGE_"
    
    def __init__(self):
        self._yaml_available = self._check_yaml_available()
    
    @staticmethod
    def _check_yaml_available() -> bool:
        """Check if PyYAML is available."""
        try:
            import yaml  # noqa: F401
            return True
        except ImportError:
            return False
    
    def load_from_file(self, path: str) -> EngineConfig:
        """
        Load configuration from a file.
        
        Supports YAML (.yaml, .yml) and JSON (.json) files.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Loaded EngineConfig
            
        Raises:
            ConfigurationError: If file not found or invalid format
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                config_key="file_path",
            )
        
        suffix = file_path.suffix.lower()
        
        if suffix in (".yaml", ".yml"):
            return self._load_yaml(file_path)
        elif suffix == ".json":
            return self._load_json(file_path)
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {suffix}",
                config_key="file_format",
            )
    
    def _load_yaml(self, path: Path) -> EngineConfig:
        """Load configuration from a YAML file."""
        if not self._yaml_available:
            raise ConfigurationError(
                "PyYAML is not installed. Install it with: pip install pyyaml",
                config_key="yaml_dependency",
            )
        
        import yaml
        
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            
            if data is None:
                data = {}
            
            # Expand environment variables in string values
            data = self._expand_env_vars(data)
            
            return EngineConfig.model_validate(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_key="yaml_parse",
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration: {e}",
                config_key="load_error",
            ) from e
    
    def _load_json(self, path: Path) -> EngineConfig:
        """Load configuration from a JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Expand environment variables in string values
            data = self._expand_env_vars(data)
            
            return EngineConfig.model_validate(data)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {e}",
                config_key="json_parse",
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration: {e}",
                config_key="load_error",
            ) from e
    
    def load_from_dict(self, data: Dict[str, Any]) -> EngineConfig:
        """
        Load configuration from a dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Loaded EngineConfig
        """
        try:
            return EngineConfig.model_validate(data)
        except Exception as e:
            raise ConfigurationError(
                f"Invalid configuration data: {e}",
                config_key="validation",
            ) from e
    
    def with_env_overrides(self, config: EngineConfig) -> EngineConfig:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables are expected in the format:
        CTXFORGE_<SECTION>_<KEY>=value
        
        Examples:
            CTXFORGE_LLM_PROVIDER=openai
            CTXFORGE_LLM_API_KEY=sk-xxx
            CTXFORGE_STORAGE_SESSION_BACKEND=redis
        
        Args:
            config: Base configuration
            
        Returns:
            Configuration with environment overrides applied
        """
        overrides = self._collect_env_overrides()
        
        if overrides:
            return config.merge_with(overrides)
        
        return config
    
    def _collect_env_overrides(self) -> Dict[str, Any]:
        """Collect configuration overrides from environment variables."""
        overrides: Dict[str, Any] = {}
        
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue
            
            # Remove prefix and convert to lowercase
            config_path = key[len(self.ENV_PREFIX):].lower()
            
            # Split by underscore to get nested path
            parts = config_path.split("_")
            
            # Convert value to appropriate type
            typed_value = self._parse_env_value(value)
            
            # Build nested dictionary
            self._set_nested(overrides, parts, typed_value)
        
        return overrides
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON (for complex values)
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value
    
    @staticmethod
    def _set_nested(data: Dict, path: List[str], value: Any) -> None:
        """Set a value in a nested dictionary using a path."""
        current = data
        
        for _i, key in enumerate(path[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        if path:
            current[path[-1]] = value
    
    def _expand_env_vars(self, data: Any) -> Any:
        """
        Recursively expand environment variables in string values.
        
        Supports ${VAR_NAME} and $VAR_NAME syntax.
        """
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._expand_string_env_vars(data)
        else:
            return data
    
    @staticmethod
    def _expand_string_env_vars(value: str) -> str:
        """
        Expand environment variables in a string.
        
        Supports:
        - ${VAR_NAME} - simple variable expansion
        - ${VAR_NAME:-default} - with default value if not set
        - $VAR_NAME - simple variable (no braces)
        """
        import re
        
        # Match ${VAR_NAME} or ${VAR_NAME:-default} or $VAR_NAME
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def replace(match):
            # ${VAR} or ${VAR:-default}
            if match.group(1):
                var_name = match.group(1)
                default_value = match.group(2)  # May be None
                env_value = os.environ.get(var_name)
                if env_value is not None:
                    return env_value
                elif default_value is not None:
                    return default_value
                else:
                    return match.group(0)  # Keep original if no value and no default
            # $VAR_NAME
            else:
                var_name = match.group(3)
                return os.environ.get(var_name, match.group(0))
        
        return re.sub(pattern, replace, value)
    
    def save_to_file(self, config: EngineConfig, path: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            config: Configuration to save
            path: Path to save to (format determined by extension)
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        data = config.model_dump()
        
        if suffix in (".yaml", ".yml"):
            if not self._yaml_available:
                raise ConfigurationError(
                    "PyYAML is not installed. Install it with: pip install pyyaml",
                    config_key="yaml_dependency",
                )
            
            import yaml
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        elif suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {suffix}",
                config_key="file_format",
            )


def load_config(
    file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    use_env: bool = True,
) -> EngineConfig:
    """
    Convenience function to load configuration.
    
    Args:
        file_path: Optional path to configuration file
        overrides: Optional dictionary of overrides
        use_env: Whether to apply environment variable overrides
        
    Returns:
        Loaded and merged EngineConfig
    """
    loader = ConfigLoader()
    
    # Start with default config
    if file_path:
        config = loader.load_from_file(file_path)
    else:
        config = EngineConfig()
    
    # Apply environment overrides
    if use_env:
        config = loader.with_env_overrides(config)
    
    # Apply programmatic overrides
    if overrides:
        config = config.merge_with(overrides)
    
    return config

