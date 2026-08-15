"""
Configuration management for examples.

Uses the framework's ConfigLoader to load engine configuration from YAML,
with environment variable overrides supported.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ctxforge.config.base import EngineConfig
from ctxforge.config.loader import ConfigLoader

# Try to load examples/.env regardless of current working directory.
try:
    from dotenv import load_dotenv

    _dotenv_path = Path(__file__).parent / ".env"
    # Load examples/.env first, then fall back to default search behavior.
    load_dotenv(dotenv_path=_dotenv_path, override=False)
    load_dotenv(override=False)
except ImportError:
    pass


# Default config file path (relative to examples directory)
DEFAULT_CONFIG_FILE = Path(__file__).parent / "engine_config.yaml"


@dataclass
class PostgresConfig:
    """PostgreSQL configuration (from extensions)."""
    host: str = "localhost"
    port: int = 5432
    database: str = "ctxforge"
    user: str = "ctxforge"
    password: str = "password"


@dataclass
class MySQLConfig:
    """MySQL configuration (from extensions)."""
    host: str = "localhost"
    port: int = 3306
    database: str = "contextengine"
    user: str = "contextengine"
    password: str = "contextengine"


@dataclass
class ChromaConfig:
    """ChromaDB configuration (from storage.memory.vector.extra_params)."""
    persist_directory: str = "./chroma_data"
    collection_name: str = "memories"


@dataclass
class DemoConfig:
    """
    Complete demo configuration.
    
    Combines the framework's EngineConfig with demo-specific settings
    extracted from the config for convenience.
    """
    engine: EngineConfig
    postgres: PostgresConfig
    mysql: MySQLConfig
    chroma: ChromaConfig
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from engine config."""
        return self.engine.llm.api_key or ""
    
    @property
    def openai_model(self) -> str:
        """Get OpenAI model from engine config."""
        return self.engine.llm.model
    
    @property
    def embedding_model(self) -> str:
        """Get embedding model from engine config."""
        return self.engine.storage.memory.vector.embedding.model
    
    @property
    def system_prompt(self) -> str:
        """Get system prompt from engine config."""
        return self.engine.prompts.system_template


def load_config(config_file: Optional[str] = None) -> DemoConfig:
    """
    Load configuration from YAML file with environment overrides.
    
    The configuration loading follows this priority (later overrides earlier):
    1. Default values in engine_config.yaml
    2. Values from the specified config file
    3. Environment variables with CTXFORGE_ prefix
    4. Direct environment variables (OPENAI_API_KEY, etc.)
    
    Args:
        config_file: Optional path to a YAML config file.
                    Defaults to examples/engine_config.yaml
    
    Returns:
        DemoConfig with all settings
        
    Raises:
        ValueError: If required settings (like API key) are missing
    """
    loader = ConfigLoader()
    
    # Determine config file path
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = DEFAULT_CONFIG_FILE
    
    # Load from YAML file
    if config_path.exists():
        print(f"   📄 Loading config from: {config_path}")
        engine_config = loader.load_from_file(str(config_path))
    else:
        print(f"   ⚠️ Config file not found: {config_path}, using defaults")
        engine_config = EngineConfig()
    
    # Apply environment variable overrides (CTXFORGE_* prefix)
    engine_config = loader.with_env_overrides(engine_config)
    
    # Determine provider type first so we can apply the right API key
    provider_name = (getattr(engine_config.llm.provider, "value", None) or str(engine_config.llm.provider)).lower()

    # Auto-detect Azure: if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set,
    # treat this as an Azure configuration even if YAML says "openai".
    # This ensures run_demo.py uses the same provider as validate_azure_openai.py.
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_api_key and azure_endpoint and provider_name not in ("azure", "azure_openai", "openrouter"):
        # User has Azure env vars but YAML says openai - switch to azure
        provider_name = "azure"
        engine_config = engine_config.merge_with({"llm": {"provider": "azure"}})

    # Handle direct Azure OpenAI environment variables (convenience).
    # For Azure providers, always prefer AZURE_OPENAI_API_KEY over OPENAI_API_KEY.
    if provider_name in ("azure", "azure_openai"):
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        patch: dict = {}

        # LLM API key - for Azure, prefer AZURE_OPENAI_API_KEY over any existing key
        if azure_api_key:
            patch.setdefault("llm", {})["api_key"] = azure_api_key

        # Azure endpoint is stored in EngineConfig.llm.api_base (used by the factory wiring).
        # If the user sets AZURE_OPENAI_ENDPOINT, it should override YAML/CTXFORGE_* values
        # so `run_demo.py` uses the same endpoint as validate_azure_openai.py.
        if azure_endpoint:
            patch.setdefault("llm", {})["api_base"] = azure_endpoint

        # Azure API version is stored in llm.extra_params.api_version (used by the factory wiring).
        # Same rule: explicit AZURE_OPENAI_API_VERSION overrides YAML/CTXFORGE_*.
        if azure_api_version:
            patch.setdefault("llm", {}).setdefault("extra_params", {})
            patch["llm"]["extra_params"]["api_version"] = azure_api_version

        # For Azure, llm.model should typically be the chat *deployment* name (not an OpenAI model id).
        if azure_chat_deployment and engine_config.llm.model != azure_chat_deployment:
            patch.setdefault("llm", {})["model"] = azure_chat_deployment

        # For Azure embeddings: switch embedding provider to azure and set deployment name.
        # This is critical - YAML may say provider: openai but we're using Azure.
        patch.setdefault("storage", {}).setdefault("memory", {}).setdefault("vector", {}).setdefault("embedding", {})
        patch["storage"]["memory"]["vector"]["embedding"]["provider"] = "azure"
        if azure_embedding_deployment:
            patch["storage"]["memory"]["vector"]["embedding"]["model"] = azure_embedding_deployment
        if azure_api_key:
            patch["storage"]["memory"]["vector"]["embedding"]["api_key"] = azure_api_key

        if patch:
            engine_config = engine_config.merge_with(patch)
    elif provider_name == "openrouter":
        # OpenRouter configuration from OPENROUTER_* env vars.
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_model = os.getenv("OPENROUTER_MODEL")
        openrouter_http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        openrouter_site_title = os.getenv("OPENROUTER_SITE_TITLE")
        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

        patch: dict = {}
        if openrouter_api_key:
            patch.setdefault("llm", {})["api_key"] = openrouter_api_key
        if openrouter_model:
            patch.setdefault("llm", {})["model"] = openrouter_model
        extra: dict = {}
        if openrouter_http_referer:
            extra["http_referer"] = openrouter_http_referer
        if openrouter_site_title:
            extra["site_title"] = openrouter_site_title
        if openrouter_base_url:
            extra["base_url"] = openrouter_base_url
        if extra:
            patch.setdefault("llm", {}).setdefault("extra_params", {})
            patch["llm"]["extra_params"].update(extra)
        if patch:
            engine_config = engine_config.merge_with(patch)
    else:
        # Handle direct OpenAI API key environment variable for non-Azure providers
        # (common pattern, more convenient than CTXFORGE_LLM_API_KEY)
        if not engine_config.llm.api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                # Create new config with api_key set
                engine_config = engine_config.merge_with({
                    "llm": {"api_key": api_key},
                    "storage": {"memory": {"vector": {"embedding": {"api_key": api_key}}}}
                })
    
    if provider_name == "openai" and not engine_config.llm.api_key:
        raise ValueError(
            "OpenAI API key is required when llm.provider=openai.\n"
            "Set it with: export OPENAI_API_KEY=sk-...\n"
            "Or in engine_config.yaml: llm.api_key: sk-..."
        )

    if provider_name in ("azure", "azure_openai"):
        missing = []
        if not engine_config.llm.api_key:
            missing.append("AZURE_OPENAI_API_KEY (or CTXFORGE_LLM_API_KEY)")
        if not engine_config.llm.api_base:
            missing.append("AZURE_OPENAI_ENDPOINT (or CTXFORGE_LLM_API_BASE)")
        if missing:
            raise ValueError(
                "Azure OpenAI is selected (llm.provider=azure) but required settings are missing:\n"
                + "\n".join([f"- {m}" for m in missing])
                + "\n\nTip: set these in ctxforge/examples/.env:\n"
                + "  AZURE_OPENAI_API_KEY=...\n"
                + "  AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com\n"
                + "  AZURE_OPENAI_API_VERSION=2024-02-15-preview\n"
                + "  AZURE_OPENAI_CHAT_DEPLOYMENT=<your-chat-deployment>\n"
                + "  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-deployment>\n"
            )

    if provider_name == "openrouter" and not engine_config.llm.api_key:
        raise ValueError(
            "OpenRouter is selected (llm.provider=openrouter) but no API key is set.\n"
            "Set it in ctxforge/examples/.env:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  OPENROUTER_MODEL=openai/gpt-4o-mini\n"
        )

    # Local embedding server configuration (TEI/Ollama/vLLM, or in-process
    # sentence-transformers via provider=local). Read from LOCAL_EMBEDDING_*
    # directly because the generic CTXFORGE_* underscore mapping cannot address
    # fields like `base_url`.
    local_emb_provider = os.getenv("LOCAL_EMBEDDING_PROVIDER")
    if local_emb_provider:
        emb_patch: dict = {"provider": local_emb_provider, "api_key": os.getenv("LOCAL_EMBEDDING_API_KEY") or ""}
        local_emb_model = os.getenv("LOCAL_EMBEDDING_MODEL")
        if local_emb_model:
            emb_patch["model"] = local_emb_model
        local_emb_base_url = os.getenv("LOCAL_EMBEDDING_BASE_URL")
        if local_emb_base_url:
            emb_patch["base_url"] = local_emb_base_url
        local_emb_dimension = os.getenv("LOCAL_EMBEDDING_DIMENSION")
        if local_emb_dimension:
            emb_patch["dimension"] = int(local_emb_dimension)
        engine_config = engine_config.merge_with(
            {"storage": {"memory": {"vector": {"embedding": emb_patch}}}}
        )

    # Extract PostgreSQL config from extensions
    pg_ext = engine_config.extensions.get("postgres", {})
    postgres_config = PostgresConfig(
        host=str(pg_ext.get("host", "localhost")),
        port=int(pg_ext.get("port", 5432)),
        database=str(pg_ext.get("database", "ctxforge")),
        user=str(pg_ext.get("user", "ctxforge")),
        password=str(pg_ext.get("password", "password")),
    )
    
    # Extract MySQL config from extensions
    mysql_ext = engine_config.extensions.get("mysql", {})
    mysql_config = MySQLConfig(
        host=os.getenv("MYSQL_HOST", str(mysql_ext.get("host", "localhost"))),
        port=int(os.getenv("MYSQL_PORT", str(mysql_ext.get("port", 3306)))),
        database=os.getenv("MYSQL_DATABASE", str(mysql_ext.get("database", "contextengine"))),
        user=os.getenv("MYSQL_USER", str(mysql_ext.get("user", "contextengine"))),
        password=os.getenv("MYSQL_PASSWORD", str(mysql_ext.get("password", "contextengine"))),
    )
    
    # Extract ChromaDB config from storage.memory.vector.extra_params
    chroma_params = engine_config.storage.memory.vector.extra_params
    chroma_config = ChromaConfig(
        persist_directory=str(chroma_params.get("persist_directory", "./chroma_data")),
        collection_name=str(chroma_params.get("collection_name", "memories")),
    )
    
    return DemoConfig(
        engine=engine_config,
        postgres=postgres_config,
        mysql=mysql_config,
        chroma=chroma_config,
    )


def print_config_summary(config: DemoConfig) -> None:
    """Print a summary of the loaded configuration."""
    print("\n📋 Configuration Summary:")
    print(f"   Engine: {config.engine.name} v{config.engine.version}")
    print(f"   LLM: {config.engine.llm.provider.value} / {config.engine.llm.model}")
    print(f"   Embedding: {config.engine.storage.memory.vector.embedding.model}")
    print(f"   Session Storage: {config.engine.storage.session.backend.value}")
    print(f"   Memory Store: {config.engine.storage.memory.store_backend.value}")
    print(f"   Memory Vector Index: {config.engine.storage.memory.vector.backend.value}")
    print(f"   Retrieval: {config.engine.retrieval.strategy.value}")
    print(f"   Extraction: {'enabled' if config.engine.extraction.enabled else 'disabled'}")
    
    prepare_count = len(config.engine.pipelines.prepare.chain)
    record_count = len(config.engine.pipelines.record.chain)
    print(f"   Pipelines: prepare={prepare_count}, record={record_count}")

    if config.engine.expertise.enabled:
        print(
            f"   Expertise: enabled "
            f"(store={config.engine.expertise.store.backend.value}, "
            f"vectorstore={config.engine.expertise.vectorstore.backend.value})"
        )
    else:
        print("   Expertise: disabled")
    
    if config.engine.debug:
        print("   Debug: enabled")
