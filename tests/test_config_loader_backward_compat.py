import json
import os
import tempfile

from ctxforge.config.base import StorageBackendType, VectorStoreType
from ctxforge.config.loader import ConfigLoader


def test_loader_accepts_legacy_storage_shape_without_memory_quality_block():
    config_data = {
        "name": "legacy-shape",
        "storage": {
            "memory": {
                "backend": "chromadb",
                "connection_string": "/tmp/chroma",
                "index_name": "legacy_memories",
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file_obj:
        json.dump(config_data, file_obj)
        file_obj.flush()
        file_path = file_obj.name

    try:
        loader = ConfigLoader()
        config = loader.load_from_file(file_path)

        assert config.name == "legacy-shape"
        assert config.storage.memory.store_backend == StorageBackendType.MEMORY
        assert config.storage.memory.vector.backend == VectorStoreType.CHROMADB
        assert config.memory_quality.entropy_gate.enabled is False
    finally:
        os.unlink(file_path)


def test_loader_preserves_existing_behavior_with_env_overrides():
    os.environ["CTXFORGE_NAME"] = "env-updated"
    os.environ["CTXFORGE_DEBUG"] = "true"

    try:
        loader = ConfigLoader()
        config = loader.load_from_dict({"name": "before-env"})
        updated = loader.with_env_overrides(config)

        assert updated.name == "env-updated"
        assert updated.debug is True
        assert updated.memory_quality.retrieval_fast_path.enabled is False
    finally:
        del os.environ["CTXFORGE_NAME"]
        del os.environ["CTXFORGE_DEBUG"]
