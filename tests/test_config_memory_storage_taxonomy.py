
from ctxforge.config.base import EngineConfig, StorageBackendType, VectorStoreType


def test_legacy_storage_memory_backend_vector_is_translated():
    cfg = EngineConfig.model_validate(
        {
            "storage": {
                "memory": {
                    "backend": "chromadb",
                    "connection_string": "/tmp/chroma",
                    "index_name": "mem_idx",
                    "extra_params": {"persist_directory": "/tmp/chroma"},
                }
            }
        }
    )
    assert cfg.storage.memory.store_backend == StorageBackendType.MEMORY
    assert cfg.storage.memory.vector.backend == VectorStoreType.CHROMADB
    assert cfg.storage.memory.vector.index_name == "mem_idx"


def test_legacy_storage_memory_backend_store_is_translated():
    cfg = EngineConfig.model_validate(
        {
            "storage": {
                "memory": {
                    "backend": "postgres",
                    "store_connection_string": "postgresql://u:p@localhost:5432/db",
                }
            }
        }
    )
    assert cfg.storage.memory.store_backend == StorageBackendType.POSTGRES
    assert cfg.storage.memory.vector.backend == VectorStoreType.MEMORY


def test_new_storage_memory_shape_is_accepted():
    cfg = EngineConfig.model_validate(
        {
            "storage": {
                "memory": {
                    "store_backend": "redis",
                    "store_connection_string": "redis://localhost:6379/0",
                    "vector": {"backend": "memory"},
                }
            }
        }
    )
    assert cfg.storage.memory.store_backend == StorageBackendType.REDIS
    assert cfg.storage.memory.vector.backend == VectorStoreType.MEMORY

