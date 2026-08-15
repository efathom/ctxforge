"""
Vector Store Protocol Interface.

Re-exports the IVectorStore protocol from the vectorstores module
for convenient access alongside other protocols.
"""

from ctxforge.vectorstores.protocol import (
    DistanceMetric,
    IVectorStore,
    QueryFilter,
    VectorQueryResult,
    VectorRecord,
    VectorStoreConfig,
)

__all__ = [
    "IVectorStore",
    "VectorRecord",
    "VectorQueryResult",
    "VectorStoreConfig",
    "DistanceMetric",
    "QueryFilter",
]

