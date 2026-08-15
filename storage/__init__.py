"""
Storage backends for the ctxforge framework.

Provides implementations of session, memory, and expertise storage:
- In-memory stores (default, for testing)
- Redis stores (production)
- PostgreSQL stores (persistence)
- MySQL stores (persistence)

Storage is organized by backend type:
- storage.memory: In-memory implementations
- storage.postgres: PostgreSQL implementations
- storage.redis: Redis implementations
- storage.mysql: MySQL implementations
"""

# Connection utilities (shared across backends)
from ctxforge.storage.connection import (
    MySQLConfig,
    MySQLConnectionManager,
    PostgresConfig,
    PostgresConnectionManager,
    RedisConfig,
    RedisConnectionManager,
)

# In-memory stores
from ctxforge.storage.memory import (
    DeduplicatingMemoryStore,
    InMemoryExpertiseStore,
    InMemoryMemoryStore,
    InMemorySessionStore,
)

# MySQL stores
from ctxforge.storage.mysql import (
    MySQLExpertiseStore,
    MySQLMemoryStore,
    MySQLSessionStore,
)

# PostgreSQL stores
from ctxforge.storage.postgres import (
    PostgresExpertiseStore,
    PostgresMemoryStore,
    PostgresSessionStore,
)

# Redis stores
from ctxforge.storage.redis import (
    RedisExpertiseStore,
    RedisMemoryStore,
    RedisSessionStore,
)

__all__ = [
    # Connection utilities
    "RedisConfig",
    "PostgresConfig",
    "MySQLConfig",
    "RedisConnectionManager",
    "PostgresConnectionManager",
    "MySQLConnectionManager",
    # In-memory stores
    "InMemorySessionStore",
    "InMemoryMemoryStore",
    "InMemoryExpertiseStore",
    # Deduplicating wrapper
    "DeduplicatingMemoryStore",
    # PostgreSQL stores
    "PostgresSessionStore",
    "PostgresMemoryStore",
    "PostgresExpertiseStore",
    # Redis stores
    "RedisSessionStore",
    "RedisMemoryStore",
    "RedisExpertiseStore",
    # MySQL stores
    "MySQLSessionStore",
    "MySQLMemoryStore",
    "MySQLExpertiseStore",
]
