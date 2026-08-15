"""
Connection utilities for storage backends.

Provides connection pool management for Redis, PostgreSQL, and MySQL.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    decode_responses: bool = True
    
    # Key prefixes for namespacing
    session_prefix: str = "ctx:session:"
    memory_prefix: str = "ctx:memory:"
    
    # TTL settings
    session_ttl_seconds: int = 86400  # 24 hours default
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for redis connection."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "ssl": self.ssl,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "decode_responses": self.decode_responses,
        }


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "context_engine"
    user: str = "postgres"
    password: Optional[str] = None
    ssl: bool = False
    min_connections: int = 1
    max_connections: int = 10
    
    # Table names
    sessions_table: str = "sessions"
    memories_table: str = "memories"
    
    # Connection pool settings
    command_timeout: float = 30.0
    
    def get_dsn(self) -> str:
        """Get PostgreSQL DSN string."""
        ssl_mode = "require" if self.ssl else "prefer"
        password_part = f":{self.password}" if self.password else ""
        return (
            f"postgresql://{self.user}{password_part}@"
            f"{self.host}:{self.port}/{self.database}?sslmode={ssl_mode}"
        )
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for asyncpg connection."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "ssl": self.ssl,
            "command_timeout": self.command_timeout,
        }


class RedisConnectionManager:
    """
    Manages Redis connections with connection pooling.
    
    Usage:
        manager = RedisConnectionManager(config)
        await manager.connect()
        
        client = manager.client
        await client.set("key", "value")
        
        await manager.disconnect()
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._client = None
        self._pool = None
    
    async def connect(self) -> None:
        """Initialize the connection pool."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis package required. Install with: pip install redis[hiredis]"
            ) from None
        
        self._pool = redis.ConnectionPool(
            max_connections=self.config.max_connections,
            **self.config.get_connection_kwargs()
        )
        client = redis.Redis(connection_pool=self._pool)
        self._client = client
        
        # Test connection
        await client.ping()
    
    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
    
    @property
    def client(self):
        """Get the Redis client."""
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._client
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._client is not None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


class PostgresConnectionManager:
    """
    Manages PostgreSQL connections with connection pooling.
    
    Usage:
        manager = PostgresConnectionManager(config)
        await manager.connect()
        
        async with manager.acquire() as conn:
            await conn.fetch("SELECT * FROM sessions")
        
        await manager.disconnect()
    """
    
    def __init__(self, config: Optional[PostgresConfig] = None):
        self.config = config or PostgresConfig()
        self._pool = None
    
    async def connect(self) -> None:
        """Initialize the connection pool."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg package required. Install with: pip install asyncpg"
            ) from None
        
        self._pool = await asyncpg.create_pool(
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            **self.config.get_connection_kwargs()
        )
    
    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    @property
    def pool(self):
        """Get the connection pool."""
        if self._pool is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._pool
    
    def acquire(self):
        """Acquire a connection from the pool."""
        return self.pool.acquire()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._pool is not None
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


@dataclass
class MySQLConfig:
    """Configuration for MySQL connection."""
    
    host: str = "localhost"
    port: int = 3306
    database: str = "context_engine"
    user: str = "root"
    password: Optional[str] = None
    ssl: bool = False
    min_connections: int = 1
    max_connections: int = 10
    
    # Table names
    sessions_table: str = "sessions"
    memories_table: str = "memories"
    expertise_table: str = "expertise"
    expertise_items_table: str = "expertise_items"
    expertise_usage_logs_table: str = "expertise_usage_logs"
    
    # Connection settings
    connect_timeout: float = 10.0
    charset: str = "utf8mb4"
    autocommit: bool = True
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for aiomysql connection."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.database,
            "user": self.user,
            "password": self.password or "",
            "charset": self.charset,
            "autocommit": self.autocommit,
            "connect_timeout": int(self.connect_timeout),
        }


class MySQLConnectionManager:
    """
    Manages MySQL connections with connection pooling.
    
    Usage:
        manager = MySQLConnectionManager(config)
        await manager.connect()
        
        async with manager.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM sessions")
                rows = await cur.fetchall()
        
        await manager.disconnect()
    """
    
    def __init__(self, config: Optional[MySQLConfig] = None):
        self.config = config or MySQLConfig()
        self._pool = None
    
    async def connect(self) -> None:
        """Initialize the connection pool."""
        try:
            import aiomysql
        except ImportError:
            raise ImportError(
                "aiomysql package required. Install with: pip install aiomysql"
            ) from None
        
        self._pool = await aiomysql.create_pool(
            minsize=self.config.min_connections,
            maxsize=self.config.max_connections,
            **self.config.get_connection_kwargs()
        )
    
    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
    
    @property
    def pool(self):
        """Get the connection pool."""
        if self._pool is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._pool
    
    def acquire(self):
        """Acquire a connection from the pool."""
        return self.pool.acquire()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._pool is not None
    
    async def execute(self, query: str, args: tuple = ()) -> int:
        """Execute a query and return affected rows."""
        async with self.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                await conn.commit()
                return cur.rowcount
    
    async def fetchone(self, query: str, args: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dict."""
        try:
            import aiomysql
        except ImportError:
            raise ImportError("aiomysql package required") from None
        
        async with self.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, args)
                return await cur.fetchone()
    
    async def fetchall(self, query: str, args: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dicts."""
        try:
            import aiomysql
        except ImportError:
            raise ImportError("aiomysql package required") from None
        
        async with self.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, args)
                return await cur.fetchall()
    
    async def fetchval(self, query: str, args: tuple = ()) -> Any:
        """Fetch a single value."""
        async with self.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                row = await cur.fetchone()
                return row[0] if row else None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

