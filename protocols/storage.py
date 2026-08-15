"""
Storage Protocol Interfaces.

Defines the contracts for session and memory storage backends.
Implementations can use any storage technology (Redis, PostgreSQL,
MongoDB, Vector DBs, etc.) as long as they implement these protocols.
"""

from typing import Dict, List, Optional, Protocol, runtime_checkable

from ctxforge.core.memory import MemoryItem, MemoryQuery
from ctxforge.core.session import Session


@runtime_checkable
class ISessionStore(Protocol):
    """
    Protocol for session storage backends.
    
    Sessions represent the active working state of conversations.
    Implementations should handle:
    - Fast read/write access (hot path)
    - Optimistic locking for concurrency
    - Session expiration/TTL
    
    Example implementations:
    - In-memory store (testing)
    - Redis store (production)
    - PostgreSQL store (persistence)
    """
    
    async def load(self, session_id: str, user_id: str) -> Session:
        """
        Load a session by ID.
        
        If the session doesn't exist, return a new session with
        the given session_id and user_id.
        
        Args:
            session_id: The unique session identifier
            user_id: The user this session belongs to
            
        Returns:
            The session object (existing or newly created)
        """
        ...
    
    async def save(self, session: Session) -> None:
        """
        Save a session.
        
        Should implement optimistic locking using the session.version field.
        Raise ConcurrencyError if version conflict detected.
        
        Args:
            session: The session to save
            
        Raises:
            ConcurrencyError: If session was modified by another process
            StorageError: If storage operation fails
        """
        ...
    
    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            True if the session was deleted, False if not found
        """
        ...
    
    async def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: The session ID to check
            
        Returns:
            True if the session exists
        """
        ...
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Session]:
        """
        List sessions for a user.
        
        Args:
            user_id: The user ID to list sessions for
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of sessions for the user
        """
        ...


@runtime_checkable
class IMemoryStore(Protocol):
    """
    Protocol for long-term memory storage backends.
    
    Memory stores handle persistent knowledge that spans sessions.
    Implementations should support:
    - Semantic/vector search
    - Filtering by type, tags, confidence
    - Deduplication and consolidation
    
    Example implementations:
    - In-memory store (testing)
    - Pinecone/Milvus/Qdrant (vector search)
    - PostgreSQL with pgvector
    """
    
    async def search(
        self,
        query: MemoryQuery,
    ) -> List[MemoryItem]:
        """
        Search for relevant memories.
        
        Should support both semantic (embedding-based) and keyword search,
        with filtering by type, tags, and confidence score.
        
        Args:
            query: The search query parameters
            
        Returns:
            List of matching memories, ordered by relevance
        """
        ...
    
    async def add(self, item: MemoryItem) -> str:
        """
        Add a new memory.
        
        Should handle embedding generation if not provided.
        May perform deduplication against existing memories.
        
        Args:
            item: The memory item to add
            
        Returns:
            The memory_id of the added item
        """
        ...
    
    async def update(self, item: MemoryItem) -> bool:
        """
        Update an existing memory.
        
        Args:
            item: The memory item with updates
            
        Returns:
            True if the memory was updated, False if not found
        """
        ...
    
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory (hard delete).
        
        For soft delete, use deactivate() on the MemoryItem.
        
        Args:
            memory_id: The memory ID to delete
            
        Returns:
            True if the memory was deleted, False if not found
        """
        ...
    
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The memory ID to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        ...
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> List[MemoryItem]:
        """
        Get all memories for a user.
        
        Args:
            user_id: The user ID to get memories for
            limit: Maximum number of memories to return
            include_inactive: Whether to include deactivated memories
            
        Returns:
            List of memories for the user
        """
        ...
    
    async def count(self, user_id: str) -> int:
        """
        Count memories for a user.
        
        Args:
            user_id: The user ID to count memories for
            
        Returns:
            The number of memories for the user
        """
        ...

    async def keyword_search(
        self,
        user_id: str,
        keywords: List[str],
        limit: int = 10,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> List[MemoryItem]:
        """
        Search memories by keyword overlap on the ``keywords`` field.

        Optionally filter by structured metadata (persons, locations, topics).

        Args:
            user_id: The user to search within.
            keywords: Keywords to match against memory keywords.
            limit: Maximum results to return.
            filters: Optional metadata filters, e.g.
                ``{"persons": ["Alice"], "topics": ["travel"]}``.

        Returns:
            Matching memories ordered by keyword overlap (descending).
        """
        ...

