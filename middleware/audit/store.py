"""
Audit Store implementations.

Provides storage for audit events.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class AuditEvent:
    """
    An audit event recording an operation.
    
    Attributes:
        event_id: Unique identifier
        timestamp: When the event occurred
        user_id: User who triggered the event
        session_id: Session identifier
        event_type: Type of event (e.g., "request", "pii_detected")
        action: Specific action taken
        details: Additional event details
        source: Source system/component
        duration_ms: Processing time in milliseconds
        success: Whether the operation succeeded
        error: Error message if failed
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    event_type: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "context_engine"
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "action": self.action,
            "details": self.details,
            "source": self.source,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@runtime_checkable
class IAuditStore(Protocol):
    """Protocol for audit event storage."""
    
    async def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.
        
        Args:
            event: The event to log
        """
        ...
    
    async def query(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            user_id: Filter by user
            session_id: Filter by session
            event_type: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        ...


class InMemoryAuditStore(IAuditStore):
    """
    In-memory audit store for development/testing.
    
    Events are stored in memory and lost on restart.
    Use a persistent store (database, log service) in production.
    
    Example:
        store = InMemoryAuditStore(max_events=10000)
        
        await store.log(AuditEvent(
            event_type="request",
            action="process_input",
            user_id="user123",
        ))
        
        events = await store.query(user_id="user123")
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize the store.
        
        Args:
            max_events: Maximum events to retain
        """
        self._max_events = max_events
        self._events: List[AuditEvent] = []
        
        # Indexes for faster querying
        self._by_user: Dict[str, List[int]] = defaultdict(list)
        self._by_session: Dict[str, List[int]] = defaultdict(list)
        self._by_type: Dict[str, List[int]] = defaultdict(list)
    
    async def log(self, event: AuditEvent) -> None:
        """Log an event."""
        # Evict old events if at capacity
        if len(self._events) >= self._max_events:
            self._evict_oldest()
        
        idx = len(self._events)
        self._events.append(event)
        
        # Update indexes
        if event.user_id:
            self._by_user[event.user_id].append(idx)
        if event.session_id:
            self._by_session[event.session_id].append(idx)
        if event.event_type:
            self._by_type[event.event_type].append(idx)
    
    async def query(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query events with filters."""
        # Start with candidate set
        if user_id:
            candidates = set(self._by_user.get(user_id, []))
        elif session_id:
            candidates = set(self._by_session.get(session_id, []))
        elif event_type:
            candidates = set(self._by_type.get(event_type, []))
        else:
            candidates = set(range(len(self._events)))
        
        # Apply additional filters
        results = []
        for idx in sorted(candidates, reverse=True):  # Most recent first
            if len(results) >= limit:
                break
            
            event = self._events[idx]
            
            # Filter checks
            if user_id and event.user_id != user_id:
                continue
            if session_id and event.session_id != session_id:
                continue
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            
            results.append(event)
        
        return results
    
    async def count(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> int:
        """Count events matching filters."""
        events = await self.query(
            user_id=user_id,
            event_type=event_type,
            limit=self._max_events,
        )
        return len(events)
    
    async def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._by_type.clear()
    
    def _evict_oldest(self) -> None:
        """Evict oldest events to make room."""
        # Simple approach: remove oldest 10%
        remove_count = max(1, self._max_events // 10)
        self._events = self._events[remove_count:]
        
        # Rebuild indexes (simplified - could be optimized)
        self._by_user.clear()
        self._by_session.clear()
        self._by_type.clear()
        
        for idx, event in enumerate(self._events):
            if event.user_id:
                self._by_user[event.user_id].append(idx)
            if event.session_id:
                self._by_session[event.session_id].append(idx)
            if event.event_type:
                self._by_type[event.event_type].append(idx)

