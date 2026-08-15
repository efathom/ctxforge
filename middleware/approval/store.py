"""
Approval store implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ctxforge.middleware.approval.models import ApprovalRequest, ApprovalStatus


class ApprovalStore(ABC):
    """
    Abstract store for approval requests.

    Implementations should persist pending requests and support
    retrieval by user/session for async approval workflows.
    """

    @abstractmethod
    async def save_request(self, request: ApprovalRequest) -> None:
        """Save an approval request."""
        ...

    @abstractmethod
    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID."""
        ...

    @abstractmethod
    async def get_pending_for_user(self, user_id: str) -> List[ApprovalRequest]:
        """Get all pending requests for a user."""
        ...

    @abstractmethod
    async def get_pending_for_session(
        self, session_id: str
    ) -> List[ApprovalRequest]:
        """Get all pending requests for a session."""
        ...

    @abstractmethod
    async def update_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        modified_content: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> Optional[ApprovalRequest]:
        """Update the status of an approval request."""
        ...

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired requests. Returns count removed."""
        ...


class InMemoryApprovalStore(ApprovalStore):
    """In-memory implementation for development/testing."""

    def __init__(self):
        self._requests: Dict[str, ApprovalRequest] = {}

    async def save_request(self, request: ApprovalRequest) -> None:
        self._requests[request.request_id] = request

    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        return self._requests.get(request_id)

    async def get_pending_for_user(self, user_id: str) -> List[ApprovalRequest]:
        return [
            r for r in self._requests.values()
            if r.user_id == user_id and r.status == ApprovalStatus.PENDING
        ]

    async def get_pending_for_session(
        self, session_id: str
    ) -> List[ApprovalRequest]:
        return [
            r for r in self._requests.values()
            if r.session_id == session_id and r.status == ApprovalStatus.PENDING
        ]

    async def update_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        modified_content: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> Optional[ApprovalRequest]:
        request = self._requests.get(request_id)
        if not request:
            return None

        request.status = status
        request.resolved_at = datetime.now(timezone.utc)
        if modified_content:
            request.modified_content = modified_content
        if rejection_reason:
            request.rejection_reason = rejection_reason

        return request

    async def cleanup_expired(self) -> int:
        now = datetime.now(timezone.utc)
        expired = [
            rid for rid, r in self._requests.items()
            if r.expires_at and r.expires_at < now
            and r.status == ApprovalStatus.PENDING
        ]
        for rid in expired:
            self._requests[rid].status = ApprovalStatus.EXPIRED
        return len(expired)

    async def clear(self) -> None:
        """Clear all requests (for testing)."""
        self._requests.clear()
