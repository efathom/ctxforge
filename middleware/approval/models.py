"""
Approval data models.

Shared models used by both middleware and store to avoid circular imports.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"  # User approved with changes
    EXPIRED = "expired"


class ApprovalRequest(BaseModel):
    """
    A pending knowledge save request awaiting user approval.

    The agent asks: "Does this answer look correct, or would you like me
    to save this to the knowledge base?"
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str

    # What we're proposing to save
    knowledge_type: str  # "expertise_item", "memory", "query_pattern", etc.
    proposed_content: str
    proposed_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Context for user decision
    source_question: Optional[str] = None
    source_answer: Optional[str] = None
    reasoning: str = ""

    # State
    status: ApprovalStatus = ApprovalStatus.PENDING
    modified_content: Optional[str] = None  # User's edited version
    rejection_reason: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
