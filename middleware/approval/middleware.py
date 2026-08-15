"""
Human Approval Middleware.

Intercepts knowledge save operations and queues them for user approval.
"""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

from ctxforge.middleware.approval.models import ApprovalRequest
from ctxforge.middleware.approval.store import ApprovalStore
from ctxforge.middleware.base import BaseMiddleware, StopChainException
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction


class HumanApprovalMiddleware(BaseMiddleware):
    """
    Middleware that intercepts knowledge save operations for approval.

    When enabled, instead of directly saving knowledge, this middleware:
    1. Creates an ApprovalRequest with the proposed knowledge
    2. Stores it in the approval queue
    3. Marks the context with approval_required=True
    4. Optionally stops the chain or continues with metadata

    The calling agent can then prompt the user:
    "I found a useful pattern. Would you like me to save it for future use?"

    Example usage in system prompt:
        After completing the task, if successful, ask:
        "Would you like me to save this [query/insight/pattern] to the knowledge base?"
        Only call save_validated_knowledge if the user explicitly agrees.
    """

    def __init__(
        self,
        approval_store: ApprovalStore,
        knowledge_types_requiring_approval: Optional[List[str]] = None,
        stop_on_pending: bool = False,
        expiry_hours: int = 24,
        enabled: bool = True,
    ):
        """
        Initialize the approval middleware.

        Args:
            approval_store: Store for pending approval requests
            knowledge_types_requiring_approval: Which types need approval
                (None = all types)
            stop_on_pending: If True, stop the chain when approval is needed
            expiry_hours: How long approval requests remain valid
            enabled: Whether this middleware is active
        """
        super().__init__(enabled=enabled)
        self._store = approval_store
        self._types = knowledge_types_requiring_approval or [
            "expertise_item",
            "validated_query",
            "procedural_memory",
        ]
        self._stop_on_pending = stop_on_pending
        self._expiry_hours = expiry_hours

    @property
    def name(self) -> str:
        return "human_approval"

    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """Process knowledge save requests for approval."""

        # Check if this context has a pending save operation
        pending_save = context.get_metadata("pending_knowledge_save")
        if not pending_save:
            return await next(context)

        knowledge_type = pending_save.get("knowledge_type", "unknown")

        # Check if this type requires approval
        if knowledge_type not in self._types:
            # Type doesn't require approval, continue
            return await next(context)

        # Create approval request
        request = ApprovalRequest(
            session_id=context.session_id or "",
            user_id=context.user_id or "",
            knowledge_type=knowledge_type,
            proposed_content=pending_save.get("content", ""),
            proposed_metadata=pending_save.get("metadata", {}),
            source_question=context.user_input,
            source_answer=context.get_metadata("assistant_response"),
            reasoning=pending_save.get("reasoning", ""),
        )

        # Set expiry
        if self._expiry_hours > 0:
            request.expires_at = request.created_at + timedelta(
                hours=self._expiry_hours
            )

        # Store the request
        await self._store.save_request(request)

        # Mark context with approval info
        context.set_metadata("approval_required", True)
        context.set_metadata("approval_request_id", request.request_id)
        context.set_metadata("approval_prompt", self._generate_approval_prompt(request))

        if self._stop_on_pending:
            raise StopChainException(
                self.name,
                f"Awaiting user approval for {knowledge_type} save"
            )

        return await next(context)

    def _generate_approval_prompt(self, request: ApprovalRequest) -> str:
        """Generate a prompt asking for user approval."""
        type_labels = {
            "expertise_item": "insight",
            "validated_query": "query pattern",
            "procedural_memory": "procedure",
            "semantic_memory": "fact",
        }
        label = type_labels.get(request.knowledge_type, "knowledge")

        return (
            f"I found a useful {label}. Would you like me to save it to the "
            f"knowledge base for future reference?\n\n"
            f"**Proposed {label}:**\n{request.proposed_content}\n\n"
            f"Reply 'yes' to save, 'no' to skip, or provide an edited version."
        )
