"""
Tests for Human Approval Middleware.

Tests the approval workflow for knowledge persistence.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.middleware.approval import (
    ApprovalRequest,
    ApprovalStatus,
    HumanApprovalMiddleware,
    InMemoryApprovalStore,
)
from ctxforge.middleware.base import StopChainException
from ctxforge.middleware.protocol import MiddlewareContext


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""
    
    def test_status_values(self):
        """Test all status values exist."""
        assert ApprovalStatus.PENDING == "pending"
        assert ApprovalStatus.APPROVED == "approved"
        assert ApprovalStatus.REJECTED == "rejected"
        assert ApprovalStatus.MODIFIED == "modified"
        assert ApprovalStatus.EXPIRED == "expired"


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""
    
    def test_create_request(self):
        """Test creating an approval request."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Always validate input before processing",
        )
        
        assert request.request_id is not None
        assert request.session_id == "session-123"
        assert request.user_id == "user-123"
        assert request.knowledge_type == "expertise_item"
        assert request.status == ApprovalStatus.PENDING
        assert request.created_at is not None
    
    def test_request_with_metadata(self):
        """Test request with additional metadata."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="validated_query",
            proposed_content="SELECT * FROM users",
            proposed_metadata={"source": "sql_agent"},
            source_question="How do I get all users?",
            source_answer="Here's the query...",
            reasoning="This pattern is commonly used",
        )
        
        assert request.proposed_metadata["source"] == "sql_agent"
        assert request.source_question == "How do I get all users?"
        assert request.reasoning == "This pattern is commonly used"


class TestInMemoryApprovalStore:
    """Tests for InMemoryApprovalStore."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemoryApprovalStore()
    
    @pytest.mark.asyncio
    async def test_save_and_get_request(self, store):
        """Test saving and retrieving a request."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Test content",
        )
        
        await store.save_request(request)
        retrieved = await store.get_request(request.request_id)
        
        assert retrieved is not None
        assert retrieved.request_id == request.request_id
        assert retrieved.proposed_content == "Test content"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_request(self, store):
        """Test retrieving a non-existent request."""
        retrieved = await store.get_request("nonexistent")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_get_pending_for_user(self, store):
        """Test getting pending requests for a user."""
        # Create multiple requests
        for i in range(3):
            request = ApprovalRequest(
                session_id=f"session-{i}",
                user_id="user-123",
                knowledge_type="expertise_item",
                proposed_content=f"Content {i}",
            )
            await store.save_request(request)
        
        # Create one for different user
        other_request = ApprovalRequest(
            session_id="session-other",
            user_id="user-456",
            knowledge_type="expertise_item",
            proposed_content="Other content",
        )
        await store.save_request(other_request)
        
        pending = await store.get_pending_for_user("user-123")
        assert len(pending) == 3
        assert all(r.user_id == "user-123" for r in pending)
    
    @pytest.mark.asyncio
    async def test_get_pending_for_session(self, store):
        """Test getting pending requests for a session."""
        # Create requests for same session
        for i in range(2):
            request = ApprovalRequest(
                session_id="session-123",
                user_id=f"user-{i}",
                knowledge_type="expertise_item",
                proposed_content=f"Content {i}",
            )
            await store.save_request(request)
        
        pending = await store.get_pending_for_session("session-123")
        assert len(pending) == 2
    
    @pytest.mark.asyncio
    async def test_update_status_approved(self, store):
        """Test updating request status to approved."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Test content",
        )
        await store.save_request(request)
        
        updated = await store.update_status(
            request.request_id,
            ApprovalStatus.APPROVED,
        )
        
        assert updated is not None
        assert updated.status == ApprovalStatus.APPROVED
        assert updated.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_update_status_rejected(self, store):
        """Test updating request status to rejected with reason."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Test content",
        )
        await store.save_request(request)
        
        updated = await store.update_status(
            request.request_id,
            ApprovalStatus.REJECTED,
            rejection_reason="Not accurate enough",
        )
        
        assert updated is not None
        assert updated.status == ApprovalStatus.REJECTED
        assert updated.rejection_reason == "Not accurate enough"
    
    @pytest.mark.asyncio
    async def test_update_status_modified(self, store):
        """Test updating request status to modified."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Original content",
        )
        await store.save_request(request)
        
        updated = await store.update_status(
            request.request_id,
            ApprovalStatus.MODIFIED,
            modified_content="Improved content",
        )
        
        assert updated is not None
        assert updated.status == ApprovalStatus.MODIFIED
        assert updated.modified_content == "Improved content"
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_request(self, store):
        """Test updating a non-existent request."""
        updated = await store.update_status(
            "nonexistent",
            ApprovalStatus.APPROVED,
        )
        assert updated is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, store):
        """Test cleaning up expired requests."""
        # Create expired request
        expired_request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Expired content",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await store.save_request(expired_request)
        
        # Create non-expired request
        valid_request = ApprovalRequest(
            session_id="session-456",
            user_id="user-456",
            knowledge_type="expertise_item",
            proposed_content="Valid content",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        await store.save_request(valid_request)
        
        expired_count = await store.cleanup_expired()
        
        assert expired_count == 1
        
        # Check expired request is marked
        expired = await store.get_request(expired_request.request_id)
        assert expired.status == ApprovalStatus.EXPIRED
        
        # Check valid request is still pending
        valid = await store.get_request(valid_request.request_id)
        assert valid.status == ApprovalStatus.PENDING


class TestHumanApprovalMiddleware:
    """Tests for HumanApprovalMiddleware."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemoryApprovalStore()
    
    @pytest.fixture
    def middleware(self, store):
        """Create middleware with default settings."""
        return HumanApprovalMiddleware(
            approval_store=store,
            expiry_hours=24,
        )
    
    @pytest.fixture
    def context_with_pending_save(self):
        """Create a context with a pending knowledge save."""
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="How do I validate input?",
        )
        context.set_metadata("pending_knowledge_save", {
            "knowledge_type": "expertise_item",
            "content": "Always validate user input before processing",
            "metadata": {"source": "agent"},
            "reasoning": "This is a common best practice",
        })
        return context
    
    @pytest.fixture
    def context_without_pending_save(self):
        """Create a context without a pending save."""
        return MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="Hello",
        )
    
    @pytest.mark.asyncio
    async def test_middleware_name(self, middleware):
        """Test middleware name property."""
        assert middleware.name == "human_approval"
    
    @pytest.mark.asyncio
    async def test_process_without_pending_save(
        self, middleware, context_without_pending_save
    ):
        """Test processing when no pending save exists."""
        next_called = False
        
        async def mock_next(ctx):
            nonlocal next_called
            next_called = True
            return ctx
        
        result = await middleware.process(context_without_pending_save, mock_next)
        
        assert next_called
        assert result.get_metadata("approval_required") is None
    
    @pytest.mark.asyncio
    async def test_process_with_pending_save(
        self, middleware, context_with_pending_save, store
    ):
        """Test processing with a pending knowledge save."""
        next_called = False
        
        async def mock_next(ctx):
            nonlocal next_called
            next_called = True
            return ctx
        
        result = await middleware.process(context_with_pending_save, mock_next)
        
        assert next_called
        assert result.get_metadata("approval_required") is True
        assert result.get_metadata("approval_request_id") is not None
        assert result.get_metadata("approval_prompt") is not None
        
        # Verify request was stored
        request_id = result.get_metadata("approval_request_id")
        stored_request = await store.get_request(request_id)
        assert stored_request is not None
        assert stored_request.proposed_content == "Always validate user input before processing"
    
    @pytest.mark.asyncio
    async def test_process_type_not_requiring_approval(
        self, middleware, store
    ):
        """Test processing a type that doesn't require approval."""
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="Test",
        )
        context.set_metadata("pending_knowledge_save", {
            "knowledge_type": "semantic_memory",  # Not in default list
            "content": "Some fact",
        })
        
        next_called = False
        
        async def mock_next(ctx):
            nonlocal next_called
            next_called = True
            return ctx
        
        result = await middleware.process(context, mock_next)
        
        assert next_called
        assert result.get_metadata("approval_required") is None
    
    @pytest.mark.asyncio
    async def test_stop_on_pending(self, store):
        """Test that stop_on_pending raises StopChainException."""
        middleware = HumanApprovalMiddleware(
            approval_store=store,
            stop_on_pending=True,
        )
        
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="Test",
        )
        context.set_metadata("pending_knowledge_save", {
            "knowledge_type": "expertise_item",
            "content": "Test content",
        })
        
        async def mock_next(ctx):
            return ctx
        
        with pytest.raises(StopChainException):
            await middleware.process(context, mock_next)
    
    @pytest.mark.asyncio
    async def test_custom_knowledge_types(self, store):
        """Test with custom knowledge types requiring approval."""
        middleware = HumanApprovalMiddleware(
            approval_store=store,
            knowledge_types_requiring_approval=["custom_type"],
        )
        
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="Test",
        )
        context.set_metadata("pending_knowledge_save", {
            "knowledge_type": "custom_type",
            "content": "Custom content",
        })
        
        async def mock_next(ctx):
            return ctx
        
        result = await middleware.process(context, mock_next)
        
        assert result.get_metadata("approval_required") is True
    
    @pytest.mark.asyncio
    async def test_disabled_middleware(self, store):
        """Test that disabled middleware passes through."""
        middleware = HumanApprovalMiddleware(
            approval_store=store,
            enabled=False,
        )
        
        context = MiddlewareContext(
            session_id="session-123",
            user_id="user-123",
            user_input="Test",
        )
        context.set_metadata("pending_knowledge_save", {
            "knowledge_type": "expertise_item",
            "content": "Content",
        })
        
        next_called = False
        
        async def mock_next(ctx):
            nonlocal next_called
            next_called = True
            return ctx
        
        await middleware.process(context, mock_next)
        
        assert next_called
    
    @pytest.mark.asyncio
    async def test_expiry_set_on_request(self, middleware, context_with_pending_save, store):
        """Test that expiry is set on the approval request."""
        async def mock_next(ctx):
            return ctx
        
        result = await middleware.process(context_with_pending_save, mock_next)
        
        request_id = result.get_metadata("approval_request_id")
        stored_request = await store.get_request(request_id)
        
        assert stored_request.expires_at is not None
        # Should be approximately 24 hours from now
        expected_expiry = stored_request.created_at + timedelta(hours=24)
        assert abs((stored_request.expires_at - expected_expiry).total_seconds()) < 1


class TestApprovalPromptGeneration:
    """Tests for approval prompt generation."""
    
    @pytest.fixture
    def store(self):
        return InMemoryApprovalStore()
    
    @pytest.fixture
    def middleware(self, store):
        return HumanApprovalMiddleware(approval_store=store)
    
    def test_prompt_for_expertise_item(self, middleware):
        """Test prompt generation for expertise item."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="expertise_item",
            proposed_content="Always use parameterized queries",
        )
        
        prompt = middleware._generate_approval_prompt(request)
        
        assert "insight" in prompt.lower()
        assert "Always use parameterized queries" in prompt
        assert "yes" in prompt.lower()
        assert "no" in prompt.lower()
    
    def test_prompt_for_validated_query(self, middleware):
        """Test prompt generation for validated query."""
        request = ApprovalRequest(
            session_id="session-123",
            user_id="user-123",
            knowledge_type="validated_query",
            proposed_content="SELECT * FROM users WHERE id = ?",
        )
        
        prompt = middleware._generate_approval_prompt(request)
        
        assert "query pattern" in prompt.lower()
        assert "SELECT * FROM users" in prompt
