"""
Comprehensive tests for the middleware module.

Tests all middleware components including:
- Protocol and base classes
- PII detection and redaction
- Rate limiting
- Audit logging
- Content filtering
- Middleware chain
"""

import asyncio

import pytest

from ctxforge.middleware import (
    AuditEvent,
    # Audit
    AuditMiddleware,
    BaseMiddleware,
    # Content Filtering
    ContentFilterMiddleware,
    # Core
    InMemoryAuditStore,
    KeywordFilter,
    MiddlewareChain,
    MiddlewareContext,
    PIIDetector,
    PIIMiddleware,
    PIIRedactor,
    PIIType,
    # Rate Limiting
    RateLimitMiddleware,
    SlidingWindowLimiter,
    TokenBucketLimiter,
)
from ctxforge.middleware.base import StopChainException
from ctxforge.middleware.content.filters import (
    CompositeFilter,
    FilterAction,
    FilterResult,
    RegexFilter,
)
from ctxforge.middleware.pii.detector import PIIMatch
from ctxforge.middleware.pii.redactor import RedactionStrategy
from ctxforge.middleware.ratelimit.limiter import (
    InMemoryRateLimiter,
)

# ============================================================================
# Test Middleware Context
# ============================================================================

class TestMiddlewareContext:
    """Tests for MiddlewareContext."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        ctx = MiddlewareContext(user_input="Hello")
        
        assert ctx.user_input == "Hello"
        assert ctx.processed_input == "Hello"  # Defaults to input
        assert ctx.agent_response is None
        assert ctx.processed_response is None
        assert len(ctx.flags) == 0
        assert len(ctx.metadata) == 0
    
    def test_init_with_response(self):
        """Test initialization with response."""
        ctx = MiddlewareContext(
            user_input="Hello",
            agent_response="Hi there!",
        )
        
        assert ctx.processed_response == "Hi there!"
    
    def test_add_flag(self):
        """Test adding flags."""
        ctx = MiddlewareContext(user_input="test")
        
        ctx.add_flag("flag1")
        ctx.add_flag("flag2")
        
        assert ctx.has_flag("flag1")
        assert ctx.has_flag("flag2")
        assert not ctx.has_flag("flag3")
    
    def test_set_get_metadata(self):
        """Test metadata operations."""
        ctx = MiddlewareContext(user_input="test")
        
        ctx.set_metadata("key1", "value1")
        ctx.set_metadata("key2", 123)
        
        assert ctx.get_metadata("key1") == "value1"
        assert ctx.get_metadata("key2") == 123
        assert ctx.get_metadata("missing") is None
        assert ctx.get_metadata("missing", "default") == "default"
    
    def test_record_modification(self):
        """Test recording modifications."""
        ctx = MiddlewareContext(user_input="test")
        
        ctx.record_modification("middleware1", {"action": "redact"})
        ctx.record_modification("middleware1", {"action": "flag"})
        ctx.record_modification("middleware2", {"action": "log"})
        
        assert len(ctx.modifications["middleware1"]) == 2
        assert len(ctx.modifications["middleware2"]) == 1


# ============================================================================
# Test PII Detector
# ============================================================================

class TestPIIDetector:
    """Tests for PIIDetector."""
    
    def test_detect_email(self):
        """Test email detection."""
        detector = PIIDetector()
        
        matches = detector.detect("Contact me at john.doe@example.com")
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "john.doe@example.com"
    
    def test_detect_phone_us(self):
        """Test US phone number detection."""
        detector = PIIDetector()
        
        test_cases = [
            "Call me at 555-123-4567",
            "Call me at (555) 123-4567",
            "Call me at 555.123.4567",
            "Call me at +1 555 123 4567",
        ]
        
        for text in test_cases:
            matches = detector.detect(text)
            assert len(matches) >= 1, f"Failed for: {text}"
            assert any(m.pii_type == PIIType.PHONE for m in matches)
    
    def test_detect_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector()
        
        matches = detector.detect("My SSN is 123-45-6789")
        
        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.SSN for m in matches)
    
    def test_detect_ip_address(self):
        """Test IP address detection."""
        detector = PIIDetector()
        
        matches = detector.detect("Server at 192.168.1.100")
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.IP_ADDRESS
        assert matches[0].value == "192.168.1.100"
    
    def test_detect_multiple(self):
        """Test detection of multiple PII types."""
        detector = PIIDetector()
        
        text = "Email: test@example.com, Phone: 555-123-4567, IP: 10.0.0.1"
        matches = detector.detect(text)
        
        types = {m.pii_type for m in matches}
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types
        assert PIIType.IP_ADDRESS in types
    
    def test_contains_pii(self):
        """Test contains_pii method."""
        detector = PIIDetector()
        
        assert detector.contains_pii("Email: test@example.com")
        assert not detector.contains_pii("No PII here")
    
    def test_detect_types(self):
        """Test detect_types method."""
        detector = PIIDetector()
        
        types = detector.detect_types("test@example.com and 555-123-4567")
        
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types
    
    def test_add_custom_pattern(self):
        """Test adding custom patterns."""
        detector = PIIDetector()
        detector.add_pattern(PIIType.CUSTOM, r'EMPLOYEE-\d{6}')
        
        matches = detector.detect("ID: EMPLOYEE-123456")
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.CUSTOM
    
    def test_disable_type(self):
        """Test disabling PII types."""
        detector = PIIDetector()
        detector.disable_type(PIIType.EMAIL)
        
        matches = detector.detect("Email: test@example.com")
        
        assert len(matches) == 0
    
    def test_empty_text(self):
        """Test with empty text."""
        detector = PIIDetector()
        
        assert detector.detect("") == []
        assert detector.detect(None) == []


class TestPIIDetectorCreditCard:
    """Tests for credit card detection with Luhn validation."""
    
    def test_valid_credit_card(self):
        """Test valid credit card detection."""
        detector = PIIDetector()
        
        # Valid Visa test number
        matches = detector.detect("Card: 4111111111111111")
        
        assert len(matches) >= 1
        cc_match = next((m for m in matches if m.pii_type == PIIType.CREDIT_CARD), None)
        assert cc_match is not None
        assert cc_match.confidence > 0.9  # High confidence for valid Luhn
    
    def test_credit_card_with_separators(self):
        """Test credit card with separators."""
        detector = PIIDetector()
        
        matches = detector.detect("Card: 4111-1111-1111-1111")
        
        # Should detect the pattern
        assert len(matches) >= 1


# ============================================================================
# Test PII Redactor
# ============================================================================

class TestPIIRedactor:
    """Tests for PIIRedactor."""
    
    def test_redact_replace(self):
        """Test REPLACE strategy."""
        detector = PIIDetector()
        redactor = PIIRedactor(strategy=RedactionStrategy.REPLACE)
        
        text = "Email: test@example.com"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)
        
        assert "[EMAIL]" in result
        assert "test@example.com" not in result
    
    def test_redact_mask(self):
        """Test MASK strategy."""
        detector = PIIDetector()
        redactor = PIIRedactor(strategy=RedactionStrategy.MASK)
        
        text = "Email: test@example.com"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)
        
        assert "@" in result  # Preserves structure
        assert "test" not in result
    
    def test_redact_partial(self):
        """Test PARTIAL strategy."""
        detector = PIIDetector()
        redactor = PIIRedactor(strategy=RedactionStrategy.PARTIAL)
        
        text = "Email: john@example.com"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)
        
        # Should show first char of local and domain
        assert "j" in result
        assert "example.com" not in result
    
    def test_redact_remove(self):
        """Test REMOVE strategy."""
        detector = PIIDetector()
        redactor = PIIRedactor(strategy=RedactionStrategy.REMOVE)
        
        text = "Email: test@example.com here"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)
        
        assert "test@example.com" not in result
        assert "Email:  here" == result
    
    def test_redact_hash(self):
        """Test HASH strategy."""
        detector = PIIDetector()
        redactor = PIIRedactor(strategy=RedactionStrategy.HASH)
        
        text = "Email: test@example.com"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)
        
        assert "test@example.com" not in result
        # Should be 8-char hash
        assert len(result.replace("Email: ", "")) == 8
    
    def test_custom_placeholders(self):
        """Test custom placeholders."""
        redactor = PIIRedactor(
            strategy=RedactionStrategy.REPLACE,
            placeholders={
                PIIType.EMAIL: "<<REDACTED_EMAIL>>",
            }
        )
        
        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            value="test@example.com",
            start=0,
            end=16,
        )
        
        result = redactor.redact("test@example.com", [match])
        assert result == "<<REDACTED_EMAIL>>"
    
    def test_multiple_matches(self):
        """Test redacting multiple matches."""
        detector = PIIDetector()
        redactor = PIIRedactor(strategy=RedactionStrategy.REPLACE)
        
        text = "Contact: test@example.com or other@example.org"
        matches = detector.detect(text)
        result = redactor.redact(text, matches)
        
        assert "[EMAIL]" in result
        assert "test@example.com" not in result
        assert "other@example.org" not in result


# ============================================================================
# Test PII Middleware
# ============================================================================

class TestPIIMiddleware:
    """Tests for PIIMiddleware."""
    
    @pytest.mark.asyncio
    async def test_detect_and_redact(self):
        """Test detection and redaction."""
        middleware = PIIMiddleware(redact=True)
        
        context = MiddlewareContext(
            user_input="My email is test@example.com",
        )
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert result.has_flag("pii_detected")
        assert result.has_flag("pii_detected_in_input")
        assert "[EMAIL]" in result.processed_input
    
    @pytest.mark.asyncio
    async def test_detect_only(self):
        """Test detection without redaction."""
        middleware = PIIMiddleware(redact=False)
        
        context = MiddlewareContext(
            user_input="My email is test@example.com",
        )
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert result.has_flag("pii_detected")
        assert "test@example.com" in result.processed_input  # Not redacted
    
    @pytest.mark.asyncio
    async def test_stop_on_pii(self):
        """Test stopping chain on PII detection."""
        middleware = PIIMiddleware(stop_on_pii=True)
        
        context = MiddlewareContext(
            user_input="My email is test@example.com",
        )
        
        async def next_fn(ctx):
            return ctx
        
        with pytest.raises(StopChainException) as exc_info:
            await middleware.process(context, next_fn)
        
        assert exc_info.value.middleware_name == "pii"
    
    @pytest.mark.asyncio
    async def test_no_pii(self):
        """Test when no PII is present."""
        middleware = PIIMiddleware()
        
        context = MiddlewareContext(
            user_input="Hello, how are you?",
        )
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert not result.has_flag("pii_detected")
    
    @pytest.mark.asyncio
    async def test_redact_response(self):
        """Test redacting agent response."""
        middleware = PIIMiddleware(redact=True, redact_response=True)
        
        context = MiddlewareContext(
            user_input="What's your email?",
            agent_response="You can reach me at agent@example.com",
        )
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert result.has_flag("pii_detected_in_response")
        assert "[EMAIL]" in result.processed_response


# ============================================================================
# Test Rate Limiters
# ============================================================================

class TestTokenBucketLimiter:
    """Tests for TokenBucketLimiter."""
    
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        """Test that requests under limit are allowed."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)
        
        result = await limiter.consume("user1")
        
        assert result.allowed
        assert result.remaining >= 8  # Allow for small timing variations
    
    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=5, initial_tokens=5)
        
        # Exhaust tokens
        for _ in range(5):
            await limiter.consume("user1")
        
        result = await limiter.consume("user1")
        
        assert not result.allowed
        assert result.remaining == 0
        assert result.retry_after > 0
    
    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketLimiter(rate=100.0, capacity=10, initial_tokens=0)
        
        # Initially empty
        result = await limiter.check("user1")
        assert result.remaining == 0
        
        # Wait for refill
        await asyncio.sleep(0.05)  # 50ms = 5 tokens at 100/s
        
        result = await limiter.check("user1")
        assert result.remaining >= 4  # Allow for timing variance
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting a bucket."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10, initial_tokens=0)
        
        await limiter.reset("user1")
        
        result = await limiter.check("user1")
        assert result.remaining == 10
    
    @pytest.mark.asyncio
    async def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens at once."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=10)
        
        result = await limiter.consume("user1", tokens=5)
        
        assert result.allowed
        assert result.remaining >= 4  # Allow for small timing variations


class TestSlidingWindowLimiter:
    """Tests for SlidingWindowLimiter."""
    
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        """Test that requests under limit are allowed."""
        limiter = SlidingWindowLimiter(limit=10, window_seconds=60.0)
        
        result = await limiter.consume("user1")
        
        assert result.allowed
        assert result.remaining == 9
    
    @pytest.mark.asyncio
    async def test_blocks_at_limit(self):
        """Test that requests at limit are blocked."""
        limiter = SlidingWindowLimiter(limit=5, window_seconds=60.0)
        
        # Hit limit
        for _ in range(5):
            await limiter.consume("user1")
        
        result = await limiter.consume("user1")
        
        assert not result.allowed
        assert result.remaining == 0
    
    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Test that old requests expire."""
        limiter = SlidingWindowLimiter(limit=2, window_seconds=0.1)  # 100ms window
        
        # Fill limit
        await limiter.consume("user1")
        await limiter.consume("user1")
        
        result = await limiter.consume("user1")
        assert not result.allowed
        
        # Wait for window to expire
        await asyncio.sleep(0.15)
        
        result = await limiter.consume("user1")
        assert result.allowed


class TestInMemoryRateLimiter:
    """Tests for InMemoryRateLimiter."""
    
    @pytest.mark.asyncio
    async def test_basic_limiting(self):
        """Test basic rate limiting."""
        limiter = InMemoryRateLimiter(limit=5, window_seconds=60.0)
        
        for i in range(5):
            result = await limiter.consume("user1")
            assert result.allowed
            assert result.remaining == 4 - i
        
        result = await limiter.consume("user1")
        assert not result.allowed


# ============================================================================
# Test Rate Limit Middleware
# ============================================================================

class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""
    
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        """Test requests under limit pass through."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)
        middleware = RateLimitMiddleware(limiter=limiter)
        
        context = MiddlewareContext(user_input="test", user_id="user1")
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert not result.has_flag("rate_limited")
        assert result.get_metadata("rate_limit_remaining") is not None
    
    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Test requests over limit are blocked."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=1, initial_tokens=0)
        middleware = RateLimitMiddleware(limiter=limiter, stop_on_limit=True)
        
        context = MiddlewareContext(user_input="test", user_id="user1")
        
        async def next_fn(ctx):
            return ctx
        
        with pytest.raises(StopChainException) as exc_info:
            await middleware.process(context, next_fn)
        
        assert exc_info.value.middleware_name == "rate_limit"
    
    @pytest.mark.asyncio
    async def test_custom_key_func(self):
        """Test custom key extraction."""
        limiter = TokenBucketLimiter(rate=10.0, capacity=100)
        
        def custom_key(ctx):
            return f"custom:{ctx.session_id}"
        
        middleware = RateLimitMiddleware(limiter=limiter, key_func=custom_key)
        
        context = MiddlewareContext(
            user_input="test",
            session_id="sess123",
        )
        
        async def next_fn(ctx):
            return ctx
        
        # Should use custom key function
        await middleware.process(context, next_fn)
        
        # Verify by checking the limiter was called with custom key
        result = await limiter.check("custom:sess123")
        assert result.remaining < 100  # Token was consumed


# ============================================================================
# Test Audit Store
# ============================================================================

class TestInMemoryAuditStore:
    """Tests for InMemoryAuditStore."""
    
    @pytest.mark.asyncio
    async def test_log_and_query(self):
        """Test logging and querying events."""
        store = InMemoryAuditStore()
        
        event = AuditEvent(
            user_id="user1",
            event_type="request",
            action="process",
        )
        
        await store.log(event)
        
        events = await store.query(user_id="user1")
        
        assert len(events) == 1
        assert events[0].user_id == "user1"
    
    @pytest.mark.asyncio
    async def test_query_by_session(self):
        """Test querying by session."""
        store = InMemoryAuditStore()
        
        await store.log(AuditEvent(session_id="sess1", event_type="a"))
        await store.log(AuditEvent(session_id="sess2", event_type="b"))
        await store.log(AuditEvent(session_id="sess1", event_type="c"))
        
        events = await store.query(session_id="sess1")
        
        assert len(events) == 2
    
    @pytest.mark.asyncio
    async def test_query_by_type(self):
        """Test querying by event type."""
        store = InMemoryAuditStore()
        
        await store.log(AuditEvent(event_type="request"))
        await store.log(AuditEvent(event_type="error"))
        await store.log(AuditEvent(event_type="request"))
        
        events = await store.query(event_type="request")
        
        assert len(events) == 2
    
    @pytest.mark.asyncio
    async def test_query_limit(self):
        """Test query limit."""
        store = InMemoryAuditStore()
        
        for _i in range(10):
            await store.log(AuditEvent(event_type="test"))
        
        events = await store.query(limit=5)
        
        assert len(events) == 5
    
    @pytest.mark.asyncio
    async def test_max_events_eviction(self):
        """Test that old events are evicted."""
        store = InMemoryAuditStore(max_events=10)
        
        for i in range(15):
            await store.log(AuditEvent(event_type=f"event_{i}"))
        
        events = await store.query(limit=100)
        
        # Should have evicted some
        assert len(events) <= 10
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the store."""
        store = InMemoryAuditStore()
        
        await store.log(AuditEvent(event_type="test"))
        await store.clear()
        
        events = await store.query()
        assert len(events) == 0


class TestAuditEvent:
    """Tests for AuditEvent."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = AuditEvent(
            event_id="123",
            user_id="user1",
            event_type="request",
            action="process",
            success=True,
        )
        
        d = event.to_dict()
        
        assert d["event_id"] == "123"
        assert d["user_id"] == "user1"
        assert d["success"] is True


# ============================================================================
# Test Audit Middleware
# ============================================================================

class TestAuditMiddleware:
    """Tests for AuditMiddleware."""
    
    @pytest.mark.asyncio
    async def test_logs_request(self):
        """Test that requests are logged."""
        store = InMemoryAuditStore()
        middleware = AuditMiddleware(store=store)
        
        context = MiddlewareContext(
            user_input="Hello",
            user_id="user1",
        )
        
        async def next_fn(ctx):
            return ctx
        
        await middleware.process(context, next_fn)
        
        events = await store.query(user_id="user1")
        
        assert len(events) == 1
        assert events[0].event_type == "request"
    
    @pytest.mark.asyncio
    async def test_logs_duration(self):
        """Test that duration is recorded."""
        store = InMemoryAuditStore()
        middleware = AuditMiddleware(store=store)
        
        context = MiddlewareContext(user_input="Hello")
        
        async def next_fn(ctx):
            await asyncio.sleep(0.01)
            return ctx
        
        await middleware.process(context, next_fn)
        
        events = await store.query()
        
        assert events[0].duration_ms >= 10  # At least 10ms
    
    @pytest.mark.asyncio
    async def test_redacts_pii_if_flagged(self):
        """Test that PII is redacted in logs if flagged."""
        store = InMemoryAuditStore()
        middleware = AuditMiddleware(store=store, redact_pii=True)
        
        context = MiddlewareContext(
            user_input="test@example.com",
        )
        context.add_flag("pii_detected_in_input")
        context.processed_input = "[EMAIL]"
        
        async def next_fn(ctx):
            return ctx
        
        await middleware.process(context, next_fn)
        
        events = await store.query()
        
        # Should use redacted version
        assert events[0].details["input"] == "[EMAIL]"
    
    @pytest.mark.asyncio
    async def test_custom_handler(self):
        """Test custom event handler."""
        store = InMemoryAuditStore()
        handler_calls = []
        
        def custom_handler(event):
            handler_calls.append(event)
        
        middleware = AuditMiddleware(
            store=store,
            custom_handler=custom_handler,
        )
        
        context = MiddlewareContext(user_input="Hello")
        
        async def next_fn(ctx):
            return ctx
        
        await middleware.process(context, next_fn)
        
        assert len(handler_calls) == 1


# ============================================================================
# Test Content Filters
# ============================================================================

class TestKeywordFilter:
    """Tests for KeywordFilter."""
    
    def test_blocks_keyword(self):
        """Test blocking a keyword."""
        f = KeywordFilter()
        f.add_keywords("profanity", ["badword"], action=FilterAction.BLOCK)
        
        result = f.filter("This contains badword")
        
        assert result.matched
        assert result.action == FilterAction.BLOCK
        assert "profanity" in result.categories
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        f = KeywordFilter(case_sensitive=False)
        f.add_keywords("test", ["keyword"])
        
        result = f.filter("Contains KEYWORD here")
        
        assert result.matched
    
    def test_case_sensitive(self):
        """Test case-sensitive matching."""
        f = KeywordFilter(case_sensitive=True)
        f.add_keywords("test", ["keyword"])
        
        result = f.filter("Contains KEYWORD here")
        
        assert not result.matched
    
    def test_multiple_categories(self):
        """Test multiple category matching."""
        f = KeywordFilter()
        f.add_keywords("category1", ["word1"])
        f.add_keywords("category2", ["word2"])
        
        result = f.filter("Contains word1 and word2")
        
        assert "category1" in result.categories
        assert "category2" in result.categories
    
    def test_no_match(self):
        """Test when no keywords match."""
        f = KeywordFilter()
        f.add_keywords("test", ["blocked"])
        
        result = f.filter("This is clean content")
        
        assert not result.matched
        assert result.action == FilterAction.ALLOW
    
    def test_word_boundary(self):
        """Test that partial matches don't count."""
        f = KeywordFilter()
        f.add_keywords("test", ["bad"])
        
        # "badger" should not match "bad"
        result = f.filter("A badger is an animal")
        
        assert not result.matched


class TestRegexFilter:
    """Tests for RegexFilter."""
    
    def test_matches_pattern(self):
        """Test pattern matching."""
        f = RegexFilter()
        f.add_pattern("spam", r"\bbuy now\b", action=FilterAction.WARN)
        
        result = f.filter("Click here to buy now!")
        
        assert result.matched
        assert result.action == FilterAction.WARN
    
    def test_sql_injection_pattern(self):
        """Test SQL injection detection."""
        f = RegexFilter()
        f.add_pattern("injection", r"(?:drop table|select \* from)", action=FilterAction.BLOCK)
        
        result = f.filter("SELECT * FROM users")
        
        assert result.matched
        assert result.action == FilterAction.BLOCK


class TestCompositeFilter:
    """Tests for CompositeFilter."""
    
    def test_combines_filters(self):
        """Test combining multiple filters."""
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("bad", ["badword"], action=FilterAction.WARN)
        
        regex_filter = RegexFilter()
        regex_filter.add_pattern("pattern", r"\d{16}", action=FilterAction.BLOCK)
        
        composite = CompositeFilter()
        composite.add(keyword_filter)
        composite.add(regex_filter)
        
        # Test keyword match
        result = composite.filter("Contains badword")
        assert result.matched
        assert result.action == FilterAction.WARN
        
        # Test regex match
        result = composite.filter("Card: 1234567890123456")
        assert result.matched
        assert result.action == FilterAction.BLOCK
    
    def test_takes_strictest_action(self):
        """Test that strictest action wins."""
        f1 = KeywordFilter()
        f1.add_keywords("warn", ["test"], action=FilterAction.WARN)
        
        f2 = KeywordFilter()
        f2.add_keywords("block", ["test"], action=FilterAction.BLOCK)
        
        composite = CompositeFilter()
        composite.add(f1)
        composite.add(f2)
        
        result = composite.filter("test")
        
        assert result.action == FilterAction.BLOCK


class TestFilterResult:
    """Tests for FilterResult."""
    
    def test_merge_actions(self):
        """Test merging results takes strictest action."""
        r1 = FilterResult(action=FilterAction.WARN, matched=True)
        r2 = FilterResult(action=FilterAction.BLOCK, matched=True)
        
        merged = r1.merge(r2)
        
        assert merged.action == FilterAction.BLOCK
    
    def test_merge_categories(self):
        """Test merging combines categories."""
        r1 = FilterResult(categories={"cat1"}, matched=True)
        r2 = FilterResult(categories={"cat2"}, matched=True)
        
        merged = r1.merge(r2)
        
        assert "cat1" in merged.categories
        assert "cat2" in merged.categories


# ============================================================================
# Test Content Filter Middleware
# ============================================================================

class TestContentFilterMiddleware:
    """Tests for ContentFilterMiddleware."""
    
    @pytest.mark.asyncio
    async def test_filters_input(self):
        """Test filtering user input."""
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("blocked", ["badword"], action=FilterAction.BLOCK)
        
        middleware = ContentFilterMiddleware(
            filter=keyword_filter,
            stop_on_block=False,
        )
        
        context = MiddlewareContext(user_input="Contains badword")
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert result.has_flag("content_filtered")
        assert result.has_flag("content_filtered_in_input")
    
    @pytest.mark.asyncio
    async def test_stops_on_block(self):
        """Test stopping chain on block."""
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("blocked", ["badword"], action=FilterAction.BLOCK)
        
        middleware = ContentFilterMiddleware(
            filter=keyword_filter,
            stop_on_block=True,
        )
        
        context = MiddlewareContext(user_input="Contains badword")
        
        async def next_fn(ctx):
            return ctx
        
        with pytest.raises(StopChainException):
            await middleware.process(context, next_fn)
    
    @pytest.mark.asyncio
    async def test_redacts_matches(self):
        """Test redacting matched content."""
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("blocked", ["badword"], action=FilterAction.WARN)
        
        middleware = ContentFilterMiddleware(
            filter=keyword_filter,
            redact_on_match=True,
            stop_on_block=False,
        )
        
        context = MiddlewareContext(user_input="Contains badword here")
        
        async def next_fn(ctx):
            return ctx
        
        result = await middleware.process(context, next_fn)
        
        assert "[REDACTED]" in result.processed_input
        assert "badword" not in result.processed_input


# ============================================================================
# Test Middleware Chain
# ============================================================================

class TestMiddlewareChain:
    """Tests for MiddlewareChain."""
    
    @pytest.mark.asyncio
    async def test_empty_chain(self):
        """Test executing empty chain."""
        chain = MiddlewareChain()
        
        context = MiddlewareContext(user_input="Hello")
        result = await chain.execute(context)
        
        assert result.success
        assert result.context.user_input == "Hello"
    
    @pytest.mark.asyncio
    async def test_single_middleware(self):
        """Test chain with single middleware."""
        chain = MiddlewareChain()
        chain.add(PIIMiddleware(redact=True))
        
        context = MiddlewareContext(user_input="Email: test@example.com")
        result = await chain.execute(context)
        
        assert result.success
        assert result.context.has_flag("pii_detected")
    
    @pytest.mark.asyncio
    async def test_multiple_middleware(self):
        """Test chain with multiple middleware."""
        store = InMemoryAuditStore()
        
        chain = MiddlewareChain()
        chain.add(PIIMiddleware(redact=True))
        chain.add(AuditMiddleware(store=store))
        
        context = MiddlewareContext(
            user_input="Email: test@example.com",
            user_id="user1",
        )
        
        result = await chain.execute(context)
        
        assert result.success
        assert result.context.has_flag("pii_detected")
        
        # Audit should have logged
        events = await store.query(user_id="user1")
        assert len(events) == 1
    
    @pytest.mark.asyncio
    async def test_chain_stop(self):
        """Test chain stopping."""
        chain = MiddlewareChain()
        chain.add(RateLimitMiddleware(
            limiter=TokenBucketLimiter(rate=0.1, capacity=1, initial_tokens=0),
            stop_on_limit=True,
        ))
        chain.add(PIIMiddleware())  # Should not be reached
        
        context = MiddlewareContext(user_input="test", user_id="user1")
        result = await chain.execute(context)
        
        assert not result.success
        assert result.stopped_by == "rate_limit"
    
    @pytest.mark.asyncio
    async def test_chain_order(self):
        """Test middleware execute in order."""
        order = []
        
        class OrderTracker(BaseMiddleware):
            def __init__(self, name: str):
                super().__init__()
                self._name = name
            
            @property
            def name(self) -> str:
                return self._name
            
            async def _do_process(self, context, next):
                order.append(f"before_{self._name}")
                result = await next(context)
                order.append(f"after_{self._name}")
                return result
        
        chain = MiddlewareChain()
        chain.add(OrderTracker("first"))
        chain.add(OrderTracker("second"))
        chain.add(OrderTracker("third"))
        
        await chain.execute(MiddlewareContext(user_input="test"))
        
        assert order == [
            "before_first",
            "before_second",
            "before_third",
            "after_third",
            "after_second",
            "after_first",
        ]
    
    @pytest.mark.asyncio
    async def test_insert_middleware(self):
        """Test inserting middleware at position."""
        chain = MiddlewareChain()
        chain.add(PIIMiddleware())
        chain.insert(0, AuditMiddleware())  # Insert at beginning
        
        assert chain.middleware[0].name == "audit"
        assert chain.middleware[1].name == "pii"
    
    @pytest.mark.asyncio
    async def test_remove_middleware(self):
        """Test removing middleware by name."""
        chain = MiddlewareChain()
        chain.add(PIIMiddleware())
        chain.add(AuditMiddleware())
        
        removed = chain.remove("pii")
        
        assert removed
        assert len(chain.middleware) == 1
        assert chain.middleware[0].name == "audit"
    
    @pytest.mark.asyncio
    async def test_get_middleware(self):
        """Test getting middleware by name."""
        chain = MiddlewareChain()
        pii = PIIMiddleware()
        chain.add(pii)
        
        found = chain.get("pii")
        
        assert found is pii
        assert chain.get("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_clear_chain(self):
        """Test clearing the chain."""
        chain = MiddlewareChain()
        chain.add(PIIMiddleware())
        chain.add(AuditMiddleware())
        
        chain.clear()
        
        assert len(chain.middleware) == 0
    
    @pytest.mark.asyncio
    async def test_processing_time(self):
        """Test that processing time is recorded."""
        chain = MiddlewareChain()
        
        class SlowMiddleware(BaseMiddleware):
            @property
            def name(self):
                return "slow"
            
            async def _do_process(self, context, next):
                await asyncio.sleep(0.01)
                return await next(context)
        
        chain.add(SlowMiddleware())
        
        result = await chain.execute(MiddlewareContext(user_input="test"))
        
        assert result.processing_time_ms >= 10


# ============================================================================
# Test Base Middleware
# ============================================================================

class TestBaseMiddleware:
    """Tests for BaseMiddleware."""
    
    @pytest.mark.asyncio
    async def test_disabled_middleware_skipped(self):
        """Test that disabled middleware is skipped."""
        pii = PIIMiddleware(enabled=False)
        
        context = MiddlewareContext(user_input="test@example.com")
        
        async def next_fn(ctx):
            return ctx
        
        result = await pii.process(context, next_fn)
        
        # PII should NOT be detected since middleware is disabled
        assert not result.has_flag("pii_detected")
    
    @pytest.mark.asyncio
    async def test_enable_disable(self):
        """Test enabling/disabling middleware."""
        pii = PIIMiddleware()
        
        assert pii.enabled
        
        pii.enabled = False
        assert not pii.enabled
        
        pii.enabled = True
        assert pii.enabled
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors are caught and recorded."""
        class ErrorMiddleware(BaseMiddleware):
            @property
            def name(self):
                return "error"
            
            async def _do_process(self, context, next):
                raise ValueError("Test error")
        
        middleware = ErrorMiddleware()
        context = MiddlewareContext(user_input="test")
        
        async def next_fn(ctx):
            return ctx
        
        # Should not raise, should record error in metadata
        result = await middleware.process(context, next_fn)
        
        assert result.get_metadata("error_error") == "Test error"


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestMiddlewareIntegration:
    """Integration tests for middleware scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete middleware pipeline."""
        audit_store = InMemoryAuditStore()
        
        # Create filter
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("spam", ["buy now", "limited offer"])
        
        # Build chain
        chain = MiddlewareChain()
        chain.add(RateLimitMiddleware(
            limiter=TokenBucketLimiter(rate=100.0, capacity=100),
        ))
        chain.add(PIIMiddleware(redact=True))
        chain.add(ContentFilterMiddleware(
            filter=keyword_filter,
            stop_on_block=False,
        ))
        chain.add(AuditMiddleware(store=audit_store))
        
        # Process request with PII
        context = MiddlewareContext(
            user_input="Contact me at john@example.com for info",
            user_id="user1",
        )
        
        result = await chain.execute(context)
        
        assert result.success
        assert result.context.has_flag("pii_detected")
        assert "[EMAIL]" in result.context.processed_input
        
        # Check audit log
        events = await audit_store.query(user_id="user1")
        assert len(events) == 1
        assert events[0].details["flags"]  # Has flags recorded
    
    @pytest.mark.asyncio
    async def test_blocked_content_scenario(self):
        """Test scenario where content is blocked."""
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("blocked", ["forbidden"], action=FilterAction.BLOCK)
        
        chain = MiddlewareChain()
        chain.add(ContentFilterMiddleware(
            filter=keyword_filter,
            stop_on_block=True,
        ))
        chain.add(PIIMiddleware())  # Should not be reached
        
        context = MiddlewareContext(user_input="This is forbidden content")
        result = await chain.execute(context)
        
        assert not result.success
        assert result.stopped_by == "content_filter"
    
    @pytest.mark.asyncio
    async def test_rate_limited_scenario(self):
        """Test scenario where user is rate limited."""
        limiter = SlidingWindowLimiter(limit=2, window_seconds=60.0)
        
        chain = MiddlewareChain()
        chain.add(RateLimitMiddleware(limiter=limiter, stop_on_limit=True))
        
        # First two requests should succeed
        for _ in range(2):
            result = await chain.execute(MiddlewareContext(
                user_input="test",
                user_id="user1",
            ))
            assert result.success
        
        # Third should be rate limited
        result = await chain.execute(MiddlewareContext(
            user_input="test",
            user_id="user1",
        ))
        
        assert not result.success
        assert result.stopped_by == "rate_limit"

