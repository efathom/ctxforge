"""
Custom Exceptions for the ctxforge framework.

Provides a hierarchy of exceptions for better error handling
and debugging across the framework.
"""

from typing import Any, Dict, Optional


class ContextEngineError(Exception):
    """
    Base exception for all ctxforge errors.
    
    All custom exceptions in the framework inherit from this class,
    allowing for broad exception catching when needed.
    """
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "CONTEXT_ENGINE_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
        }


class ConfigurationError(ContextEngineError):
    """
    Raised when there's an error in configuration.
    
    Examples:
    - Invalid configuration values
    - Missing required configuration
    - Configuration file not found
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class StorageError(ContextEngineError):
    """
    Raised when there's an error with storage operations.
    
    Examples:
    - Database connection errors
    - Read/write failures
    - Concurrency conflicts
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="STORAGE_ERROR", details=details)
        self.operation = operation
        if operation:
            self.details["operation"] = operation


class ConcurrencyError(StorageError):
    """
    Raised when there's a concurrency conflict (optimistic locking failure).
    """
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        actual_version: Optional[int] = None,
    ):
        details = {}
        if session_id:
            details["session_id"] = session_id
        if expected_version is not None:
            details["expected_version"] = expected_version
        if actual_version is not None:
            details["actual_version"] = actual_version
        
        super().__init__(message, operation="optimistic_lock", details=details)
        self.session_id = session_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class LLMError(ContextEngineError):
    """
    Raised when there's an error with LLM operations.
    
    Examples:
    - API errors
    - Rate limiting
    - Token limit exceeded
    - Model not found
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="LLM_ERROR", details=details)
        self.provider = provider
        self.model = model
        if provider:
            self.details["provider"] = provider
        if model:
            self.details["model"] = model


class RateLimitError(LLMError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message, provider=provider)
        self.code = "RATE_LIMIT_ERROR"
        self.retry_after = retry_after
        if retry_after is not None:
            self.details["retry_after"] = retry_after


class TokenLimitError(LLMError):
    """Raised when token limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message, provider=provider)
        self.code = "TOKEN_LIMIT_ERROR"
        self.token_count = token_count
        self.token_limit = token_limit
        if token_count is not None:
            self.details["token_count"] = token_count
        if token_limit is not None:
            self.details["token_limit"] = token_limit


class ValidationError(ContextEngineError):
    """
    Raised when input validation fails.
    
    Examples:
    - Invalid event type
    - Malformed input
    - Missing required fields
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="VALIDATION_ERROR", details=details)
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class MiddlewareError(ContextEngineError):
    """
    Raised when a middleware component fails.
    """
    
    def __init__(
        self,
        message: str,
        middleware_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="MIDDLEWARE_ERROR", details=details)
        self.middleware_name = middleware_name
        if middleware_name:
            self.details["middleware"] = middleware_name


class RetrievalError(ContextEngineError):
    """
    Raised when memory retrieval fails.
    """
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="RETRIEVAL_ERROR", details=details)
        self.query = query
        if query:
            self.details["query"] = query[:100]  # Truncate for safety


class ToolError(ContextEngineError):
    """
    Raised when a tool execution fails.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="TOOL_ERROR", details=details)
        self.tool_name = tool_name
        self.tool_args = tool_args
        if tool_name:
            self.details["tool_name"] = tool_name


class ExtractionError(ContextEngineError):
    """
    Raised when memory extraction fails.
    """
    
    def __init__(
        self,
        message: str,
        extractor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="EXTRACTION_ERROR", details=details)
        self.extractor = extractor
        if extractor:
            self.details["extractor"] = extractor


class CompactionError(ContextEngineError):
    """
    Raised when context compaction fails.
    """
    
    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code="COMPACTION_ERROR", details=details)
        self.strategy = strategy
        if strategy:
            self.details["strategy"] = strategy

