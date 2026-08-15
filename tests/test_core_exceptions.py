"""
Tests for core exceptions.
"""


from ctxforge.core.exceptions import (
    CompactionError,
    ConcurrencyError,
    ConfigurationError,
    ContextEngineError,
    ExtractionError,
    LLMError,
    MiddlewareError,
    RateLimitError,
    RetrievalError,
    StorageError,
    TokenLimitError,
    ToolError,
    ValidationError,
)


class TestContextEngineError:
    """Tests for base exception."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = ContextEngineError("Something went wrong")
        
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code == "CONTEXT_ENGINE_ERROR"
        assert error.details == {}
    
    def test_error_with_code_and_details(self):
        """Test error with code and details."""
        error = ContextEngineError(
            "Error occurred",
            code="CUSTOM_ERROR",
            details={"key": "value"},
        )
        
        assert error.code == "CUSTOM_ERROR"
        assert error.details["key"] == "value"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = ContextEngineError(
            "Error occurred",
            code="TEST_ERROR",
            details={"info": "data"},
        )
        
        d = error.to_dict()
        assert d["error"] == "ContextEngineError"
        assert d["message"] == "Error occurred"
        assert d["code"] == "TEST_ERROR"
        assert d["details"]["info"] == "data"


class TestConfigurationError:
    """Tests for ConfigurationError."""
    
    def test_basic_config_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid configuration")
        
        assert error.code == "CONFIGURATION_ERROR"
        assert error.config_key is None
    
    def test_config_error_with_key(self):
        """Test configuration error with key."""
        error = ConfigurationError(
            "Invalid value",
            config_key="llm.model",
        )
        
        assert error.config_key == "llm.model"
        assert error.details["config_key"] == "llm.model"


class TestStorageError:
    """Tests for StorageError."""
    
    def test_basic_storage_error(self):
        """Test basic storage error."""
        error = StorageError("Database connection failed")
        
        assert error.code == "STORAGE_ERROR"
        assert error.operation is None
    
    def test_storage_error_with_operation(self):
        """Test storage error with operation."""
        error = StorageError(
            "Write failed",
            operation="save_session",
        )
        
        assert error.operation == "save_session"
        assert error.details["operation"] == "save_session"


class TestConcurrencyError:
    """Tests for ConcurrencyError."""
    
    def test_concurrency_error(self):
        """Test concurrency error."""
        error = ConcurrencyError(
            "Version conflict",
            session_id="sess_123",
            expected_version=5,
            actual_version=7,
        )
        
        assert error.session_id == "sess_123"
        assert error.expected_version == 5
        assert error.actual_version == 7
        assert error.details["session_id"] == "sess_123"


class TestLLMError:
    """Tests for LLMError."""
    
    def test_basic_llm_error(self):
        """Test basic LLM error."""
        error = LLMError("API call failed")
        
        assert error.code == "LLM_ERROR"
        assert error.provider is None
        assert error.model is None
    
    def test_llm_error_with_provider(self):
        """Test LLM error with provider info."""
        error = LLMError(
            "Generation failed",
            provider="openai",
            model="gpt-4",
        )
        
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.details["provider"] == "openai"
        assert error.details["model"] == "gpt-4"


class TestRateLimitError:
    """Tests for RateLimitError."""
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60.0,
            provider="openai",
        )
        
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.retry_after == 60.0
        assert error.details["retry_after"] == 60.0


class TestTokenLimitError:
    """Tests for TokenLimitError."""
    
    def test_token_limit_error(self):
        """Test token limit error."""
        error = TokenLimitError(
            "Context too long",
            token_count=10000,
            token_limit=8000,
        )
        
        assert error.code == "TOKEN_LIMIT_ERROR"
        assert error.token_count == 10000
        assert error.token_limit == 8000


class TestValidationError:
    """Tests for ValidationError."""
    
    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError(
            "Invalid input",
            field="email",
            value="not-an-email",
        )
        
        assert error.code == "VALIDATION_ERROR"
        assert error.field == "email"
        assert error.value == "not-an-email"


class TestMiddlewareError:
    """Tests for MiddlewareError."""
    
    def test_middleware_error(self):
        """Test middleware error."""
        error = MiddlewareError(
            "Processing failed",
            middleware_name="PIIRedactor",
        )
        
        assert error.code == "MIDDLEWARE_ERROR"
        assert error.middleware_name == "PIIRedactor"


class TestRetrievalError:
    """Tests for RetrievalError."""
    
    def test_retrieval_error(self):
        """Test retrieval error."""
        error = RetrievalError(
            "Search failed",
            query="vegetarian recipes",
        )
        
        assert error.code == "RETRIEVAL_ERROR"
        assert error.query == "vegetarian recipes"


class TestToolError:
    """Tests for ToolError."""
    
    def test_tool_error(self):
        """Test tool error."""
        error = ToolError(
            "Tool execution failed",
            tool_name="calculator",
            tool_args={"expression": "1/0"},
        )
        
        assert error.code == "TOOL_ERROR"
        assert error.tool_name == "calculator"
        assert error.tool_args["expression"] == "1/0"


class TestExtractionError:
    """Tests for ExtractionError."""
    
    def test_extraction_error(self):
        """Test extraction error."""
        error = ExtractionError(
            "Extraction failed",
            extractor="LLMExtractor",
        )
        
        assert error.code == "EXTRACTION_ERROR"
        assert error.extractor == "LLMExtractor"


class TestCompactionError:
    """Tests for CompactionError."""
    
    def test_compaction_error(self):
        """Test compaction error."""
        error = CompactionError(
            "Summarization failed",
            strategy="recursive_summary",
        )
        
        assert error.code == "COMPACTION_ERROR"
        assert error.strategy == "recursive_summary"


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""
    
    def test_all_inherit_from_base(self):
        """Test that all exceptions inherit from ContextEngineError."""
        exceptions = [
            ConfigurationError("test"),
            StorageError("test"),
            ConcurrencyError("test"),
            LLMError("test"),
            RateLimitError("test"),
            TokenLimitError("test"),
            ValidationError("test"),
            MiddlewareError("test"),
            RetrievalError("test"),
            ToolError("test"),
            ExtractionError("test"),
            CompactionError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, ContextEngineError)
    
    def test_can_catch_by_base(self):
        """Test that base exception catches all."""
        try:
            raise LLMError("Test error")
        except ContextEngineError as e:
            assert e.message == "Test error"
    
    def test_storage_errors_are_storage_errors(self):
        """Test storage exception hierarchy."""
        assert isinstance(ConcurrencyError("test"), StorageError)
    
    def test_llm_errors_are_llm_errors(self):
        """Test LLM exception hierarchy."""
        assert isinstance(RateLimitError("test"), LLMError)
        assert isinstance(TokenLimitError("test"), LLMError)

