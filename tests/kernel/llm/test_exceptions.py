"""Tests for exceptions.py."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.kernel.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMResponseConsumedError,
    LLMTimeoutError,
    LLMTokenLimitError,
    classify_exception,
)


class TestLLMError:
    """Test cases for LLMError base class."""

    def test_llm_error_is_runtime_error(self) -> None:
        """Test that LLMError is a RuntimeError."""
        assert issubclass(LLMError, RuntimeError)

    def test_llm_error_can_be_raised(self) -> None:
        """Test that LLMError can be raised."""
        with pytest.raises(LLMError):
            raise LLMError("Test error")

    def test_llm_error_message(self) -> None:
        """Test LLMError message."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert "Something went wrong" in repr(error)


class TestLLMConfigurationError:
    """Test cases for LLMConfigurationError."""

    def test_configuration_error_is_llm_error(self) -> None:
        """Test that LLMConfigurationError is an LLMError."""
        assert issubclass(LLMConfigurationError, LLMError)

    def test_configuration_error_creation(self) -> None:
        """Test creating LLMConfigurationError."""
        error = LLMConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"

    def test_configuration_error_can_be_raised(self) -> None:
        """Test that LLMConfigurationError can be raised."""
        with pytest.raises(LLMConfigurationError):
            raise LLMConfigurationError("Config error")


class TestLLMResponseConsumedError:
    """Test cases for LLMResponseConsumedError."""

    def test_response_consumed_error_is_llm_error(self) -> None:
        """Test that LLMResponseConsumedError is an LLMError."""
        assert issubclass(LLMResponseConsumedError, LLMError)

    def test_response_consumed_error_creation(self) -> None:
        """Test creating LLMResponseConsumedError."""
        error = LLMResponseConsumedError("Response already consumed")
        assert str(error) == "Response already consumed"


class TestLLMRateLimitError:
    """Test cases for LLMRateLimitError."""

    def test_rate_limit_error_is_llm_error(self) -> None:
        """Test that LLMRateLimitError is an LLMError."""
        assert issubclass(LLMRateLimitError, LLMError)

    def test_rate_limit_error_with_retry_after(self) -> None:
        """Test LLMRateLimitError with retry_after."""
        error = LLMRateLimitError(
            message="Rate limit exceeded", retry_after=60.0, model="gpt-4"
        )
        assert error.retry_after == 60.0
        assert error.model == "gpt-4"

    def test_rate_limit_error_without_optional_params(self) -> None:
        """Test LLMRateLimitError without optional parameters."""
        error = LLMRateLimitError(message="Rate limit exceeded")
        assert error.retry_after is None
        assert error.model is None


class TestLLMTimeoutError:
    """Test cases for LLMTimeoutError."""

    def test_timeout_error_is_llm_error(self) -> None:
        """Test that LLMTimeoutError is an LLMError."""
        assert issubclass(LLMTimeoutError, LLMError)

    def test_timeout_error_with_timeout(self) -> None:
        """Test LLMTimeoutError with timeout."""
        error = LLMTimeoutError(
            message="Request timeout", timeout=30.0, model="gpt-3.5-turbo"
        )
        assert error.timeout == 30.0
        assert error.model == "gpt-3.5-turbo"

    def test_timeout_error_without_optional_params(self) -> None:
        """Test LLMTimeoutError without optional parameters."""
        error = LLMTimeoutError(message="Request timeout")
        assert error.timeout is None
        assert error.model is None


class TestLLMContentFilterError:
    """Test cases for LLMContentFilterError."""

    def test_content_filter_error_is_llm_error(self) -> None:
        """Test that LLMContentFilterError is an LLMError."""
        assert issubclass(LLMContentFilterError, LLMError)

    def test_content_filter_error_with_filter_type(self) -> None:
        """Test LLMContentFilterError with filter_type."""
        error = LLMContentFilterError(
            message="Content filtered", filter_type="violence", model="gpt-4"
        )
        assert error.filter_type == "violence"
        assert error.model == "gpt-4"

    def test_content_filter_error_without_optional_params(self) -> None:
        """Test LLMContentFilterError without optional parameters."""
        error = LLMContentFilterError(message="Content filtered")
        assert error.filter_type is None
        assert error.model is None


class TestLLMTokenLimitError:
    """Test cases for LLMTokenLimitError."""

    def test_token_limit_error_is_llm_error(self) -> None:
        """Test that LLMTokenLimitError is an LLMError."""
        assert issubclass(LLMTokenLimitError, LLMError)

    def test_token_limit_error_with_limits(self) -> None:
        """Test LLMTokenLimitError with max_tokens and requested_tokens."""
        error = LLMTokenLimitError(
            message="Token limit exceeded",
            max_tokens=4096,
            requested_tokens=5000,
            model="gpt-4",
        )
        assert error.max_tokens == 4096
        assert error.requested_tokens == 5000
        assert error.model == "gpt-4"

    def test_token_limit_error_without_optional_params(self) -> None:
        """Test LLMTokenLimitError without optional parameters."""
        error = LLMTokenLimitError(message="Token limit exceeded")
        assert error.max_tokens is None
        assert error.requested_tokens is None
        assert error.model is None


class TestLLMAuthenticationError:
    """Test cases for LLMAuthenticationError."""

    def test_authentication_error_is_llm_error(self) -> None:
        """Test that LLMAuthenticationError is an LLMError."""
        assert issubclass(LLMAuthenticationError, LLMError)

    def test_authentication_error_with_model(self) -> None:
        """Test LLMAuthenticationError with model."""
        error = LLMAuthenticationError(message="Invalid API key", model="gpt-4")
        assert error.model == "gpt-4"

    def test_authentication_error_without_optional_params(self) -> None:
        """Test LLMAuthenticationError without optional parameters."""
        error = LLMAuthenticationError(message="Invalid API key")
        assert error.model is None


class TestLLMAPIError:
    """Test cases for LLMAPIError."""

    def test_api_error_is_llm_error(self) -> None:
        """Test that LLMAPIError is an LLMError."""
        assert issubclass(LLMAPIError, LLMError)

    def test_api_error_with_all_params(self) -> None:
        """Test LLMAPIError with all parameters."""
        error = LLMAPIError(
            message="API error",
            status_code=500,
            error_code="server_error",
            model="gpt-4",
        )
        assert error.status_code == 500
        assert error.error_code == "server_error"
        assert error.model == "gpt-4"

    def test_api_error_without_optional_params(self) -> None:
        """Test LLMAPIError without optional parameters."""
        error = LLMAPIError(message="API error")
        assert error.status_code is None
        assert error.error_code is None
        assert error.model is None


class TestClassifyException:
    """Test cases for classify_exception function."""

    def test_classify_generic_exception(self) -> None:
        """Test classifying a generic exception."""
        error = ValueError("Some error")
        classified = classify_exception(error)
        assert classified == error  # Returns original error

    def test_classify_with_model_param(self) -> None:
        """Test classify_exception with model parameter."""
        error = ValueError("Some error")
        classified = classify_exception(error, model="gpt-4")
        assert classified == error

    @pytest.mark.parametrize(
        "error_msg,expected_class",
        [
            ("rate limit exceeded", LLMRateLimitError),
            ("Rate limit reached", LLMRateLimitError),
            ("request timed out", LLMTimeoutError),
            ("operation timeout", LLMTimeoutError),
            ("authentication failed", LLMAuthenticationError),
            ("unauthorized access", LLMAuthenticationError),
            ("invalid api key", LLMAuthenticationError),
            ("token limit exceeded", LLMTokenLimitError),
            ("maximum context length", LLMTokenLimitError),
            ("content filter triggered", LLMContentFilterError),
            ("content policy violation", LLMContentFilterError),
        ],
    )
    def test_classify_by_error_message(
        self, error_msg: str, expected_class: type[LLMError]
    ) -> None:
        """Test classifying exceptions by error message."""
        error = ValueError(error_msg)
        classified = classify_exception(error, model="gpt-4")
        assert isinstance(classified, expected_class)
        if hasattr(classified, "model"):
            assert classified.model == "gpt-4"

    @patch("src.kernel.llm.exceptions.openai")
    def test_classify_openai_rate_limit_error(self, mock_openai: Mock) -> None:
        """Test classifying OpenAI RateLimitError."""
        # Create a mock RateLimitError
        mock_rate_limit = MagicMock()
        mock_rate_limit.__class__ = mock_openai.RateLimitError
        mock_rate_limit.__class__.__name__ = "RateLimitError"
        mock_rate_limit.retry_after = 60
        mock_rate_limit.__str__ = lambda self: "Rate limit exceeded"

        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_rate_limit_instance = mock_openai.RateLimitError("Rate limit exceeded")
        mock_rate_limit_instance.retry_after = 60

        classified = classify_exception(mock_rate_limit_instance, model="gpt-4")
        assert isinstance(classified, LLMRateLimitError)
        assert classified.retry_after == 60

    @patch("src.kernel.llm.exceptions.openai")
    def test_classify_openai_timeout_error(self, mock_openai: Mock) -> None:
        """Test classifying OpenAI APITimeoutError."""
        mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_timeout = mock_openai.APITimeoutError("Request timeout")

        classified = classify_exception(mock_timeout, model="gpt-4")
        assert isinstance(classified, LLMTimeoutError)

    @patch("src.kernel.llm.exceptions.openai")
    def test_classify_openai_auth_error(self, mock_openai: Mock) -> None:
        """Test classifying OpenAI AuthenticationError."""
        mock_openai.AuthenticationError = type(
            "AuthenticationError", (Exception,), {}
        )
        mock_auth = mock_openai.AuthenticationError("Invalid API key")

        classified = classify_exception(mock_auth, model="gpt-4")
        assert isinstance(classified, LLMAuthenticationError)

    @patch("src.kernel.llm.exceptions.openai")
    def test_classify_openai_bad_request_token_limit(
        self, mock_openai: Mock
    ) -> None:
        """Test classifying OpenAI BadRequestError with token limit."""
        mock_openai.BadRequestError = type("BadRequestError", (Exception,), {})
        mock_bad = mock_openai.BadRequestError(
            "maximum context length exceeded"
        )

        classified = classify_exception(mock_bad, model="gpt-4")
        assert isinstance(classified, LLMTokenLimitError)

    @patch("src.kernel.llm.exceptions.openai")
    def test_classify_openai_bad_request_content_filter(
        self, mock_openai: Mock
    ) -> None:
        """Test classifying OpenAI BadRequestError with content filter."""
        mock_openai.BadRequestError = type("BadRequestError", (Exception,), {})
        mock_bad = mock_openai.BadRequestError("content filter triggered")

        classified = classify_exception(mock_bad, model="gpt-4")
        assert isinstance(classified, LLMContentFilterError)

    @patch("src.kernel.llm.exceptions.openai")
    def test_classify_openai_api_error(self, mock_openai: Mock) -> None:
        """Test classifying OpenAI APIError."""
        mock_api_error_class = type("APIError", (Exception,), {})
        mock_api_error = mock_api_error_class("API error occurred")
        mock_api_error.status_code = 500
        mock_api_error.code = "server_error"

        mock_openai.APIError = mock_api_error_class

        classified = classify_exception(mock_api_error, model="gpt-4")
        assert isinstance(classified, LLMAPIError)
        assert classified.status_code == 500

    @patch("src.kernel.llm.exceptions.openai", None)
    def test_classify_without_openai_installed(self) -> None:
        """Test classify_exception when openai is not installed."""
        error = ValueError("rate limit exceeded")
        # Should still work based on error message
        classified = classify_exception(error)
        assert isinstance(classified, LLMRateLimitError)

    def test_classify_unknown_error_returns_original(self) -> None:
        """Test that unknown error returns original exception."""
        error = RuntimeError("Unknown error")
        classified = classify_exception(error)
        assert classified == error

    def test_classify_case_insensitive(self) -> None:
        """Test that error message matching is case insensitive."""
        error = ValueError("RATE LIMIT EXCEEDED")
        classified = classify_exception(error)
        assert isinstance(classified, LLMRateLimitError)


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_llm_error(self) -> None:
        """Test that all custom exceptions inherit from LLMError."""
        exceptions = [
            LLMConfigurationError,
            LLMResponseConsumedError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMContentFilterError,
            LLMTokenLimitError,
            LLMAuthenticationError,
            LLMAPIError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, LLMError), f"{exc_class} should inherit from LLMError"

    def test_all_exceptions_catchable_as_llm_error(self) -> None:
        """Test that all exceptions can be caught as LLMError."""
        caught_errors = []

        try:
            raise LLMRateLimitError("test")
        except LLMError:
            caught_errors.append("rate_limit")

        try:
            raise LLMTimeoutError("test")
        except LLMError:
            caught_errors.append("timeout")

        try:
            raise LLMAuthenticationError("test")
        except LLMError:
            caught_errors.append("auth")

        assert len(caught_errors) == 3
