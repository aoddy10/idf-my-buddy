"""Custom exceptions and error handlers for My Buddy application.

This module defines application-specific exceptions and FastAPI error handlers
to provide consistent error responses across all endpoints.
"""

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


class MyBuddyException(Exception):
    """Base exception for My Buddy application errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None
    ) -> None:
        """Initialize exception.

        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code.
            status_code: HTTP status code.
            details: Additional error details.
        """
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(MyBuddyException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str = "Invalid input data",
        field: str | None = None,
        value: Any | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = str(value)

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=error_details
        )


class ServiceUnavailableError(MyBuddyException):
    """Raised when an external service is unavailable."""

    def __init__(
        self,
        service: str,
        message: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        message = message or f"{service} service is currently unavailable"
        error_details = details or {}
        error_details["service"] = service

        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=error_details
        )


class AIServiceError(MyBuddyException):
    """Raised when AI/ML service operations fail."""

    def __init__(
        self,
        operation: str,
        model: str,
        message: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        message = message or f"AI service error during {operation}"
        error_details = details or {}
        error_details.update({
            "operation": operation,
            "model": model
        })

        super().__init__(
            message=message,
            error_code="AI_SERVICE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=error_details
        )


class AuthenticationError(MyBuddyException):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationError(MyBuddyException):
    """Raised when authorization fails."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        resource: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        error_details = details or {}
        if resource:
            error_details["resource"] = resource

        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            details=error_details
        )


class RateLimitError(MyBuddyException):
    """Raised when rate limiting is triggered."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        error_details = details or {}
        if retry_after:
            error_details["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=error_details
        )


class ResourceNotFoundError(MyBuddyException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        message: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        message = message or f"{resource_type} with ID {resource_id} not found"
        error_details = details or {}
        error_details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })

        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=error_details
        )


# Error Handlers

async def my_buddy_exception_handler(
    request: Request,
    exc: MyBuddyException
) -> JSONResponse:
    """Handle My Buddy application exceptions.

    Args:
        request: FastAPI request object.
        exc: My Buddy exception instance.

    Returns:
        JSONResponse: Formatted error response.
    """
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    # Log error with correlation ID if available
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.error(
        "Application error",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
    )

    response_data = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        }
    }

    if correlation_id:
        response_data["correlation_id"] = correlation_id

    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
    )


async def validation_exception_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """Handle validation errors with detailed field information.

    Args:
        request: FastAPI request object.
        exc: Validation error instance.

    Returns:
        JSONResponse: Formatted validation error response.
    """
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    correlation_id = getattr(request.state, "correlation_id", None)

    logger.warning(
        "Validation error",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
            "correlation_id": correlation_id,
        },
    )


async def service_unavailable_handler(
    request: Request,
    exc: ServiceUnavailableError
) -> JSONResponse:
    """Handle service unavailable errors.

    Args:
        request: FastAPI request object.
        exc: Service unavailable error instance.

    Returns:
        JSONResponse: Formatted service error response.
    """
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    correlation_id = getattr(request.state, "correlation_id", None)

    logger.error(
        "Service unavailable",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
            "correlation_id": correlation_id,
        },
        headers={
            "Retry-After": "60"  # Suggest retry after 60 seconds
        }
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: FastAPI request object.
        exc: Generic exception instance.

    Returns:
        JSONResponse: Generic error response.
    """
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    correlation_id = getattr(request.state, "correlation_id", None)

    logger.error(
        "Unexpected error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    # Don't expose internal error details in production
    from app.core.config import settings

    if settings.is_production:
        message = "An unexpected error occurred"
        details = {}
    else:
        message = str(exc)
        details = {"type": type(exc).__name__}

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": message,
                "details": details,
            },
            "correlation_id": correlation_id,
        },
    )
