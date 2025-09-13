"""Structured logging configuration for My Buddy application.

This module provides JSON-formatted logging with request correlation IDs
and performance timing for production observability.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if settings.log_format == "json":
        # JSON output for production
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable output for development
        processors.extend([
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True)
        ])

    # Configure structlog
    structlog.configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance with optional name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        structlog.BoundLogger: Configured logger instance.
    """
    return structlog.get_logger(name or "my_buddy")


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log function call with parameters.

    Args:
        func_name: Name of the function being called.
        **kwargs: Function parameters to log.
    """
    logger = get_logger()

    # Filter out sensitive parameters
    safe_kwargs = {
        key: "***REDACTED***" if key.lower() in ["password", "token", "key", "secret"]
        else value
        for key, value in kwargs.items()
    }

    logger.debug("Function called", function=func_name, parameters=safe_kwargs)


def log_performance(operation: str, duration: float, **context: Any) -> None:
    """Log performance metrics for operations.

    Args:
        operation: Name of the operation being measured.
        duration: Duration in seconds.
        **context: Additional context for the operation.
    """
    logger = get_logger()

    level = "warning" if duration > 1.0 else "info"

    getattr(logger, level)(
        "Performance metric",
        operation=operation,
        duration_seconds=round(duration, 4),
        **context
    )


def log_ai_operation(
    operation: str,
    model: str,
    input_size: int | None = None,
    output_size: int | None = None,
    confidence: float | None = None,
    duration: float | None = None,
    **context: Any
) -> None:
    """Log AI/ML operation metrics.

    Args:
        operation: Type of AI operation (e.g., "transcription", "translation").
        model: Model name or identifier.
        input_size: Size of input data (e.g., audio duration, image pixels).
        output_size: Size of output data (e.g., text length).
        confidence: Model confidence score (0.0 to 1.0).
        duration: Operation duration in seconds.
        **context: Additional context for the operation.
    """
    logger = get_logger()

    log_data = {
        "ai_operation": operation,
        "model": model,
        **context
    }

    if input_size is not None:
        log_data["input_size"] = input_size

    if output_size is not None:
        log_data["output_size"] = output_size

    if confidence is not None:
        log_data["confidence"] = round(confidence, 3)

    if duration is not None:
        log_data["duration_seconds"] = round(duration, 4)

    logger.info("AI operation completed", **log_data)


def log_user_action(
    user_id: str,
    action: str,
    feature: str,
    success: bool = True,
    **context: Any
) -> None:
    """Log user actions for analytics (privacy-preserving).

    Args:
        user_id: Anonymized user identifier.
        action: User action type.
        feature: Feature area (navigation, restaurant, shopping, safety).
        success: Whether the action was successful.
        **context: Additional non-PII context.
    """
    if not settings.enable_telemetry:
        return

    logger = get_logger()

    logger.info(
        "User action",
        user_id=user_id,
        action=action,
        feature=feature,
        success=success,
        **context
    )
