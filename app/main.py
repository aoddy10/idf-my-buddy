"""FastAPI application factory and configuration.

This module creates and configures the FastAPI application with middleware,
exception handlers, and router inclusions for all feature domains.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import routers
from app.api import health, navigation, restaurant, safety, shopping
from app.api.v1 import voice
from app.core.config import settings
from app.core.errors import (
    MyBuddyException,
    ServiceUnavailableError,
    ValidationError,
    generic_exception_handler,
    my_buddy_exception_handler,
    service_unavailable_handler,
    validation_exception_handler,
)
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.

    Args:
        app: FastAPI application instance.

    Yields:
        None: Control back to the application.
    """
    # Startup
    logger.info("Starting My Buddy AI Travel Assistant", app_env=settings.app_env)

    # Initialize voice services
    try:
        from app.api.v1.voice import voice_service
        logger.info("Initializing voice services...")
        voice_service.initialize_services()
        logger.info("Voice services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize voice services: {e}")
        # Continue without voice services in case of failure

    # TODO: Initialize database connections
    # TODO: Load AI models
    # TODO: Setup external service clients

    yield

    # Shutdown
    logger.info("Shutting down My Buddy AI Travel Assistant")

    # Cleanup voice services
    try:
        from app.api.v1.voice import voice_service
        logger.info("Cleaning up voice services...")
        # Voice service cleanup is handled internally by the services
        logger.info("Voice services cleanup completed")
    except Exception as e:
        logger.error(f"Error during voice services cleanup: {e}")

    # TODO: Cleanup database connections
    # TODO: Cleanup AI model resources
    # TODO: Cleanup external service clients


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    # Setup logging first
    setup_logging()

    # Create FastAPI app with conditional docs
    app = FastAPI(
        title="My Buddy API",
        description="AI Travel Assistant - Your Smart Travel Companion",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    # Add security middleware
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual domains in production
        )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )

    # Add request logging middleware
    if settings.enable_request_logging:
        @app.middleware("http")
        async def log_requests(request: Request, call_next) -> Response:
            """Log HTTP requests with correlation IDs."""
            import time
            import uuid

            # Generate correlation ID
            correlation_id = str(uuid.uuid4())
            request.state.correlation_id = correlation_id

            start_time = time.time()

            logger.info(
                "Request started",
                method=request.method,
                url=str(request.url),
                correlation_id=correlation_id,
                user_agent=request.headers.get("user-agent"),
            )

            response = await call_next(request)

            process_time = time.time() - start_time

            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=f"{process_time:.4f}s",
                correlation_id=correlation_id,
            )

            return response

    # Add exception handlers
    app.add_exception_handler(MyBuddyException, my_buddy_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(ServiceUnavailableError, service_unavailable_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Include feature routers
    app.include_router(
        health.router,
        prefix="/health",
        tags=["health"]
    )

    app.include_router(
        navigation.router,
        prefix="/navigation",
        tags=["navigation"]
    )

    app.include_router(
        restaurant.router,
        prefix="/restaurant",
        tags=["restaurant"]
    )

    app.include_router(
        shopping.router,
        prefix="/shopping",
        tags=["shopping"]
    )

    app.include_router(
        safety.router,
        prefix="/safety",
        tags=["safety"]
    )

    app.include_router(
        voice.router,
        prefix="/api/v1/voice",
        tags=["voice", "speech", "audio"]
    )

    # Add debug endpoints in development
    if settings.enable_debug_endpoints and settings.is_development:
        @app.get("/debug/config")
        async def debug_config() -> dict:
            """Debug endpoint to view configuration (development only)."""
            return {
                "app_env": settings.app_env,
                "debug": settings.debug,
                "log_level": settings.log_level,
                "enable_offline_mode": settings.enable_offline_mode,
                "voice_config": {
                    "whisper": {
                        "model_size": settings.whisper.model_size,
                        "use_local": settings.whisper.use_local,
                        "device": settings.whisper.device,
                    },
                    "tts": {
                        "default_voice": settings.tts.default_voice,
                        "speech_rate": settings.tts.speech_rate,
                        "enable_openai": settings.tts.enable_openai,
                        "enable_speechbrain": settings.tts.enable_speechbrain,
                    },
                    "audio": {
                        "max_file_size": settings.audio.max_file_size,
                        "max_duration": settings.audio.max_duration,
                        "sample_rate": settings.audio.sample_rate,
                    },
                    "pipeline": {
                        "enable_cloud_fallback": settings.voice_pipeline.enable_cloud_fallback,
                        "session_timeout": settings.voice_pipeline.session_timeout,
                        "enable_metrics": settings.voice_pipeline.enable_metrics,
                    }
                }
            }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    """Run the application directly."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.reload and settings.is_development,
        log_level=settings.log_level.lower(),
    )
