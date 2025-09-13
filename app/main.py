"""FastAPI application factory and configuration.

This module creates and configures the FastAPI application with middleware,
exception handlers, and router inclusions for all feature domains.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.errors import (
    MyBuddyException,
    ValidationError,
    ServiceUnavailableError,
    my_buddy_exception_handler,
    validation_exception_handler,
    service_unavailable_handler,
    generic_exception_handler,
)

# Import routers
from app.api import health, navigation, restaurant, shopping, safety

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
    
    # TODO: Initialize database connections
    # TODO: Load AI models
    # TODO: Setup external service clients
    
    yield
    
    # Shutdown
    logger.info("Shutting down My Buddy AI Travel Assistant")
    
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
                "whisper_model_size": settings.whisper_model_size,
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
