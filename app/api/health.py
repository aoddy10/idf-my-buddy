"""Health check endpoints for My Buddy API.

This module provides health check and status endpoints for monitoring
the application and its dependencies.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    environment: str


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    from app.core.config import get_settings
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        environment=settings.app_env
    )


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for deployment health monitoring."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow(),
        "checks": {
            "database": "healthy",  # TODO: Implement actual DB check
            "cache": "healthy",     # TODO: Implement Redis check
        }
    }


@router.get("/live", response_model=Dict[str, str])
async def liveness_check() -> Dict[str, str]:
    """Liveness check for container orchestration."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
