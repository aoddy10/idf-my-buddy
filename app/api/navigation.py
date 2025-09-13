"""Navigation API endpoints for My Buddy application.

Provides location-based routing, directions, and navigation assistance.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class NavigationResponse(BaseModel):
    """Navigation response model."""
    message: str = "Navigation service is under development"


@router.get("/", response_model=NavigationResponse)
async def navigation_status() -> NavigationResponse:
    """Get navigation service status."""
    return NavigationResponse()
