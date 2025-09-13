"""Restaurant API endpoints for My Buddy application.

Provides menu translation, allergen detection, and restaurant recommendations.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RestaurantResponse(BaseModel):
    """Restaurant response model."""
    message: str = "Restaurant service is under development"


@router.get("/", response_model=RestaurantResponse)
async def restaurant_status() -> RestaurantResponse:
    """Get restaurant service status."""
    return RestaurantResponse()
