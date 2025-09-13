"""Shopping API endpoints for My Buddy application.

Provides product information, price comparisons, and shopping assistance.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ShoppingResponse(BaseModel):
    """Shopping response model."""
    message: str = "Shopping service is under development"


@router.get("/", response_model=ShoppingResponse)
async def shopping_status() -> ShoppingResponse:
    """Get shopping service status."""
    return ShoppingResponse()
