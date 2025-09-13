"""Safety API endpoints for My Buddy application.

Provides emergency assistance, safety tips, and alert services.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SafetyResponse(BaseModel):
    """Safety response model."""
    message: str = "Safety service is under development"


@router.get("/", response_model=SafetyResponse)
async def safety_status() -> SafetyResponse:
    """Get safety service status."""
    return SafetyResponse()
