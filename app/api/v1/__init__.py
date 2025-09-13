"""API v1 package marker."""

from fastapi import APIRouter

# Import all v1 routers
from .auth import router as auth_router
from .navigation import router as navigation_router  
from .restaurant import router as restaurant_router
from .shopping import router as shopping_router
from .safety import router as safety_router
from .voice import router as voice_router

# Create main v1 router
api_router = APIRouter(prefix="/v1")

# Include all feature routers
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(navigation_router, prefix="/navigation", tags=["Navigation"])
api_router.include_router(restaurant_router, prefix="/restaurants", tags=["Restaurants"])
api_router.include_router(shopping_router, prefix="/shopping", tags=["Shopping"])
api_router.include_router(safety_router, prefix="/safety", tags=["Safety"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice"])

__all__ = ["api_router"]
