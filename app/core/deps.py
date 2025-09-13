"""Database and external service dependencies for My Buddy application.

This module provides dependency injection functions for database sessions,
external services, and other shared resources used across the application.
"""

import logging
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.config import get_settings
from app.core.database import get_session
from app.models.entities.user import User
from app.models.entities.session import Session

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)

settings = get_settings()


# Database Dependencies
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async for session in get_session():
        yield session


# Authentication Dependencies
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> Optional[User]:
    """Get current user from JWT token (optional, returns None if not authenticated)."""
    if not credentials:
        return None
    
    try:
        # Here you would decode and validate the JWT token
        # For now, this is a placeholder implementation
        token = credentials.credentials
        
        # TODO: Implement JWT token validation
        # user_id = decode_jwt_token(token)
        # For now, assume the token contains the user_id directly
        
        logger.warning("JWT token validation not implemented - using placeholder")
        return None
        
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> User:
    """Get current user from JWT token (required)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await get_current_user_optional(credentials, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_session(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
) -> Optional[Session]:
    """Get current active session for the user."""
    try:
        statement = select(Session).where(
            Session.user_id == user.id,
            Session.is_active == True
        ).order_by(Session.last_activity_at.desc())
        
        result = await db.exec(statement)
        session = result.first()
        
        if session and session.is_expired:
            # Session is expired, mark as inactive
            session.is_active = False
            session.end_session()
            db.add(session)
            await db.commit()
            return None
        
        return session
        
    except Exception as e:
        logger.error(f"Failed to get current session: {e}")
        return None


# Service Dependencies
class AIServiceDependencies:
    """Dependencies for AI services."""
    
    def __init__(self):
        self._whisper_model = None
        self._nllb_model = None
        self._ocr_service = None
        self._tts_service = None
    
    async def get_whisper_service(self):
        """Get Whisper ASR service."""
        if not self._whisper_model:
            # Lazy loading of Whisper model
            logger.info("Loading Whisper ASR model")
            # TODO: Implement Whisper model loading
            pass
        return self._whisper_model
    
    async def get_nllb_service(self):
        """Get NLLB translation service."""
        if not self._nllb_model:
            # Lazy loading of NLLB model
            logger.info("Loading NLLB translation model")
            # TODO: Implement NLLB model loading
            pass
        return self._nllb_model
    
    async def get_ocr_service(self):
        """Get OCR service."""
        if not self._ocr_service:
            logger.info("Initializing OCR service")
            # TODO: Implement OCR service initialization
            pass
        return self._ocr_service
    
    async def get_tts_service(self):
        """Get TTS service."""
        if not self._tts_service:
            logger.info("Initializing TTS service")
            # TODO: Implement TTS service initialization
            pass
        return self._tts_service


# Global AI service dependencies instance
ai_services = AIServiceDependencies()


async def get_whisper_service():
    """Dependency for Whisper ASR service."""
    return await ai_services.get_whisper_service()


async def get_nllb_service():
    """Dependency for NLLB translation service."""
    return await ai_services.get_nllb_service()


async def get_ocr_service():
    """Dependency for OCR service."""
    return await ai_services.get_ocr_service()


async def get_tts_service():
    """Dependency for TTS service."""
    return await ai_services.get_tts_service()


# External API Dependencies
class ExternalServiceDependencies:
    """Dependencies for external services."""
    
    def __init__(self):
        self._maps_client = None
        self._weather_client = None
        self._places_client = None
    
    async def get_maps_service(self):
        """Get maps/geocoding service."""
        if not self._maps_client:
            logger.info("Initializing maps service")
            # TODO: Implement maps service (Google Maps, OpenStreetMap, etc.)
            pass
        return self._maps_client
    
    async def get_weather_service(self):
        """Get weather service."""
        if not self._weather_client:
            logger.info("Initializing weather service")
            # TODO: Implement weather service (OpenWeatherMap, etc.)
            pass
        return self._weather_client
    
    async def get_places_service(self):
        """Get places/POI service."""
        if not self._places_client:
            logger.info("Initializing places service")
            # TODO: Implement places service (Google Places, Foursquare, etc.)
            pass
        return self._places_client


# Global external service dependencies instance
external_services = ExternalServiceDependencies()


async def get_maps_service():
    """Dependency for maps service."""
    return await external_services.get_maps_service()


async def get_weather_service():
    """Dependency for weather service."""
    return await external_services.get_weather_service()


async def get_places_service():
    """Dependency for places service."""
    return await external_services.get_places_service()


# Validation Dependencies
def validate_uuid(uuid_str: str) -> UUID:
    """Validate and convert string to UUID."""
    try:
        return UUID(uuid_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )


def validate_language_code(language_code: str) -> str:
    """Validate language code format."""
    # Basic validation for language codes (ISO 639-1 or locale codes)
    if not language_code or len(language_code) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid language code"
        )
    return language_code.lower()


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    """Validate coordinate values."""
    if not (-90 <= latitude <= 90):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid latitude value"
        )
    if not (-180 <= longitude <= 180):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid longitude value"
        )
    return latitude, longitude


# Cache Dependencies (for future implementation)
class CacheDependencies:
    """Dependencies for caching services."""
    
    def __init__(self):
        self._redis_client = None
        self._memory_cache = {}
    
    async def get_cache_service(self):
        """Get caching service (Redis or in-memory)."""
        if settings.REDIS_URL and not self._redis_client:
            # TODO: Initialize Redis client
            logger.info("Initializing Redis cache")
            pass
        
        # Fallback to in-memory cache
        return self._memory_cache


cache_services = CacheDependencies()


async def get_cache_service():
    """Dependency for cache service."""
    return await cache_services.get_cache_service()


# Rate Limiting Dependencies (for future implementation)
async def rate_limit_check(
    user: Optional[User] = Depends(get_current_user_optional)
) -> bool:
    """Check rate limits for the current user/IP."""
    # TODO: Implement rate limiting logic
    # For now, always allow requests
    return True
