"""Common schemas for My Buddy API.

This module contains shared Pydantic schemas used across multiple API endpoints,
including base models, common response types, and utility schemas.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, validator


class BaseResponse(BaseModel):
    """Base response schema for all API responses."""

    success: bool = Field(description="Indicates if the request was successful")
    message: str | None = Field(None, description="Human-readable response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat() if v else None
        }
    )


class ErrorResponse(BaseResponse):
    """Error response schema."""

    success: bool = Field(default=False, description="Always false for error responses")
    error_code: str | None = Field(None, description="Machine-readable error code")
    error_details: dict[str, Any] | None = Field(None, description="Additional error context")


class SuccessResponse(BaseResponse):
    """Success response schema with optional data payload."""

    success: bool = Field(default=True, description="Always true for success responses")
    data: dict[str, Any] | None = Field(None, description="Response payload data")


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    page: int = Field(ge=1, description="Current page number")
    per_page: int = Field(ge=1, le=100, description="Items per page")
    total_items: int = Field(ge=0, description="Total number of items")
    total_pages: int = Field(ge=1, description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_prev: bool = Field(description="Whether there are previous pages")


class PaginatedResponse(BaseResponse):
    """Paginated response schema."""

    success: bool = Field(default=True)
    data: list[dict[str, Any]] = Field(default_factory=list, description="List of items")
    pagination: PaginationMeta = Field(description="Pagination metadata")


# Location and Coordinate Schemas
class Coordinates(BaseModel):
    """Geographic coordinates schema."""

    latitude: float = Field(ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(ge=-180, le=180, description="Longitude coordinate")
    accuracy: float | None = Field(None, ge=0, description="Location accuracy in meters")

    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v):
        """Validate coordinate precision."""
        if v is not None:
            # Limit to 7 decimal places (approximately 1cm precision)
            return round(float(v), 7)
        return v


class Address(BaseModel):
    """Address information schema."""

    street_number: str | None = Field(None, description="Street number")
    street_name: str | None = Field(None, description="Street name")
    city: str | None = Field(None, description="City name")
    state: str | None = Field(None, description="State or province")
    country: str | None = Field(None, description="Country name")
    country_code: str | None = Field(None, min_length=2, max_length=3, description="ISO country code")
    postal_code: str | None = Field(None, description="Postal/ZIP code")
    formatted_address: str | None = Field(None, description="Complete formatted address")


class Location(BaseModel):
    """Location schema combining coordinates and address."""

    coordinates: Coordinates = Field(description="Geographic coordinates")
    address: Address | None = Field(None, description="Address information")
    place_id: str | None = Field(None, description="External place identifier")
    timezone: str | None = Field(None, description="Timezone identifier")


# Language and Translation Schemas
class LanguageCode(str, Enum):
    """Supported language codes (ISO 639-1)."""
    EN = "en"  # English
    TH = "th"  # Thai
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    RU = "ru"  # Russian
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    ZH = "zh"  # Chinese (Simplified)
    AR = "ar"  # Arabic
    HI = "hi"  # Hindi


class TranslationRequest(BaseModel):
    """Translation request schema."""

    text: str = Field(min_length=1, max_length=5000, description="Text to translate")
    source_language: LanguageCode | None = Field(None, description="Source language (auto-detect if None)")
    target_language: LanguageCode = Field(description="Target language")
    context: str | None = Field(None, max_length=500, description="Context for better translation")


class TranslationResponse(BaseModel):
    """Translation response schema."""

    original_text: str = Field(description="Original input text")
    translated_text: str = Field(description="Translated text")
    source_language: LanguageCode = Field(description="Detected/specified source language")
    target_language: LanguageCode = Field(description="Target language")
    confidence_score: float = Field(ge=0, le=1, description="Translation confidence (0-1)")
    processing_time_ms: int | None = Field(None, description="Processing time in milliseconds")


# File Upload and Media Schemas
class FileType(str, Enum):
    """Supported file types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class FileUpload(BaseModel):
    """File upload metadata schema."""

    filename: str = Field(description="Original filename")
    file_type: FileType = Field(description="Type of file")
    file_size: int = Field(ge=0, description="File size in bytes")
    mime_type: str = Field(description="MIME type of the file")
    upload_url: str | None = Field(None, description="Temporary upload URL")


class ProcessedMedia(BaseModel):
    """Processed media result schema."""

    file_id: UUID = Field(description="Unique file identifier")
    original_filename: str = Field(description="Original filename")
    file_type: FileType = Field(description="Type of file")
    processing_status: str = Field(description="Processing status")
    result: dict[str, Any] | None = Field(None, description="Processing result data")
    error_message: str | None = Field(None, description="Error message if processing failed")


# Voice and Audio Schemas
class VoiceInput(BaseModel):
    """Voice input request schema."""

    audio_format: str = Field(description="Audio format (wav, mp3, m4a, etc.)")
    sample_rate: int | None = Field(None, description="Audio sample rate")
    duration_seconds: float | None = Field(None, ge=0, description="Audio duration")
    language_hint: LanguageCode | None = Field(None, description="Language hint for ASR")


class VoiceResponse(BaseModel):
    """Voice processing response schema."""

    transcribed_text: str = Field(description="Transcribed speech text")
    detected_language: LanguageCode = Field(description="Detected language")
    confidence_score: float = Field(ge=0, le=1, description="Transcription confidence")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    audio_quality: str | None = Field(None, description="Audio quality assessment")


# User Preference Schemas
class DietaryRestriction(str, Enum):
    """Dietary restrictions."""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"
    GLUTEN_FREE = "gluten_free"
    LACTOSE_FREE = "lactose_free"
    NUT_FREE = "nut_free"
    SHELLFISH_FREE = "shellfish_free"
    DIABETIC = "diabetic"
    LOW_SODIUM = "low_sodium"


class AccessibilityNeed(str, Enum):
    """Accessibility needs."""
    WHEELCHAIR = "wheelchair"
    VISUAL_IMPAIRMENT = "visual_impairment"
    HEARING_IMPAIRMENT = "hearing_impairment"
    MOBILITY_ASSISTANCE = "mobility_assistance"
    COGNITIVE_ASSISTANCE = "cognitive_assistance"


class UserPreferences(BaseModel):
    """User preferences schema."""

    dietary_restrictions: list[DietaryRestriction] = Field(default_factory=list)
    accessibility_needs: list[AccessibilityNeed] = Field(default_factory=list)
    preferred_languages: list[LanguageCode] = Field(default_factory=list)
    budget_range: str | None = Field(None, description="Budget preference (low, medium, high)")
    travel_style: str | None = Field(None, description="Travel style preference")


# Validation Schemas
class ValidationError(BaseModel):
    """Validation error detail schema."""

    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Validation error message")
    invalid_value: Any | None = Field(None, description="The invalid value provided")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details."""

    validation_errors: list[ValidationError] = Field(description="List of validation errors")


# Health Check Schema
class HealthCheck(BaseModel):
    """Health check response schema."""

    status: str = Field(description="Service status")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database: bool = Field(description="Database connectivity status")
    ai_services: dict[str, bool] = Field(description="AI service availability")
    external_services: dict[str, bool] = Field(description="External service status")


# Rate Limiting Schema
class RateLimitInfo(BaseModel):
    """Rate limit information schema."""

    limit: int = Field(description="Rate limit threshold")
    remaining: int = Field(description="Remaining requests")
    reset_time: datetime = Field(description="When the rate limit resets")
    retry_after: int | None = Field(None, description="Seconds to wait before retry")
