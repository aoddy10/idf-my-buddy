"""Authentication and user schemas for My Buddy API.

This module contains Pydantic schemas for authentication endpoints,
user management, and session handling.
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator

from app.schemas.common import (
    AccessibilityNeed,
    BaseResponse,
    Coordinates,
    DietaryRestriction,
    LanguageCode,
)


class UserRegistrationRequest(BaseModel):
    """User registration request schema."""

    # Required fields
    email: EmailStr = Field(description="User email address")
    password: str = Field(min_length=8, max_length=128, description="User password")
    confirm_password: str = Field(description="Password confirmation")

    # Personal information
    first_name: str = Field(min_length=1, max_length=50, description="First name")
    last_name: str = Field(min_length=1, max_length=50, description="Last name")
    date_of_birth: date | None = Field(None, description="Date of birth")

    # Preferences
    preferred_language: LanguageCode = Field(default=LanguageCode.EN, description="Preferred language")
    home_country: str | None = Field(None, max_length=100, description="Home country")
    home_city: str | None = Field(None, max_length=100, description="Home city")

    # Terms and privacy
    accept_terms: bool = Field(description="Accept terms of service")
    accept_privacy: bool = Field(description="Accept privacy policy")
    marketing_consent: bool = Field(default=False, description="Marketing communications consent")

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

    @validator('accept_terms', 'accept_privacy')
    def validate_required_consents(cls, v):
        """Validate required consents are given."""
        if not v:
            raise ValueError('Required consent must be accepted')
        return v


class UserLoginRequest(BaseModel):
    """User login request schema."""

    email: EmailStr = Field(description="User email address")
    password: str = Field(description="User password")
    remember_me: bool = Field(default=False, description="Remember user session")
    device_id: str | None = Field(None, description="Device identifier")
    device_name: str | None = Field(None, description="Human-readable device name")


class LoginResponse(BaseResponse):
    """Login response schema."""

    access_token: str = Field(description="JWT access token")
    refresh_token: str = Field(description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")
    user_id: UUID = Field(description="User identifier")
    session_id: UUID = Field(description="Session identifier")


class TokenRefreshRequest(BaseModel):
    """Token refresh request schema."""

    refresh_token: str = Field(description="Valid refresh token")


class TokenRefreshResponse(BaseResponse):
    """Token refresh response schema."""

    access_token: str = Field(description="New JWT access token")
    expires_in: int = Field(description="Token expiration time in seconds")


class LogoutRequest(BaseModel):
    """Logout request schema."""

    refresh_token: str | None = Field(None, description="Refresh token to invalidate")
    logout_all_devices: bool = Field(default=False, description="Logout from all devices")


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""

    email: EmailStr = Field(description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""

    reset_token: str = Field(description="Password reset token")
    new_password: str = Field(min_length=8, max_length=128, description="New password")
    confirm_password: str = Field(description="Password confirmation")

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if 'password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""

    current_password: str = Field(description="Current password")
    new_password: str = Field(min_length=8, max_length=128, description="New password")
    confirm_password: str = Field(description="Password confirmation")

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class UserProfileResponse(BaseModel):
    """User profile response schema."""

    user_id: UUID = Field(description="User identifier")
    email: str = Field(description="User email address")
    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")
    date_of_birth: date | None = Field(None, description="Date of birth")

    # Profile information
    preferred_language: LanguageCode = Field(description="Preferred language")
    secondary_languages: list[LanguageCode] = Field(default_factory=list, description="Secondary languages")
    home_country: str | None = Field(None, description="Home country")
    home_city: str | None = Field(None, description="Home city")

    # Preferences
    dietary_restrictions: list[DietaryRestriction] = Field(default_factory=list, description="Dietary restrictions")
    accessibility_needs: list[AccessibilityNeed] = Field(default_factory=list, description="Accessibility needs")

    # Profile metadata
    profile_complete: bool = Field(description="Profile completion status")
    email_verified: bool = Field(description="Email verification status")
    phone_verified: bool = Field(description="Phone verification status")

    # Timestamps
    created_at: datetime = Field(description="Account creation timestamp")
    updated_at: datetime = Field(description="Last profile update timestamp")
    last_login_at: datetime | None = Field(None, description="Last login timestamp")


class UserProfileUpdateRequest(BaseModel):
    """User profile update request schema."""

    first_name: str | None = Field(None, min_length=1, max_length=50, description="First name")
    last_name: str | None = Field(None, min_length=1, max_length=50, description="Last name")
    date_of_birth: date | None = Field(None, description="Date of birth")

    # Contact information
    phone: str | None = Field(None, description="Phone number")

    # Location
    home_country: str | None = Field(None, max_length=100, description="Home country")
    home_city: str | None = Field(None, max_length=100, description="Home city")
    timezone: str | None = Field(None, description="User timezone")

    # Language preferences
    preferred_language: LanguageCode | None = Field(None, description="Preferred language")
    secondary_languages: list[LanguageCode] | None = Field(None, description="Secondary languages")

    # Dietary and accessibility
    dietary_restrictions: list[DietaryRestriction] | None = Field(None, description="Dietary restrictions")
    accessibility_needs: list[AccessibilityNeed] | None = Field(None, description="Accessibility needs")

    # Travel preferences
    travel_style: str | None = Field(None, description="Travel style preference")
    budget_preference: str | None = Field(None, description="Budget preference")

    # Emergency contact
    emergency_contact_name: str | None = Field(None, description="Emergency contact name")
    emergency_contact_phone: str | None = Field(None, description="Emergency contact phone")
    emergency_contact_relationship: str | None = Field(None, description="Emergency contact relationship")


class UserPreferencesResponse(BaseModel):
    """User preferences response schema."""

    # Language preferences
    preferred_language: LanguageCode = Field(description="Primary language")
    secondary_languages: list[LanguageCode] = Field(description="Secondary languages")
    auto_translate: bool = Field(description="Automatic translation enabled")

    # Dietary preferences
    dietary_restrictions: list[DietaryRestriction] = Field(description="Dietary restrictions")
    cuisine_preferences: list[str] = Field(description="Preferred cuisines")
    allergies: list[str] = Field(description="Food allergies")

    # Accessibility preferences
    accessibility_needs: list[AccessibilityNeed] = Field(description="Accessibility requirements")
    mobility_assistance: bool = Field(description="Mobility assistance required")

    # Travel preferences
    travel_style: str | None = Field(None, description="Travel style")
    budget_preference: str | None = Field(None, description="Budget preference")
    transport_preferences: list[str] = Field(description="Preferred transport modes")
    accommodation_preferences: list[str] = Field(description="Accommodation preferences")

    # Privacy preferences
    location_sharing: bool = Field(description="Location sharing enabled")
    data_collection_consent: bool = Field(description="Data collection consent")
    analytics_consent: bool = Field(description="Analytics consent")

    # Notification preferences
    push_notifications: bool = Field(description="Push notifications enabled")
    email_notifications: bool = Field(description="Email notifications enabled")
    safety_alerts: bool = Field(description="Safety alerts enabled")

    # AI preferences
    voice_commands: bool = Field(description="Voice commands enabled")
    personalization: bool = Field(description="AI personalization enabled")
    conversation_memory: bool = Field(description="Conversation memory enabled")


class EmailVerificationRequest(BaseModel):
    """Email verification request schema."""

    verification_token: str = Field(description="Email verification token")


class PhoneVerificationRequest(BaseModel):
    """Phone verification request schema."""

    phone: str = Field(description="Phone number to verify")


class PhoneVerificationConfirm(BaseModel):
    """Phone verification confirmation schema."""

    phone: str = Field(description="Phone number being verified")
    verification_code: str = Field(min_length=4, max_length=8, description="Verification code")


class UserSessionInfo(BaseModel):
    """User session information schema."""

    session_id: UUID = Field(description="Session identifier")
    device_id: str | None = Field(None, description="Device identifier")
    device_name: str | None = Field(None, description="Device name")
    device_type: str | None = Field(None, description="Device type")

    # Location context
    current_location: Coordinates | None = Field(None, description="Current location")
    current_city: str | None = Field(None, description="Current city")
    current_country: str | None = Field(None, description="Current country")

    # Session metadata
    created_at: datetime = Field(description="Session start time")
    last_activity_at: datetime = Field(description="Last activity time")
    expires_at: datetime | None = Field(None, description="Session expiration time")
    is_active: bool = Field(description="Session is active")

    # Usage statistics
    interaction_count: int = Field(description="Number of interactions")
    features_used: dict[str, int] = Field(description="Feature usage count")


class UserAccountDeletion(BaseModel):
    """User account deletion request schema."""

    password: str = Field(description="Current password for verification")
    deletion_reason: str | None = Field(None, description="Reason for account deletion")
    feedback: str | None = Field(None, description="Additional feedback")

    # Data handling preferences
    delete_all_data: bool = Field(default=True, description="Delete all user data")
    export_data_first: bool = Field(default=False, description="Export data before deletion")


class DataExportRequest(BaseModel):
    """User data export request schema."""

    export_types: list[str] = Field(description="Types of data to export")
    format: str = Field(default="json", description="Export format")
    include_media: bool = Field(default=False, description="Include media files")


class TwoFactorSetupRequest(BaseModel):
    """Two-factor authentication setup request."""

    method: str = Field(description="2FA method (totp, sms)")
    phone: str | None = Field(None, description="Phone number for SMS 2FA")


class TwoFactorVerifyRequest(BaseModel):
    """Two-factor authentication verification."""

    code: str = Field(min_length=6, max_length=8, description="2FA verification code")
    remember_device: bool = Field(default=False, description="Remember this device")
