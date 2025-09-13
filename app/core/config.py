"""Configuration management with Pydantic Settings.

This module provides environment-based configuration for the My Buddy application,
supporting development, testing, and production environments with proper validation.
"""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment-based configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Environment
    app_env: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Application environment"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/mybuddy_dev",
        description="Database connection URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching and sessions"
    )
    
    # External API Keys
    google_maps_api_key: Optional[str] = Field(
        default=None,
        description="Google Maps API key for location services"
    )
    mapbox_access_token: Optional[str] = Field(
        default=None,
        description="Mapbox access token for offline maps"
    )
    azure_speech_key: Optional[str] = Field(
        default=None,
        description="Azure Speech Services key"
    )
    azure_speech_region: Optional[str] = Field(
        default=None,
        description="Azure Speech Services region"
    )
    
    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for JWT and session encryption"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="JWT access token expiration time in minutes"
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    # AI Model Configuration
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base",
        description="Whisper model size for speech recognition"
    )
    nllb_model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="NLLB model for machine translation"
    )
    enable_cloud_fallback: bool = Field(
        default=True,
        description="Enable cloud API fallback for AI services"
    )
    model_cache_dir: str = Field(
        default="./models",
        description="Directory for caching AI models"
    )
    
    # Performance Settings
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker processes"
    )
    request_timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds"
    )
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate for processing"
    )
    max_audio_duration: int = Field(
        default=60,
        description="Maximum audio duration in seconds"
    )
    max_image_size: int = Field(
        default=5242880,  # 5MB
        description="Maximum image size in bytes"
    )
    
    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )
    enable_request_logging: bool = Field(
        default=True,
        description="Enable HTTP request logging"
    )
    
    # Feature Flags
    enable_offline_mode: bool = Field(
        default=True,
        description="Enable offline mode capabilities"
    )
    enable_telemetry: bool = Field(
        default=False,
        description="Enable telemetry collection"
    )
    enable_debug_endpoints: bool = Field(
        default=False,
        description="Enable debug API endpoints"
    )
    enable_safety_features: bool = Field(
        default=True,
        description="Enable safety and emergency features"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit: requests per window"
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    # Health Check Configuration
    health_check_timeout: int = Field(
        default=5,
        description="Health check timeout in seconds"
    )
    database_pool_size: int = Field(
        default=10,
        description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=20,
        description="Database max overflow connections"
    )
    
    # Development Settings
    reload: bool = Field(
        default=False,
        description="Enable auto-reload in development"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    enable_docs: bool = Field(
        default=True,
        description="Enable OpenAPI documentation endpoints"
    )
    
    @validator("app_env")
    def validate_app_env(cls, v: str) -> str:
        """Validate application environment."""
        if v not in ["development", "testing", "production"]:
            raise ValueError("app_env must be one of: development, testing, production")
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v: str, values: dict) -> str:
        """Validate secret key is not default in production."""
        if values.get("app_env") == "production" and v == "dev-secret-key-change-in-production":
            raise ValueError("Must set a secure secret_key in production")
        if len(v) < 32:
            raise ValueError("secret_key must be at least 32 characters long")
        return v
    
    @validator("database_url")
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        allowed_schemes = [
            "postgresql://", "postgresql+asyncpg://",
            "sqlite://", "sqlite+aiosqlite:///"
        ]
        if not any(v.startswith(scheme) for scheme in allowed_schemes):
            raise ValueError("database_url must be a PostgreSQL or SQLite URL")
        return v
    
    @validator("max_audio_duration")
    def validate_max_audio_duration(cls, v: int) -> int:
        """Validate audio duration limits."""
        if v > 300:  # 5 minutes max
            raise ValueError("max_audio_duration cannot exceed 300 seconds")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.app_env == "testing"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"
    
    @property
    def database_echo(self) -> bool:
        """Enable SQLAlchemy echo in development."""
        return self.is_development and self.debug


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings: Cached application settings instance.
    """
    return Settings()


# Global settings instance
settings = get_settings()
