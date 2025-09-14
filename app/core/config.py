"""Configuration management with Pydantic Settings.

This module provides environment-based configuration for the My Buddy application,
supporting development, testing, and production environments with proper validation.
"""

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WhisperSettings(BaseModel):
    """Whisper ASR configuration settings."""

    model_size: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] = Field(
        default="base",
        description="Whisper model size for speech recognition"
    )
    use_local: bool = Field(
        default=True,
        description="Use local Whisper model (vs OpenAI API)"
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for model inference"
    )
    compute_type: Literal["auto", "int8", "int16", "float16", "float32"] = Field(
        default="auto",
        description="Compute type for faster inference"
    )

    # Performance settings
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam size for decoding")
    patience: float = Field(default=1.0, ge=0.5, le=2.0, description="Patience for beam search")
    length_penalty: float = Field(default=1.0, ge=0.5, le=2.0, description="Length penalty")
    repetition_penalty: float = Field(default=1.0, ge=0.8, le=1.5, description="Repetition penalty")

    # Audio processing
    vad_filter: bool = Field(default=True, description="Enable voice activity detection")
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="VAD threshold")

    # Output options
    word_timestamps: bool = Field(default=False, description="Include word-level timestamps")
    hallucination_silence_threshold: float | None = Field(
        default=None, ge=0.0, le=5.0, description="Silence threshold to prevent hallucinations"
    )


class TTSSettings(BaseModel):
    """Text-to-speech configuration settings."""

    # Voice settings
    default_voice: str = Field(default="neutral", description="Default voice identifier")
    default_language: str = Field(default="en", description="Default language code")
    default_gender: Literal["male", "female", "neutral"] = Field(default="neutral", description="Default voice gender")

    # Speech parameters
    speech_rate: float = Field(default=1.0, ge=0.25, le=4.0, description="Default speech rate multiplier")
    pitch_adjustment: float = Field(default=0.0, ge=-20.0, le=20.0, description="Pitch adjustment in semitones")
    volume_gain: float = Field(default=0.0, ge=-20.0, le=20.0, description="Volume gain in dB")

    # Audio output
    sample_rate: int = Field(default=22050, description="Output audio sample rate")
    audio_format: Literal["wav", "mp3", "ogg", "aac"] = Field(default="wav", description="Default audio format")
    audio_quality: Literal["low", "standard", "high", "premium"] = Field(default="standard", description="Audio quality level")

    # Engine settings
    enable_openai: bool = Field(default=True, description="Enable OpenAI TTS")
    enable_speechbrain: bool = Field(default=False, description="Enable SpeechBrain TTS")
    enable_pyttsx3: bool = Field(default=True, description="Enable pyttsx3 TTS")
    enable_gtts: bool = Field(default=True, description="Enable Google TTS")

    # Performance
    streaming_chunk_size: int = Field(default=1024, description="Chunk size for streaming TTS")
    cache_enabled: bool = Field(default=True, description="Enable TTS response caching")
    max_text_length: int = Field(default=5000, description="Maximum text length for TTS")


class AudioProcessingSettings(BaseModel):
    """Audio processing configuration settings."""

    # Input constraints
    max_file_size: int = Field(default=52428800, description="Maximum audio file size (50MB)")
    max_duration: int = Field(default=300, description="Maximum audio duration in seconds")
    supported_formats: list[str] = Field(
        default=["wav", "mp3", "m4a", "aac", "ogg", "flac", "amr", "webm"],
        description="Supported audio formats"
    )

    # Processing settings
    sample_rate: int = Field(default=16000, description="Processing sample rate")
    channels: int = Field(default=1, description="Audio channels (1=mono, 2=stereo)")

    # Audio enhancement
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    normalize_audio: bool = Field(default=True, description="Normalize audio levels")
    auto_gain_control: bool = Field(default=True, description="Enable auto gain control")

    # Quality levels
    quality_settings: dict[str, dict[str, Any]] = Field(
        default={
            "fast": {"sample_rate": 8000, "quality": "low", "realtime_factor": 0.1},
            "balanced": {"sample_rate": 16000, "quality": "standard", "realtime_factor": 0.3},
            "accurate": {"sample_rate": 22050, "quality": "high", "realtime_factor": 0.8},
            "premium": {"sample_rate": 48000, "quality": "premium", "realtime_factor": 1.5}
        },
        description="Quality level configurations"
    )


class VoicePipelineSettings(BaseModel):
    """Voice pipeline configuration settings."""

    # Session settings
    session_timeout: int = Field(default=3600, description="Voice session timeout in seconds")
    max_conversation_history: int = Field(default=50, description="Max conversation exchanges to keep")

    # Performance thresholds
    max_processing_time: float = Field(default=10.0, description="Maximum processing time per request")
    realtime_factor_threshold: float = Field(default=0.8, description="RTF threshold for quality adjustment")

    # Fallback settings
    enable_cloud_fallback: bool = Field(default=True, description="Enable cloud API fallback")
    fallback_timeout: float = Field(default=5.0, description="Timeout before fallback activation")

    # WebSocket settings
    websocket_max_connections: int = Field(default=100, description="Max concurrent WebSocket connections")
    websocket_message_size: int = Field(default=1048576, description="Max WebSocket message size (1MB)")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    metrics_retention_hours: int = Field(default=24, description="Metrics retention period")


class NavigationSettings(BaseModel):
    """Navigation and location services configuration settings."""

    # Google Maps settings
    use_google_maps: bool = Field(default=True, description="Use Google Maps API for routing")
    google_maps_api_key: str | None = Field(default=None, description="Google Maps API key")
    
    # OpenStreetMap settings
    use_openstreetmap: bool = Field(default=True, description="Use OpenStreetMap as fallback")
    osm_base_url: str = Field(default="https://api.openstreetmap.org", description="OpenStreetMap API base URL")
    overpass_url: str = Field(default="https://overpass-api.de/api/interpreter", description="Overpass API URL for OSM queries")
    
    # Location settings
    location_accuracy_meters: float = Field(default=10.0, ge=1.0, le=100.0, description="Desired GPS accuracy in meters")
    location_timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0, description="Location request timeout")
    enable_background_location: bool = Field(default=False, description="Enable background location updates")
    
    # Points of Interest (POI) settings
    poi_search_radius_km: float = Field(default=1.0, ge=0.1, le=50.0, description="POI search radius in kilometers")
    max_poi_results: int = Field(default=20, ge=1, le=100, description="Maximum POI results to return")
    poi_categories: list[str] = Field(
        default=["restaurant", "hotel", "gas_station", "atm", "hospital", "pharmacy", "tourist_attraction"],
        description="Default POI categories to search"
    )
    
    # Routing settings
    default_travel_mode: Literal["walking", "driving", "bicycling", "transit"] = Field(
        default="walking", description="Default travel mode for route calculation"
    )
    avoid_tolls: bool = Field(default=False, description="Avoid toll roads by default")
    avoid_highways: bool = Field(default=False, description="Avoid highways by default")
    avoid_ferries: bool = Field(default=False, description="Avoid ferries by default")
    
    # Performance settings
    route_calculation_timeout: float = Field(default=10.0, ge=2.0, le=30.0, description="Route calculation timeout")
    max_waypoints: int = Field(default=8, ge=2, le=25, description="Maximum waypoints per route")
    
    # Caching settings
    enable_route_caching: bool = Field(default=True, description="Cache calculated routes")
    route_cache_ttl_minutes: int = Field(default=30, ge=5, le=1440, description="Route cache TTL in minutes")
    enable_poi_caching: bool = Field(default=True, description="Cache POI search results")
    poi_cache_ttl_minutes: int = Field(default=60, ge=10, le=1440, description="POI cache TTL in minutes")
    
    # Offline settings
    enable_offline_maps: bool = Field(default=True, description="Enable offline map functionality")
    offline_map_storage_mb: int = Field(default=500, ge=100, le=5000, description="Offline map storage limit in MB")
    
    # Voice navigation settings
    enable_voice_navigation: bool = Field(default=True, description="Enable voice-guided navigation")
    voice_instruction_language: str = Field(default="en", description="Default language for voice instructions")
    announce_distance_units: Literal["metric", "imperial"] = Field(default="metric", description="Distance units for announcements")
    
    # Privacy settings
    store_location_history: bool = Field(default=False, description="Store user location history")
    anonymize_location_data: bool = Field(default=True, description="Anonymize location data before storage")
    location_data_retention_days: int = Field(default=7, ge=1, le=365, description="Location data retention period")


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
    google_maps_api_key: str | None = Field(
        default=None,
        description="Google Maps API key for location services"
    )
    mapbox_access_token: str | None = Field(
        default=None,
        description="Mapbox access token for offline maps"
    )
    azure_speech_key: str | None = Field(
        default=None,
        description="Azure Speech Services key"
    )
    azure_speech_region: str | None = Field(
        default=None,
        description="Azure Speech Services region"
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for Whisper and other services"
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
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    allowed_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    allowed_headers: list[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )

    # Voice Service Configuration
    whisper: WhisperSettings = Field(default_factory=WhisperSettings, description="Whisper ASR settings")
    tts: TTSSettings = Field(default_factory=TTSSettings, description="Text-to-speech settings")
    audio: AudioProcessingSettings = Field(default_factory=AudioProcessingSettings, description="Audio processing settings")
    voice_pipeline: VoicePipelineSettings = Field(default_factory=VoicePipelineSettings, description="Voice pipeline settings")
    
    # Navigation Service Configuration
    navigation: NavigationSettings = Field(default_factory=NavigationSettings, description="Navigation and location services settings")

    # Legacy AI Model Configuration (for backward compatibility)
    nllb_model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="NLLB model for machine translation"
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
    # Legacy audio settings (for backward compatibility)
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate for processing (use audio.sample_rate for new code)"
    )
    max_audio_duration: int = Field(
        default=60,
        description="Maximum audio duration in seconds (use audio.max_duration for new code)"
    )
    max_audio_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum audio file size in bytes (use audio.max_file_size for new code)"
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

    @validator("whisper")
    def validate_whisper_settings(cls, v: WhisperSettings) -> WhisperSettings:
        """Validate Whisper configuration."""
        # Ensure compatible settings
        if v.model_size in ["large", "large-v2", "large-v3"] and v.device == "cpu":
            import warnings
            warnings.warn("Large Whisper models may be slow on CPU. Consider using GPU or smaller model.", stacklevel=2)
        return v

    @validator("tts")
    def validate_tts_settings(cls, v: TTSSettings) -> TTSSettings:
        """Validate TTS configuration."""
        # Ensure at least one TTS engine is enabled
        engines_enabled = [v.enable_openai, v.enable_speechbrain, v.enable_pyttsx3, v.enable_gtts]
        if not any(engines_enabled):
            raise ValueError("At least one TTS engine must be enabled")
        return v

    @validator("audio")
    def validate_audio_settings(cls, v: AudioProcessingSettings) -> AudioProcessingSettings:
        """Validate audio processing configuration."""
        # Validate max file size (50MB limit)
        max_allowed = 52428800  # 50MB
        if v.max_file_size > max_allowed:
            raise ValueError(f"max_file_size cannot exceed {max_allowed} bytes (50MB)")
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

    # Voice service convenience properties
    @property
    def whisper_use_local(self) -> bool:
        """Get Whisper local usage setting (backward compatibility)."""
        return self.whisper.use_local

    @property
    def whisper_model_size(self) -> str:
        """Get Whisper model size (backward compatibility)."""
        return self.whisper.model_size

    @property
    def tts_default_voice(self) -> str:
        """Get default TTS voice (backward compatibility)."""
        return self.tts.default_voice

    @property
    def tts_speech_rate(self) -> float:
        """Get TTS speech rate (backward compatibility)."""
        return self.tts.speech_rate

    @property
    def speechbrain_enabled(self) -> bool:
        """Get SpeechBrain enabled status (backward compatibility)."""
        return self.tts.enable_speechbrain

    @property
    def enable_cloud_fallback(self) -> bool:
        """Get cloud fallback setting (backward compatibility)."""
        return self.voice_pipeline.enable_cloud_fallback


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Cached application settings instance.
    """
    return Settings()


# Global settings instance
settings = get_settings()
