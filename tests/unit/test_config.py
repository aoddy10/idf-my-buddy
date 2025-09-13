"""Unit tests for core configuration.

Tests the Settings class and configuration loading functionality.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from app.core.config import Settings, get_settings


class TestSettings:
    """Test Settings configuration class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.APP_NAME == "IDF My Buddy"
        assert settings.VERSION == "1.0.0"
        assert settings.ENVIRONMENT == "development"
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "INFO"
    
    def test_database_url_construction(self):
        """Test database URL construction."""
        settings = Settings(
            DB_HOST="localhost",
            DB_PORT=5432,
            DB_NAME="testdb",
            DB_USER="testuser",
            DB_PASSWORD="testpass"
        )
        
        expected_url = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
        assert settings.DATABASE_URL == expected_url
    
    def test_sqlite_database_url(self):
        """Test SQLite database URL."""
        settings = Settings(DATABASE_URL="sqlite+aiosqlite:///./test.db")
        assert "sqlite+aiosqlite" in settings.DATABASE_URL
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'DEBUG': 'false',
        'LOG_LEVEL': 'WARNING'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = Settings()
        
        assert settings.ENVIRONMENT == "production"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "WARNING"
    
    def test_api_keys_configuration(self):
        """Test API keys configuration."""
        settings = Settings(
            GOOGLE_MAPS_API_KEY="test_google_key",
            OPENWEATHER_API_KEY="test_weather_key",
            GOOGLE_PLACES_API_KEY="test_places_key"
        )
        
        assert settings.GOOGLE_MAPS_API_KEY == "test_google_key"
        assert settings.OPENWEATHER_API_KEY == "test_weather_key"
        assert settings.GOOGLE_PLACES_API_KEY == "test_places_key"
    
    def test_ai_service_settings(self):
        """Test AI service configuration."""
        settings = Settings(
            USE_GPU_FOR_OCR=True,
            USE_GPU_FOR_TTS=False,
            WHISPER_MODEL_SIZE="base",
            NLLB_MODEL_SIZE="600M"
        )
        
        assert settings.USE_GPU_FOR_OCR is True
        assert settings.USE_GPU_FOR_TTS is False
        assert settings.WHISPER_MODEL_SIZE == "base"
        assert settings.NLLB_MODEL_SIZE == "600M"
    
    def test_caching_settings(self):
        """Test caching configuration."""
        settings = Settings(
            ENABLE_CACHING=True,
            CACHE_TTL=3600,
            MODEL_CACHE_DIR="/tmp/models"
        )
        
        assert settings.ENABLE_CACHING is True
        assert settings.CACHE_TTL == 3600
        assert settings.MODEL_CACHE_DIR == "/tmp/models"
    
    def test_cors_settings(self):
        """Test CORS configuration."""
        settings = Settings(
            ALLOWED_ORIGINS=["http://localhost:3000", "https://example.com"]
        )
        
        assert len(settings.ALLOWED_ORIGINS) == 2
        assert "http://localhost:3000" in settings.ALLOWED_ORIGINS
    
    def test_rate_limiting_settings(self):
        """Test rate limiting configuration."""
        settings = Settings(
            RATE_LIMIT_ENABLED=True,
            RATE_LIMIT_REQUESTS=100,
            RATE_LIMIT_WINDOW=60
        )
        
        assert settings.RATE_LIMIT_ENABLED is True
        assert settings.RATE_LIMIT_REQUESTS == 100
        assert settings.RATE_LIMIT_WINDOW == 60
    
    def test_security_settings(self):
        """Test security configuration."""
        settings = Settings(
            SECRET_KEY="test_secret_key",
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            REFRESH_TOKEN_EXPIRE_DAYS=7
        )
        
        assert settings.SECRET_KEY == "test_secret_key"
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30
        assert settings.REFRESH_TOKEN_EXPIRE_DAYS == 7
    
    def test_testing_mode_settings(self):
        """Test testing mode configuration."""
        settings = Settings(
            TESTING=True,
            DISABLE_EXTERNAL_APIS=True,
            USE_MOCK_SERVICES=True
        )
        
        assert settings.TESTING is True
        assert settings.DISABLE_EXTERNAL_APIS is True
        assert settings.USE_MOCK_SERVICES is True
    
    def test_file_upload_settings(self):
        """Test file upload configuration."""
        settings = Settings(
            MAX_FILE_SIZE_MB=10,
            ALLOWED_FILE_TYPES=["image/jpeg", "image/png", "audio/wav"]
        )
        
        assert settings.MAX_FILE_SIZE_MB == 10
        assert len(settings.ALLOWED_FILE_TYPES) == 3
        assert "image/jpeg" in settings.ALLOWED_FILE_TYPES


class TestGetSettings:
    """Test get_settings dependency function."""
    
    def test_get_settings_returns_instance(self):
        """Test that get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        # Clear any existing cache
        get_settings.cache_clear() if hasattr(get_settings, 'cache_clear') else None
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should return the same instance (cached)
        assert settings1 is settings2
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'test_env'})
    def test_get_settings_with_env_vars(self):
        """Test get_settings with environment variables."""
        # Clear cache to ensure new settings are loaded
        get_settings.cache_clear() if hasattr(get_settings, 'cache_clear') else None
        
        settings = get_settings()
        assert settings.ENVIRONMENT == "test_env"


@pytest.mark.integration
class TestSettingsIntegration:
    """Integration tests for settings with external dependencies."""
    
    def test_settings_validation(self):
        """Test settings validation with invalid values."""
        # Test invalid database URL
        with pytest.raises(Exception):
            Settings(DATABASE_URL="invalid_url")
    
    def test_settings_with_env_file(self, tmp_path):
        """Test loading settings from .env file."""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "APP_NAME=Test App\n"
            "DEBUG=false\n"
            "LOG_LEVEL=DEBUG\n"
        )
        
        # Test loading from custom env file
        settings = Settings(_env_file=str(env_file))
        assert settings.APP_NAME == "Test App"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "DEBUG"
    
    def test_production_settings_security(self):
        """Test production settings for security."""
        settings = Settings(ENVIRONMENT="production")
        
        # In production, debug should be False
        if settings.ENVIRONMENT == "production":
            assert settings.DEBUG is False
    
    def test_database_connection_string_formats(self):
        """Test various database connection string formats."""
        # PostgreSQL
        settings = Settings(
            DB_HOST="postgres.example.com",
            DB_PORT=5432,
            DB_NAME="myapp",
            DB_USER="user",
            DB_PASSWORD="pass123"
        )
        assert "postgresql+asyncpg://" in settings.DATABASE_URL
        
        # SQLite
        settings = Settings(DATABASE_URL="sqlite+aiosqlite:///./data.db")
        assert "sqlite+aiosqlite" in settings.DATABASE_URL
    
    def test_api_key_validation(self):
        """Test API key validation and requirements."""
        settings = Settings()
        
        # API keys should be configurable
        assert hasattr(settings, 'GOOGLE_MAPS_API_KEY')
        assert hasattr(settings, 'OPENWEATHER_API_KEY')
        assert hasattr(settings, 'GOOGLE_PLACES_API_KEY')
    
    def test_model_cache_directory_creation(self):
        """Test model cache directory configuration."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(MODEL_CACHE_DIR=temp_dir)
            assert settings.MODEL_CACHE_DIR == temp_dir
    
    def test_logging_configuration(self):
        """Test logging configuration settings."""
        settings = Settings(
            LOG_LEVEL="DEBUG",
            LOG_FORMAT="json",
            LOG_FILE="app.log"
        )
        
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.LOG_FORMAT == "json"
        assert settings.LOG_FILE == "app.log"
