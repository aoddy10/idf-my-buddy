"""Main conftest.py for pytest configuration.

This module provides shared fixtures, test configuration, and utilities
for the entire test suite.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import os

# FastAPI and database imports
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

# Application imports
from app.main import app
from app.core.config import get_settings, Settings
from app.core.database import get_async_session
from app.models.user import User
from app.models.session import UserSession
from app.models.travel import TravelContext


# Test settings configuration
class TestSettings(Settings):
    """Test-specific settings."""
    
    class Config:
        env_file = ".env.test"
    
    ENVIRONMENT: str = "testing"
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    TESTING: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Disable external services in tests
    DISABLE_EXTERNAL_APIS: bool = True
    USE_MOCK_SERVICES: bool = True
    
    # AI/ML settings for tests
    USE_GPU_FOR_OCR: bool = False
    USE_GPU_FOR_TTS: bool = False
    ENABLE_MODEL_CACHING: bool = False
    
    # External API settings (use test keys or disable)
    GOOGLE_MAPS_API_KEY: str = "test_google_maps_key"
    OPENWEATHER_API_KEY: str = "test_openweather_key"
    GOOGLE_PLACES_API_KEY: str = "test_places_key"


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    """Test settings fixture."""
    return TestSettings()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(test_settings):
    """Setup test environment variables."""
    # Set test environment variables
    os.environ.update({
        "ENVIRONMENT": "testing",
        "DATABASE_URL": test_settings.DATABASE_URL,
        "TESTING": "true",
        "DISABLE_EXTERNAL_APIS": "true",
        "USE_MOCK_SERVICES": "true",
    })
    
    # Override settings in app
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    yield
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine(test_settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session_maker = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
def client(db_session) -> Generator[TestClient, None, None]:
    """Create test client with database override."""
    
    def get_test_session():
        return db_session
    
    app.dependency_overrides[get_async_session] = get_test_session
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def test_user(db_session: AsyncSession) -> User:
    """Create test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        preferred_language="en",
        hashed_password="fake_hash_for_testing"
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest.fixture(scope="function")
async def test_session(db_session: AsyncSession, test_user: User) -> UserSession:
    """Create test user session."""
    session = UserSession(
        user_id=test_user.id,
        session_token="test_session_token",
        device_info={"platform": "test", "version": "1.0"}
    )
    
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    
    return session


@pytest.fixture(scope="function")
async def test_travel_context(
    db_session: AsyncSession, 
    test_user: User
) -> TravelContext:
    """Create test travel context."""
    context = TravelContext(
        user_id=test_user.id,
        current_location={"lat": 40.7128, "lng": -74.0060},
        destination={"lat": 34.0522, "lng": -118.2437},
        travel_mode="walking",
        preferences={
            "cuisine": ["italian", "japanese"],
            "budget": "moderate",
            "interests": ["museums", "parks"]
        }
    )
    
    db_session.add(context)
    await db_session.commit()
    await db_session.refresh(context)
    
    return context


@pytest.fixture(scope="function")
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_image_data() -> bytes:
    """Create sample image data for testing."""
    # Simple 1x1 pixel PNG image (base64 decoded)
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x04\x00\x01]U\xaa\x00\x00\x00\x00IEND\xaeB`\x82'
    return png_data


@pytest.fixture(scope="function")
def sample_audio_data() -> bytes:
    """Create sample audio data for testing."""
    # Minimal WAV header + silence
    wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    return wav_header


# Mock fixtures for external services
@pytest.fixture(scope="function")
def mock_google_maps():
    """Mock Google Maps API responses."""
    mock = Mock()
    mock.directions = AsyncMock(return_value={
        "routes": [{
            "legs": [{
                "distance": {"text": "1.2 km", "value": 1200},
                "duration": {"text": "15 mins", "value": 900},
                "steps": []
            }]
        }]
    })
    mock.geocode = AsyncMock(return_value={
        "results": [{
            "geometry": {
                "location": {"lat": 40.7128, "lng": -74.0060}
            },
            "formatted_address": "New York, NY, USA"
        }]
    })
    return mock


@pytest.fixture(scope="function")
def mock_weather_service():
    """Mock weather service responses."""
    mock = Mock()
    mock.get_current_weather = AsyncMock(return_value={
        "temperature": 22.5,
        "condition": "sunny",
        "humidity": 65,
        "wind_speed": 10,
        "description": "Clear sky"
    })
    mock.get_forecast = AsyncMock(return_value={
        "daily": [
            {
                "date": "2023-09-13",
                "temperature": {"min": 18, "max": 25},
                "condition": "partly_cloudy"
            }
        ]
    })
    return mock


@pytest.fixture(scope="function")
def mock_places_service():
    """Mock places service responses."""
    mock = Mock()
    mock.search_places = AsyncMock(return_value={
        "results": [
            {
                "name": "Test Restaurant",
                "place_id": "test_place_123",
                "rating": 4.5,
                "price_level": 2,
                "location": {"lat": 40.7128, "lng": -74.0060},
                "types": ["restaurant", "food"]
            }
        ]
    })
    mock.get_place_details = AsyncMock(return_value={
        "name": "Test Restaurant",
        "formatted_address": "123 Test St, New York, NY",
        "phone": "+1234567890",
        "website": "https://testrestaurant.com",
        "opening_hours": {"open_now": True}
    })
    return mock


@pytest.fixture(scope="function")
def mock_whisper_service():
    """Mock Whisper ASR service."""
    mock = Mock()
    mock.transcribe_audio = AsyncMock(return_value={
        "text": "Hello, this is a test transcription",
        "language": "en",
        "confidence": 0.95
    })
    return mock


@pytest.fixture(scope="function")
def mock_nllb_service():
    """Mock NLLB translation service."""
    mock = Mock()
    mock.translate = AsyncMock(return_value={
        "translated_text": "Hola, esta es una traducción de prueba",
        "source_language": "en",
        "target_language": "es",
        "confidence": 0.98
    })
    return mock


@pytest.fixture(scope="function")
def mock_ocr_service():
    """Mock OCR service."""
    mock = Mock()
    mock.extract_text = AsyncMock(return_value={
        "text": "Sample text extracted from image",
        "words": [
            {
                "text": "Sample",
                "confidence": 0.95,
                "bbox": {"x": 10, "y": 10, "width": 50, "height": 20}
            }
        ],
        "language": "en",
        "engine": "tesseract",
        "confidence": 0.95
    })
    mock.extract_structured_text = AsyncMock(return_value={
        "text": "Menu items and prices",
        "document_type": "menu",
        "structured": {
            "sections": ["appetizers", "mains"],
            "items": ["Pizza", "Pasta"],
            "prices": ["$12", "$15"]
        }
    })
    return mock


@pytest.fixture(scope="function")
def mock_tts_service():
    """Mock TTS service."""
    mock = Mock()
    mock.synthesize_speech = AsyncMock(return_value={
        "audio_data": b"fake_audio_data",
        "format": "wav",
        "sample_rate": 22050,
        "duration": 2.5
    })
    return mock


# AI/ML Mock fixtures
@pytest.fixture(scope="function")
def mock_model_loader():
    """Mock ModelLoader for testing."""
    mock = Mock()
    mock.load_model = AsyncMock(return_value="fake_model_object")
    mock.get_model_info = Mock(return_value={
        "type": "test_model",
        "parameters": 1000000
    })
    mock.get_cache_stats = Mock(return_value={
        "entries": 1,
        "memory_used_mb": 100
    })
    return mock


@pytest.fixture(scope="function")
def mock_edge_computer():
    """Mock EdgeComputer for testing."""
    mock = Mock()
    mock.prepare_model = AsyncMock(return_value="model_key_123")
    mock.run_inference = AsyncMock(return_value=["result1", "result2"])
    mock.get_performance_stats = Mock(return_value={
        "optimized_models": 1,
        "inference_engine": {"max_batch_size": 8}
    })
    return mock


@pytest.fixture(scope="function")
def mock_performance_monitor():
    """Mock PerformanceMonitor for testing."""
    mock = Mock()
    mock.track_inference = Mock()
    mock.get_inference_stats = Mock(return_value={
        "test_model": {
            "total_inferences": 10,
            "avg_time": 0.1,
            "throughput": 10.0
        }
    })
    mock.start_profiling = Mock(return_value="profile_123")
    mock.stop_profiling = Mock(return_value={
        "statistics": {"total_inferences": 5}
    })
    return mock


# Authentication fixtures
@pytest.fixture(scope="function")
def auth_headers(test_session: UserSession) -> Dict[str, str]:
    """Create authentication headers for API requests."""
    return {
        "Authorization": f"Bearer {test_session.session_token}",
        "Content-Type": "application/json"
    }


# Utility functions for tests
def assert_valid_response(response, expected_status: int = 200):
    """Assert that API response is valid."""
    assert response.status_code == expected_status
    assert "application/json" in response.headers.get("content-type", "")


def assert_error_response(response, expected_status: int, error_code: str = None):
    """Assert that API response is a valid error."""
    assert response.status_code == expected_status
    data = response.json()
    assert "error" in data or "detail" in data
    
    if error_code:
        assert data.get("error_code") == error_code or error_code in str(data)


# Pytest plugins and hooks
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(marker.name in ['integration', 'e2e'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)


# Async testing utilities
class AsyncTestCase:
    """Base class for async test cases."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def create_test_data(self):
        """Override in subclasses to create test data."""
        pass
    
    async def cleanup_test_data(self):
        """Override in subclasses to cleanup test data."""
        pass
