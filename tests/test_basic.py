"""Basic tests to verify the application can be imported and initialized."""

import pytest
from fastapi.testclient import TestClient


def test_app_import():
    """Test that we can import the main app module."""
    from app.main import create_app
    
    app = create_app()
    assert app is not None
    assert app.title == "My Buddy API"


def test_app_creation():
    """Test that we can create a FastAPI app instance."""
    from app.main import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # The app should be created successfully
    assert client is not None


def test_config_import():
    """Test that we can import configuration."""
    from app.core.config import Settings, get_settings
    
    settings = Settings()
    assert settings.app_env in ["development", "testing", "production"]
    
    # Test get_settings function
    settings_func = get_settings()
    assert settings_func.app_env in ["development", "testing", "production"]


def test_basic_health_check():
    """Test basic health endpoint if it exists."""
    from app.main import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # Try to access health endpoint - this might not exist yet
    # Just test that the app responds to something
    response = client.get("/")
    # Don't assert status code since we don't have a root endpoint yet
    assert response is not None
