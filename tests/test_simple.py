"""Simple tests that don't depend on complex fixtures."""

import pytest
from fastapi.testclient import TestClient


def test_fastapi_app_creation():
    """Test that we can create FastAPI app without database."""
    import os
    # Set test database URL to avoid PostgreSQL connection
    os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test.db'
    
    from app.main import create_app
    
    app = create_app()
    assert app is not None
    assert app.title == "My Buddy API"
    assert app.version == "1.0.0"


def test_health_endpoints():
    """Test health check endpoints."""
    import os
    # Set test environment
    os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test.db'
    os.environ['APP_ENV'] = 'testing'
    
    from app.main import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"
    
    # Test readiness endpoint
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    
    # Test liveness endpoint
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


def test_api_endpoints_exist():
    """Test that API endpoints are accessible."""
    import os
    os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test.db'
    
    from app.main import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # Test navigation endpoint
    response = client.get("/navigation/")
    assert response.status_code == 200
    
    # Test restaurant endpoint  
    response = client.get("/restaurant/")
    assert response.status_code == 200
    
    # Test shopping endpoint
    response = client.get("/shopping/")
    assert response.status_code == 200
    
    # Test safety endpoint
    response = client.get("/safety/")
    assert response.status_code == 200


def test_config_settings():
    """Test configuration settings."""
    from app.core.config import Settings, get_settings
    
    # Test default settings
    settings = Settings()
    assert settings.app_env in ["development", "testing", "production"]
    assert isinstance(settings.debug, bool)
    
    # Test get_settings function
    settings_func = get_settings()
    assert settings_func.app_env in ["development", "testing", "production"]
