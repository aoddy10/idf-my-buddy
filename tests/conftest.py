"""Minimal test configuration for basic testing."""

import pytest
import os
from typing import Generator
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["APP_ENV"] = "testing"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
    os.environ["DEBUG"] = "false"


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create test client."""
    from app.main import create_app
    
    app = create_app()
    with TestClient(app) as client:
        yield client
