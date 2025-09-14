"""Simple test configuration for navigation tests."""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing."""
    from app.schemas.common import Coordinates
    return {
        "origin": Coordinates(latitude=40.7128, longitude=-74.0060, accuracy=10.0),  # NYC
        "destination": Coordinates(latitude=40.7589, longitude=-73.9851, accuracy=10.0)  # Times Square
    }
