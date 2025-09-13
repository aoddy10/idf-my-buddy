"""Test utilities and helpers.

This module provides common utilities, fixtures, and helpers for testing
the IDF My Buddy application.
"""

import asyncio
import json
import tempfile
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import io
import base64

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user_data(**overrides) -> Dict[str, Any]:
        """Create test user data."""
        default_data = {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "preferred_language": "en",
            "profile_image_url": None,
            "preferences": {
                "notifications": True,
                "theme": "light",
                "default_currency": "USD"
            }
        }
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def create_travel_context_data(**overrides) -> Dict[str, Any]:
        """Create test travel context data."""
        default_data = {
            "current_location": {"lat": 40.7128, "lng": -74.0060},
            "destination": {"lat": 34.0522, "lng": -118.2437},
            "travel_mode": "walking",
            "preferences": {
                "cuisine": ["italian", "japanese"],
                "budget": "moderate",
                "interests": ["museums", "parks"],
                "accessibility": []
            },
            "context": "tourist",
            "group_size": 2,
            "duration_hours": 4
        }
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def create_navigation_request(**overrides) -> Dict[str, Any]:
        """Create test navigation request."""
        default_data = {
            "origin": {"lat": 40.7128, "lng": -74.0060},
            "destination": {"lat": 40.7589, "lng": -73.9851},
            "travel_mode": "walking",
            "language": "en",
            "avoid": [],
            "waypoints": []
        }
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def create_places_search_request(**overrides) -> Dict[str, Any]:
        """Create test places search request."""
        default_data = {
            "query": "restaurants near me",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "radius": 1000,
            "type": "restaurant",
            "min_rating": 4.0,
            "price_level": [2, 3],
            "open_now": True
        }
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def create_voice_request(**overrides) -> Dict[str, Any]:
        """Create test voice processing request."""
        default_data = {
            "audio_format": "wav",
            "sample_rate": 16000,
            "language": "en",
            "task": "transcribe"
        }
        default_data.update(overrides)
        return default_data


class MockResponseBuilder:
    """Builder for creating mock API responses."""
    
    @staticmethod
    def google_maps_directions_response(distance_km: float = 1.2, duration_min: int = 15) -> Dict[str, Any]:
        """Create mock Google Maps directions response."""
        return {
            "routes": [{
                "legs": [{
                    "distance": {"text": f"{distance_km} km", "value": int(distance_km * 1000)},
                    "duration": {"text": f"{duration_min} mins", "value": duration_min * 60},
                    "start_location": {"lat": 40.7128, "lng": -74.0060},
                    "end_location": {"lat": 40.7589, "lng": -73.9851},
                    "steps": [
                        {
                            "html_instructions": "Head north on Broadway",
                            "distance": {"text": "0.5 km", "value": 500},
                            "duration": {"text": "6 mins", "value": 360},
                            "start_location": {"lat": 40.7128, "lng": -74.0060},
                            "end_location": {"lat": 40.7178, "lng": -74.0060}
                        }
                    ]
                }],
                "overview_polyline": {"points": "fake_polyline_data"},
                "summary": "Broadway"
            }],
            "status": "OK"
        }
    
    @staticmethod
    def weather_current_response(temp: float = 22.5, condition: str = "sunny") -> Dict[str, Any]:
        """Create mock weather current response."""
        return {
            "temperature": temp,
            "feels_like": temp + 2,
            "condition": condition,
            "description": f"{condition.title()} skies",
            "humidity": 65,
            "pressure": 1013,
            "wind_speed": 10,
            "wind_direction": 180,
            "visibility": 10000,
            "uv_index": 6,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def places_search_response(num_results: int = 3) -> Dict[str, Any]:
        """Create mock places search response."""
        results = []
        for i in range(num_results):
            results.append({
                "place_id": f"test_place_{i}",
                "name": f"Test Restaurant {i+1}",
                "formatted_address": f"{100+i} Test St, New York, NY",
                "geometry": {
                    "location": {"lat": 40.7128 + i*0.001, "lng": -74.0060 + i*0.001}
                },
                "rating": 4.0 + (i % 5) * 0.2,
                "price_level": (i % 4) + 1,
                "types": ["restaurant", "food", "establishment"],
                "opening_hours": {"open_now": True},
                "photos": [{"photo_reference": f"photo_ref_{i}"}]
            })
        
        return {
            "results": results,
            "status": "OK",
            "html_attributions": []
        }
    
    @staticmethod
    def ocr_response(text: str = "Sample extracted text") -> Dict[str, Any]:
        """Create mock OCR response."""
        return {
            "text": text,
            "words": [
                {
                    "text": word,
                    "confidence": 0.95,
                    "bbox": {"x": i*50, "y": 10, "width": 45, "height": 20}
                }
                for i, word in enumerate(text.split())
            ],
            "language": "en",
            "engine": "tesseract",
            "confidence": 0.95
        }
    
    @staticmethod
    def whisper_response(text: str = "Hello world") -> Dict[str, Any]:
        """Create mock Whisper ASR response."""
        return {
            "text": text,
            "language": "en",
            "confidence": 0.98,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": text
                }
            ]
        }
    
    @staticmethod
    def translation_response(text: str = "Hola mundo", source: str = "en", target: str = "es") -> Dict[str, Any]:
        """Create mock translation response."""
        return {
            "translated_text": text,
            "source_language": source,
            "target_language": target,
            "confidence": 0.97,
            "alternatives": [text]
        }


class AsyncTestClient:
    """Async wrapper for test client operations."""
    
    def __init__(self, client):
        self.client = client
    
    async def get(self, url: str, **kwargs):
        """Async GET request."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.get(url, **kwargs)
        )
    
    async def post(self, url: str, **kwargs):
        """Async POST request."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.post(url, **kwargs)
        )
    
    async def put(self, url: str, **kwargs):
        """Async PUT request."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.put(url, **kwargs)
        )
    
    async def delete(self, url: str, **kwargs):
        """Async DELETE request."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.delete(url, **kwargs)
        )


class FileTestHelper:
    """Helper for file-related testing."""
    
    @staticmethod
    def create_test_image(width: int = 100, height: int = 100) -> bytes:
        """Create a test image file."""
        try:
            from PIL import Image
            import io
            
            # Create a simple test image
            img = Image.new('RGB', (width, height), color='red')
            
            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return img_bytes.getvalue()
            
        except ImportError:
            # Fallback: minimal PNG data
            return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    
    @staticmethod
    def create_test_audio(duration_seconds: float = 1.0) -> bytes:
        """Create test audio file."""
        # Simple WAV header for 16kHz mono
        sample_rate = 16000
        samples = int(sample_rate * duration_seconds)
        
        # WAV header
        header = b'RIFF'
        header += (36 + samples * 2).to_bytes(4, 'little')  # File size
        header += b'WAVE'
        header += b'fmt '
        header += (16).to_bytes(4, 'little')  # Format chunk size
        header += (1).to_bytes(2, 'little')   # Audio format (PCM)
        header += (1).to_bytes(2, 'little')   # Channels
        header += sample_rate.to_bytes(4, 'little')  # Sample rate
        header += (sample_rate * 2).to_bytes(4, 'little')  # Byte rate
        header += (2).to_bytes(2, 'little')   # Block align
        header += (16).to_bytes(2, 'little')  # Bits per sample
        header += b'data'
        header += (samples * 2).to_bytes(4, 'little')  # Data size
        
        # Silence data
        audio_data = b'\x00\x00' * samples
        
        return header + audio_data
    
    @staticmethod
    def create_base64_image(width: int = 100, height: int = 100) -> str:
        """Create base64-encoded test image."""
        image_bytes = FileTestHelper.create_test_image(width, height)
        return base64.b64encode(image_bytes).decode('utf-8')


class DatabaseTestHelper:
    """Helper for database testing operations."""
    
    @staticmethod
    async def create_test_user(db_session, **overrides):
        """Create test user in database."""
        from app.models.user import User
        
        user_data = TestDataFactory.create_user_data(**overrides)
        user = User(**user_data, hashed_password="fake_hash")
        
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        return user
    
    @staticmethod
    async def create_test_session(db_session, user_id: int, **overrides):
        """Create test session in database."""
        from app.models.session import UserSession
        
        session_data = {
            "user_id": user_id,
            "session_token": "test_token_123",
            "device_info": {"platform": "test"},
            **overrides
        }
        
        session = UserSession(**session_data)
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        return session
    
    @staticmethod
    async def cleanup_test_data(db_session, model_class, **filters):
        """Clean up test data from database."""
        from sqlmodel import select
        
        stmt = select(model_class)
        for key, value in filters.items():
            stmt = stmt.where(getattr(model_class, key) == value)
        
        result = await db_session.execute(stmt)
        instances = result.scalars().all()
        
        for instance in instances:
            await db_session.delete(instance)
        
        await db_session.commit()


class APITestHelper:
    """Helper for API endpoint testing."""
    
    @staticmethod
    def assert_success_response(response, expected_status: int = 200):
        """Assert successful API response."""
        assert response.status_code == expected_status
        assert "application/json" in response.headers.get("content-type", "")
        
        data = response.json()
        assert "error" not in data
        
        return data
    
    @staticmethod
    def assert_error_response(response, expected_status: int, expected_error: str = None):
        """Assert error API response."""
        assert response.status_code == expected_status
        
        data = response.json()
        assert "error" in data or "detail" in data
        
        if expected_error:
            error_message = data.get("error", data.get("detail", ""))
            assert expected_error.lower() in error_message.lower()
        
        return data
    
    @staticmethod
    def assert_validation_error(response, field_name: str = None):
        """Assert validation error response."""
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        
        if field_name:
            errors = data["detail"]
            field_errors = [e for e in errors if field_name in str(e.get("loc", []))]
            assert len(field_errors) > 0, f"No validation error found for field: {field_name}"
        
        return data


class MockServicePatcher:
    """Context manager for patching services in tests."""
    
    def __init__(self, service_patches: Dict[str, Any]):
        self.service_patches = service_patches
        self.patchers = []
    
    def __enter__(self):
        for service_path, mock_obj in self.service_patches.items():
            patcher = patch(service_path, mock_obj)
            patcher.start()
            self.patchers.append(patcher)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patcher in self.patchers:
            patcher.stop()


class PerformanceTestHelper:
    """Helper for performance testing."""
    
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time."""
        import time
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    @staticmethod
    def assert_performance_threshold(execution_time: float, max_time: float):
        """Assert that execution time is within threshold."""
        assert execution_time <= max_time, f"Execution time {execution_time:.3f}s exceeded threshold {max_time:.3f}s"


# Pytest fixtures for commonly used test helpers
@pytest.fixture
def test_data_factory():
    """Test data factory fixture."""
    return TestDataFactory()


@pytest.fixture
def mock_response_builder():
    """Mock response builder fixture."""
    return MockResponseBuilder()


@pytest.fixture
def file_test_helper():
    """File test helper fixture."""
    return FileTestHelper()


@pytest.fixture
def db_test_helper():
    """Database test helper fixture."""
    return DatabaseTestHelper()


@pytest.fixture
def api_test_helper():
    """API test helper fixture."""
    return APITestHelper()


@pytest.fixture
def performance_test_helper():
    """Performance test helper fixture."""
    return PerformanceTestHelper()
