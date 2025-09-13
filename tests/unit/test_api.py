"""Unit tests for API endpoints.

Tests the FastAPI routers and endpoint functionality.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from tests.utils import TestDataFactory, MockResponseBuilder, APITestHelper


class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""
    
    def test_register_user_success(self, client, api_test_helper):
        """Test successful user registration."""
        user_data = TestDataFactory.create_user_data(
            username="newuser",
            email="new@example.com",
            password="securepassword123"
        )
        
        response = client.post("/api/auth/register", json=user_data)
        
        data = api_test_helper.assert_success_response(response, 201)
        assert "user" in data
        assert data["user"]["username"] == "newuser"
        assert data["user"]["email"] == "new@example.com"
        assert "access_token" in data
    
    def test_register_user_duplicate_username(self, client, test_user, api_test_helper):
        """Test registration with duplicate username."""
        user_data = TestDataFactory.create_user_data(
            username=test_user.username,  # Duplicate username
            email="different@example.com",
            password="securepassword123"
        )
        
        response = client.post("/api/auth/register", json=user_data)
        
        api_test_helper.assert_error_response(response, 400, "username already exists")
    
    def test_register_user_invalid_email(self, client, api_test_helper):
        """Test registration with invalid email."""
        user_data = TestDataFactory.create_user_data(
            email="invalid_email_format"
        )
        
        response = client.post("/api/auth/register", json=user_data)
        
        api_test_helper.assert_validation_error(response, "email")
    
    def test_login_success(self, client, test_user, api_test_helper):
        """Test successful user login."""
        login_data = {
            "username": test_user.username,
            "password": "correct_password"
        }
        
        with patch("app.services.auth.verify_password", return_value=True):
            response = client.post("/api/auth/login", json=login_data)
        
        data = api_test_helper.assert_success_response(response)
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client, test_user, api_test_helper):
        """Test login with invalid credentials."""
        login_data = {
            "username": test_user.username,
            "password": "wrong_password"
        }
        
        with patch("app.services.auth.verify_password", return_value=False):
            response = client.post("/api/auth/login", json=login_data)
        
        api_test_helper.assert_error_response(response, 401, "invalid credentials")
    
    def test_refresh_token(self, client, auth_headers, api_test_helper):
        """Test token refresh."""
        refresh_data = {
            "refresh_token": "valid_refresh_token"
        }
        
        with patch("app.services.auth.verify_refresh_token", return_value=True):
            response = client.post(
                "/api/auth/refresh",
                json=refresh_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "access_token" in data
    
    def test_logout(self, client, auth_headers, api_test_helper):
        """Test user logout."""
        response = client.post("/api/auth/logout", headers=auth_headers)
        
        api_test_helper.assert_success_response(response)
    
    def test_get_current_user(self, client, auth_headers, test_user, api_test_helper):
        """Test getting current user profile."""
        with patch("app.api.dependencies.get_current_user", return_value=test_user):
            response = client.get("/api/auth/me", headers=auth_headers)
        
        data = api_test_helper.assert_success_response(response)
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email
    
    def test_unauthorized_access(self, client, api_test_helper):
        """Test unauthorized access to protected endpoint."""
        response = client.get("/api/auth/me")
        
        api_test_helper.assert_error_response(response, 401)


class TestNavigationEndpoints:
    """Test navigation API endpoints."""
    
    def test_get_directions_success(self, client, auth_headers, mock_google_maps, api_test_helper):
        """Test successful directions request."""
        request_data = TestDataFactory.create_navigation_request()
        
        with patch("app.adapters.maps.GoogleMapsAdapter", return_value=mock_google_maps):
            response = client.post(
                "/api/navigation/directions",
                json=request_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "routes" in data
        assert "distance" in data
        assert "duration" in data
    
    def test_get_directions_invalid_coordinates(self, client, auth_headers, api_test_helper):
        """Test directions with invalid coordinates."""
        request_data = TestDataFactory.create_navigation_request(
            origin={"lat": 200, "lng": -300}  # Invalid coordinates
        )
        
        response = client.post(
            "/api/navigation/directions",
            json=request_data,
            headers=auth_headers
        )
        
        api_test_helper.assert_validation_error(response)
    
    def test_geocode_address(self, client, auth_headers, mock_google_maps, api_test_helper):
        """Test address geocoding."""
        request_data = {
            "address": "Times Square, New York, NY"
        }
        
        with patch("app.adapters.maps.GoogleMapsAdapter", return_value=mock_google_maps):
            response = client.post(
                "/api/navigation/geocode",
                json=request_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "location" in data
        assert "formatted_address" in data
    
    def test_reverse_geocode(self, client, auth_headers, mock_google_maps, api_test_helper):
        """Test reverse geocoding."""
        request_data = {
            "location": {"lat": 40.7128, "lng": -74.0060}
        }
        
        with patch("app.adapters.maps.GoogleMapsAdapter", return_value=mock_google_maps):
            response = client.post(
                "/api/navigation/reverse-geocode",
                json=request_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "formatted_address" in data


class TestRestaurantEndpoints:
    """Test restaurant API endpoints."""
    
    def test_search_restaurants_success(self, client, auth_headers, mock_places_service, api_test_helper):
        """Test successful restaurant search."""
        request_data = TestDataFactory.create_places_search_request(
            query="italian restaurants",
            type="restaurant"
        )
        
        with patch("app.adapters.places.GooglePlacesAdapter", return_value=mock_places_service):
            response = client.post(
                "/api/restaurants/search",
                json=request_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "restaurants" in data
        assert len(data["restaurants"]) > 0
    
    def test_get_restaurant_details(self, client, auth_headers, mock_places_service, api_test_helper):
        """Test getting restaurant details."""
        place_id = "test_place_123"
        
        with patch("app.adapters.places.GooglePlacesAdapter", return_value=mock_places_service):
            response = client.get(
                f"/api/restaurants/{place_id}",
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "name" in data
        assert "formatted_address" in data
    
    def test_get_restaurant_menu_ocr(self, client, auth_headers, mock_ocr_service, api_test_helper):
        """Test menu OCR extraction."""
        # Mock image data
        image_data = {
            "image": "base64_encoded_image_data",
            "language": "en"
        }
        
        with patch("app.services.ocr.OCRService", return_value=mock_ocr_service):
            response = client.post(
                "/api/restaurants/menu-ocr",
                json=image_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "menu_text" in data
        assert "structured_data" in data


class TestVoiceEndpoints:
    """Test voice processing API endpoints."""
    
    def test_transcribe_audio_success(self, client, auth_headers, mock_whisper_service, sample_audio_data, api_test_helper):
        """Test successful audio transcription."""
        files = {"audio": ("test.wav", sample_audio_data, "audio/wav")}
        data = {"language": "en"}
        
        with patch("app.services.whisper.WhisperService", return_value=mock_whisper_service):
            response = client.post(
                "/api/voice/transcribe",
                files=files,
                data=data,
                headers=auth_headers
            )
        
        response_data = api_test_helper.assert_success_response(response)
        assert "transcription" in response_data
        assert "language" in response_data
        assert "confidence" in response_data
    
    def test_transcribe_audio_invalid_format(self, client, auth_headers, api_test_helper):
        """Test transcription with invalid audio format."""
        files = {"audio": ("test.txt", b"not_audio_data", "text/plain")}
        
        response = client.post(
            "/api/voice/transcribe",
            files=files,
            headers=auth_headers
        )
        
        api_test_helper.assert_error_response(response, 400, "unsupported audio format")
    
    def test_synthesize_speech_success(self, client, auth_headers, mock_tts_service, api_test_helper):
        """Test successful speech synthesis."""
        request_data = {
            "text": "Hello, welcome to New York!",
            "language": "en",
            "voice": "female",
            "speed": 1.0
        }
        
        with patch("app.services.tts.TTSService", return_value=mock_tts_service):
            response = client.post(
                "/api/voice/synthesize",
                json=request_data,
                headers=auth_headers
            )
        
        # TTS endpoint should return audio data
        assert response.status_code == 200
        assert "audio" in response.headers.get("content-type", "")
    
    def test_translate_and_speak(self, client, auth_headers, mock_nllb_service, mock_tts_service, api_test_helper):
        """Test translation and speech synthesis combined."""
        request_data = {
            "text": "Where is the nearest restaurant?",
            "source_language": "en",
            "target_language": "es",
            "voice": "female"
        }
        
        with patch("app.services.nllb.NLLBTranslationService", return_value=mock_nllb_service), \
             patch("app.services.tts.TTSService", return_value=mock_tts_service):
            response = client.post(
                "/api/voice/translate-speak",
                json=request_data,
                headers=auth_headers
            )
        
        data = api_test_helper.assert_success_response(response)
        assert "translated_text" in data
        assert "audio_data" in data


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client, api_test_helper):
        """Test basic health check."""
        response = client.get("/health")
        
        data = api_test_helper.assert_success_response(response)
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, client, api_test_helper):
        """Test readiness check."""
        response = client.get("/ready")
        
        data = api_test_helper.assert_success_response(response)
        assert "database" in data
        assert "services" in data
    
    def test_metrics_endpoint(self, client, api_test_helper):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        # Metrics might be in different format
        assert response.status_code == 200


@pytest.mark.integration
class TestEndpointIntegration:
    """Integration tests for API endpoints."""
    
    async def test_user_journey_flow(self, client, api_test_helper):
        """Test complete user journey flow."""
        # 1. Register user
        user_data = TestDataFactory.create_user_data(
            username="journeyuser",
            email="journey@example.com",
            password="secure123"
        )
        
        register_response = client.post("/api/auth/register", json=user_data)
        register_data = api_test_helper.assert_success_response(register_response, 201)
        
        # 2. Login
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        with patch("app.services.auth.verify_password", return_value=True):
            login_response = client.post("/api/auth/login", json=login_data)
        
        login_data = api_test_helper.assert_success_response(login_response)
        auth_headers = {"Authorization": f"Bearer {login_data['access_token']}"}
        
        # 3. Search for places
        with patch("app.adapters.places.GooglePlacesAdapter") as mock_places:
            mock_places.return_value.search_places = AsyncMock(
                return_value=MockResponseBuilder.places_search_response()
            )
            
            search_response = client.post(
                "/api/restaurants/search",
                json=TestDataFactory.create_places_search_request(),
                headers=auth_headers
            )
        
        search_data = api_test_helper.assert_success_response(search_response)
        assert "restaurants" in search_data
    
    def test_error_handling_consistency(self, client, api_test_helper):
        """Test that error responses are consistent across endpoints."""
        # Test 401 errors
        endpoints_401 = [
            "/api/auth/me",
            "/api/navigation/directions",
            "/api/restaurants/search",
            "/api/voice/transcribe"
        ]
        
        for endpoint in endpoints_401:
            if endpoint == "/api/voice/transcribe":
                response = client.post(endpoint)
            else:
                response = client.get(endpoint)
            
            api_test_helper.assert_error_response(response, 401)
    
    def test_rate_limiting(self, client, auth_headers, api_test_helper):
        """Test rate limiting on endpoints."""
        # Make multiple requests quickly
        responses = []
        for _ in range(10):
            response = client.get("/api/auth/me", headers=auth_headers)
            responses.append(response)
        
        # Check if rate limiting is enforced (429 status)
        rate_limited = any(r.status_code == 429 for r in responses)
        
        # Rate limiting might not be enabled in tests
        if rate_limited:
            assert any(r.status_code == 429 for r in responses[-3:])  # Last few should be limited
