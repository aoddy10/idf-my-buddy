"""Integration tests for Navigation API endpoints.

Tests the complete navigation API functionality including route calculation,
POI search, geocoding, and voice navigation integration.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.schemas.navigation import TransportMode, POICategory
from app.schemas.common import LanguageCode


class TestNavigationAPI:
    """Integration tests for Navigation API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_route_request(self):
        """Sample route calculation request."""
        return {
            "origin": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "accuracy": 10.0
            },
            "destination": {
                "latitude": 40.7589,
                "longitude": -73.9851,
                "accuracy": 10.0
            },
            "transport_mode": "driving",
            "language": "en",
            "avoid_tolls": False,
            "avoid_highways": False,
            "avoid_ferries": False
        }
    
    def test_navigation_status_endpoint(self, client):
        """Test navigation service status endpoint."""
        response = client.get("/navigation/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert "google_maps_available" in data["data"]
        assert "supported_transport_modes" in data["data"]
        assert "supported_poi_categories" in data["data"]
        assert "version" in data["data"]
    
    def test_calculate_route_success(self, client, sample_route_request):
        """Test successful route calculation."""
        # Mock the NavigationService
        mock_route = {
            "distance_meters": 4023,
            "duration_seconds": 480,
            "overview_polyline": "encoded_polyline_string",
            "steps": [{
                "instruction": "Head north on 5th Ave",
                "distance_meters": 804,
                "duration_seconds": 120,
                "start_location": {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 10.0},
                "end_location": {"latitude": 40.7589, "longitude": -73.9851, "accuracy": 10.0},
                "maneuver": "straight"
            }]
        }
        
        with patch("app.api.navigation.navigation_service.calculate_route") as mock_calc:
            mock_calc.return_value = [mock_route]
            
            response = client.post("/navigation/routes/calculate", json=sample_route_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["routes"]) == 1
            assert data["routes"][0]["distance_meters"] == 4023
            assert data["routes"][0]["duration_seconds"] == 480
    
    def test_calculate_route_invalid_coordinates(self, client):
        """Test route calculation with invalid coordinates."""
        invalid_request = {
            "origin": {
                "latitude": 91.0,  # Invalid latitude
                "longitude": -74.0060,
                "accuracy": 10.0
            },
            "destination": {
                "latitude": 40.7589,
                "longitude": -73.9851,
                "accuracy": 10.0
            },
            "transport_mode": "driving",
            "language": "en"
        }
        
        response = client.post("/navigation/routes/calculate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_poi_search_success(self, client):
        """Test successful POI search."""
        search_request = {
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "accuracy": 10.0
            },
            "categories": ["restaurant", "tourist_attraction"],
            "radius_meters": 1000,
            "max_results": 10,
            "language": "en"
        }
        
        mock_pois = [{
            "name": "Central Park",
            "category": "tourist_attraction",
            "coordinates": {"latitude": 40.7829, "longitude": -73.9654, "accuracy": 10.0},
            "rating": 4.5,
            "price_level": 0,
            "place_id": "ChIJqaIXmTlawokR1k0_1Nt_2Gc"
        }]
        
        with patch("app.api.navigation.navigation_service.search_nearby_pois") as mock_search:
            mock_search.return_value = mock_pois
            
            response = client.post("/navigation/poi/search", json=search_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["pois"]) == 1
            assert data["pois"][0]["name"] == "Central Park"
    
    def test_geocode_address_success(self, client):
        """Test successful address geocoding."""
        geocode_request = {
            "address": "Empire State Building, New York"
        }
        
        mock_result = {
            "coordinates": {"latitude": 40.7484405, "longitude": -73.9856644, "accuracy": 10.0},
            "formatted_address": "350 5th Ave, New York, NY 10118, USA",
            "place_id": "ChIJaXQRs6lZwokRY6tbFzl5AiU",
            "accuracy": "ROOFTOP"
        }
        
        with patch("app.api.navigation.navigation_service.geocode_address") as mock_geocode:
            mock_geocode.return_value = mock_result
            
            response = client.post("/navigation/geocode", json=geocode_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["coordinates"]["latitude"] == 40.7484405
            assert data["formatted_address"] == "350 5th Ave, New York, NY 10118, USA"
    
    def test_reverse_geocode_success(self, client):
        """Test successful reverse geocoding."""
        reverse_request = {
            "coordinates": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "accuracy": 10.0
            }
        }
        
        mock_result = {
            "formatted_address": "New York, NY 10007, USA",
            "place_id": "ChIJOwg_06VPwokRYv534QaPC8g"
        }
        
        with patch("app.api.navigation.navigation_service.reverse_geocode") as mock_reverse:
            mock_reverse.return_value = mock_result
            
            response = client.post("/navigation/geocode/reverse", json=reverse_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "formatted_addresses" in data
    
    def test_current_location_info_success(self, client):
        """Test successful current location info retrieval."""
        location_request = {
            "coordinates": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "accuracy": 10.0
            },
            "include_address": True,
            "include_nearby_pois": True,
            "poi_categories": ["restaurant", "tourist_attraction"]
        }
        
        mock_info = {
            "address": "New York, NY, USA",
            "formatted_address": "New York, NY 10007, USA",
            "nearby_pois": [{
                "name": "Battery Park",
                "category": "tourist_attraction",
                "coordinates": {"latitude": 40.7033, "longitude": -74.0170, "accuracy": 10.0}
            }],
            "timezone": "America/New_York",
            "country_code": "US",
            "administrative_area": "NY",
            "locality": "New York"
        }
        
        with patch("app.api.navigation.navigation_service.get_current_location_info") as mock_info_call:
            mock_info_call.return_value = mock_info
            
            response = client.post("/navigation/location/current", json=location_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["formatted_address"] == "New York, NY 10007, USA"
            assert len(data["nearby_pois"]) == 1
    
    def test_navigation_session_lifecycle(self, client, sample_route_request):
        """Test complete navigation session lifecycle."""
        # Mock route calculation
        mock_route = {
            "distance_meters": 4023,
            "duration_seconds": 480,
            "overview_polyline": "encoded_polyline_string",
            "steps": [{
                "instruction": "Head north on 5th Ave",
                "distance_meters": 804,
                "duration_seconds": 120,
                "start_location": {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 10.0},
                "end_location": {"latitude": 40.7589, "longitude": -73.9851, "accuracy": 10.0},
                "maneuver": "straight"
            }]
        }
        
        with patch("app.api.navigation.navigation_service.calculate_route") as mock_calc:
            mock_calc.return_value = [mock_route]
            
            # Start navigation session
            session_request = {
                "navigation_request": sample_route_request,
                "enable_voice_guidance": True,
                "voice_language": "en"
            }
            
            response = client.post("/navigation/sessions/start", json=session_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "session_id" in data
            assert data["voice_enabled"] is True
            
            session_id = data["session_id"]
            
            # Configure voice settings
            voice_request = {
                "enable_voice": True,
                "language": "en",
                "voice_speed": 1.0,
                "distance_units": "metric",
                "announce_traffic": True
            }
            
            response = client.post(f"/navigation/sessions/{session_id}/voice", json=voice_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["voice_settings"]["enabled"] is True
            
            # Update navigation location
            update_request = {
                "coordinates": {
                    "latitude": 40.7200,
                    "longitude": -74.0000,
                    "accuracy": 5.0
                },
                "heading": 45.0,
                "speed_mps": 10.0
            }
            
            response = client.post(f"/navigation/sessions/{session_id}/update", json=update_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == session_id
            
            # Stop navigation session
            response = client.delete(f"/navigation/sessions/{session_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_multilingual_support(self, client, sample_route_request):
        """Test multilingual navigation support."""
        languages_to_test = ["en", "th", "es", "fr"]
        
        for language in languages_to_test:
            request = sample_route_request.copy()
            request["language"] = language
            
            mock_route = {
                "distance_meters": 4023,
                "duration_seconds": 480,
                "overview_polyline": "encoded_polyline_string",
                "steps": [{
                    "instruction": f"Instruction in {language}",
                    "distance_meters": 804,
                    "duration_seconds": 120,
                    "start_location": {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 10.0},
                    "end_location": {"latitude": 40.7589, "longitude": -73.9851, "accuracy": 10.0},
                    "maneuver": "straight"
                }]
            }
            
            with patch("app.api.navigation.navigation_service.calculate_route") as mock_calc:
                mock_calc.return_value = [mock_route]
                
                response = client.post("/navigation/routes/calculate", json=request)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
    
    def test_error_handling_service_unavailable(self, client, sample_route_request):
        """Test error handling when navigation service is unavailable."""
        with patch("app.api.navigation.navigation_service.calculate_route") as mock_calc:
            mock_calc.side_effect = Exception("Service unavailable")
            
            response = client.post("/navigation/routes/calculate", json=sample_route_request)
            
            assert response.status_code == 400
            data = response.json()
            assert "Route calculation failed" in data["detail"]
    
    def test_performance_requirements(self, client, sample_route_request):
        """Test that API endpoints meet performance requirements."""
        import time
        
        mock_route = {
            "distance_meters": 4023,
            "duration_seconds": 480,
            "overview_polyline": "encoded_polyline_string",
            "steps": []
        }
        
        with patch("app.api.navigation.navigation_service.calculate_route") as mock_calc:
            mock_calc.return_value = [mock_route]
            
            start_time = time.time()
            response = client.post("/navigation/routes/calculate", json=sample_route_request)
            end_time = time.time()
            
            assert response.status_code == 200
            # API response should be fast (under 1 second for mocked service)
            assert (end_time - start_time) < 1.0
    
    def test_voice_navigation_integration(self, client):
        """Test voice navigation integration with TTS and translation services."""
        # Test voice instruction generation
        session_request = {
            "navigation_request": {
                "origin": {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 10.0},
                "destination": {"latitude": 40.7589, "longitude": -73.9851, "accuracy": 10.0},
                "transport_mode": "driving",
                "language": "en"
            },
            "enable_voice_guidance": True,
            "voice_language": "th"  # Test Thai language
        }
        
        mock_route = {
            "distance_meters": 4023,
            "duration_seconds": 480,
            "overview_polyline": "encoded_polyline_string",
            "steps": [{
                "instruction": "Head north",
                "distance_meters": 804,
                "duration_seconds": 120,
                "start_location": {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 10.0},
                "end_location": {"latitude": 40.7589, "longitude": -73.9851, "accuracy": 10.0},
                "maneuver": "straight"
            }]
        }
        
        with patch("app.api.navigation.navigation_service.calculate_route") as mock_calc:
            mock_calc.return_value = [mock_route]
            
            response = client.post("/navigation/sessions/start", json=session_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["voice_enabled"] is True
