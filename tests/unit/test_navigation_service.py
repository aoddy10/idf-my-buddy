"""Unit tests for NavigationService.

Tests the core navigation functionality including route calculation,
POI discovery, geocoding, and location services.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.services.maps import NavigationService
from app.schemas.navigation import (
    NavigationRequest, TransportMode, POICategory
)
from app.schemas.common import LanguageCode, Coordinates


class TestNavigationService:
    """Test cases for NavigationService."""
    
    @pytest.fixture
    def navigation_service(self):
        """Create NavigationService instance for testing."""
        return NavigationService()
    
    @pytest.fixture
    def sample_coordinates(self):
        """Sample coordinates for testing."""
        return {
            "origin": Coordinates(latitude=40.7128, longitude=-74.0060, accuracy=10.0),  # NYC
            "destination": Coordinates(latitude=40.7589, longitude=-73.9851, accuracy=10.0)  # Times Square
        }
    
    @pytest.fixture
    def sample_navigation_request(self, sample_coordinates):
        """Sample navigation request for testing."""
        return NavigationRequest(
            origin=sample_coordinates["origin"],
            destination=sample_coordinates["destination"],
            transport_mode=TransportMode.DRIVING,
            language=LanguageCode.EN,
            avoid_tolls=False,
            avoid_highways=False,
            avoid_ferries=False,
            departure_time=None
        )

    @pytest.mark.asyncio
    async def test_calculate_route_success(self, navigation_service, sample_navigation_request):
        """Test successful route calculation."""
        # Mock Google Maps client
        mock_directions_result = [{
            'legs': [{
                'distance': {'text': '2.5 mi', 'value': 4023},
                'duration': {'text': '8 mins', 'value': 480},
                'steps': [{
                    'html_instructions': 'Head north on 5th Ave',
                    'distance': {'text': '0.5 mi', 'value': 804},
                    'duration': {'text': '2 mins', 'value': 120},
                    'maneuver': 'straight',
                    'start_location': {'lat': 40.7128, 'lng': -74.0060},
                    'end_location': {'lat': 40.7589, 'lng': -73.9851}
                }]
            }],
            'overview_polyline': {'points': 'encoded_polyline_string'}
        }]
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.directions.return_value = mock_directions_result
            
            routes = await navigation_service.calculate_route(sample_navigation_request)
            
            assert len(routes) == 1
            route = routes[0]
            assert route.distance_meters == 4023
            assert route.duration_seconds == 480
            assert len(route.steps) == 1
            assert route.steps[0].instruction == "Head north on 5th Ave"
            
            # Verify Google Maps client was called correctly
            mock_client.directions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_route_fallback_to_osm(self, navigation_service, sample_navigation_request):
        """Test fallback to OpenStreetMap when Google Maps fails."""
        # Mock Google Maps client failure
        with patch.object(navigation_service, '_google_client') as mock_google_client:
            mock_google_client.directions.side_effect = Exception("Google Maps API error")
            
            # Mock OSM fallback
            with patch.object(navigation_service, '_calculate_route_osm') as mock_osm:
                mock_route = MagicMock()
                mock_route.distance_meters = 4000
                mock_route.duration_seconds = 500
                mock_osm.return_value = [mock_route]
                
                routes = await navigation_service.calculate_route(sample_navigation_request)
                
                assert len(routes) == 1
                assert routes[0].distance_meters == 4000
                mock_osm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_nearby_pois_success(self, navigation_service, sample_coordinates):
        """Test successful POI search."""
        mock_places_result = {
            'results': [{
                'name': 'Central Park',
                'place_id': 'ChIJqaIXmTlawokR1k0_1Nt_2Gc',
                'geometry': {
                    'location': {'lat': 40.7829, 'lng': -73.9654}
                },
                'types': ['park', 'tourist_attraction'],
                'rating': 4.5,
                'price_level': 0
            }]
        }
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.places_nearby.return_value = mock_places_result
            
            pois = await navigation_service.search_nearby_pois(
                coordinates=sample_coordinates["origin"],
                categories=[POICategory.TOURIST_ATTRACTION],
                radius_meters=1000,
                keyword="park",
                max_results=10
            )
            
            assert len(pois) == 1
            poi = pois[0]
            assert poi.name == "Central Park"
            assert poi.category == POICategory.TOURIST_ATTRACTION
            assert poi.rating == 4.5
            
            # Verify Google Places client was called correctly
            mock_client.places_nearby.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_geocode_address_success(self, navigation_service):
        """Test successful address geocoding."""
        mock_geocode_result = [{
            'formatted_address': '350 5th Ave, New York, NY 10118, USA',
            'geometry': {
                'location': {'lat': 40.7484405, 'lng': -73.9856644}
            },
            'place_id': 'ChIJaXQRs6lZwokRY6tbFzl5AiU'
        }]
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.geocode.return_value = mock_geocode_result
            
            result = await navigation_service.geocode_address(
                address="Empire State Building, New York",
                country_code="US"
            )
            
            assert result["formatted_address"] == "350 5th Ave, New York, NY 10118, USA"
            assert result["coordinates"].latitude == 40.7484405
            assert result["coordinates"].longitude == -73.9856644
            assert result["place_id"] == "ChIJaXQRs6lZwokRY6tbFzl5AiU"
    
    @pytest.mark.asyncio
    async def test_reverse_geocode_success(self, navigation_service, sample_coordinates):
        """Test successful reverse geocoding."""
        mock_reverse_result = [{
            'formatted_address': 'New York, NY 10007, USA',
            'place_id': 'ChIJOwg_06VPwokRYv534QaPC8g'
        }]
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.reverse_geocode.return_value = mock_reverse_result
            
            result = await navigation_service.reverse_geocode(sample_coordinates["origin"])
            
            assert result["formatted_address"] == "New York, NY 10007, USA"
            assert result["place_id"] == "ChIJOwg_06VPwokRYv534QaPC8g"
    
    @pytest.mark.asyncio
    async def test_get_current_location_info_complete(self, navigation_service, sample_coordinates):
        """Test getting comprehensive location information."""
        # Mock reverse geocoding
        mock_reverse_result = [{
            'formatted_address': 'New York, NY 10007, USA',
            'address_components': [
                {'long_name': 'New York', 'types': ['locality']},
                {'long_name': 'NY', 'types': ['administrative_area_level_1']},
                {'long_name': 'US', 'types': ['country']}
            ]
        }]
        
        # Mock nearby POIs
        mock_places_result = {
            'results': [{
                'name': 'Battery Park',
                'geometry': {'location': {'lat': 40.7033, 'lng': -74.0170}},
                'types': ['park']
            }]
        }
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.reverse_geocode.return_value = mock_reverse_result
            mock_client.places_nearby.return_value = mock_places_result
            
            location_info = await navigation_service.get_current_location_info(
                coordinates=sample_coordinates["origin"],
                include_address=True,
                include_pois=True,
                poi_categories=[POICategory.TOURIST_ATTRACTION]
            )
            
            assert location_info["formatted_address"] == "New York, NY 10007, USA"
            assert location_info["locality"] == "New York"
            assert location_info["administrative_area"] == "NY" 
            assert location_info["country_code"] == "US"
            assert len(location_info["nearby_pois"]) == 1
            assert location_info["nearby_pois"][0].name == "Battery Park"
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, navigation_service, sample_navigation_request):
        """Test that results are properly cached."""
        mock_directions_result = [{
            'legs': [{'distance': {'value': 4023}, 'duration': {'value': 480}, 'steps': []}],
            'overview_polyline': {'points': 'encoded_polyline_string'}
        }]
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.directions.return_value = mock_directions_result
            
            # First call should hit the API
            routes1 = await navigation_service.calculate_route(sample_navigation_request)
            
            # Second identical call should use cache (mock should only be called once)
            routes2 = await navigation_service.calculate_route(sample_navigation_request)
            
            assert routes1 == routes2
            assert mock_client.directions.call_count == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_coordinates(self, navigation_service):
        """Test error handling for invalid coordinates."""
        invalid_request = NavigationRequest(
            origin=Coordinates(latitude=91.0, longitude=-74.0060, accuracy=10.0),  # Invalid latitude
            destination=Coordinates(latitude=40.7589, longitude=-73.9851, accuracy=10.0),
            transport_mode=TransportMode.DRIVING,
            language=LanguageCode.EN,
            departure_time=None
        )
        
        with pytest.raises(ValueError, match="Invalid coordinates"):
            await navigation_service.calculate_route(invalid_request)
    
    @pytest.mark.asyncio 
    async def test_transport_mode_handling(self, navigation_service, sample_coordinates):
        """Test different transport modes are handled correctly."""
        for transport_mode in [TransportMode.DRIVING, TransportMode.WALKING, TransportMode.BICYCLING, TransportMode.TRANSIT]:
            request = NavigationRequest(
                origin=sample_coordinates["origin"],
                destination=sample_coordinates["destination"],
                transport_mode=transport_mode,
                language=LanguageCode.EN,
                departure_time=None
            )
            
            with patch.object(navigation_service, '_google_client') as mock_client:
                mock_client.directions.return_value = [{
                    'legs': [{'distance': {'value': 1000}, 'duration': {'value': 300}, 'steps': []}],
                    'overview_polyline': {'points': 'test'}
                }]
                
                routes = await navigation_service.calculate_route(request)
                assert len(routes) == 1
                
                # Verify correct mode was passed to Google Maps API
                call_args = mock_client.directions.call_args
                assert call_args[1]['mode'] == transport_mode.value.lower()
    
    def test_poi_category_mapping(self, navigation_service):
        """Test POI category mapping to Google Places types."""
        category_mapping = navigation_service._get_google_place_types(POICategory.RESTAURANT)
        assert "restaurant" in category_mapping
        
        category_mapping = navigation_service._get_google_place_types(POICategory.GAS_STATION)
        assert "gas_station" in category_mapping
        
        category_mapping = navigation_service._get_google_place_types(POICategory.HOSPITAL)
        assert "hospital" in category_mapping
    
    @pytest.mark.asyncio
    async def test_performance_route_calculation(self, navigation_service, sample_navigation_request):
        """Test that route calculation completes within performance requirements."""
        import time
        
        mock_directions_result = [{
            'legs': [{'distance': {'value': 4023}, 'duration': {'value': 480}, 'steps': []}],
            'overview_polyline': {'points': 'encoded_polyline_string'}
        }]
        
        with patch.object(navigation_service, '_google_client') as mock_client:
            mock_client.directions.return_value = mock_directions_result
            
            start_time = time.time()
            routes = await navigation_service.calculate_route(sample_navigation_request)
            end_time = time.time()
            
            # Route calculation should complete in under 3 seconds (PRP requirement)
            assert (end_time - start_time) < 3.0
            assert len(routes) == 1
