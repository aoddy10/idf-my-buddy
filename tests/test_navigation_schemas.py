"""Simple navigation schema validation test."""

import pytest
from datetime import datetime, timezone
from app.schemas.navigation import NavigationRequest, TransportMode
from app.schemas.common import Coordinates, LanguageCode
from app.services.voice_navigation import VoiceInstructionTemplates


def test_navigation_coordinates():
    """Test Coordinates schema validation."""
    coords = Coordinates(latitude=13.7563, longitude=100.5018, accuracy=None)
    assert coords.latitude == 13.7563
    assert coords.longitude == 100.5018
    assert coords.accuracy is None


def test_navigation_request():
    """Test NavigationRequest schema validation."""
    origin = Coordinates(latitude=13.7563, longitude=100.5018, accuracy=None)
    destination = Coordinates(latitude=13.7650, longitude=100.5379, accuracy=None)
    
    nav_request = NavigationRequest(
        origin=origin,
        destination=destination,
        transport_mode=TransportMode.DRIVING,
        departure_time=datetime.now(timezone.utc),
        language=LanguageCode.EN
    )
    
    assert nav_request.origin.latitude == 13.7563
    assert nav_request.destination.latitude == 13.7650
    assert nav_request.transport_mode == TransportMode.DRIVING
    assert nav_request.language == LanguageCode.EN


def test_voice_instruction_templates():
    """Test voice instruction template basic functionality."""
    voice = VoiceInstructionTemplates()
    
    # Test that the class can be instantiated
    assert voice is not None
    assert hasattr(voice, 'get_instruction_template')


def test_transport_modes():
    """Test transport mode enumeration."""
    assert TransportMode.WALKING == "walking"
    assert TransportMode.DRIVING == "driving"
    assert TransportMode.BICYCLING == "bicycling" 
    assert TransportMode.TRANSIT == "transit"


def test_language_codes():
    """Test language code enumeration."""
    assert LanguageCode.EN == "en"
    assert LanguageCode.TH == "th"
    assert LanguageCode.ES == "es"
    
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
