"""Unit tests for Voice Navigation Service.

Tests the voice instruction templates and voice navigation functionality
including multilingual support and instruction generation.
"""

import pytest
from unittest.mock import MagicMock

from app.services.voice_navigation import (
    VoiceInstructionTemplates, VoiceNavigationService, NavigationInstructionType
)
from app.schemas.common import LanguageCode


class TestVoiceInstructionTemplates:
    """Test cases for VoiceInstructionTemplates."""
    
    def test_get_instruction_template_english(self):
        """Test getting instruction templates in English."""
        template = VoiceInstructionTemplates.get_instruction_template(
            NavigationInstructionType.TURN_LEFT,
            LanguageCode.EN
        )
        
        assert "turn left" in template.lower()
        assert "{distance}" in template
        assert "{street}" in template
    
    def test_get_instruction_template_thai(self):
        """Test getting instruction templates in Thai."""
        template = VoiceInstructionTemplates.get_instruction_template(
            NavigationInstructionType.TURN_LEFT,
            LanguageCode.TH
        )
        
        assert "เลี้ยวซ้าย" in template  # "turn left" in Thai
        assert "{distance}" in template
        assert "{street}" in template
    
    def test_get_instruction_template_fallback(self):
        """Test fallback to English for unsupported languages."""
        # Test with a language that might not have full template coverage
        template = VoiceInstructionTemplates.get_instruction_template(
            NavigationInstructionType.TURN_LEFT,
            LanguageCode.ZH  # Chinese - might fall back to English
        )
        
        assert "{distance}" in template
        assert "{street}" in template
    
    def test_format_distance_metric_english(self):
        """Test distance formatting in metric units (English)."""
        # Test various distances
        assert "now" in VoiceInstructionTemplates.format_distance(5, LanguageCode.EN, False).lower()
        assert "soon" in VoiceInstructionTemplates.format_distance(25, LanguageCode.EN, False).lower()
        assert "100 meters" in VoiceInstructionTemplates.format_distance(100, LanguageCode.EN, False)
        assert "1.5 kilometers" in VoiceInstructionTemplates.format_distance(1500, LanguageCode.EN, False)
    
    def test_format_distance_imperial_english(self):
        """Test distance formatting in imperial units (English)."""
        # Test various distances in imperial
        distance_ft = VoiceInstructionTemplates.format_distance(100, LanguageCode.EN, True)
        assert "feet" in distance_ft
        
        distance_mi = VoiceInstructionTemplates.format_distance(2000, LanguageCode.EN, True)
        assert "mile" in distance_mi
    
    def test_format_distance_thai(self):
        """Test distance formatting in Thai."""
        distance = VoiceInstructionTemplates.format_distance(100, LanguageCode.TH, False)
        assert "เมตร" in distance  # "meters" in Thai
        
        distance = VoiceInstructionTemplates.format_distance(1500, LanguageCode.TH, False)
        assert "กิโลเมตร" in distance  # "kilometers" in Thai
    
    def test_create_voice_instruction_turn_left(self):
        """Test creating turn left instruction."""
        instruction = VoiceInstructionTemplates.create_voice_instruction(
            NavigationInstructionType.TURN_LEFT,
            LanguageCode.EN,
            distance_meters=200,
            street_name="Main Street"
        )
        
        assert "200 meters" in instruction
        assert "turn left" in instruction.lower()
        assert "Main Street" in instruction
    
    def test_create_voice_instruction_roundabout(self):
        """Test creating roundabout instruction."""
        instruction = VoiceInstructionTemplates.create_voice_instruction(
            NavigationInstructionType.ENTER_ROUNDABOUT,
            LanguageCode.EN,
            distance_meters=100,
            exit_number=2
        )
        
        assert "100 meters" in instruction
        assert "roundabout" in instruction.lower()
        assert "2nd exit" in instruction
    
    def test_create_voice_instruction_destination_reached(self):
        """Test creating destination reached instruction."""
        instruction = VoiceInstructionTemplates.create_voice_instruction(
            NavigationInstructionType.DESTINATION_REACHED,
            LanguageCode.EN
        )
        
        assert "arrived" in instruction.lower() or "destination" in instruction.lower()
    
    def test_create_voice_instruction_multilingual(self):
        """Test creating instructions in multiple languages."""
        languages = [LanguageCode.EN, LanguageCode.TH, LanguageCode.ES, LanguageCode.FR]
        
        for language in languages:
            instruction = VoiceInstructionTemplates.create_voice_instruction(
                NavigationInstructionType.TURN_RIGHT,
                language,
                distance_meters=150,
                street_name="Oak Avenue"
            )
            
            assert instruction  # Should not be empty
            assert "Oak Avenue" in instruction
            # Each language should have different text
    
    def test_exit_number_formatting_languages(self):
        """Test exit number formatting in different languages."""
        # English: 1st, 2nd, 3rd, 4th
        instruction_en = VoiceInstructionTemplates.create_voice_instruction(
            NavigationInstructionType.ENTER_ROUNDABOUT,
            LanguageCode.EN,
            distance_meters=100,
            exit_number=1
        )
        assert "1st" in instruction_en
        
        # Test other exit numbers
        instruction_en_2 = VoiceInstructionTemplates.create_voice_instruction(
            NavigationInstructionType.ENTER_ROUNDABOUT,
            LanguageCode.EN,
            distance_meters=100,
            exit_number=2
        )
        assert "2nd" in instruction_en_2
    
    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = VoiceInstructionTemplates.get_supported_languages()
        
        assert LanguageCode.EN in languages
        assert LanguageCode.TH in languages
        assert LanguageCode.ES in languages
        assert LanguageCode.FR in languages
        assert len(languages) >= 4
    
    def test_detect_instruction_type(self):
        """Test detecting instruction type from maneuver strings."""
        # Test common Google Maps maneuvers
        assert VoiceInstructionTemplates.detect_instruction_type("turn-left") == NavigationInstructionType.TURN_LEFT
        assert VoiceInstructionTemplates.detect_instruction_type("turn-right") == NavigationInstructionType.TURN_RIGHT
        assert VoiceInstructionTemplates.detect_instruction_type("straight") == NavigationInstructionType.CONTINUE_STRAIGHT
        assert VoiceInstructionTemplates.detect_instruction_type("merge") == NavigationInstructionType.MERGE
        assert VoiceInstructionTemplates.detect_instruction_type("roundabout-left") == NavigationInstructionType.ENTER_ROUNDABOUT
        
        # Test unknown maneuver defaults to straight
        assert VoiceInstructionTemplates.detect_instruction_type("unknown-maneuver") == NavigationInstructionType.CONTINUE_STRAIGHT


class TestVoiceNavigationService:
    """Test cases for VoiceNavigationService."""
    
    @pytest.fixture
    def voice_nav_service(self):
        """Create VoiceNavigationService instance for testing."""
        return VoiceNavigationService()
    
    def test_set_voice_settings(self, voice_nav_service):
        """Test setting voice settings for a session."""
        session_id = "test-session-123"
        
        voice_nav_service.set_voice_settings(
            session_id=session_id,
            language=LanguageCode.TH,
            use_imperial=True,
            voice_speed=1.2,
            announce_distance_threshold=300
        )
        
        settings = voice_nav_service.active_voice_settings[session_id]
        assert settings["language"] == LanguageCode.TH
        assert settings["use_imperial"] is True
        assert settings["voice_speed"] == 1.2
        assert settings["announce_distance_threshold"] == 300
    
    def test_get_voice_instruction_within_threshold(self, voice_nav_service):
        """Test getting voice instruction within announcement threshold."""
        session_id = "test-session-123"
        
        # Set voice settings with 500m threshold
        voice_nav_service.set_voice_settings(
            session_id=session_id,
            language=LanguageCode.EN,
            announce_distance_threshold=500
        )
        
        # Test distance within threshold
        instruction = voice_nav_service.get_voice_instruction_for_step(
            session_id=session_id,
            maneuver="turn-left",
            distance_meters=200,
            street_name="Main Street"
        )
        
        assert instruction is not None
        assert "turn left" in instruction.lower()
        assert "Main Street" in instruction
    
    def test_get_voice_instruction_beyond_threshold(self, voice_nav_service):
        """Test that no instruction is returned beyond announcement threshold."""
        session_id = "test-session-123"
        
        # Set voice settings with 500m threshold
        voice_nav_service.set_voice_settings(
            session_id=session_id,
            language=LanguageCode.EN,
            announce_distance_threshold=500
        )
        
        # Test distance beyond threshold
        instruction = voice_nav_service.get_voice_instruction_for_step(
            session_id=session_id,
            maneuver="turn-left",
            distance_meters=800,  # Beyond 500m threshold
            street_name="Main Street"
        )
        
        assert instruction is None
    
    def test_get_voice_instruction_default_settings(self, voice_nav_service):
        """Test getting instruction with default settings for unknown session."""
        instruction = voice_nav_service.get_voice_instruction_for_step(
            session_id="unknown-session",
            maneuver="turn-right",
            distance_meters=100,
            street_name="Oak Street"
        )
        
        assert instruction is not None
        assert "turn right" in instruction.lower()
        assert "Oak Street" in instruction
    
    def test_get_voice_instruction_multilingual(self, voice_nav_service):
        """Test getting instructions in different languages."""
        sessions = {
            "en-session": LanguageCode.EN,
            "th-session": LanguageCode.TH,
            "es-session": LanguageCode.ES
        }
        
        for session_id, language in sessions.items():
            voice_nav_service.set_voice_settings(
                session_id=session_id,
                language=language
            )
            
            instruction = voice_nav_service.get_voice_instruction_for_step(
                session_id=session_id,
                maneuver="turn-left",
                distance_meters=150,
                street_name="Test Street"
            )
            
            assert instruction is not None
            assert "Test Street" in instruction
            # Instructions should be different for different languages
    
    def test_get_voice_instruction_imperial_units(self, voice_nav_service):
        """Test getting instructions with imperial units."""
        session_id = "imperial-session"
        
        voice_nav_service.set_voice_settings(
            session_id=session_id,
            language=LanguageCode.EN,
            use_imperial=True
        )
        
        instruction = voice_nav_service.get_voice_instruction_for_step(
            session_id=session_id,
            maneuver="turn-right",
            distance_meters=400,  # Should be converted to feet
            street_name="Highway 101"
        )
        
        assert instruction is not None
        assert ("feet" in instruction or "mile" in instruction)
        assert "Highway 101" in instruction
    
    def test_cleanup_session(self, voice_nav_service):
        """Test cleaning up voice settings for a session."""
        session_id = "test-session-cleanup"
        
        # Set voice settings
        voice_nav_service.set_voice_settings(
            session_id=session_id,
            language=LanguageCode.ES
        )
        
        assert session_id in voice_nav_service.active_voice_settings
        
        # Cleanup session
        voice_nav_service.cleanup_session(session_id)
        
        assert session_id not in voice_nav_service.active_voice_settings
    
    def test_multiple_sessions_isolation(self, voice_nav_service):
        """Test that multiple sessions are properly isolated."""
        session_1 = "session-1"
        session_2 = "session-2"
        
        # Set different settings for each session
        voice_nav_service.set_voice_settings(
            session_id=session_1,
            language=LanguageCode.EN,
            use_imperial=False
        )
        
        voice_nav_service.set_voice_settings(
            session_id=session_2,
            language=LanguageCode.TH,
            use_imperial=True
        )
        
        # Test instructions are generated with correct settings
        instruction_1 = voice_nav_service.get_voice_instruction_for_step(
            session_id=session_1,
            maneuver="turn-left",
            distance_meters=100
        )
        
        instruction_2 = voice_nav_service.get_voice_instruction_for_step(
            session_id=session_2,
            maneuver="turn-left", 
            distance_meters=100
        )
        
        # Instructions should be different due to different language settings
        assert instruction_1 != instruction_2
        
        # Test cleanup doesn't affect other sessions
        voice_nav_service.cleanup_session(session_1)
        assert session_1 not in voice_nav_service.active_voice_settings
        assert session_2 in voice_nav_service.active_voice_settings
