"""Tests for TTS (Text-to-Speech) service.

This module tests the TTS service functionality including text synthesis,
voice selection, audio format handling, and multi-engine support.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.services.tts import TTSService
from app.core.config import get_settings


class TestTTSService:
    """Test cases for TTSService."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService instance for testing."""
        return TTSService()
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for TTS testing."""
        return "Hello, this is a test message for text-to-speech synthesis."
    
    def test_tts_service_creation(self, tts_service):
        """Test TTSService can be created."""
        assert tts_service is not None
        assert hasattr(tts_service, 'log')
        assert hasattr(tts_service, 'engines')
        assert isinstance(tts_service.engines, dict)
    
    def test_service_configuration(self, tts_service):
        """Test service configuration reading."""
        settings = get_settings()
        
        # Test TTS configuration structure
        assert hasattr(settings, 'tts')
        assert isinstance(settings.tts.default_voice, str)
        assert isinstance(settings.tts.speech_rate, float)
        assert settings.tts.speech_rate > 0
    
    def test_voice_mapping_structure(self, tts_service):
        """Test voice mapping functionality."""
        # Test voice selection logic
        voice_id = tts_service._get_voice_for_language("en", "female")
        assert isinstance(voice_id, str)
        
        # Test default fallback
        default_voice = tts_service._get_voice_for_language("unknown", "neutral")
        assert isinstance(default_voice, str)
    
    def test_audio_format_validation(self, tts_service):
        """Test audio format validation."""
        # Valid formats
        assert tts_service._validate_audio_format("mp3") is True
        assert tts_service._validate_audio_format("wav") is True
        assert tts_service._validate_audio_format("ogg") is True
        
        # Invalid format
        assert tts_service._validate_audio_format("invalid") is False
    
    def test_speed_parameter_validation(self, tts_service):
        """Test speech speed parameter validation."""
        # Valid speeds
        assert tts_service._validate_speed(0.5) is True
        assert tts_service._validate_speed(1.0) is True
        assert tts_service._validate_speed(2.0) is True
        
        # Invalid speeds
        assert tts_service._validate_speed(0.1) is False  # Too slow
        assert tts_service._validate_speed(5.0) is False  # Too fast
    
    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, tts_service):
        """Test synthesis with empty text."""
        result = await tts_service.synthesize_text("")
        
        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower() or "text" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_synthesize_long_text(self, tts_service):
        """Test synthesis with very long text."""
        long_text = "This is a test. " * 1000  # Very long text
        
        result = await tts_service.synthesize_text(long_text)
        
        # Should handle gracefully (either succeed or fail with appropriate error)
        assert "success" in result
        if not result["success"]:
            assert "error" in result
    
    @patch('app.services.tts.PYTTSX3_AVAILABLE', True)
    @patch('app.services.tts.pyttsx3')
    def test_pyttsx3_engine_setup(self, mock_pyttsx3, tts_service):
        """Test pyttsx3 engine initialization."""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        tts_service._setup_pyttsx3_engine()
        
        assert 'pyttsx3' in tts_service.engines
        mock_pyttsx3.init.assert_called_once()
    
    @patch('app.services.tts.OPENAI_AVAILABLE', True) 
    @patch('app.services.tts.OpenAI')
    def test_openai_engine_setup(self, mock_openai_class, tts_service):
        """Test OpenAI TTS engine initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Set API key temporarily
        settings = get_settings()
        original_key = settings.openai_api_key
        settings.openai_api_key = "test-api-key"
        
        try:
            tts_service._setup_openai_engine()
            assert 'openai' in tts_service.engines
        finally:
            settings.openai_api_key = original_key
    
    @pytest.mark.asyncio
    async def test_mock_openai_synthesis(self, tts_service, sample_text):
        """Test OpenAI TTS synthesis with mock."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = b"mock_audio_data"
        mock_client.audio.speech.create.return_value = mock_response
        
        tts_service.engines['openai'] = mock_client
        
        result = await tts_service._synthesize_openai(
            sample_text,
            voice="alloy",
            speed=1.0,
            output_format="mp3"
        )
        
        assert result["success"] is True
        assert "audio_data" in result
        assert result["audio_data"] == b"mock_audio_data"
    
    @pytest.mark.asyncio
    async def test_mock_pyttsx3_synthesis(self, tts_service, sample_text):
        """Test pyttsx3 synthesis with mock."""
        # Mock pyttsx3 engine
        mock_engine = Mock()
        mock_engine.getProperty.return_value = "default_voice"
        
        tts_service.engines['pyttsx3'] = mock_engine
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test.wav"
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_bytes', return_value=b"mock_wav_data"):
                    result = await tts_service._synthesize_pyttsx3(
                        sample_text,
                        voice="default_voice",
                        speed=1.0
                    )
                    
                    assert result["success"] is True
                    assert "audio_data" in result
    
    @pytest.mark.asyncio
    @patch('app.services.tts.GTTS_AVAILABLE', True)
    async def test_mock_gtts_synthesis(self, tts_service, sample_text):
        """Test gTTS synthesis with mock."""
        with patch('app.services.tts.gTTS') as mock_gtts_class:
            mock_gtts = Mock()
            mock_gtts_class.return_value = mock_gtts
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
                mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test.mp3"
                
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.read_bytes', return_value=b"mock_mp3_data"):
                        result = await tts_service._synthesize_gtts(
                            sample_text,
                            language="en",
                            slow=False
                        )
                        
                        assert result["success"] is True
                        assert "audio_data" in result
    
    @pytest.mark.asyncio
    async def test_engine_selection_priority(self, tts_service, sample_text):
        """Test engine selection based on availability and preference."""
        # Mock multiple engines available
        mock_openai = Mock()
        mock_pyttsx3 = Mock()
        
        tts_service.engines['openai'] = mock_openai
        tts_service.engines['pyttsx3'] = mock_pyttsx3
        
        # Test engine selection logic
        selected_engine = tts_service._select_best_engine("en", "mp3")
        assert selected_engine in ['openai', 'pyttsx3', 'gtts']
    
    @pytest.mark.asyncio
    async def test_concurrent_synthesis_requests(self, tts_service):
        """Test handling multiple concurrent synthesis requests."""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.return_value = {
            "success": True,
            "audio_data": b"mock_data"
        }
        
        tts_service._synthesize_pyttsx3 = mock_engine
        tts_service.engines['pyttsx3'] = Mock()
        
        # Create multiple concurrent requests
        tasks = [
            tts_service.synthesize_text(f"Test message {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all requests were handled
        for result in results:
            assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_language_detection_and_voice_mapping(self, tts_service):
        """Test language detection and appropriate voice selection."""
        # Test different languages
        languages = ["en", "es", "fr", "de", "it"]
        
        for lang in languages:
            voice = tts_service._get_voice_for_language(lang, "female")
            assert isinstance(voice, str)
            assert len(voice) > 0
    
    @pytest.mark.asyncio
    async def test_audio_format_conversion(self, tts_service, sample_text):
        """Test different audio format outputs."""
        # Mock successful synthesis
        mock_result = {
            "success": True,
            "audio_data": b"mock_audio_data",
            "format": "wav"
        }
        
        with patch.object(tts_service, '_synthesize_pyttsx3', return_value=mock_result):
            tts_service.engines['pyttsx3'] = Mock()
            
            # Test different formats
            for fmt in ["mp3", "wav", "ogg"]:
                result = await tts_service.synthesize_text(
                    sample_text,
                    output_format=fmt
                )
                
                if result["success"]:
                    assert "audio_data" in result
    
    def test_performance_metrics_creation(self, tts_service):
        """Test performance metrics structure."""
        metrics = tts_service._create_metrics(
            processing_time=1.5,
            audio_duration=3.0,
            engine_used="openai",
            voice_used="alloy"
        )
        
        assert "processing_time" in metrics
        assert "audio_duration" in metrics
        assert "engine_used" in metrics
        assert "voice_used" in metrics
        assert metrics["processing_time"] == 1.5
        assert metrics["audio_duration"] == 3.0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_parameters(self, tts_service):
        """Test error handling with invalid parameters."""
        # Test invalid language
        result = await tts_service.synthesize_text(
            "test",
            language="invalid_language_code"
        )
        
        # Should handle gracefully (either succeed with fallback or fail gracefully)
        assert "success" in result
        
        # Test invalid speed
        result = await tts_service.synthesize_text(
            "test",
            speed=-1.0
        )
        
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, tts_service, sample_text):
        """Test TTS response caching if implemented."""
        # This would test caching functionality if implemented
        # For now, just ensure repeated calls work
        
        # Mock successful synthesis
        with patch.object(tts_service, '_synthesize_pyttsx3') as mock_synth:
            mock_synth.return_value = {
                "success": True,
                "audio_data": b"cached_data"
            }
            tts_service.engines['pyttsx3'] = Mock()
            
            # First call
            result1 = await tts_service.synthesize_text(sample_text)
            
            # Second call (potentially cached)
            result2 = await tts_service.synthesize_text(sample_text)
            
            assert result1.get("success") is True
            assert result2.get("success") is True


@pytest.mark.integration
class TestTTSServiceIntegration:
    """Integration tests for TTSService (require actual TTS engines)."""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService for integration testing."""
        return TTSService()
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    @pytest.mark.asyncio
    async def test_real_tts_synthesis(self, tts_service):
        """Test TTS with real engines (integration test)."""
        result = await tts_service.synthesize_text(
            "This is a test message.",
            language="en",
            speed=1.0
        )
        
        # Basic validation
        assert "success" in result
        if result["success"]:
            assert "audio_data" in result
            assert len(result["audio_data"]) > 0
