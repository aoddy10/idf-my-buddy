"""Tests for Whisper ASR service.

This module tests the Whisper speech recognition service functionality
including model loading, audio transcription, and error handling.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.services.whisper import WhisperService
from app.core.config import get_settings


class TestWhisperService:
    """Test cases for WhisperService."""
    
    @pytest.fixture
    def whisper_service(self):
        """Create WhisperService instance for testing."""
        return WhisperService()
    
    @pytest.fixture
    def sample_audio_data(self) -> bytes:
        """Create mock audio data for testing."""
        # Return minimal WAV header + silence
        return b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    @pytest.fixture
    def temp_audio_file(self, sample_audio_data: bytes):
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(sample_audio_data)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_whisper_service_creation(self, whisper_service):
        """Test WhisperService can be created."""
        assert whisper_service is not None
        assert hasattr(whisper_service, 'log')
        assert whisper_service.local_model is None
        assert whisper_service.openai_client is None
    
    def test_service_setup_configuration(self, whisper_service):
        """Test service setup reads configuration correctly."""
        settings = get_settings()
        
        # Test that service respects configuration
        assert isinstance(settings.whisper.use_local, bool)
        assert settings.whisper.model_size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        assert settings.openai_api_key is not None or settings.openai_api_key == ""
    
    @patch('app.services.whisper.WHISPER_AVAILABLE', True)
    @patch('app.services.whisper.whisper')
    def test_setup_local_model(self, mock_whisper, whisper_service):
        """Test local model setup."""
        # Mock whisper.load_model
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        whisper_service._setup_service()
        
        assert whisper_service.local_model is not None
        mock_whisper.load_model.assert_called_once()
    
    @patch('app.services.whisper.OPENAI_AVAILABLE', True)
    @patch('app.services.whisper.OpenAI')
    def test_setup_openai_client(self, mock_openai_class, whisper_service):
        """Test OpenAI client setup."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Temporarily set API key
        settings = get_settings()
        original_key = settings.openai_api_key
        settings.openai_api_key = "test-api-key"
        
        try:
            whisper_service._setup_service()
            assert whisper_service.openai_client is not None
            mock_openai_class.assert_called_once()
        finally:
            # Restore original key
            settings.openai_api_key = original_key
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_found(self, whisper_service):
        """Test transcription with non-existent file."""
        non_existent_path = Path("/non/existent/file.wav")
        
        result = await whisper_service.transcribe_audio(non_existent_path)
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_invalid_format(self, whisper_service):
        """Test transcription with invalid audio format."""
        # Create a text file with audio extension
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"This is not audio data")
            temp_path = Path(f.name)
        
        try:
            result = await whisper_service.transcribe_audio(temp_path)
            
            assert result["success"] is False
            assert "error" in result
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.asyncio
    @patch('app.services.whisper.WHISPER_AVAILABLE', True)
    async def test_transcribe_with_mock_local_model(self, whisper_service, temp_audio_file):
        """Test transcription using mocked local model."""
        # Mock the local model
        mock_model = AsyncMock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": []
        }
        whisper_service.local_model = mock_model
        
        result = await whisper_service.transcribe_audio(
            temp_audio_file,
            language="en",
            return_timestamps=False
        )
        
        assert result["success"] is True
        assert "text" in result
        assert "language" in result
        assert "processing_time" in result
    
    @pytest.mark.asyncio
    @patch('app.services.whisper.OPENAI_AVAILABLE', True)
    async def test_transcribe_with_mock_openai(self, whisper_service, temp_audio_file):
        """Test transcription using mocked OpenAI API."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_transcription = Mock()
        mock_transcription.text = "Hello from OpenAI"
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        whisper_service.openai_client = mock_client
        
        result = await whisper_service.transcribe_audio(
            temp_audio_file,
            language="en"
        )
        
        assert result["success"] is True
        assert "text" in result
        mock_client.audio.transcriptions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_bytes(self, whisper_service, sample_audio_data):
        """Test transcription with raw audio bytes."""
        # Mock local model for this test
        mock_model = AsyncMock()
        mock_model.transcribe.return_value = {
            "text": "Test transcription",
            "language": "en"
        }
        whisper_service.local_model = mock_model
        
        result = await whisper_service.transcribe_audio(sample_audio_data)
        
        assert result["success"] is True
        assert "text" in result
    
    def test_language_detection_parameter(self, whisper_service):
        """Test language detection parameter handling."""
        # Test with None (auto-detect)
        assert whisper_service._get_language_code(None) is None
        
        # Test with specific language
        assert whisper_service._get_language_code("en") == "en"
        assert whisper_service._get_language_code("es") == "es"
    
    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, whisper_service, temp_audio_file):
        """Test multiple concurrent transcriptions."""
        # Mock local model
        mock_model = AsyncMock()
        mock_model.transcribe.return_value = {
            "text": "Concurrent test",
            "language": "en"
        }
        whisper_service.local_model = mock_model
        
        # Run multiple transcriptions concurrently
        tasks = [
            whisper_service.transcribe_audio(temp_audio_file)
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result["success"] is True
            assert "text" in result
    
    @pytest.mark.asyncio
    async def test_transcription_with_timestamps(self, whisper_service, temp_audio_file):
        """Test transcription with timestamps enabled."""
        # Mock local model with segments
        mock_model = AsyncMock()
        mock_model.transcribe.return_value = {
            "text": "Timestamped transcription",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Timestamped transcription"
                }
            ]
        }
        whisper_service.local_model = mock_model
        
        result = await whisper_service.transcribe_audio(
            temp_audio_file,
            return_timestamps=True
        )
        
        assert result["success"] is True
        assert "segments" in result
        assert len(result["segments"]) > 0
    
    def test_performance_metrics_structure(self, whisper_service):
        """Test that performance metrics are properly structured."""
        metrics = whisper_service._create_performance_metrics(1.5, "en")
        
        assert "processing_time" in metrics
        assert "language" in metrics
        assert metrics["processing_time"] == 1.5
        assert metrics["language"] == "en"
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self, whisper_service):
        """Test error recovery and fallback mechanisms."""
        # Test with corrupted audio data
        corrupted_data = b"corrupted audio data"
        
        result = await whisper_service.transcribe_audio(corrupted_data)
        
        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result
        assert isinstance(result["error"], str)


@pytest.mark.integration
class TestWhisperServiceIntegration:
    """Integration tests for WhisperService (require actual models)."""
    
    @pytest.fixture
    def whisper_service(self):
        """Create WhisperService for integration testing."""
        return WhisperService()
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    @pytest.mark.asyncio
    async def test_real_audio_transcription(self, whisper_service, temp_audio_file):
        """Test transcription with real audio file (integration test)."""
        # This would require a real audio file and models
        # Skip by default to avoid heavy model downloads in CI
        result = await whisper_service.transcribe_audio(temp_audio_file)
        
        # Basic validation without requiring specific text
        assert "success" in result
        assert "text" in result or "error" in result
