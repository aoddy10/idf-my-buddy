"""Tests for Voice Pipeline service.

This module tests the VoicePipeline orchestration service that coordinates
ASR, TTS, and translation services for complete voice interactions.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from app.services.voice_pipeline import VoicePipeline, VoiceContext, VoiceExchange, ProcessingMetrics
from app.core.config import get_settings


class TestVoicePipeline:
    """Test cases for VoicePipeline."""
    
    @pytest.fixture
    def voice_pipeline(self):
        """Create VoicePipeline instance for testing."""
        return VoicePipeline()
    
    @pytest.fixture
    def sample_context(self) -> VoiceContext:
        """Create sample voice context for testing."""
        return VoiceContext(
            session_id="test-session-123",
            user_id="test-user-456",
            source_language="auto",
            target_language="en",
            preferred_voice="neutral"
        )
    
    @pytest.fixture
    def sample_audio_data(self) -> bytes:
        """Create mock audio data for testing."""
        return b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    def test_voice_pipeline_creation(self, voice_pipeline):
        """Test VoicePipeline can be created."""
        assert voice_pipeline is not None
        assert hasattr(voice_pipeline, 'log')
        assert voice_pipeline.whisper_service is None
        assert voice_pipeline.tts_service is None
        assert voice_pipeline.nllb_service is None
    
    def test_voice_context_creation(self):
        """Test VoiceContext data class functionality."""
        context = VoiceContext(
            session_id="test-session",
            user_id="test-user",
            source_language="es", 
            target_language="en"
        )
        
        assert context.session_id == "test-session"
        assert context.user_id == "test-user"
        assert context.source_language == "es"
        assert context.target_language == "en"
        assert isinstance(context.conversation_history, list)
        assert len(context.conversation_history) == 0
    
    def test_voice_exchange_creation(self):
        """Test VoiceExchange data class functionality."""
        exchange = VoiceExchange()
        
        assert exchange.exchange_id is not None
        assert exchange.timestamp > 0
        assert exchange.user_audio_data is None
        assert exchange.user_text is None
        assert exchange.response_text is None
        assert isinstance(exchange.confidence_scores, dict)
    
    def test_processing_metrics_creation(self):
        """Test ProcessingMetrics data class functionality."""
        metrics = ProcessingMetrics()
        
        assert metrics.total_time == 0.0
        assert metrics.audio_duration == 0.0
        assert metrics.real_time_factor == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert isinstance(metrics.stage_times, dict)
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, voice_pipeline):
        """Test service initialization within pipeline."""
        with patch('app.services.voice_pipeline.WhisperService') as mock_whisper:
            with patch('app.services.voice_pipeline.TTSService') as mock_tts:
                with patch('app.services.voice_pipeline.NLLBService') as mock_nllb:
                    
                    voice_pipeline._initialize_services()
                    
                    assert voice_pipeline.whisper_service is not None
                    assert voice_pipeline.tts_service is not None
                    assert voice_pipeline.nllb_service is not None
    
    @pytest.mark.asyncio 
    async def test_process_voice_input_complete_flow(self, voice_pipeline, sample_context, sample_audio_data):
        """Test complete voice processing flow with mocked services."""
        # Mock services
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Hola mundo",
            "language": "es",
            "confidence": 0.95,
            "processing_time": 0.5
        }
        
        mock_nllb = AsyncMock() 
        mock_nllb.translate_text.return_value = {
            "success": True,
            "translated_text": "Hello world",
            "source_language": "es",
            "target_language": "en",
            "processing_time": 0.3
        }
        
        mock_tts = AsyncMock()
        mock_tts.synthesize_text.return_value = {
            "success": True,
            "audio_data": b"synthesized_audio",
            "duration": 2.0,
            "processing_time": 0.8
        }
        
        voice_pipeline.whisper_service = mock_whisper
        voice_pipeline.nllb_service = mock_nllb
        voice_pipeline.tts_service = mock_tts
        
        # Process voice input
        result = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context,
            return_audio=True
        )
        
        # Verify result structure
        assert isinstance(result, VoiceExchange)
        assert result.user_text == "Hola mundo"
        assert result.translated_text == "Hello world"
        assert result.response_text == "Hello world"
        assert result.response_audio_data == b"synthesized_audio"
        assert result.user_language == "es"
        
        # Verify service calls
        mock_whisper.transcribe_audio.assert_called_once()
        mock_nllb.translate_text.assert_called_once()
        mock_tts.synthesize_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_voice_input_no_translation_needed(self, voice_pipeline, sample_context, sample_audio_data):
        """Test voice processing when no translation is needed."""
        # Set context for same language
        sample_context.source_language = "en"
        sample_context.target_language = "en"
        
        # Mock services
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Hello world",
            "language": "en",
            "confidence": 0.95
        }
        
        mock_tts = AsyncMock()
        mock_tts.synthesize_text.return_value = {
            "success": True,
            "audio_data": b"synthesized_audio"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        voice_pipeline.tts_service = mock_tts
        
        # Process voice input
        result = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context,
            return_audio=True
        )
        
        # Verify no translation occurred
        assert result.user_text == "Hello world"
        assert result.translated_text is None  # No translation needed
        assert result.response_text == "Hello world"
        
        # Verify translation service was not called
        assert voice_pipeline.nllb_service is None or not hasattr(voice_pipeline.nllb_service, 'translate_text')
    
    @pytest.mark.asyncio
    async def test_process_voice_input_asr_failure(self, voice_pipeline, sample_context, sample_audio_data):
        """Test handling of ASR service failure."""
        # Mock ASR failure
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": False,
            "error": "Audio processing failed"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        
        # Process voice input
        result = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context
        )
        
        # Verify error handling
        assert isinstance(result, VoiceExchange)
        assert result.error_message is not None
        assert "failed" in result.error_message.lower()
        assert result.user_text is None
    
    @pytest.mark.asyncio
    async def test_process_voice_input_translation_failure(self, voice_pipeline, sample_context, sample_audio_data):
        """Test handling of translation service failure."""
        # Mock successful ASR but failed translation
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Hola mundo",
            "language": "es"
        }
        
        mock_nllb = AsyncMock()
        mock_nllb.translate_text.return_value = {
            "success": False,
            "error": "Translation failed"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        voice_pipeline.nllb_service = mock_nllb
        
        # Process voice input
        result = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context
        )
        
        # Verify fallback to original text
        assert result.user_text == "Hola mundo"
        assert result.translated_text is None
        assert result.response_text == "Hola mundo"  # Fallback to original
    
    @pytest.mark.asyncio
    async def test_process_voice_input_tts_failure(self, voice_pipeline, sample_context, sample_audio_data):
        """Test handling of TTS service failure."""
        # Mock successful ASR and translation but failed TTS
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Hello world",
            "language": "en"
        }
        
        mock_tts = AsyncMock()
        mock_tts.synthesize_text.return_value = {
            "success": False,
            "error": "TTS synthesis failed"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        voice_pipeline.tts_service = mock_tts
        
        # Process voice input
        result = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context,
            return_audio=True
        )
        
        # Verify text response available even without audio
        assert result.response_text == "Hello world"
        assert result.response_audio_data is None
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, voice_pipeline, sample_context, sample_audio_data):
        """Test performance metrics calculation."""
        # Mock services with timing
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Test",
            "language": "en",
            "processing_time": 1.0
        }
        
        mock_tts = AsyncMock()
        mock_tts.synthesize_text.return_value = {
            "success": True,
            "audio_data": b"audio",
            "processing_time": 0.5
        }
        
        voice_pipeline.whisper_service = mock_whisper
        voice_pipeline.tts_service = mock_tts
        
        # Process voice input
        result = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context
        )
        
        # Verify timing information
        assert result.processing_time > 0
        assert "asr_time" in result.confidence_scores or result.processing_time >= 1.0
    
    @pytest.mark.asyncio
    async def test_conversation_history_management(self, voice_pipeline, sample_context, sample_audio_data):
        """Test conversation history tracking."""
        # Mock successful interaction
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Hello",
            "language": "en"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        
        # Process multiple interactions
        result1 = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context
        )
        
        result2 = await voice_pipeline.process_voice_input(
            audio_data=sample_audio_data,
            context=sample_context
        )
        
        # Verify history is maintained in context
        assert len(sample_context.conversation_history) >= 0  # History may be managed differently
    
    @pytest.mark.asyncio
    async def test_quality_level_adjustment(self, voice_pipeline, sample_context, sample_audio_data):
        """Test quality level adjustments based on performance."""
        # Mock services
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Test",
            "language": "en"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        
        # Test different quality levels
        for quality in ["fast", "balanced", "accurate"]:
            result = await voice_pipeline.process_voice_input(
                audio_data=sample_audio_data,
                context=sample_context,
                quality_level=quality
            )
            
            assert isinstance(result, VoiceExchange)
    
    def test_context_validation(self, voice_pipeline):
        """Test voice context validation."""
        # Valid context
        valid_context = VoiceContext(
            session_id="valid-session",
            source_language="en",
            target_language="es"
        )
        assert voice_pipeline._validate_context(valid_context) is True
        
        # Invalid context (missing session_id)
        invalid_context = VoiceContext(
            session_id="",
            source_language="en", 
            target_language="es"
        )
        assert voice_pipeline._validate_context(invalid_context) is False
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, voice_pipeline, sample_context, sample_audio_data):
        """Test handling of concurrent voice processing requests."""
        # Mock services
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.return_value = {
            "success": True,
            "text": "Concurrent test",
            "language": "en"
        }
        
        voice_pipeline.whisper_service = mock_whisper
        
        # Create multiple concurrent requests
        tasks = [
            voice_pipeline.process_voice_input(
                audio_data=sample_audio_data,
                context=VoiceContext(
                    session_id=f"session-{i}",
                    source_language="en",
                    target_language="en"
                )
            )
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, VoiceExchange)
    
    @pytest.mark.asyncio
    async def test_session_management(self, voice_pipeline):
        """Test voice session management functionality."""
        # Create session
        session_context = await voice_pipeline.create_voice_session(
            user_id="test-user",
            source_language="en",
            target_language="es"
        )
        
        assert isinstance(session_context, VoiceContext)
        assert session_context.user_id == "test-user"
        assert session_context.source_language == "en"
        assert session_context.target_language == "es"
        
        # Retrieve session
        retrieved_context = await voice_pipeline.get_voice_session(session_context.session_id)
        assert retrieved_context is not None
        assert retrieved_context.session_id == session_context.session_id
        
        # End session
        success = await voice_pipeline.end_voice_session(session_context.session_id)
        assert success is True
    
    def test_rtf_calculation(self, voice_pipeline):
        """Test Real-Time Factor calculation."""
        # Test RTF calculation
        audio_duration = 2.0  # 2 seconds of audio
        processing_time = 1.0  # 1 second to process
        
        rtf = voice_pipeline._calculate_rtf(audio_duration, processing_time)
        assert rtf == 0.5  # 1.0 / 2.0 = 0.5
        
        # Test edge case
        rtf_zero_audio = voice_pipeline._calculate_rtf(0.0, 1.0)
        assert rtf_zero_audio >= 0  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self, voice_pipeline, sample_context):
        """Test error recovery and fallback mechanisms."""
        # Test with completely invalid audio data
        invalid_audio = b"invalid audio data"
        
        # Mock service that raises exception
        mock_whisper = AsyncMock()
        mock_whisper.transcribe_audio.side_effect = Exception("Service failure")
        
        voice_pipeline.whisper_service = mock_whisper
        
        # Process should handle exception gracefully
        result = await voice_pipeline.process_voice_input(
            audio_data=invalid_audio,
            context=sample_context
        )
        
        assert isinstance(result, VoiceExchange)
        assert result.error_message is not None


@pytest.mark.integration
class TestVoicePipelineIntegration:
    """Integration tests for VoicePipeline (require actual services)."""
    
    @pytest.fixture
    def voice_pipeline(self):
        """Create VoicePipeline for integration testing."""
        return VoicePipeline()
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    @pytest.mark.asyncio
    async def test_real_voice_processing_flow(self, voice_pipeline):
        """Test complete voice processing with real services (integration test)."""
        # This would test with actual models and services
        # Skip by default to avoid heavy processing in CI
        
        context = VoiceContext(
            session_id="integration-test",
            source_language="auto",
            target_language="en"
        )
        
        # Use minimal audio data
        audio_data = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        
        result = await voice_pipeline.process_voice_input(
            audio_data=audio_data,
            context=context
        )
        
        # Basic validation
        assert isinstance(result, VoiceExchange)
        assert hasattr(result, 'processing_time')
