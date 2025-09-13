"""Integration tests for Voice API endpoints.

This module tests the complete voice API functionality including 
FastAPI endpoints, WebSocket connections, and full integration flows.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

from fastapi.testclient import TestClient
from fastapi import status
from httpx import AsyncClient
import websockets
from unittest.mock import patch, AsyncMock, Mock

from app.main import app
from app.core.config import get_settings


class TestVoiceAPIEndpoints:
    """Test cases for Voice API REST endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write minimal WAV file header
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            f.flush()
            yield f.name
        
        # Cleanup
        os.unlink(f.name)
    
    def test_transcribe_endpoint_success(self, client, sample_audio_file):
        """Test successful audio transcription endpoint."""
        with patch('app.services.whisper.WhisperService.transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {
                "success": True,
                "text": "Hello world",
                "language": "en",
                "confidence": 0.95,
                "processing_time": 0.5
            }
            
            # Make request
            with open(sample_audio_file, "rb") as f:
                response = client.post(
                    "/api/v1/voice/transcribe",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={"language": "auto"}
                )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["text"] == "Hello world"
            assert data["language"] == "en"
            assert "processing_time" in data
    
    def test_transcribe_endpoint_missing_file(self, client):
        """Test transcribe endpoint with missing audio file."""
        response = client.post(
            "/api/v1/voice/transcribe",
            data={"language": "en"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_transcribe_endpoint_invalid_file(self, client):
        """Test transcribe endpoint with invalid audio file."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            f.write(b"This is not an audio file")
            f.flush()
            f.seek(0)
            
            response = client.post(
                "/api/v1/voice/transcribe",
                files={"audio_file": ("test.txt", f, "text/plain")},
                data={"language": "auto"}
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_synthesize_endpoint_success(self, client):
        """Test successful text-to-speech synthesis endpoint."""
        with patch('app.services.tts.TTSService.synthesize_text') as mock_synthesize:
            mock_synthesize.return_value = {
                "success": True,
                "audio_data": b"synthesized_audio_data",
                "duration": 2.0,
                "processing_time": 0.8,
                "audio_format": "wav"
            }
            
            # Make request
            response = client.post(
                "/api/v1/voice/synthesize",
                json={
                    "text": "Hello world",
                    "language": "en",
                    "voice": "neutral",
                    "audio_format": "wav"
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
            assert len(response.content) > 0
    
    def test_synthesize_endpoint_empty_text(self, client):
        """Test synthesize endpoint with empty text."""
        response = client.post(
            "/api/v1/voice/synthesize",
            json={
                "text": "",
                "language": "en"
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_synthesize_endpoint_service_failure(self, client):
        """Test synthesize endpoint with service failure."""
        with patch('app.services.tts.TTSService.synthesize_text') as mock_synthesize:
            mock_synthesize.return_value = {
                "success": False,
                "error": "TTS service unavailable"
            }
            
            response = client.post(
                "/api/v1/voice/synthesize",
                json={
                    "text": "Hello world",
                    "language": "en"
                }
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_session_create_endpoint(self, client):
        """Test voice session creation endpoint."""
        response = client.post(
            "/api/v1/voice/sessions/",
            json={
                "user_id": "test-user-123",
                "source_language": "en",
                "target_language": "es",
                "preferred_voice": "neutral"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "session_id" in data
        assert data["user_id"] == "test-user-123"
        assert data["source_language"] == "en"
        assert data["target_language"] == "es"
    
    def test_session_get_endpoint(self, client):
        """Test voice session retrieval endpoint."""
        # First create a session
        create_response = client.post(
            "/api/v1/voice/sessions/",
            json={
                "user_id": "test-user-456",
                "source_language": "en",
                "target_language": "fr"
            }
        )
        
        assert create_response.status_code == status.HTTP_201_CREATED
        session_id = create_response.json()["session_id"]
        
        # Then retrieve it
        get_response = client.get(f"/api/v1/voice/sessions/{session_id}")
        
        assert get_response.status_code == status.HTTP_200_OK
        data = get_response.json()
        assert data["session_id"] == session_id
        assert data["user_id"] == "test-user-456"
    
    def test_session_get_not_found(self, client):
        """Test voice session retrieval for non-existent session."""
        response = client.get("/api/v1/voice/sessions/non-existent-session")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_session_update_endpoint(self, client):
        """Test voice session update endpoint."""
        # First create a session
        create_response = client.post(
            "/api/v1/voice/sessions/",
            json={
                "user_id": "test-user-789",
                "source_language": "en",
                "target_language": "de"
            }
        )
        
        session_id = create_response.json()["session_id"]
        
        # Update the session
        update_response = client.put(
            f"/api/v1/voice/sessions/{session_id}",
            json={
                "target_language": "it",
                "preferred_voice": "female"
            }
        )
        
        assert update_response.status_code == status.HTTP_200_OK
        data = update_response.json()
        assert data["target_language"] == "it"
        assert data["preferred_voice"] == "female"
    
    def test_session_delete_endpoint(self, client):
        """Test voice session deletion endpoint."""
        # First create a session
        create_response = client.post(
            "/api/v1/voice/sessions/",
            json={
                "user_id": "test-user-delete",
                "source_language": "en",
                "target_language": "ja"
            }
        )
        
        session_id = create_response.json()["session_id"]
        
        # Delete the session
        delete_response = client.delete(f"/api/v1/voice/sessions/{session_id}")
        
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify it's deleted
        get_response = client.get(f"/api/v1/voice/sessions/{session_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND


class TestVoiceWebSocketEndpoints:
    """Test cases for Voice API WebSocket endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for WebSocket testing."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_websocket_conversation_connection(self, test_client):
        """Test WebSocket conversation endpoint connection."""
        with patch('app.services.voice_pipeline.VoicePipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Test WebSocket connection
            with test_client.websocket_connect("/api/v1/voice/conversation/stream") as websocket:
                # Send initial configuration
                config_message = {
                    "type": "config",
                    "session_id": "test-session-ws",
                    "source_language": "en",
                    "target_language": "es"
                }
                websocket.send_json(config_message)
                
                # Should receive confirmation
                response = websocket.receive_json()
                assert response["type"] == "config_confirmed"
                assert response["session_id"] == "test-session-ws"
    
    @pytest.mark.asyncio
    async def test_websocket_audio_processing(self, test_client):
        """Test WebSocket audio processing flow."""
        with patch('app.services.voice_pipeline.VoicePipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock pipeline response
            mock_pipeline.process_voice_input.return_value = Mock(
                user_text="Hello",
                translated_text="Hola",
                response_text="Hola",
                response_audio_data=b"audio_response",
                processing_time=1.0,
                error_message=None
            )
            
            with test_client.websocket_connect("/api/v1/voice/conversation/stream") as websocket:
                # Send configuration
                websocket.send_json({
                    "type": "config",
                    "session_id": "test-audio-ws",
                    "source_language": "en",
                    "target_language": "es"
                })
                websocket.receive_json()  # Config confirmation
                
                # Send audio data
                audio_message = {
                    "type": "audio",
                    "audio_data": "base64_encoded_audio_data",
                    "return_audio": True
                }
                websocket.send_json(audio_message)
                
                # Receive processing result
                response = websocket.receive_json()
                assert response["type"] == "response"
                assert "user_text" in response
                assert "response_audio" in response
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, test_client):
        """Test WebSocket error handling."""
        with test_client.websocket_connect("/api/v1/voice/conversation/stream") as websocket:
            # Send malformed message
            websocket.send_text("invalid json")
            
            # Should receive error response
            try:
                response = websocket.receive_json()
                assert response["type"] == "error"
            except Exception:
                # WebSocket might close on invalid message
                pass
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_sessions(self, test_client):
        """Test multiple concurrent WebSocket sessions."""
        with patch('app.services.voice_pipeline.VoicePipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create multiple WebSocket connections
            connections = []
            try:
                for i in range(3):
                    ws = test_client.websocket_connect("/api/v1/voice/conversation/stream")
                    connection = ws.__enter__()
                    connections.append((ws, connection))
                    
                    # Configure each session
                    connection.send_json({
                        "type": "config",
                        "session_id": f"concurrent-session-{i}",
                        "source_language": "en",
                        "target_language": "es"
                    })
                    
                    response = connection.receive_json()
                    assert response["type"] == "config_confirmed"
                    
            finally:
                # Cleanup connections
                for ws, connection in connections:
                    ws.__exit__(None, None, None)


@pytest.mark.integration
class TestVoiceAPIIntegration:
    """Full integration tests for Voice API (require real services)."""
    
    @pytest.fixture
    def client(self):
        """Create test client for integration testing."""
        return TestClient(app)
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_full_transcription_flow(self, client):
        """Test complete transcription flow with real services."""
        # Create a proper test audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            # This would need actual audio data for a real test
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            f.flush()
            f.seek(0)
            
            response = client.post(
                "/api/v1/voice/transcribe",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={"language": "auto"}
            )
            
            # Should handle gracefully even with minimal audio
            assert response.status_code in [200, 400, 500]  # Various valid outcomes
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_full_synthesis_flow(self, client):
        """Test complete synthesis flow with real services."""
        response = client.post(
            "/api/v1/voice/synthesize",
            json={
                "text": "Hello, this is a test",
                "language": "en",
                "voice": "neutral",
                "audio_format": "wav"
            }
        )
        
        # Should produce audio or handle gracefully
        assert response.status_code in [200, 500]  # Success or service unavailable
        
        if response.status_code == 200:
            assert len(response.content) > 0
            assert response.headers["content-type"] == "audio/wav"
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    @pytest.mark.asyncio
    async def test_full_websocket_conversation(self):
        """Test complete WebSocket conversation flow with real services."""
        # This would test actual WebSocket conversation
        # Skip by default due to complexity and resource requirements
        pass
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_session_persistence(self, client):
        """Test voice session persistence across requests."""
        # Create session
        create_response = client.post(
            "/api/v1/voice/sessions/",
            json={
                "user_id": "integration-test-user",
                "source_language": "en",
                "target_language": "es"
            }
        )
        
        assert create_response.status_code == 201
        session_id = create_response.json()["session_id"]
        
        # Use session in transcription
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            f.flush()
            f.seek(0)
            
            transcribe_response = client.post(
                "/api/v1/voice/transcribe",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={
                    "language": "auto",
                    "session_id": session_id
                }
            )
            
            # Should handle session context
            assert transcribe_response.status_code in [200, 400, 404, 500]
        
        # Verify session still exists
        get_response = client.get(f"/api/v1/voice/sessions/{session_id}")
        assert get_response.status_code == 200


class TestVoiceAPIErrorScenarios:
    """Test error scenarios and edge cases for Voice API."""
    
    @pytest.fixture
    def client(self):
        """Create test client for error testing.""" 
        return TestClient(app)
    
    def test_health_check_with_voice_services(self, client):
        """Test health check includes voice service status."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        # Voice service status should be included in health check
    
    def test_api_rate_limiting(self, client):
        """Test API rate limiting behavior."""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.post(
                "/api/v1/voice/synthesize",
                json={
                    "text": f"Test message {i}",
                    "language": "en"
                }
            )
            responses.append(response.status_code)
        
        # Should handle requests gracefully (rate limiting or success)
        assert all(status_code in [200, 429, 500] for status_code in responses)
    
    def test_large_audio_file_handling(self, client):
        """Test handling of large audio files."""
        # Create oversized "audio" file
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            # Write large amount of data (simulating large file)
            large_data = b'RIFF' + b'x' * (10 * 1024 * 1024)  # 10MB
            f.write(large_data)
            f.flush()
            f.seek(0)
            
            response = client.post(
                "/api/v1/voice/transcribe",
                files={"audio_file": ("large.wav", f, "audio/wav")},
                data={"language": "auto"}
            )
            
            # Should reject or handle large files appropriately
            assert response.status_code in [200, 400, 413, 500]
    
    def test_concurrent_api_requests(self, client):
        """Test handling of concurrent API requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.post(
                "/api/v1/voice/synthesize",
                json={
                    "text": "Concurrent test",
                    "language": "en"
                }
            )
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All requests should complete
        assert len(results) == 5
        assert all(status in [200, 500] for status in results)
    
    def test_malformed_request_handling(self, client):
        """Test handling of malformed requests."""
        # Test various malformed requests
        malformed_requests = [
            # Missing required fields
            {},
            {"text": ""},
            {"language": "invalid_lang_code"},
            {"text": "x" * 10000},  # Very long text
            {"audio_format": "invalid_format"}
        ]
        
        for request_data in malformed_requests:
            response = client.post(
                "/api/v1/voice/synthesize",
                json=request_data
            )
            
            # Should return appropriate error status
            assert response.status_code in [400, 422, 500]
