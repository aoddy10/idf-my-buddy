"""Real-time audio streaming utilities.

This module provides functionality for handling real-time audio streams,
WebSocket audio communication, and streaming audio processing.
"""

import asyncio
import base64
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.logging import LoggerMixin

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class AudioChunk:
    """Represents a chunk of audio data for streaming."""
    data: bytes
    timestamp: float
    sequence: int
    format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


class AudioStreamer(LoggerMixin):
    """Handles real-time audio streaming and buffering."""

    def __init__(
        self,
        buffer_size_ms: int = 1000,
        sample_rate: int = 16000,
        chunk_size_ms: int = 100
    ):
        super().__init__()
        self.buffer_size_ms = buffer_size_ms
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms

        # Calculate buffer sizes
        self.buffer_size_samples = int(buffer_size_ms * sample_rate / 1000)
        self.chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)

        # Audio buffer
        self._audio_buffer = bytearray()
        self._sequence_counter = 0
        self._is_streaming = False

        # Callbacks
        self._chunk_callback: Callable[[AudioChunk], None] | None = None

    def set_chunk_callback(self, callback: Callable[[AudioChunk], None]):
        """Set callback function for processing audio chunks."""
        self._chunk_callback = callback

    async def start_streaming(self):
        """Start audio streaming."""
        self._is_streaming = True
        self.logger.info("Audio streaming started")

    async def stop_streaming(self):
        """Stop audio streaming."""
        self._is_streaming = False
        self._audio_buffer.clear()
        self._sequence_counter = 0
        self.logger.info("Audio streaming stopped")

    async def add_audio_data(self, audio_data: bytes, timestamp: float | None = None):
        """Add audio data to the streaming buffer.

        Args:
            audio_data: Raw audio data
            timestamp: Timestamp of the audio data
        """
        if not self._is_streaming:
            return

        # Add to buffer
        self._audio_buffer.extend(audio_data)

        # Process complete chunks
        while len(self._audio_buffer) >= self.chunk_size_samples * 2:  # 16-bit samples
            # Extract chunk
            chunk_bytes = self.chunk_size_samples * 2
            chunk_data = bytes(self._audio_buffer[:chunk_bytes])
            del self._audio_buffer[:chunk_bytes]

            # Create audio chunk
            chunk = AudioChunk(
                data=chunk_data,
                timestamp=timestamp or asyncio.get_event_loop().time(),
                sequence=self._sequence_counter,
                sample_rate=self.sample_rate
            )

            self._sequence_counter += 1

            # Process chunk
            if self._chunk_callback:
                try:
                    await asyncio.create_task(
                        self._process_chunk_safely(chunk)
                    )
                except Exception as e:
                    self.logger.error(f"Chunk processing failed: {e}")

    async def _process_chunk_safely(self, chunk: AudioChunk):
        """Safely process audio chunk with error handling."""
        try:
            if asyncio.iscoroutinefunction(self._chunk_callback):
                await self._chunk_callback(chunk)
            else:
                self._chunk_callback(chunk)
        except Exception as e:
            self.logger.error(f"Chunk callback failed: {e}")

    async def get_buffered_audio(self) -> bytes | None:
        """Get all buffered audio data."""
        if self._audio_buffer:
            data = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            return data
        return None

    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming

    def get_buffer_level(self) -> float:
        """Get current buffer level as percentage of maximum."""
        current_samples = len(self._audio_buffer) // 2
        return min(current_samples / self.buffer_size_samples, 1.0)


class WebSocketAudioHandler(LoggerMixin):
    """Handles WebSocket-based audio communication."""

    def __init__(self):
        super().__init__()
        self._connections: dict[str, dict] = {}
        self._streamers: dict[str, AudioStreamer] = {}

    async def handle_connection(
        self,
        websocket,
        connection_id: str,
        on_audio_chunk: Callable[[str, AudioChunk], Any] | None = None,
        on_transcription: Callable[[str, str], Any] | None = None
    ):
        """Handle WebSocket audio connection.

        Args:
            websocket: WebSocket connection
            connection_id: Unique connection identifier
            on_audio_chunk: Callback for audio chunks
            on_transcription: Callback for transcription results
        """
        try:
            # Initialize connection
            self._connections[connection_id] = {
                'websocket': websocket,
                'active': True,
                'audio_format': 'wav',
                'sample_rate': 16000
            }

            # Create audio streamer
            streamer = AudioStreamer(
                buffer_size_ms=1000,
                sample_rate=16000,
                chunk_size_ms=200
            )

            if on_audio_chunk:
                def chunk_callback(chunk: AudioChunk):
                    asyncio.create_task(on_audio_chunk(connection_id, chunk))
                streamer.set_chunk_callback(chunk_callback)

            self._streamers[connection_id] = streamer
            await streamer.start_streaming()

            self.logger.info(f"WebSocket audio connection established: {connection_id}")

            # Handle messages
            await self._message_loop(websocket, connection_id, streamer, on_transcription)

        except Exception as e:
            self.logger.error(f"WebSocket handler error: {e}")
        finally:
            await self._cleanup_connection(connection_id)

    async def _message_loop(
        self,
        websocket,
        connection_id: str,
        streamer: AudioStreamer,
        on_transcription: Callable[[str, str], Any] | None
    ):
        """Main message handling loop."""
        async for message in websocket:
            try:
                if isinstance(message, str):
                    # Text message - JSON command
                    await self._handle_text_message(
                        message, connection_id, websocket, on_transcription
                    )
                else:
                    # Binary message - audio data
                    await self._handle_audio_message(
                        message, connection_id, streamer
                    )

            except Exception as e:
                self.logger.error(f"Message handling error: {e}")
                await self._send_error(websocket, str(e))

    async def _handle_text_message(
        self,
        message: str,
        connection_id: str,
        websocket,
        on_transcription: Callable[[str, str], Any] | None
    ):
        """Handle text/JSON messages."""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'audio_config':
                # Update audio configuration
                config = data.get('config', {})
                conn = self._connections[connection_id]
                conn['audio_format'] = config.get('format', 'wav')
                conn['sample_rate'] = config.get('sample_rate', 16000)

                await self._send_message(websocket, {
                    'type': 'config_updated',
                    'config': config
                })

            elif message_type == 'audio_data':
                # Handle base64 encoded audio
                audio_b64 = data.get('data')
                if audio_b64:
                    audio_data = base64.b64decode(audio_b64)
                    streamer = self._streamers[connection_id]
                    await streamer.add_audio_data(audio_data)

            elif message_type == 'start_streaming':
                # Start audio streaming
                streamer = self._streamers[connection_id]
                await streamer.start_streaming()

                await self._send_message(websocket, {
                    'type': 'streaming_started'
                })

            elif message_type == 'stop_streaming':
                # Stop audio streaming
                streamer = self._streamers[connection_id]
                await streamer.stop_streaming()

                # Get any remaining buffered audio
                remaining_audio = await streamer.get_buffered_audio()
                if remaining_audio and on_transcription:
                    # Process final chunk
                    # This would typically trigger transcription
                    pass

                await self._send_message(websocket, {
                    'type': 'streaming_stopped'
                })

            elif message_type == 'ping':
                await self._send_message(websocket, {
                    'type': 'pong',
                    'timestamp': asyncio.get_event_loop().time()
                })

        except json.JSONDecodeError as e:
            await self._send_error(websocket, f"Invalid JSON: {e}")
        except Exception as e:
            await self._send_error(websocket, f"Message processing error: {e}")

    async def _handle_audio_message(
        self,
        message: bytes,
        connection_id: str,
        streamer: AudioStreamer
    ):
        """Handle binary audio messages."""
        await streamer.add_audio_data(
            message,
            timestamp=asyncio.get_event_loop().time()
        )

    async def _send_message(self, websocket, data: dict[str, Any]):
        """Send JSON message to WebSocket."""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket message: {e}")

    async def _send_error(self, websocket, error_message: str):
        """Send error message to WebSocket."""
        await self._send_message(websocket, {
            'type': 'error',
            'message': error_message
        })

    async def send_transcription(
        self,
        connection_id: str,
        transcription: str,
        confidence: float = 1.0,
        is_final: bool = True
    ):
        """Send transcription result to WebSocket client.

        Args:
            connection_id: Connection identifier
            transcription: Transcription text
            confidence: Transcription confidence score
            is_final: Whether this is a final result
        """
        if connection_id in self._connections:
            websocket = self._connections[connection_id]['websocket']
            await self._send_message(websocket, {
                'type': 'transcription',
                'text': transcription,
                'confidence': confidence,
                'is_final': is_final,
                'timestamp': asyncio.get_event_loop().time()
            })

    async def send_audio_response(
        self,
        connection_id: str,
        audio_data: bytes,
        format: str = 'mp3'
    ):
        """Send audio response to WebSocket client.

        Args:
            connection_id: Connection identifier
            audio_data: Audio data to send
            format: Audio format
        """
        if connection_id in self._connections:
            websocket = self._connections[connection_id]['websocket']

            # Send as base64 encoded data
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

            await self._send_message(websocket, {
                'type': 'audio_response',
                'data': audio_b64,
                'format': format,
                'timestamp': asyncio.get_event_loop().time()
            })

    async def _cleanup_connection(self, connection_id: str):
        """Clean up connection resources."""
        if connection_id in self._streamers:
            await self._streamers[connection_id].stop_streaming()
            del self._streamers[connection_id]

        if connection_id in self._connections:
            del self._connections[connection_id]

        self.logger.info(f"WebSocket audio connection cleaned up: {connection_id}")

    def get_active_connections(self) -> list:
        """Get list of active connection IDs."""
        return list(self._connections.keys())

    def is_connection_active(self, connection_id: str) -> bool:
        """Check if connection is active."""
        return connection_id in self._connections and \
               self._connections[connection_id]['active']


async def create_audio_stream_processor(
    chunk_callback: Callable[[AudioChunk], Any],
    buffer_size_ms: int = 1000,
    sample_rate: int = 16000
) -> AudioStreamer:
    """Create and configure an audio stream processor.

    Args:
        chunk_callback: Function to process audio chunks
        buffer_size_ms: Buffer size in milliseconds
        sample_rate: Audio sample rate

    Returns:
        Configured AudioStreamer instance
    """
    streamer = AudioStreamer(
        buffer_size_ms=buffer_size_ms,
        sample_rate=sample_rate
    )

    streamer.set_chunk_callback(chunk_callback)
    await streamer.start_streaming()

    return streamer
