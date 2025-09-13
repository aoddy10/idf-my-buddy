"""Audio processing utilities package.

This package provides audio processing functionality including:
- Audio preprocessing (VAD, noise reduction, normalization)
- Audio format conversion and validation
- Real-time audio streaming handlers
"""

from .formats import (
    compress_audio,
    convert_audio_format,
    detect_audio_format,
    validate_audio_file,
)
from .preprocessing import (
    audio_normalization,
    chunk_audio,
    noise_reduction,
    voice_activity_detection,
)
from .streaming import AudioStreamer, WebSocketAudioHandler

__all__ = [
    # Preprocessing
    "voice_activity_detection",
    "noise_reduction",
    "audio_normalization",
    "chunk_audio",
    # Format handling
    "convert_audio_format",
    "detect_audio_format",
    "compress_audio",
    "validate_audio_file",
    # Streaming
    "AudioStreamer",
    "WebSocketAudioHandler"
]
