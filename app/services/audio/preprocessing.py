"""Audio preprocessing utilities.

This module provides audio preprocessing functionality including voice activity
detection (VAD), noise reduction, normalization, and chunking for streaming.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from app.core.config import settings
from app.core.logging import LoggerMixin

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False


class AudioPreprocessor(LoggerMixin):
    """Audio preprocessing service."""

    def __init__(self):
        super().__init__()
        self.sample_rate = settings.audio_sample_rate

        # Initialize VAD if available
        self.vad = None
        if WEBRTCVAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
            except Exception as e:
                self.logger.warning(f"Failed to initialize WebRTC VAD: {e}")


async def voice_activity_detection(
    audio_data: bytes | np.ndarray,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2
) -> list[tuple[float, float]]:
    """Detect voice activity in audio using WebRTC VAD.

    Args:
        audio_data: Raw audio data or numpy array
        sample_rate: Audio sample rate in Hz
        frame_duration_ms: VAD frame duration (10, 20, or 30ms)
        aggressiveness: VAD aggressiveness level (0-3, higher = more aggressive)

    Returns:
        List of (start_time, end_time) tuples for voice segments in seconds
    """
    if not WEBRTCVAD_AVAILABLE:
        # Fallback: assume entire audio contains voice
        if isinstance(audio_data, bytes):
            duration = len(audio_data) / (sample_rate * 2)  # 16-bit audio
        else:
            duration = len(audio_data) / sample_rate
        return [(0.0, duration)]

    try:
        # Convert to numpy if needed
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_np = audio_data.astype(np.int16)

        # Initialize VAD
        vad = webrtcvad.Vad(aggressiveness)

        # Calculate frame size
        frame_length = int(sample_rate * frame_duration_ms / 1000)

        # Detect voice activity
        voice_segments = []
        current_start = None

        for i in range(0, len(audio_np), frame_length):
            frame = audio_np[i:i + frame_length]

            # Pad frame if necessary
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))

            # Check if frame contains voice
            frame_bytes = frame.tobytes()
            is_voice = vad.is_speech(frame_bytes, sample_rate)

            time_s = i / sample_rate

            if is_voice and current_start is None:
                current_start = time_s
            elif not is_voice and current_start is not None:
                voice_segments.append((current_start, time_s))
                current_start = None

        # Handle case where voice extends to end of audio
        if current_start is not None:
            voice_segments.append((current_start, len(audio_np) / sample_rate))

        return voice_segments

    except Exception as e:
        logging.error(f"VAD failed: {e}")
        # Fallback: return entire audio as voice
        duration = len(audio_np) / sample_rate if isinstance(audio_np, np.ndarray) else 1.0
        return [(0.0, duration)]


async def noise_reduction(
    audio_data: bytes | np.ndarray,
    sample_rate: int = 16000,
    noise_reduce_strength: float = 0.8
) -> np.ndarray:
    """Reduce noise in audio using spectral subtraction.

    Args:
        audio_data: Raw audio data or numpy array
        sample_rate: Audio sample rate in Hz
        noise_reduce_strength: Noise reduction strength (0.0-1.0)

    Returns:
        Denoised audio as numpy array
    """
    if not NOISEREDUCE_AVAILABLE or not LIBROSA_AVAILABLE:
        # Fallback: return original audio
        if isinstance(audio_data, bytes):
            return np.frombuffer(audio_data, dtype=np.float32)
        return audio_data.astype(np.float32)

    try:
        # Convert to numpy if needed
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = audio_data.astype(np.float32)

        # Apply noise reduction
        reduced_noise = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: nr.reduce_noise(
                y=audio_np,
                sr=sample_rate,
                prop_decrease=noise_reduce_strength
            )
        )

        return reduced_noise

    except Exception as e:
        logging.error(f"Noise reduction failed: {e}")
        # Return original audio
        if isinstance(audio_data, bytes):
            return np.frombuffer(audio_data, dtype=np.float32)
        return audio_data.astype(np.float32)


async def audio_normalization(
    audio_data: bytes | np.ndarray,
    target_db: float = -20.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Normalize audio to target RMS level.

    Args:
        audio_data: Raw audio data or numpy array
        target_db: Target RMS level in dB
        sample_rate: Audio sample rate in Hz

    Returns:
        Normalized audio as numpy array
    """
    try:
        # Convert to numpy if needed
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = audio_data.astype(np.float32)

        # Calculate current RMS
        rms = np.sqrt(np.mean(audio_np ** 2))

        if rms == 0:
            return audio_np

        # Calculate target RMS from dB
        target_rms = 10 ** (target_db / 20)

        # Apply normalization
        normalization_factor = target_rms / rms
        normalized_audio = audio_np * normalization_factor

        # Prevent clipping
        max_val = np.max(np.abs(normalized_audio))
        if max_val > 1.0:
            normalized_audio = normalized_audio / max_val * 0.95

        return normalized_audio

    except Exception as e:
        logging.error(f"Audio normalization failed: {e}")
        # Return original audio
        if isinstance(audio_data, bytes):
            return np.frombuffer(audio_data, dtype=np.float32)
        return audio_data.astype(np.float32)


async def chunk_audio(
    audio_data: bytes | np.ndarray,
    chunk_duration_s: float = 5.0,
    sample_rate: int = 16000,
    overlap_s: float = 0.5
) -> AsyncIterator[np.ndarray]:
    """Split audio into overlapping chunks for streaming processing.

    Args:
        audio_data: Raw audio data or numpy array
        chunk_duration_s: Duration of each chunk in seconds
        sample_rate: Audio sample rate in Hz
        overlap_s: Overlap between chunks in seconds

    Yields:
        Audio chunks as numpy arrays
    """
    try:
        # Convert to numpy if needed
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = audio_data.astype(np.float32)

        # Calculate chunk and overlap sizes in samples
        chunk_size = int(chunk_duration_s * sample_rate)
        overlap_size = int(overlap_s * sample_rate)
        step_size = chunk_size - overlap_size

        # Generate chunks
        for start in range(0, len(audio_np), step_size):
            end = min(start + chunk_size, len(audio_np))
            chunk = audio_np[start:end]

            # Pad last chunk if necessary
            if len(chunk) < chunk_size and start + chunk_size >= len(audio_np):
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            yield chunk

            # Break if we've reached the end
            if end >= len(audio_np):
                break

    except Exception as e:
        logging.error(f"Audio chunking failed: {e}")
        # Yield original audio as single chunk
        if isinstance(audio_data, bytes):
            yield np.frombuffer(audio_data, dtype=np.float32)
        else:
            yield audio_data.astype(np.float32)


async def preprocess_audio_pipeline(
    audio_data: bytes | np.ndarray,
    sample_rate: int = 16000,
    enable_vad: bool = True,
    enable_noise_reduction: bool = True,
    enable_normalization: bool = True,
    target_sample_rate: int | None = None
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Complete audio preprocessing pipeline.

    Args:
        audio_data: Raw audio data or numpy array
        sample_rate: Input audio sample rate in Hz
        enable_vad: Enable voice activity detection
        enable_noise_reduction: Enable noise reduction
        enable_normalization: Enable audio normalization
        target_sample_rate: Target sample rate for output (None = keep original)

    Returns:
        Tuple of (processed_audio, voice_segments)
    """
    try:
        # Convert to numpy
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = audio_data.astype(np.float32)

        processed_audio = audio_np
        voice_segments = []

        # Voice activity detection (before other processing)
        if enable_vad:
            voice_segments = await voice_activity_detection(
                processed_audio, sample_rate
            )

        # Noise reduction
        if enable_noise_reduction:
            processed_audio = await noise_reduction(
                processed_audio, sample_rate
            )

        # Normalization
        if enable_normalization:
            processed_audio = await audio_normalization(
                processed_audio, sample_rate=sample_rate
            )

        # Resample if needed
        if target_sample_rate and target_sample_rate != sample_rate and LIBROSA_AVAILABLE:
            processed_audio = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: librosa.resample(
                    processed_audio,
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate
                )
            )

            # Adjust voice segments timing for new sample rate
            ratio = target_sample_rate / sample_rate
            voice_segments = [(start * ratio, end * ratio) for start, end in voice_segments]

        return processed_audio, voice_segments

    except Exception as e:
        logging.error(f"Audio preprocessing pipeline failed: {e}")
        # Return original audio
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio_np = audio_data.astype(np.float32)
        return audio_np, [(0.0, len(audio_np) / sample_rate)]
