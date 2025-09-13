"""Audio format conversion and validation utilities.

This module provides functionality for converting between audio formats,
detecting audio file types, compressing audio, and validating audio files.
"""

import asyncio
import io
import logging
import tempfile
from typing import Any

try:
    import librosa
    import numpy as np
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
    # Check if ffmpeg is available
    FFMPEG_AVAILABLE = which("ffmpeg") is not None
except ImportError:
    PYDUB_AVAILABLE = False
    FFMPEG_AVAILABLE = False


class AudioFormatError(Exception):
    """Exception raised for audio format related errors."""
    pass


async def detect_audio_format(audio_data: bytes) -> str | None:
    """Detect audio format from file header magic numbers.

    Args:
        audio_data: Raw audio file data

    Returns:
        Detected format string or None if unknown
    """
    if len(audio_data) < 12:
        return None

    # Check common audio format signatures
    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        return 'wav'
    elif audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
        return 'mp3'
    elif audio_data[:4] == b'OggS':
        return 'ogg'
    elif audio_data[:4] == b'fLaC':
        return 'flac'
    elif audio_data[:4] == b'ftypM4A ':
        return 'm4a'
    elif audio_data[:4] == b'#!AMR':
        return 'amr'
    elif audio_data[:4] == b'\x1a\x45\xdf\xa3':  # WebM/Matroska
        return 'webm'

    return None


async def validate_audio_file(
    audio_data: bytes,
    max_size_bytes: int | None = None,
    max_duration_s: int | None = None,
    allowed_formats: list | None = None
) -> dict[str, Any]:
    """Validate audio file and extract metadata.

    Args:
        audio_data: Raw audio file data
        max_size_bytes: Maximum allowed file size in bytes
        max_duration_s: Maximum allowed duration in seconds
        allowed_formats: List of allowed audio formats

    Returns:
        Dict with validation results and metadata

    Raises:
        AudioFormatError: If validation fails
    """
    # Check file size
    if max_size_bytes and len(audio_data) > max_size_bytes:
        raise AudioFormatError(f"Audio file too large: {len(audio_data)} bytes > {max_size_bytes}")

    # Detect format
    detected_format = await detect_audio_format(audio_data)
    if not detected_format:
        raise AudioFormatError("Cannot detect audio format")

    # Check allowed formats
    if allowed_formats and detected_format not in allowed_formats:
        raise AudioFormatError(f"Audio format {detected_format} not allowed")

    # Extract metadata
    metadata = {
        'format': detected_format,
        'size_bytes': len(audio_data),
        'valid': True
    }

    try:
        # Try to get more detailed info using librosa
        if LIBROSA_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix=f'.{detected_format}') as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()

                y, sr = librosa.load(temp_file.name, sr=None)
                duration = len(y) / sr

                metadata.update({
                    'duration_s': duration,
                    'sample_rate': sr,
                    'channels': 1 if len(y.shape) == 1 else y.shape[0]
                })

                # Check duration limit
                if max_duration_s and duration > max_duration_s:
                    raise AudioFormatError(f"Audio too long: {duration}s > {max_duration_s}s")

    except Exception as e:
        logging.warning(f"Could not extract detailed metadata: {e}")

    return metadata


async def convert_audio_format(
    audio_data: bytes,
    source_format: str,
    target_format: str,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
    quality: str = 'high'
) -> bytes:
    """Convert audio between formats.

    Args:
        audio_data: Source audio data
        source_format: Source format (wav, mp3, etc.)
        target_format: Target format (wav, mp3, etc.)
        target_sample_rate: Target sample rate (None = keep original)
        target_channels: Target channel count (None = keep original)
        quality: Quality level ('low', 'medium', 'high')

    Returns:
        Converted audio data as bytes

    Raises:
        AudioFormatError: If conversion fails
    """
    if source_format == target_format and not target_sample_rate and not target_channels:
        return audio_data

    try:
        # Use pydub for conversion if available
        if PYDUB_AVAILABLE and FFMPEG_AVAILABLE:
            return await _convert_with_pydub(
                audio_data, source_format, target_format,
                target_sample_rate, target_channels, quality
            )

        # Fallback to librosa + soundfile
        elif LIBROSA_AVAILABLE:
            return await _convert_with_librosa(
                audio_data, source_format, target_format,
                target_sample_rate, target_channels
            )

        else:
            raise AudioFormatError("No audio conversion libraries available")

    except Exception as e:
        raise AudioFormatError(f"Audio conversion failed: {e}")


async def _convert_with_pydub(
    audio_data: bytes,
    source_format: str,
    target_format: str,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
    quality: str = 'high'
) -> bytes:
    """Convert audio using pydub."""

    def _pydub_conversion():
        # Load audio
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_data),
            format=source_format
        )

        # Apply transformations
        if target_channels:
            if target_channels == 1:
                audio_segment = audio_segment.set_channels(1)
            elif target_channels == 2:
                audio_segment = audio_segment.set_channels(2)

        if target_sample_rate:
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)

        # Set export parameters based on quality
        export_params = _get_export_params(target_format, quality)

        # Export to target format
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format=target_format, **export_params)

        return output_buffer.getvalue()

    return await asyncio.get_event_loop().run_in_executor(None, _pydub_conversion)


async def _convert_with_librosa(
    audio_data: bytes,
    source_format: str,
    target_format: str,
    target_sample_rate: int | None = None,
    target_channels: int | None = None
) -> bytes:
    """Convert audio using librosa + soundfile."""

    def _librosa_conversion():
        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix=f'.{source_format}') as input_file:
            input_file.write(audio_data)
            input_file.flush()

            # Load audio
            y, sr = librosa.load(input_file.name, sr=target_sample_rate)

            # Handle channel conversion
            if target_channels == 2 and len(y.shape) == 1:
                # Mono to stereo
                y = np.stack([y, y], axis=0)
            elif target_channels == 1 and len(y.shape) > 1:
                # Stereo to mono
                y = librosa.to_mono(y)

            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=f'.{target_format}') as output_file:
                sf.write(output_file.name, y.T if len(y.shape) > 1 else y, sr or 16000)

                # Read converted data
                with open(output_file.name, 'rb') as f:
                    return f.read()

    return await asyncio.get_event_loop().run_in_executor(None, _librosa_conversion)


def _get_export_params(format: str, quality: str) -> dict[str, Any]:
    """Get export parameters for different formats and quality levels."""

    quality_settings = {
        'low': {'bitrate': '64k'},
        'medium': {'bitrate': '128k'},
        'high': {'bitrate': '192k'}
    }

    base_params = quality_settings.get(quality, quality_settings['medium'])

    if format == 'mp3':
        return {**base_params, 'codec': 'mp3'}
    elif format == 'ogg':
        return {**base_params, 'codec': 'libvorbis'}
    elif format == 'aac':
        return {**base_params, 'codec': 'aac'}
    elif format == 'wav':
        return {'codec': 'pcm_s16le'}
    elif format == 'flac':
        return {'codec': 'flac'}
    else:
        return base_params


async def compress_audio(
    audio_data: bytes,
    source_format: str,
    compression_level: str = 'medium',
    target_bitrate: str | None = None
) -> bytes:
    """Compress audio for bandwidth optimization.

    Args:
        audio_data: Source audio data
        source_format: Source audio format
        compression_level: Compression level ('low', 'medium', 'high')
        target_bitrate: Specific target bitrate (e.g., '64k')

    Returns:
        Compressed audio data
    """
    # Define compression settings
    compression_settings = {
        'low': {'bitrate': '128k', 'format': 'mp3'},
        'medium': {'bitrate': '96k', 'format': 'mp3'},
        'high': {'bitrate': '64k', 'format': 'ogg'}
    }

    settings = compression_settings.get(compression_level, compression_settings['medium'])

    if target_bitrate:
        settings['bitrate'] = target_bitrate

    # Convert to compressed format
    try:
        return await convert_audio_format(
            audio_data,
            source_format=source_format,
            target_format=settings['format'],
            quality=compression_level
        )
    except AudioFormatError:
        # Fallback: return original if compression fails
        return audio_data


async def get_audio_info(audio_data: bytes) -> dict[str, Any]:
    """Extract comprehensive audio information.

    Args:
        audio_data: Raw audio file data

    Returns:
        Dict with detailed audio information
    """
    info = {
        'format': await detect_audio_format(audio_data),
        'size_bytes': len(audio_data),
        'mime_type': None,
        'duration_s': None,
        'sample_rate': None,
        'channels': None,
        'bit_depth': None
    }

    # Get MIME type
    if info['format']:
        mime_map = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4',
            'aac': 'audio/aac',
            'webm': 'audio/webm'
        }
        info['mime_type'] = mime_map.get(info['format'], 'audio/unknown')

    # Try to extract detailed metadata
    try:
        if LIBROSA_AVAILABLE and info['format']:
            with tempfile.NamedTemporaryFile(suffix=f'.{info["format"]}') as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()

                y, sr = librosa.load(temp_file.name, sr=None)
                info.update({
                    'duration_s': len(y) / sr,
                    'sample_rate': sr,
                    'channels': 1 if len(y.shape) == 1 else y.shape[0]
                })

    except Exception as e:
        logging.warning(f"Could not extract detailed audio info: {e}")

    return info
