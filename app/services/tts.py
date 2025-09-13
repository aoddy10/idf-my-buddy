"""TTS (Text-to-Speech) service.

This module provides text-to-speech functionality using various TTS engines
with support for multiple languages and voice characteristics.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import LoggerMixin

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import torch
    import torchaudio
    from speechbrain.pretrained import HIFIGAN, Tacotron2
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TTSService(LoggerMixin):
    """Text-to-Speech service with multiple backend support.

    Supports local TTS engines (pyttsx3, SpeechBrain) and cloud APIs (OpenAI, gTTS)
    with intelligent fallback and voice customization.
    """

    def __init__(self):
        super().__init__()
        self._pyttsx3_engine = None
        self._speechbrain_tacotron = None
        self._speechbrain_vocoder = None
        self._openai_client = None
        self._default_voice = "en"
        self._device = None

        # Initialize available TTS engines
        self._setup_service()

    def _setup_service(self):
        """Setup TTS service with available engines."""
        try:
            # Setup pyttsx3 (local, offline)
            if PYTTSX3_AVAILABLE:
                self._setup_pyttsx3()

            # Setup SpeechBrain (local, high quality)
            if SPEECHBRAIN_AVAILABLE and getattr(settings, 'speechbrain_enabled', True):
                self._setup_speechbrain()

            # Setup OpenAI TTS (cloud, high quality)
            if OPENAI_AVAILABLE and settings.openai_api_key:
                self._setup_openai()

            if not any([self._pyttsx3_engine, self._speechbrain_tacotron, self._openai_client]):
                self.logger.warning("No TTS backend available - speech synthesis will be disabled")

        except Exception as e:
            self.logger.error(f"Failed to setup TTS service: {e}")

    def _setup_pyttsx3(self):
        """Setup pyttsx3 TTS engine."""
        try:
            self._pyttsx3_engine = pyttsx3.init()

            # Configure engine properties
            self._pyttsx3_engine.setProperty('rate', 180)  # Speed
            self._pyttsx3_engine.setProperty('volume', 0.8)  # Volume

            self.logger.info("pyttsx3 TTS engine initialized")

        except Exception as e:
            self.logger.warning(f"pyttsx3 not available: {e}")
            self._pyttsx3_engine = None

    def _setup_speechbrain(self):
        """Setup SpeechBrain TTS models."""
        try:
            # Setup device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load Tacotron2 model for text-to-mel conversion
            self._speechbrain_tacotron = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech",
                savedir=Path(settings.model_cache_dir) / "tacotron2",
                run_opts={"device": self._device}
            )

            # Load HiFi-GAN vocoder for mel-to-audio conversion
            self._speechbrain_vocoder = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech",
                savedir=Path(settings.model_cache_dir) / "hifigan",
                run_opts={"device": self._device}
            )

            self.logger.info("SpeechBrain TTS models loaded")

        except Exception as e:
            self.logger.error(f"Failed to initialize SpeechBrain TTS: {e}")
            self._speechbrain_tacotron = None
            self._speechbrain_vocoder = None

    def _setup_openai(self):
        """Setup OpenAI TTS client."""
        try:
            self._openai_client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=30.0
            )
            self.logger.info("OpenAI TTS client initialized")

        except Exception as e:
            self.logger.error(f"Failed to setup OpenAI TTS client: {e}")
            self._openai_client = None

    async def synthesize_text(
        self,
        text: str,
        language: str = "en",
        voice: str | None = None,
        engine: str | None = None,
        speed: float = 1.0,
        output_format: str = "mp3"
    ) -> dict[str, Any]:
        """Synthesize speech from text.

        Args:
            text: Text to convert to speech
            language: Language code
            voice: Voice identifier or characteristics
            engine: Specific TTS engine to use
            speed: Speech rate multiplier (0.5-2.0)
            output_format: Output audio format ("mp3", "wav", "ogg")

        Returns:
            Dict containing audio data and metadata
        """
        try:
            # Validate inputs
            if not text.strip():
                raise ValueError("Text cannot be empty")

            if speed < 0.5 or speed > 2.0:
                speed = 1.0

            # Choose TTS engine
            if engine is None:
                engine = self._choose_best_engine(language, voice)

            # Synthesize speech based on engine
            if engine == "openai" and self._openai_client:
                return await self._synthesize_openai(text, voice, speed, output_format)
            elif engine == "speechbrain" and self._speechbrain_tacotron:
                return await self._synthesize_speechbrain(text, speed, output_format)
            elif engine == "gtts":
                return await self._synthesize_gtts(text, language, speed, output_format)
            elif engine == "pyttsx3" and self._pyttsx3_engine:
                return await self._synthesize_pyttsx3(text, voice, speed, output_format)
            else:
                # Try available engines in order of preference
                for fallback_engine in ["openai", "speechbrain", "gtts", "pyttsx3"]:
                    if self._is_engine_available(fallback_engine):
                        self.logger.info(f"Falling back to {fallback_engine}")
                        return await self.synthesize_text(
                            text, language, voice, fallback_engine, speed, output_format
                        )

                raise RuntimeError("No TTS engine available")

        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise

    def _choose_best_engine(self, language: str, voice: str | None) -> str:
        """Choose the best TTS engine for the given parameters."""

        # Prefer high-quality engines for production
        if self._openai_client and language in ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]:
            return "openai"
        elif self._speechbrain_tacotron and language == "en":
            return "speechbrain"
        elif language in ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"]:
            return "gtts"
        elif self._pyttsx3_engine:
            return "pyttsx3"
        else:
            # Return first available
            for engine in ["openai", "speechbrain", "gtts", "pyttsx3"]:
                if self._is_engine_available(engine):
                    return engine
            raise RuntimeError("No TTS engine available")

    def _is_engine_available(self, engine: str) -> bool:
        """Check if TTS engine is available."""
        if engine == "openai":
            return self._openai_client is not None
        elif engine == "speechbrain":
            return self._speechbrain_tacotron is not None
        elif engine == "gtts":
            return GTTS_AVAILABLE
        elif engine == "pyttsx3":
            return self._pyttsx3_engine is not None
        return False

    async def _synthesize_openai(
        self,
        text: str,
        voice: str | None,
        speed: float,
        output_format: str
    ) -> dict[str, Any]:
        """Synthesize speech using OpenAI TTS."""

        try:
            # Map voice parameter to OpenAI voices
            openai_voice = self._map_voice_to_openai(voice)

            # Call OpenAI TTS API
            response = await self._openai_client.audio.speech.create(
                model="tts-1-hd",  # High quality model
                voice=openai_voice,
                input=text,
                response_format=output_format if output_format in ["mp3", "opus", "aac", "flac"] else "mp3",
                speed=speed
            )

            # Get audio data
            audio_data = await response.aread()

            return {
                "audio_data": audio_data,
                "format": output_format,
                "sample_rate": 24000,  # OpenAI default
                "duration": len(audio_data) / 24000,  # Approximate
                "engine": "openai",
                "voice": openai_voice,
                "language": "auto-detected"
            }

        except Exception as e:
            self.logger.error(f"OpenAI TTS failed: {e}")
            raise

    def _map_voice_to_openai(self, voice: str | None) -> str:
        """Map voice parameter to OpenAI voice names."""
        voice_mapping = {
            "male": "onyx",
            "female": "nova",
            "neutral": "echo",
            "alloy": "alloy",
            "echo": "echo",
            "fable": "fable",
            "onyx": "onyx",
            "nova": "nova",
            "shimmer": "shimmer"
        }

        return voice_mapping.get(voice or "neutral", "nova")  # Default to nova

    async def _synthesize_speechbrain(
        self,
        text: str,
        speed: float,
        output_format: str
    ) -> dict[str, Any]:
        """Synthesize speech using SpeechBrain."""

        # Run synthesis in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._speechbrain_synthesis,
            text, speed
        )

        # Convert to requested format if needed
        audio_data = await self._convert_audio_format(result["audio_data"], "wav", output_format)

        result["audio_data"] = audio_data
        result["format"] = output_format

        return result

    def _speechbrain_synthesis(self, text: str, speed: float) -> dict[str, Any]:
        """SpeechBrain synthesis (blocking operation)."""
        try:
            # Generate mel spectrogram
            mel_output, mel_length, alignment = self._speechbrain_tacotron.encode_text(text)

            # Generate audio from mel spectrogram
            waveforms = self._speechbrain_vocoder.decode_batch(mel_output)

            # Apply speed adjustment (simple time stretching)
            if speed != 1.0:
                # Simple implementation - in practice, use proper time stretching
                target_length = int(waveforms.shape[-1] / speed)
                waveforms = torch.nn.functional.interpolate(
                    waveforms.unsqueeze(0),
                    size=target_length,
                    mode='linear'
                ).squeeze(0)

            # Convert to numpy and save as WAV
            audio_np = waveforms.squeeze().cpu().numpy()

            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

            torchaudio.save(
                temp_file.name,
                waveforms.unsqueeze(0),
                sample_rate=22050
            )

            # Read back as bytes
            with open(temp_file.name, "rb") as f:
                audio_data = f.read()

            # Cleanup
            Path(temp_file.name).unlink(missing_ok=True)

            return {
                "audio_data": audio_data,
                "format": "wav",
                "sample_rate": 22050,
                "duration": len(audio_np) / 22050,
                "engine": "speechbrain",
                "voice": "ljspeech",
                "language": "en"
            }

        except Exception as e:
            self.logger.error(f"SpeechBrain synthesis failed: {e}")
            raise

    async def _synthesize_gtts(
        self,
        text: str,
        language: str,
        speed: float,
        output_format: str
    ) -> dict[str, Any]:
        """Synthesize speech using gTTS (Google Text-to-Speech)."""

        # Run synthesis in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._gtts_synthesis,
            text, language, speed
        )

        # Convert format if needed
        if output_format != "mp3":
            audio_data = await self._convert_audio_format(result["audio_data"], "mp3", output_format)
            result["audio_data"] = audio_data
            result["format"] = output_format

        return result

    def _gtts_synthesis(self, text: str, language: str, speed: float) -> dict[str, Any]:
        """gTTS synthesis (blocking operation)."""
        try:
            # Create gTTS object
            # Note: gTTS doesn't support speed adjustment directly
            tts = gTTS(text=text, lang=language, slow=(speed < 0.8))

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(temp_file.name)

            # Read audio data
            with open(temp_file.name, "rb") as f:
                audio_data = f.read()

            # Cleanup
            Path(temp_file.name).unlink(missing_ok=True)

            return {
                "audio_data": audio_data,
                "format": "mp3",
                "sample_rate": 24000,  # gTTS default
                "duration": len(audio_data) / 3000,  # Rough estimate
                "engine": "gtts",
                "voice": "google",
                "language": language
            }

        except Exception as e:
            self.logger.error(f"gTTS synthesis failed: {e}")
            raise

    async def _synthesize_pyttsx3(
        self,
        text: str,
        voice: str | None,
        speed: float,
        output_format: str
    ) -> dict[str, Any]:
        """Synthesize speech using pyttsx3."""

        # Run synthesis in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._pyttsx3_synthesis,
            text, voice, speed
        )

        # Convert format if needed
        if output_format != "wav":
            audio_data = await self._convert_audio_format(result["audio_data"], "wav", output_format)
            result["audio_data"] = audio_data
            result["format"] = output_format

        return result

    def _pyttsx3_synthesis(self, text: str, voice: str | None, speed: float) -> dict[str, Any]:
        """pyttsx3 synthesis (blocking operation)."""
        try:
            # Configure engine
            self._pyttsx3_engine.setProperty('rate', int(180 * speed))

            if voice:
                voices = self._pyttsx3_engine.getProperty('voices')
                # Simple voice selection logic
                for v in voices:
                    if voice.lower() in v.name.lower() or voice.lower() in v.id.lower():
                        self._pyttsx3_engine.setProperty('voice', v.id)
                        break

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self._pyttsx3_engine.save_to_file(text, temp_file.name)
            self._pyttsx3_engine.runAndWait()

            # Read audio data
            with open(temp_file.name, "rb") as f:
                audio_data = f.read()

            # Cleanup
            Path(temp_file.name).unlink(missing_ok=True)

            return {
                "audio_data": audio_data,
                "format": "wav",
                "sample_rate": 22050,
                "duration": len(audio_data) / 22050 / 2,  # Rough estimate
                "engine": "pyttsx3",
                "voice": voice or "default",
                "language": "en"
            }

        except Exception as e:
            self.logger.error(f"pyttsx3 synthesis failed: {e}")
            raise

    async def _convert_audio_format(
        self,
        audio_data: bytes,
        source_format: str,
        target_format: str
    ) -> bytes:
        """Convert audio between formats."""

        if source_format == target_format:
            return audio_data

        # TODO: Implement audio format conversion
        # For now, return original data
        self.logger.warning(f"Audio format conversion {source_format}->{target_format} not implemented")
        return audio_data

    async def get_available_voices(self, engine: str | None = None) -> dict[str, list[str]]:
        """Get available voices for each engine."""

        voices = {}

        if (engine is None or engine == "openai") and self._openai_client:
            voices["openai"] = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

        if (engine is None or engine == "pyttsx3") and self._pyttsx3_engine:
            # Run in thread pool since this might be blocking
            pyttsx3_voices = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: [v.name for v in self._pyttsx3_engine.getProperty('voices')]
            )
            voices["pyttsx3"] = pyttsx3_voices

        if engine is None or engine == "speechbrain":
            if self._speechbrain_tacotron:
                voices["speechbrain"] = ["ljspeech"]

        if engine is None or engine == "gtts":
            voices["gtts"] = ["google"]

        return voices

    def get_supported_languages(self) -> dict[str, list[str]]:
        """Get supported languages by engine."""
        return {
            "openai": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            "gtts": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "th", "vi"],
            "speechbrain": ["en"],
            "pyttsx3": ["en"]  # Depends on system voices
        }

    def is_available(self) -> bool:
        """Check if TTS service is available."""
        return any([
            self._openai_client,
            self._speechbrain_tacotron,
            GTTS_AVAILABLE,
            self._pyttsx3_engine
        ])
