"""Whisper ASR (Automatic Speech Recognition) service.

This module provides speech-to-text functionality using OpenAI's Whisper model
with support for local edge computing and cloud API fallback.
"""

import logging
import tempfile
import asyncio
from typing import Optional, Dict, Any, List, Union, BinaryIO
from pathlib import Path
import io

from app.core.logging import LoggerMixin
from app.core.config import settings

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class WhisperService(LoggerMixin):
    """Whisper-based speech recognition service.
    
    Supports both local Whisper models and OpenAI API with intelligent fallback.
    Optimized for edge computing with model caching and efficient loading.
    """
    
    def __init__(self):
        super().__init__()
        self._local_model = None
        self._model_name = "base"  # Default model size
        self._device = None
        self._openai_client = None
        
        # Initialize based on available resources
        self._setup_service()
    
    def _setup_service(self):
        """Setup the ASR service based on available resources."""
        try:
            if WHISPER_AVAILABLE and settings.WHISPER_USE_LOCAL:
                self._setup_local_model()
            
            if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
                self._setup_openai_client()
                
            if not self._local_model and not self._openai_client:
                self.logger.warning("No ASR backend available - speech recognition will be disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to setup Whisper service: {e}")
    
    def _setup_local_model(self):
        """Setup local Whisper model for edge computing."""
        try:
            # Determine optimal model size based on available memory
            self._model_name = self._get_optimal_model_size()
            
            # Setup device (CPU/GPU)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading Whisper {self._model_name} model on {self._device}")
            
            # Load model with caching
            self._local_model = whisper.load_model(
                self._model_name,
                device=self._device,
                download_root=settings.MODEL_CACHE_DIR
            )
            
            self.logger.info("Local Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load local Whisper model: {e}")
            self._local_model = None
    
    def _setup_openai_client(self):
        """Setup OpenAI client for cloud ASR."""
        try:
            self._openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=30.0
            )
            self.logger.info("OpenAI Whisper API client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup OpenAI client: {e}")
            self._openai_client = None
    
    def _get_optimal_model_size(self) -> str:
        """Determine optimal Whisper model size based on system resources."""
        try:
            # Get available memory
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            # Choose model based on memory
            if available_gb >= 8:
                return "large"
            elif available_gb >= 4:
                return "medium"
            elif available_gb >= 2:
                return "small"
            else:
                return "tiny"
                
        except ImportError:
            # Fallback if psutil not available
            return "base"
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, BinaryIO, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """Transcribe audio to text using Whisper.
        
        Args:
            audio_data: Audio file data, file object, or path
            language: ISO language code (None for auto-detection)
            task: "transcribe" or "translate" 
            return_timestamps: Include word-level timestamps
            return_confidence: Include confidence scores
            
        Returns:
            Dict containing transcription results
        """
        try:
            # Try local model first if available
            if self._local_model:
                return await self._transcribe_local(
                    audio_data, language, task, return_timestamps, return_confidence
                )
            
            # Fallback to OpenAI API
            elif self._openai_client:
                return await self._transcribe_openai(
                    audio_data, language, task, return_timestamps
                )
            
            else:
                raise RuntimeError("No ASR backend available")
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    async def _transcribe_local(
        self,
        audio_data: Union[bytes, BinaryIO, Path],
        language: Optional[str] = None,
        task: str = "transcribe", 
        return_timestamps: bool = False,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        
        # Prepare audio file
        audio_path = await self._prepare_audio_file(audio_data)
        
        try:
            # Run transcription in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._local_model.transcribe(
                    str(audio_path),
                    language=language,
                    task=task,
                    word_timestamps=return_timestamps,
                    verbose=False
                )
            )
            
            # Format result
            formatted_result = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", 0),
                "backend": "local_whisper"
            }
            
            if return_timestamps and "segments" in result:
                formatted_result["segments"] = [
                    {
                        "text": segment["text"].strip(),
                        "start": segment["start"],
                        "end": segment["end"],
                        "words": segment.get("words", []) if return_timestamps else None
                    }
                    for segment in result["segments"]
                ]
            
            if return_confidence:
                # Whisper doesn't provide confidence scores directly
                # We could implement this with additional processing
                formatted_result["confidence"] = 0.85  # Placeholder
            
            return formatted_result
            
        finally:
            # Cleanup temp file if created
            if isinstance(audio_data, bytes):
                audio_path.unlink(missing_ok=True)
    
    async def _transcribe_openai(
        self,
        audio_data: Union[bytes, BinaryIO, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        
        # Prepare audio for API
        if isinstance(audio_data, Path):
            with open(audio_data, "rb") as f:
                audio_file = f.read()
        elif isinstance(audio_data, bytes):
            audio_file = audio_data
        else:
            audio_file = audio_data.read()
        
        try:
            # Call OpenAI API
            if task == "translate":
                response = await self._openai_client.audio.translations.create(
                    model="whisper-1",
                    file=io.BytesIO(audio_file),
                    response_format="verbose_json" if return_timestamps else "json"
                )
            else:
                response = await self._openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=io.BytesIO(audio_file),
                    language=language,
                    response_format="verbose_json" if return_timestamps else "json"
                )
            
            # Format result
            result = {
                "text": response.text.strip(),
                "backend": "openai_api"
            }
            
            if hasattr(response, 'language'):
                result["language"] = response.language
            
            if hasattr(response, 'duration'):
                result["duration"] = response.duration
                
            if return_timestamps and hasattr(response, 'segments'):
                result["segments"] = [
                    {
                        "text": segment["text"].strip(),
                        "start": segment["start"], 
                        "end": segment["end"]
                    }
                    for segment in response.segments
                ]
            
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI API transcription failed: {e}")
            raise
    
    async def _prepare_audio_file(self, audio_data: Union[bytes, BinaryIO, Path]) -> Path:
        """Prepare audio file for processing."""
        
        if isinstance(audio_data, Path):
            return audio_data
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        
        try:
            if isinstance(audio_data, bytes):
                temp_file.write(audio_data)
            else:
                temp_file.write(audio_data.read())
            
            temp_file.flush()
            return Path(temp_file.name)
            
        finally:
            temp_file.close()
    
    async def detect_language(self, audio_data: Union[bytes, BinaryIO, Path]) -> str:
        """Detect the language of the audio."""
        try:
            if self._local_model:
                audio_path = await self._prepare_audio_file(audio_data)
                
                # Load first 30 seconds for language detection
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._local_model.transcribe(
                        str(audio_path),
                        task="transcribe",
                        language=None  # Auto-detect
                    )
                )
                
                return result.get("language", "unknown")
            
            else:
                # Use OpenAI API for language detection
                result = await self._transcribe_openai(audio_data)
                return result.get("language", "unknown")
                
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown"
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Whisper supports these languages
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    def is_available(self) -> bool:
        """Check if ASR service is available."""
        return self._local_model is not None or self._openai_client is not None
