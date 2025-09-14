"""Voice processing pipeline service.

This module provides unified voice conversation management by coordinating
ASR (Whisper), TTS, and machine translation services for seamless
speech-to-speech communication.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.services.audio.formats import get_audio_info, validate_audio_file
from app.services.audio.preprocessing import preprocess_audio_pipeline
from app.services.nllb import NLLBTranslationService
from app.services.tts import TTSService
from app.services.whisper import WhisperService

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ProcessingStage(str, Enum):
    """Voice processing pipeline stages."""
    AUDIO_PREPROCESSING = "audio_preprocessing"
    SPEECH_RECOGNITION = "speech_recognition"
    TRANSLATION = "translation"
    RESPONSE_GENERATION = "response_generation"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_POSTPROCESSING = "audio_postprocessing"


class PipelineStatus(str, Enum):
    """Pipeline processing status."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class VoiceContext:
    """Context for voice conversation."""
    session_id: str
    user_id: str | None = None
    source_language: str = "auto"
    target_language: str = "en"
    preferred_voice: str = "neutral"
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    travel_context: dict[str, Any] | None = None
    location: dict[str, Any] | None = None
    preferences: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, 'session_id') or not self.session_id:
            self.session_id = str(uuid.uuid4())


@dataclass
class VoiceExchange:
    """Single voice interaction exchange."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    user_audio_data: bytes | None = None
    user_text: str | None = None
    user_language: str | None = None
    translated_text: str | None = None
    response_text: str | None = None
    response_audio_data: bytes | None = None
    response_audio_url: str | None = None
    processing_time: float = 0.0
    confidence_scores: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class ProcessingMetrics:
    """Performance metrics for voice processing."""
    total_time: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)
    audio_duration: float = 0.0
    real_time_factor: float = 0.0  # processing_time / audio_duration
    memory_usage_mb: float = 0.0

    def calculate_rtf(self):
        """Calculate real-time factor."""
        if self.audio_duration > 0:
            self.real_time_factor = self.total_time / self.audio_duration


class VoicePipeline(LoggerMixin):
    """Unified voice processing pipeline.

    Coordinates ASR, TTS, and translation services for complete
    speech-to-speech conversation flows.
    """

    def __init__(self):
        super().__init__()

        # Service instances
        self.whisper_service = WhisperService()
        self.tts_service = TTSService()
        self.nllb_service = NLLBTranslationService()

        # Pipeline state
        self._status = PipelineStatus.IDLE
        self._current_stage = None
        self._active_sessions: dict[str, VoiceContext] = {}

        # Performance monitoring
        self._performance_history: list[ProcessingMetrics] = []
        self._max_history_size = 100

        # Quality thresholds
        self.min_confidence_threshold = 0.6
        self.max_processing_time = 10.0  # seconds
        self.target_rtf = 0.3  # 30% of real-time

    async def process_voice_input(
        self,
        audio_data: bytes,
        context: VoiceContext,
        return_audio: bool = True,
        quality_level: str = "balanced"
    ) -> VoiceExchange:
        """Process complete voice input through the pipeline.

        Args:
            audio_data: Raw audio data from user
            context: Voice conversation context
            return_audio: Whether to generate audio response
            quality_level: Processing quality ('fast', 'balanced', 'high')

        Returns:
            VoiceExchange with complete interaction results
        """
        start_time = time.time()
        exchange = VoiceExchange(user_audio_data=audio_data)
        metrics = ProcessingMetrics()

        try:
            self._status = PipelineStatus.PROCESSING

            # Stage 1: Audio preprocessing
            exchange, metrics = await self._stage_audio_preprocessing(
                exchange, context, metrics, quality_level
            )

            # Stage 2: Speech recognition
            exchange, metrics = await self._stage_speech_recognition(
                exchange, context, metrics, quality_level
            )

            # Stage 3: Translation (if needed)
            exchange, metrics = await self._stage_translation(
                exchange, context, metrics
            )

            # Stage 4: Response generation
            exchange, metrics = await self._stage_response_generation(
                exchange, context, metrics
            )

            # Stage 5: Text-to-speech (if requested)
            if return_audio:
                exchange, metrics = await self._stage_text_to_speech(
                    exchange, context, metrics, quality_level
                )

            # Update metrics
            metrics.total_time = time.time() - start_time
            metrics.calculate_rtf()
            exchange.processing_time = metrics.total_time

            # Update context
            await self._update_conversation_context(exchange, context)

            self._status = PipelineStatus.COMPLETED
            self.logger.info(
                "Voice pipeline completed",
                processing_time=metrics.total_time,
                rtf=metrics.real_time_factor,
                session_id=context.session_id
            )

        except Exception as e:
            self._status = PipelineStatus.ERROR
            exchange.error_message = str(e)
            self.logger.error(f"Voice pipeline failed: {e}", session_id=context.session_id)

        finally:
            # Store performance metrics
            self._performance_history.append(metrics)
            if len(self._performance_history) > self._max_history_size:
                self._performance_history.pop(0)

        return exchange

    async def _stage_audio_preprocessing(
        self,
        exchange: VoiceExchange,
        context: VoiceContext,
        metrics: ProcessingMetrics,
        quality_level: str
    ) -> tuple[VoiceExchange, ProcessingMetrics]:
        """Stage 1: Audio preprocessing and validation."""
        stage_start = time.time()
        self._current_stage = ProcessingStage.AUDIO_PREPROCESSING

        try:
            # Validate audio
            if not exchange.user_audio_data:
                raise ValueError("No audio data provided")

            audio_info = await get_audio_info(exchange.user_audio_data)
            metrics.audio_duration = audio_info.get('duration_s', 0)

            # Validate constraints
            await validate_audio_file(
                exchange.user_audio_data,
                max_size_bytes=settings.max_audio_size,
                max_duration_s=settings.max_audio_duration,
                allowed_formats=['wav', 'mp3', 'ogg', 'm4a', 'webm']
            )

            # Preprocess audio based on quality level
            enable_noise_reduction = quality_level in ['balanced', 'high']
            enable_vad = quality_level == 'high'

            if NUMPY_AVAILABLE:
                processed_audio, voice_segments = await preprocess_audio_pipeline(
                    exchange.user_audio_data,
                    sample_rate=audio_info.get('sample_rate', 16000),
                    enable_vad=enable_vad,
                    enable_noise_reduction=enable_noise_reduction,
                    enable_normalization=True,
                    target_sample_rate=16000  # Whisper optimal
                )

                # Convert back to bytes for Whisper
                if hasattr(processed_audio, 'tobytes'):
                    exchange.user_audio_data = (processed_audio * 32767).astype('int16').tobytes()

        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise

        finally:
            metrics.stage_times[ProcessingStage.AUDIO_PREPROCESSING] = time.time() - stage_start

        return exchange, metrics

    async def _stage_speech_recognition(
        self,
        exchange: VoiceExchange,
        context: VoiceContext,
        metrics: ProcessingMetrics,
        quality_level: str
    ) -> tuple[VoiceExchange, ProcessingMetrics]:
        """Stage 2: Speech-to-text recognition."""
        stage_start = time.time()
        self._current_stage = ProcessingStage.SPEECH_RECOGNITION

        try:
            # Choose model size based on quality level
            model_size_map = {
                'fast': 'tiny',
                'balanced': 'base',
                'high': 'small'
            }
            model_size_map.get(quality_level, 'base')

            # Transcribe audio
            if not exchange.user_audio_data:
                raise ValueError("No audio data for transcription")

            transcription_result = await self.whisper_service.transcribe_audio(
                exchange.user_audio_data,
                language=context.source_language if context.source_language != "auto" else None,
                return_confidence=True
            )

            # Extract results
            exchange.user_text = transcription_result["text"]
            exchange.user_language = transcription_result.get("language", "unknown")

            # Store confidence
            confidence = transcription_result.get("confidence", 0.8)
            exchange.confidence_scores["transcription"] = confidence

            # Check quality threshold
            if confidence < self.min_confidence_threshold:
                self.logger.warning(
                    f"Low transcription confidence: {confidence}",
                    session_id=context.session_id
                )

            # Update context language if auto-detected
            if context.source_language == "auto" and exchange.user_language:
                context.source_language = exchange.user_language

        except Exception as e:
            self.logger.error(f"Speech recognition failed: {e}")
            raise

        finally:
            metrics.stage_times[ProcessingStage.SPEECH_RECOGNITION] = time.time() - stage_start

        return exchange, metrics

    async def _stage_translation(
        self,
        exchange: VoiceExchange,
        context: VoiceContext,
        metrics: ProcessingMetrics
    ) -> tuple[VoiceExchange, ProcessingMetrics]:
        """Stage 3: Translation (if needed)."""
        stage_start = time.time()
        self._current_stage = ProcessingStage.TRANSLATION

        try:
            # Check if translation is needed
            source_lang = exchange.user_language or context.source_language
            target_lang = context.target_language

            if source_lang != target_lang and exchange.user_text:
                # Translate user text
                translation_result = await self.nllb_service.translate_text(
                    exchange.user_text,
                    target_language=target_lang,
                    source_language=source_lang
                )

                exchange.translated_text = translation_result["translated_text"]
                exchange.confidence_scores["translation"] = translation_result.get("confidence", 0.9)
            else:
                # No translation needed
                exchange.translated_text = exchange.user_text
                exchange.confidence_scores["translation"] = 1.0

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            # Fall back to original text
            exchange.translated_text = exchange.user_text
            exchange.confidence_scores["translation"] = 0.5

        finally:
            metrics.stage_times[ProcessingStage.TRANSLATION] = time.time() - stage_start

        return exchange, metrics

    async def _stage_response_generation(
        self,
        exchange: VoiceExchange,
        context: VoiceContext,
        metrics: ProcessingMetrics
    ) -> tuple[VoiceExchange, ProcessingMetrics]:
        """Stage 4: Generate contextual response."""
        stage_start = time.time()
        self._current_stage = ProcessingStage.RESPONSE_GENERATION

        try:
            # For now, implement a simple echo/acknowledgment response
            # In a full implementation, this would interface with:
            # - Intent recognition
            # - Knowledge bases
            # - Travel domain logic
            # - Conversation management

            input_text = exchange.translated_text or exchange.user_text or ""

            # Simple response generation based on travel context
            if context.travel_context:
                location = context.travel_context.get('current_location', 'unknown location')
                response = f"I understand you said: '{input_text}'. I'm here to help you with travel in {location}. What would you like to know?"
            else:
                response = f"I heard: '{input_text}'. How can I assist you with your travel today?"

            exchange.response_text = response
            exchange.confidence_scores["response"] = 1.0

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            exchange.response_text = "I'm sorry, I didn't understand that. Could you please try again?"
            exchange.confidence_scores["response"] = 0.5

        finally:
            metrics.stage_times[ProcessingStage.RESPONSE_GENERATION] = time.time() - stage_start

        return exchange, metrics

    async def _stage_text_to_speech(
        self,
        exchange: VoiceExchange,
        context: VoiceContext,
        metrics: ProcessingMetrics,
        quality_level: str
    ) -> tuple[VoiceExchange, ProcessingMetrics]:
        """Stage 5: Text-to-speech synthesis."""
        stage_start = time.time()
        self._current_stage = ProcessingStage.TEXT_TO_SPEECH

        try:
            # Choose TTS quality settings
            tts_engine = None
            if quality_level == "fast":
                tts_engine = "pyttsx3"
            elif quality_level == "high":
                tts_engine = "openai"

            # Synthesize speech
            if not exchange.response_text:
                raise ValueError("No response text for TTS")

            tts_result = await self.tts_service.synthesize_text(
                exchange.response_text,
                language=context.target_language,
                voice=context.preferred_voice,
                engine=tts_engine,
                speed=context.preferences.get('speech_rate', 1.0),
                output_format='mp3'
            )

            exchange.response_audio_data = tts_result["audio_data"]
            exchange.confidence_scores["tts"] = 1.0

            # TODO: In a full implementation, this would:
            # - Save audio to temporary storage
            # - Generate accessible URL
            # - Handle streaming for long responses
            exchange.response_audio_url = f"/audio/temp/{exchange.exchange_id}.mp3"

        except Exception as e:
            self.logger.error(f"TTS failed: {e}")
            exchange.confidence_scores["tts"] = 0.0

        finally:
            metrics.stage_times[ProcessingStage.TEXT_TO_SPEECH] = time.time() - stage_start

        return exchange, metrics

    async def _update_conversation_context(
        self,
        exchange: VoiceExchange,
        context: VoiceContext
    ):
        """Update conversation context with exchange results."""
        # Add to conversation history
        context.conversation_history.append({
            'exchange_id': exchange.exchange_id,
            'timestamp': exchange.timestamp,
            'user_text': exchange.user_text,
            'user_language': exchange.user_language,
            'response_text': exchange.response_text,
            'processing_time': exchange.processing_time,
            'confidence_scores': exchange.confidence_scores
        })

        # Limit history size
        max_history = 20
        if len(context.conversation_history) > max_history:
            context.conversation_history = context.conversation_history[-max_history:]

        # Update session in active sessions
        self._active_sessions[context.session_id] = context

    async def create_voice_session(
        self,
        user_id: str | None = None,
        source_language: str = "auto",
        target_language: str = "en",
        preferred_voice: str = "neutral",
        travel_context: dict[str, Any] | None = None
    ) -> VoiceContext:
        """Create a new voice conversation session.

        Args:
            user_id: Optional user identifier
            source_language: Source language code or 'auto'
            target_language: Target language code
            preferred_voice: Preferred TTS voice
            travel_context: Travel-specific context

        Returns:
            New VoiceContext instance
        """
        context = VoiceContext(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            source_language=source_language,
            target_language=target_language,
            preferred_voice=preferred_voice,
            travel_context=travel_context
        )

        self._active_sessions[context.session_id] = context

        self.logger.info(f"Created voice session: {context.session_id}")
        return context

    async def get_voice_session(self, session_id: str) -> VoiceContext | None:
        """Get existing voice session."""
        return self._active_sessions.get(session_id)

    async def end_voice_session(self, session_id: str) -> bool:
        """End and cleanup voice session."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            self.logger.info(f"Ended voice session: {session_id}")
            return True
        return False

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get aggregated performance metrics."""
        if not self._performance_history:
            return {}

        recent_metrics = self._performance_history[-10:]  # Last 10 executions

        return {
            'average_processing_time': sum(m.total_time for m in recent_metrics) / len(recent_metrics),
            'average_rtf': sum(m.real_time_factor for m in recent_metrics) / len(recent_metrics),
            'stage_breakdown': {
                stage: sum(m.stage_times.get(stage, 0) for m in recent_metrics) / len(recent_metrics)
                for stage in ProcessingStage
            },
            'total_sessions': len(self._active_sessions),
            'pipeline_status': self._status.value
        }

    def is_healthy(self) -> bool:
        """Check if pipeline is healthy and responsive."""
        try:
            # Check service availability
            if not self.whisper_service.is_available():
                return False

            if not self.tts_service.is_available():
                return False

            # Check recent performance
            if self._performance_history:
                recent_rtf = self._performance_history[-1].real_time_factor
                if recent_rtf > self.target_rtf * 3:  # More than 3x target
                    return False

            return True

        except Exception:
            return False
