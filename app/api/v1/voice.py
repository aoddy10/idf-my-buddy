"""Voice API router for My Buddy application.

This module provides voice and audio processing endpoints including speech recognition,
text-to-speech synthesis, voice conversation, and audio analysis.
"""

import contextlib
import io
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user_optional, get_db_session
from app.core.logging import LoggerMixin
from app.models.entities.user import User
from app.models.entities.voice_session import VoiceSession, VoiceSessionStatus
from app.schemas.common import BaseResponse, LanguageCode
from app.schemas.voice import (
    AudioAnalysisResponse,
    AudioFormat,
    AudioQuality,
    SpeechRecognitionResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VoiceConversationResponse,
    VoiceGender,
)
from app.services.audio.preprocessing import AudioPreprocessor
from app.services.tts import TTSService
from app.services.voice_pipeline import VoicePipeline
from app.services.whisper import WhisperService

logger = logging.getLogger(__name__)
router = APIRouter()


class VoiceService(LoggerMixin):
    """Voice service with speech and audio processing logic."""

    def __init__(self):
        super().__init__()
        self.whisper_service = None
        self.tts_service = None
        self.voice_pipeline = None
        self.audio_preprocessor = None

    def initialize_services(self):
        """Initialize voice services on first use."""
        if not self.whisper_service:
            self.whisper_service = WhisperService()

        if not self.tts_service:
            self.tts_service = TTSService()

        if not self.voice_pipeline:
            self.voice_pipeline = VoicePipeline()

        if not self.audio_preprocessor:
            self.audio_preprocessor = AudioPreprocessor()


voice_service = VoiceService()


@router.post(
    "/speech-to-text",
    response_model=SpeechRecognitionResponse,
    summary="Speech to text",
    description="Convert audio speech to text using ASR."
)
async def speech_to_text(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: LanguageCode | None = Form(None, description="Audio language hint"),
    detect_language: bool = Form(True, description="Auto-detect language"),
    include_confidence: bool = Form(False, description="Include confidence scores"),
    include_timestamps: bool = Form(False, description="Include word timestamps"),
    current_user: User | None = Depends(get_current_user_optional)
):
    """Convert speech to text."""
    try:
        # Initialize services
        voice_service.initialize_services()

        # Validate file
        if not audio_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is required"
            )

        # Read audio data
        audio_data = await audio_file.read()

        logger.info(
            "Speech recognition requested",
            extra={
                "filename": audio_file.filename,
                "content_type": audio_file.content_type,
                "size": len(audio_data),
                "language": language.value if language else None
            }
        )

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        try:
            # Transcribe audio
            result = await voice_service.whisper_service.transcribe_audio(
                Path(temp_file_path),
                language=language.value if language else None,
                return_timestamps=include_timestamps,
                return_confidence=include_confidence
            )

            # Format response
            response_data = {
                "success": True,
                "message": "Speech recognition completed",
                "data": {
                    "text": result.get("text", ""),
                    "language": result.get("language", "unknown"),
                    "confidence": result.get("confidence") if include_confidence else None,
                    "segments": result.get("segments") if include_timestamps else None,
                    "processing_time": result.get("processing_time"),
                }
            }

            return SpeechRecognitionResponse(**response_data)

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech recognition failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech recognition failed"
        )


@router.post(
    "/text-to-speech",
    summary="Text to speech",
    description="Convert text to speech audio. Returns streaming audio data."
)
async def text_to_speech(
    tts_request: TextToSpeechRequest,
    current_user: User | None = Depends(get_current_user_optional)
):
    """Convert text to speech."""
    try:
        # Initialize services
        voice_service.initialize_services()

        # Validate text
        if not tts_request.text or not tts_request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is required"
            )

        logger.info(
            "TTS requested",
            extra={
                "text_length": len(tts_request.text),
                "language": tts_request.language,
                "voice_gender": tts_request.voice_gender
            }
        )

        # Generate speech
        voice_preference = f"{tts_request.voice_gender.value}_{tts_request.language.value}" if tts_request.voice_gender else None

        audio_result = await voice_service.tts_service.synthesize_text(
            text=tts_request.text,
            language=tts_request.language.value if tts_request.language else "en",
            voice=voice_preference,
            speed=tts_request.speaking_rate,
            output_format=tts_request.output_format.value
        )

        if audio_result.get("audio_data"):
            # Return streaming response for audio with proper headers
            return StreamingResponse(
                io.BytesIO(audio_result["audio_data"]),
                media_type=f"audio/{tts_request.output_format.value}",
                headers={
                    "Content-Disposition": f"attachment; filename=tts_output.{tts_request.output_format.value}",
                    "X-Audio-Duration": str(audio_result.get("duration", 0)),
                    "X-Audio-Engine": audio_result.get("engine", "unknown"),
                    "X-Audio-Voice": audio_result.get("voice", "default"),
                    "X-Processing-Time": str(audio_result.get("processing_time", 0))
                }
            )
        else:
            # Return error if no audio data was generated
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate audio data"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text-to-speech failed"
        )


@router.post(
    "/conversation",
    response_model=VoiceConversationResponse,
    summary="Voice conversation",
    description="Process voice input for conversational AI interaction."
)
async def voice_conversation(
    audio_file: UploadFile = File(..., description="Voice input audio"),
    conversation_mode: str = Form("assistant", description="Conversation mode"),
    context_id: str | None = Form(None, description="Conversation context ID"),
    language: LanguageCode | None = Form(None, description="Input language"),
    response_language: LanguageCode = Form(LanguageCode.EN, description="Response language"),
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db_session)
):
    """Process voice conversation input."""
    try:
        # TODO: Implement voice conversation
        # 1. Transcribe audio to text
        # 2. Process with conversational AI
        # 3. Generate voice response
        # 4. Return both text and audio

        logger.info(
            "Voice conversation requested",
            extra={
                "filename": audio_file.filename,
                "mode": conversation_mode,
                "context_id": context_id
            }
        )

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Voice conversation not yet implemented"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice conversation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice conversation failed"
        )


@router.websocket("/conversation/stream")
async def voice_conversation_websocket(websocket: WebSocket):
    """Real-time voice conversation via WebSocket."""
    await websocket.accept()

    # Initialize services
    voice_service.initialize_services()

    try:
        while True:
            try:
                # Receive audio data from client
                data = await websocket.receive_bytes()

                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(data)
                    temp_file_path = temp_file.name

                try:
                    # Create minimal voice context
                    from app.services.voice_pipeline import VoiceContext

                    context = VoiceContext(
                        session_id="websocket_session",
                        user_id=None,
                        source_language="auto",
                        target_language="en"
                    )

                    # Process with voice pipeline
                    result = await voice_service.voice_pipeline.process_voice_input(
                        audio_data=data,
                        context=context,
                        return_audio=True
                    )

                    # Send response back to client
                    await websocket.send_json({
                        "type": "response",
                        "data": {
                            "text": result.response_text or "",
                            "audio_url": result.response_audio_url,
                            "confidence": result.confidence_scores,
                            "processing_time": result.processing_time
                        }
                    })

                finally:
                    # Cleanup temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


@router.post("/sessions", response_model=dict[str, Any])
async def create_voice_session(
    language: LanguageCode | None = Query(None, description="Preferred language"),
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new voice conversation session."""
    try:
        voice_service.initialize_services()

        # Create session in database
        session = VoiceSession(
            user_id=current_user.id if current_user else None,
            source_language="auto",
            target_language=language.value if language else "en",
            status=VoiceSessionStatus.ACTIVE
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        return {
            "success": True,
            "data": {
                "session_id": str(session.id),
                "status": session.status.value,
                "source_language": session.source_language,
                "target_language": session.target_language,
                "created_at": session.created_at.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to create voice session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create voice session"
        )


@router.get("/sessions/{session_id}", response_model=dict[str, Any])
async def get_voice_session(
    session_id: UUID,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db_session)
):
    """Get voice session details."""
    try:
        session = await db.get(VoiceSession, session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice session not found"
            )

        return {
            "success": True,
            "data": {
                "session_id": str(session.id),
                "status": session.status.value,
                "source_language": session.source_language,
                "target_language": session.target_language,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "exchange_count": len(session.voice_exchanges)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get voice session"
        )


@router.delete("/sessions/{session_id}", response_model=dict[str, Any])
async def end_voice_session(
    session_id: UUID,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db_session)
):
    """End voice conversation session."""
    try:
        session = await db.get(VoiceSession, session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice session not found"
            )

        session.status = VoiceSessionStatus.COMPLETED
        session.ended_at = datetime.utcnow()

        await db.commit()

        return {
            "success": True,
            "message": "Voice session ended successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end voice session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end voice session"
        )


@router.get("/voices", response_model=dict[str, Any])
async def get_available_voices():
    """Get available TTS voices."""
    try:
        voice_service.initialize_services()

        # Get voices from TTS service
        voices = []
        if voice_service.tts_service and hasattr(voice_service.tts_service, 'get_available_voices'):
            voices = await voice_service.tts_service.get_available_voices()
        else:
            # Default voice information
            voices = [
                {"id": "neutral", "name": "Neutral Voice", "gender": "neutral"},
                {"id": "male_en", "name": "English Male", "gender": "male"},
                {"id": "female_en", "name": "English Female", "gender": "female"}
            ]

        return {
            "success": True,
            "data": {
                "voices": voices,
                "voice_genders": [gender.value for gender in VoiceGender],
                "audio_formats": [format.value for format in AudioFormat],
                "audio_qualities": [quality.value for quality in AudioQuality]
            }
        }

    except Exception as e:
        logger.error(f"Failed to get available voices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available voices"
        )


@router.post(
    "/analysis/audio",
    response_model=AudioAnalysisResponse,
    summary="Analyze audio",
    description="Analyze audio for various properties and content."
)
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    analyze_emotion: bool = Form(False, description="Analyze emotional content"),
    analyze_quality: bool = Form(True, description="Analyze audio quality"),
    analyze_language: bool = Form(True, description="Detect spoken language"),
    analyze_speaker: bool = Form(False, description="Analyze speaker characteristics"),
    current_user: User | None = Depends(get_current_user_optional)
):
    """Analyze audio content and properties."""
    try:
        # TODO: Implement audio analysis
        # 1. Load and validate audio
        # 2. Run requested analysis types
        # 3. Return comprehensive analysis

        logger.info(
            "Audio analysis requested",
            extra={
                "filename": audio_file.filename,
                "analyze_emotion": analyze_emotion,
                "analyze_quality": analyze_quality
            }
        )

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Audio analysis not yet implemented"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio analysis failed"
        )


@router.post(
    "/commands/voice",
    response_model=VoiceConversationResponse,
    summary="Process voice command",
    description="Process and execute voice commands."
)
async def process_voice_command(
    audio_file: UploadFile = File(..., description="Voice command audio"),
    command_context: str | None = Form(None, description="Command context/domain"),
    language: LanguageCode | None = Form(None, description="Command language"),
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db_session)
):
    """Process voice commands."""
    try:
        # TODO: Implement voice command processing
        # 1. Transcribe voice command
        # 2. Parse command intent and parameters
        # 3. Execute command action
        # 4. Return execution result

        logger.info(
            "Voice command requested",
            extra={
                "filename": audio_file.filename,
                "context": command_context
            }
        )

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Voice command processing not yet implemented"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice command processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice command processing failed"
        )


@router.get(
    "/formats/supported",
    response_model=BaseResponse,
    summary="Get supported formats",
    description="Get list of supported audio formats and codecs."
)
async def get_supported_formats():
    """Get supported audio formats."""
    return {
        "success": True,
        "message": "Supported formats retrieved",
        "input_formats": [format.value for format in AudioFormat],
        "output_formats": ["mp3", "wav", "ogg", "m4a"],
        "sample_rates": [16000, 22050, 44100, 48000],
        "max_file_size": "50MB",
        "max_duration": "300s"
    }


@router.get(
    "/languages/supported",
    response_model=BaseResponse,
    summary="Get supported languages",
    description="Get list of supported speech languages."
)
async def get_supported_languages():
    """Get supported speech languages."""
    return {
        "success": True,
        "message": "Supported languages retrieved",
        "speech_languages": [lang.value for lang in LanguageCode],
        "auto_detection": True,
        "response_languages": [lang.value for lang in LanguageCode]
    }
