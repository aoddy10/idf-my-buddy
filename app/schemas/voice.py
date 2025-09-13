"""Voice and audio processing schemas for My Buddy API.

This module contains Pydantic schemas for voice-related endpoints,
including speech recognition, text-to-speech, and audio processing.
"""

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, validator

from app.schemas.common import BaseResponse, FileUpload, LanguageCode


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    AAC = "aac"
    OGG = "ogg"
    FLAC = "flac"
    AMR = "amr"
    WEBM = "webm"


class AudioQuality(str, Enum):
    """Audio quality levels."""
    LOW = "low"          # 8kHz, mono, compressed
    STANDARD = "standard" # 16kHz, mono, good quality
    HIGH = "high"        # 22kHz+, stereo, high quality
    LOSSLESS = "lossless" # Uncompressed, studio quality


class VoiceGender(str, Enum):
    """Voice gender options for TTS."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class SpeechRecognitionRequest(BaseModel):
    """Speech recognition (ASR) request schema."""

    audio_file: FileUpload = Field(description="Audio file to transcribe")

    # Audio metadata
    audio_format: AudioFormat = Field(description="Audio format")
    sample_rate: int | None = Field(None, description="Audio sample rate (Hz)")
    duration_seconds: float | None = Field(None, ge=0, description="Audio duration")
    channels: int = Field(default=1, ge=1, le=2, description="Number of audio channels")

    # Recognition parameters
    language_hint: LanguageCode | None = Field(None, description="Expected language hint")
    model_preference: str | None = Field(None, description="ASR model preference")
    enable_automatic_punctuation: bool = Field(default=True, description="Add punctuation")
    enable_word_time_offsets: bool = Field(default=False, description="Include word timestamps")
    enable_speaker_diarization: bool = Field(default=False, description="Identify different speakers")
    max_speakers: int | None = Field(None, ge=2, le=10, description="Maximum number of speakers")

    # Content filtering
    profanity_filter: bool = Field(default=False, description="Filter profanity")

    # Context and domain
    context_phrases: list[str] = Field(default_factory=list, description="Context phrases for better accuracy")
    domain_hint: str | None = Field(None, description="Domain hint (medical, legal, etc.)")

    @validator('context_phrases')
    def validate_context_phrases(cls, v):
        """Limit context phrases."""
        if len(v) > 20:
            raise ValueError("Maximum 20 context phrases allowed")
        return v


class SpeechSegment(BaseModel):
    """Speech segment with timing information."""

    text: str = Field(description="Segment text")
    start_time_seconds: float = Field(ge=0, description="Segment start time")
    end_time_seconds: float = Field(ge=0, description="Segment end time")
    confidence_score: float = Field(ge=0, le=1, description="Segment confidence")
    speaker_id: int | None = Field(None, description="Speaker identifier (if diarization enabled)")

    @validator('end_time_seconds')
    def validate_end_time(cls, v, values):
        """Validate end time is after start time."""
        if 'start_time_seconds' in values and v <= values['start_time_seconds']:
            raise ValueError("End time must be after start time")
        return v


class WordTimestamp(BaseModel):
    """Word-level timestamp information."""

    word: str = Field(description="Word text")
    start_time_seconds: float = Field(ge=0, description="Word start time")
    end_time_seconds: float = Field(ge=0, description="Word end time")
    confidence_score: float = Field(ge=0, le=1, description="Word confidence")


class SpeechRecognitionResponse(BaseResponse):
    """Speech recognition response."""

    # Primary result
    transcript: str = Field(description="Full transcript text")
    detected_language: LanguageCode = Field(description="Detected language")
    overall_confidence: float = Field(ge=0, le=1, description="Overall confidence score")

    # Detailed results
    segments: list[SpeechSegment] = Field(default_factory=list, description="Speech segments")
    word_timestamps: list[WordTimestamp] = Field(default_factory=list, description="Word-level timing")

    # Audio analysis
    audio_quality_score: float = Field(ge=0, le=1, description="Audio quality assessment")
    noise_level: str | None = Field(None, description="Background noise level")
    speech_rate_wpm: int | None = Field(None, description="Speech rate (words per minute)")

    # Processing metadata
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    model_used: str = Field(description="ASR model used")

    # Speakers (if diarization enabled)
    speaker_count: int | None = Field(None, description="Number of detected speakers")
    speaker_labels: list[str] = Field(default_factory=list, description="Speaker labels")


class TextToSpeechRequest(BaseModel):
    """Text-to-speech (TTS) request schema."""

    text: str = Field(min_length=1, max_length=5000, description="Text to synthesize")
    language: LanguageCode = Field(description="Speech language")

    # Voice parameters
    voice_gender: VoiceGender = Field(default=VoiceGender.FEMALE, description="Voice gender")
    voice_name: str | None = Field(None, description="Specific voice name")

    # Audio parameters
    output_format: AudioFormat = Field(default=AudioFormat.MP3, description="Output audio format")
    sample_rate: int = Field(default=22050, description="Output sample rate")
    audio_quality: AudioQuality = Field(default=AudioQuality.STANDARD, description="Audio quality")

    # Speech parameters
    speaking_rate: float = Field(default=1.0, ge=0.25, le=4.0, description="Speaking rate multiplier")
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0, description="Pitch adjustment (semitones)")
    volume_gain_db: float = Field(default=0.0, ge=-96.0, le=16.0, description="Volume gain (dB)")

    # SSML support
    use_ssml: bool = Field(default=False, description="Text contains SSML markup")

    # Effects
    add_echo: bool = Field(default=False, description="Add echo effect")
    add_reverb: bool = Field(default=False, description="Add reverb effect")


class TextToSpeechResponse(BaseResponse):
    """Text-to-speech response."""

    audio_url: str = Field(description="URL to generated audio file")
    audio_format: AudioFormat = Field(description="Generated audio format")
    duration_seconds: float = Field(ge=0, description="Audio duration")
    file_size_bytes: int = Field(ge=0, description="Audio file size")

    # Generation metadata
    voice_used: str = Field(description="Voice used for synthesis")
    processing_time_ms: int = Field(description="Processing time")
    character_count: int = Field(description="Characters synthesized")

    # Audio properties
    sample_rate: int = Field(description="Audio sample rate")
    bit_rate: int | None = Field(None, description="Audio bit rate")
    channels: int = Field(description="Number of audio channels")


class VoiceConversationRequest(BaseModel):
    """Voice conversation processing request."""

    audio_file: FileUpload = Field(description="Voice input audio")
    conversation_id: UUID | None = Field(None, description="Conversation context ID")

    # Processing preferences
    source_language: LanguageCode | None = Field(None, description="Input language (auto-detect if None)")
    target_language: LanguageCode = Field(default=LanguageCode.EN, description="Response language")

    # Voice response preferences
    generate_voice_response: bool = Field(default=True, description="Generate TTS response")
    voice_gender: VoiceGender = Field(default=VoiceGender.FEMALE, description="Response voice gender")
    response_language: LanguageCode | None = Field(None, description="Voice response language")

    # Context
    location: dict[str, float] | None = Field(None, description="User location context")
    user_preferences: dict[str, Any] | None = Field(None, description="User preferences")


class VoiceConversationResponse(BaseResponse):
    """Voice conversation response."""

    conversation_id: UUID = Field(description="Conversation identifier")

    # Input processing results
    user_speech: str = Field(description="Transcribed user speech")
    detected_language: LanguageCode = Field(description="Detected input language")
    speech_confidence: float = Field(ge=0, le=1, description="Transcription confidence")

    # Response generation
    ai_response: str = Field(description="AI text response")
    translated_response: str | None = Field(None, description="Translated response")

    # Voice output (if requested)
    voice_response_url: str | None = Field(None, description="Voice response audio URL")
    voice_duration_seconds: float | None = Field(None, description="Voice response duration")

    # Processing metadata
    total_processing_time_ms: int = Field(description="Total processing time")
    asr_time_ms: int = Field(description="ASR processing time")
    ai_time_ms: int = Field(description="AI response generation time")
    tts_time_ms: int | None = Field(None, description="TTS processing time")

    # Context
    conversation_context: dict[str, Any] = Field(description="Updated conversation context")


class AudioTranslationRequest(BaseModel):
    """Audio translation request schema."""

    audio_file: FileUpload = Field(description="Audio file to translate")
    source_language: LanguageCode | None = Field(None, description="Source language")
    target_language: LanguageCode = Field(description="Target language")

    # Output preferences
    include_original_transcript: bool = Field(default=True, description="Include original transcript")
    include_translated_audio: bool = Field(default=True, description="Generate translated audio")
    voice_gender: VoiceGender = Field(default=VoiceGender.FEMALE, description="Translated voice gender")

    # Processing options
    preserve_speaking_style: bool = Field(default=False, description="Preserve original speaking style")
    normalize_volume: bool = Field(default=True, description="Normalize audio volume")


class AudioTranslationResponse(BaseResponse):
    """Audio translation response."""

    # Original content
    original_transcript: str = Field(description="Original transcribed text")
    detected_language: LanguageCode = Field(description="Detected source language")

    # Translation
    translated_text: str = Field(description="Translated text")
    target_language: LanguageCode = Field(description="Target language")
    translation_confidence: float = Field(ge=0, le=1, description="Translation confidence")

    # Translated audio (if requested)
    translated_audio_url: str | None = Field(None, description="Translated audio URL")
    translated_duration_seconds: float | None = Field(None, description="Translated audio duration")

    # Metadata
    processing_time_ms: int = Field(description="Total processing time")
    original_duration_seconds: float = Field(description="Original audio duration")


class VoiceCloneRequest(BaseModel):
    """Voice cloning/voice synthesis request (future feature)."""

    reference_audio: list[FileUpload] = Field(description="Reference audio samples")
    target_text: str = Field(description="Text to synthesize with cloned voice")
    language: LanguageCode = Field(description="Speech language")

    # Cloning parameters
    quality_level: str = Field(default="standard", description="Cloning quality level")
    training_duration: int | None = Field(None, description="Minimum training duration")

    @validator('reference_audio')
    def validate_reference_audio(cls, v):
        """Validate reference audio samples."""
        if len(v) < 1:
            raise ValueError("At least one reference audio sample required")
        if len(v) > 10:
            raise ValueError("Maximum 10 reference audio samples allowed")
        return v


class VoiceCloneResponse(BaseResponse):
    """Voice cloning response."""

    cloned_audio_url: str = Field(description="Cloned voice audio URL")
    similarity_score: float = Field(ge=0, le=1, description="Voice similarity score")
    quality_score: float = Field(ge=0, le=1, description="Audio quality score")
    processing_time_ms: int = Field(description="Processing time")

    # Model info
    voice_model_id: str = Field(description="Generated voice model identifier")
    training_duration_seconds: float = Field(description="Training audio duration used")


class AudioAnalysisRequest(BaseModel):
    """Audio analysis request for quality, content, etc."""

    audio_file: FileUpload = Field(description="Audio file to analyze")

    # Analysis types
    analyze_quality: bool = Field(default=True, description="Analyze audio quality")
    analyze_content: bool = Field(default=True, description="Analyze speech content")
    analyze_emotion: bool = Field(default=False, description="Analyze emotional tone")
    analyze_language: bool = Field(default=True, description="Detect language")
    analyze_speakers: bool = Field(default=False, description="Analyze speaker characteristics")


class AudioAnalysisResponse(BaseResponse):
    """Audio analysis response."""

    # Technical analysis
    duration_seconds: float = Field(description="Audio duration")
    sample_rate: int = Field(description="Sample rate")
    bit_rate: int | None = Field(None, description="Bit rate")
    file_format: str = Field(description="Audio format")

    # Quality metrics
    signal_to_noise_ratio: float | None = Field(None, description="SNR in dB")
    audio_quality_score: float = Field(ge=0, le=1, description="Overall quality score")
    clipping_detected: bool = Field(description="Audio clipping detected")
    silence_percentage: float = Field(ge=0, le=100, description="Percentage of silence")

    # Content analysis
    speech_detected: bool = Field(description="Speech content detected")
    detected_language: LanguageCode | None = Field(None, description="Detected language")
    estimated_words: int | None = Field(None, description="Estimated word count")
    speaking_rate_wpm: int | None = Field(None, description="Speaking rate")

    # Emotional analysis (if enabled)
    emotional_tone: str | None = Field(None, description="Detected emotional tone")
    emotion_confidence: float | None = Field(None, description="Emotion detection confidence")

    # Speaker analysis (if enabled)
    speaker_count: int | None = Field(None, description="Number of speakers detected")
    primary_speaker_gender: VoiceGender | None = Field(None, description="Primary speaker gender")

    # Processing metadata
    processing_time_ms: int = Field(description="Analysis processing time")
