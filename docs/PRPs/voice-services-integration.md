name: "Voice Services Integration PRP"
description: |

## Purpose

Implement comprehensive voice interaction services for the My Buddy AI travel assistant, establishing the core speech-to-speech pipeline that enables natural conversation, real-time translation, and voice-guided navigation. This foundational feature enables all user interaction modalities and serves as the backbone for the multimodal AI experience.

## Core Principles

1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal

Create a complete voice services integration that provides:

-   High-quality speech recognition using Whisper models (edge + cloud)
-   Natural text-to-speech output with multilingual support
-   Voice activity detection and audio processing
-   Unified voice API for all travel assistant features
-   Performance-optimized edge-first approach with cloud fallback

## Why

-   **Business value**: Enables core travel assistant interactions through natural voice
-   **Integration foundation**: Provides voice interface for navigation, restaurant, shopping, safety
-   **Problems this solves**:
    -   Language barriers during travel (real-time voice translation)
    -   Hands-free interaction while navigating or carrying items
    -   Accessibility for users who prefer/need voice interaction
    -   Natural multimodal experience combining voice, camera, and location

## What

A fully functional voice services system with:

-   Working Whisper ASR with configurable model sizes and quality levels
-   TTS service supporting multiple languages and voice styles
-   Audio preprocessing (noise reduction, VAD, normalization)
-   Voice conversation flow management
-   Performance monitoring and automatic quality adjustment
-   Fallback mechanisms for poor network/device conditions
-   Integration endpoints for all feature domains

### Success Criteria

-   [ ] Voice recognition works with <2s latency for 5-second utterances
-   [ ] TTS produces natural speech in target languages (EN, ES, FR, DE, JA, KO, ZH)
-   [ ] Voice conversation flows handle interruptions and context switching
-   [ ] All linting and type checks pass (ruff, mypy)
-   [ ] Full test suite passes with >90% coverage
-   [ ] Performance budgets met: P50 < 1.8s, P95 < 2.2s for speech-to-speech
-   [ ] Offline mode works for common languages
-   [ ] Voice API integrates seamlessly with existing feature endpoints
-   [ ] Audio processing handles noisy environments (café, street)

## All Needed Context

### Documentation & References (list all context needed to implement the feature)

```yaml
# MUST READ - Include these in your context window
- url: https://github.com/openai/whisper
  why: Primary ASR model - architecture, usage patterns, model sizes
  critical: Model selection (tiny/base/small), quantization for mobile

- url: https://github.com/ggml-org/whisper.cpp
  why: C++ implementation for edge deployment, optimization patterns
  critical: Performance optimizations, memory management, mobile integration

- url: https://huggingface.co/openai/whisper-tiny
  why: Model specifications and performance characteristics
  critical: Size/accuracy tradeoffs, language support coverage

- url: https://ai.pydantic.dev/
  why: Data validation patterns for audio processing and API responses

- url: https://fastapi.tiangolo.com/advanced/background-tasks/
  why: Background processing for audio transcription and TTS generation

- url: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/
  why: Cloud TTS/ASR API patterns and fallback implementations

- url: https://cloud.google.com/text-to-speech/docs/
  why: Neural TTS voices and SSML markup for natural speech

- url: https://developer.apple.com/documentation/avfaudio/avspeechsynthesizer/
  why: System TTS integration patterns for offline capability

- file: docs/project-research.md
  why: Performance budgets, edge vs cloud strategy, privacy requirements
  critical: <2s latency targets, local-first policy, multilingual support

- file: app/services/whisper.py
  why: Existing ASR service implementation patterns
  critical: Service structure, error handling, model management

- file: app/services/tts.py
  why: Existing TTS service implementation patterns
  critical: Audio format handling, voice selection, caching

- file: app/core/config.py
  why: Configuration management patterns and environment variables
  critical: Model settings, API keys, feature flags

- file: docs/CLAUDE.md
  why: Project architecture rules, async patterns, error handling
  critical: 500-line limit, feature-first structure, testing requirements

- file: tests/test_simple.py
  why: Testing patterns and FastAPI integration testing
  critical: Test client usage, environment setup, assertion patterns
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase

```bash
./app/
├── core/
│   ├── config.py          # Settings and environment configuration
│   ├── errors.py          # Exception handling patterns
│   ├── logging.py         # Structured logging
│   └── deps.py            # Dependency injection
├── services/
│   ├── whisper.py         # Whisper ASR service (partial implementation)
│   ├── tts.py             # TTS service (partial implementation)
│   ├── nllb.py            # Machine translation service
│   └── ocr.py             # OCR service
├── api/
│   ├── health.py          # Health check endpoints
│   ├── navigation.py      # Navigation endpoints (stub)
│   ├── restaurant.py      # Restaurant endpoints (stub)
│   ├── shopping.py        # Shopping endpoints (stub)
│   └── safety.py          # Safety endpoints (stub)
├── schemas/
│   ├── voice.py           # Voice API schemas
│   ├── common.py          # Shared response models
│   └── [domain].py        # Feature-specific schemas
├── models/
│   └── entities/          # Database models
└── main.py                # FastAPI application factory

./tests/
├── test_simple.py         # Basic integration tests
├── conftest.py            # Test configuration and fixtures
└── [feature]/             # Feature-specific test modules
```

### Desired Codebase tree with files to be added and responsibility of file

```bash
./app/
├── services/
│   ├── whisper.py         # COMPLETE: Full ASR implementation with edge/cloud
│   ├── tts.py             # COMPLETE: Full TTS with neural voices and system fallback
│   ├── audio/             # NEW: Audio processing utilities
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # VAD, noise reduction, normalization
│   │   ├── formats.py        # Audio format conversion (wav, mp3, ogg)
│   │   └── streaming.py      # Real-time audio streaming handlers
│   └── voice_pipeline.py  # NEW: Unified voice conversation management
├── api/
│   └── v1/
│       └── voice.py       # NEW: Voice interaction endpoints
├── schemas/
│   └── voice.py           # ENHANCED: Complete voice API schemas
└── models/
    └── entities/
        └── voice_session.py  # NEW: Voice conversation state management

./tests/
├── services/
│   ├── test_whisper.py    # NEW: ASR service tests
│   ├── test_tts.py        # NEW: TTS service tests
│   └── test_voice_pipeline.py  # NEW: Integration tests
└── integration/
    └── test_voice_api.py   # NEW: End-to-end voice API tests
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: Whisper models require specific torch versions and CUDA setup
# Example: whisper.load_model() can fail silently if CUDA unavailable
# Solution: Graceful fallback to CPU with performance warnings

# CRITICAL: FastAPI file uploads have size limits and memory implications
# Example: Large audio files (>10MB) can cause memory issues
# Solution: Streaming upload with chunked processing

# CRITICAL: TTS audio generation is CPU intensive and blocking
# Example: Generating 30s of speech can take 5-10s on CPU
# Solution: Background tasks with WebSocket progress updates

# CRITICAL: Audio format compatibility across platforms
# Example: iOS prefers AAC, Android prefers MP3, web prefers OGG
# Solution: Dynamic format selection based on User-Agent headers

# CRITICAL: Whisper model loading is slow (5-15s for first load)
# Example: Cold start penalty affects user experience
# Solution: Model preloading in lifespan events with health checks

# CRITICAL: Voice Activity Detection (VAD) false positives in noisy environments
# Example: Background music triggers continuous transcription
# Solution: Adaptive VAD thresholds with environment classification

# CRITICAL: We use pydantic v2 - schema validation patterns differ from v1
# Example: Field validation and serialization syntax changes
# Solution: Follow existing patterns in app/schemas/

# CRITICAL: Async context managers required for audio processing
# Example: Audio streams must be properly closed to prevent resource leaks
# Solution: Always use async with statements for audio handles
```

## Implementation Blueprint

### Data models and structure

Create the core voice interaction data models ensuring type safety and real-time performance.

```python
# Voice Request/Response Models
class VoiceRequest(BaseModel):
    audio_data: bytes | str          # Raw audio or base64 encoded
    format: AudioFormat             # wav, mp3, ogg, webm
    language: str | None = None     # Auto-detect if None
    context: VoiceContext | None = None

class VoiceResponse(BaseModel):
    text: str                       # Transcribed text
    confidence: float               # 0.0 to 1.0
    language: str                   # Detected/specified language
    audio_url: str | None = None    # TTS response URL if requested
    processing_time: float          # Latency metrics

# Voice Session Models
class VoiceSession(BaseModel):
    session_id: str
    user_id: str | None = None
    conversation_history: list[VoiceExchange]
    current_context: TravelContext | None = None
    preferences: VoicePreferences

# Audio Processing Models
class AudioMetadata(BaseModel):
    duration: float                 # Seconds
    sample_rate: int               # Hz (16000 recommended for Whisper)
    channels: int                  # 1 (mono) or 2 (stereo)
    format: AudioFormat
    size_bytes: int
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Complete Whisper ASR Service Implementation
MODIFY app/services/whisper.py:
    - FIND pattern: "class WhisperService(LoggerMixin)"
    - COMPLETE _setup_service() method with model loading and device selection
    - IMPLEMENT transcribe_audio() with error handling and confidence scoring
    - ADD batch processing for multiple audio files
    - IMPLEMENT model switching based on quality/performance requirements
    - ADD comprehensive logging and performance metrics

Task 2: Complete TTS Service Implementation
MODIFY app/services/tts.py:
    - FIND pattern: "class TTSService(LoggerMixin)"
    - COMPLETE synthesize_text() with multi-language support
    - IMPLEMENT voice selection and SSML processing
    - ADD audio format conversion and optimization
    - IMPLEMENT caching for repeated phrases
    - ADD streaming TTS for long texts

Task 3: Create Audio Processing Utilities
CREATE app/services/audio/__init__.py:
    - EXPORT all audio processing functions and classes

CREATE app/services/audio/preprocessing.py:
    - IMPLEMENT voice_activity_detection() using librosa/webrtcvad
    - ADD noise_reduction() for noisy environments
    - IMPLEMENT audio_normalization() for consistent levels
    - ADD chunk_audio() for streaming processing

CREATE app/services/audio/formats.py:
    - IMPLEMENT convert_audio_format() with ffmpeg/pydub
    - ADD detect_audio_format() with magic number detection
    - IMPLEMENT compress_audio() for bandwidth optimization
    - ADD validate_audio_file() with format verification

CREATE app/services/audio/streaming.py:
    - IMPLEMENT WebSocket audio streaming handlers
    - ADD real-time transcription with partial results
    - IMPLEMENT audio buffering and chunking strategies

Task 4: Create Unified Voice Pipeline Service
CREATE app/services/voice_pipeline.py:
    - IMPLEMENT VoicePipeline class coordinating ASR + TTS + MT
    - ADD conversation state management with context retention
    - IMPLEMENT speech-to-speech translation pipeline
    - ADD performance monitoring and adaptive quality adjustment
    - IMPLEMENT fallback strategies for service failures

Task 5: Enhance Voice API Schemas
MODIFY app/schemas/voice.py:
    - ADD comprehensive request/response models from blueprint
    - IMPLEMENT validation for audio formats and parameters
    - ADD conversation flow schemas with context management
    - IMPLEMENT streaming response schemas for real-time updates

Task 6: Create Voice Session Entity Model
CREATE app/models/entities/voice_session.py:
    - IMPLEMENT VoiceSession database model with SQLModel
    - ADD relationship to User and TravelContext entities
    - IMPLEMENT conversation history storage with JSON fields
    - ADD session lifecycle management (creation, updates, cleanup)

Task 7: Create Voice API Endpoints
CREATE app/api/v1/voice.py:
    - IMPLEMENT /transcribe endpoint with file upload
    - ADD /synthesize endpoint for TTS generation
    - IMPLEMENT /conversation WebSocket for real-time interaction
    - ADD /voices endpoint to list available TTS voices
    - IMPLEMENT session management endpoints

Task 8: Integration with Main Application
MODIFY app/main.py:
    - FIND pattern: "# TODO: Load AI models"
    - ADD voice services initialization in lifespan startup
    - IMPLEMENT proper cleanup in lifespan shutdown
    - ADD voice router to application with proper middleware

Task 9: Update Configuration
MODIFY app/core/config.py:
    - ADD voice service configuration variables
    - IMPLEMENT WhisperSettings with model size options
    - ADD TTSSettings with voice preferences and API keys
    - IMPLEMENT audio processing settings with quality levels

Task 10: Comprehensive Testing Suite
CREATE tests/services/test_whisper.py:
    - TEST model loading and inference with sample audio files
    - ADD performance benchmarks for different model sizes
    - IMPLEMENT error handling tests (invalid audio, model failures)
    - ADD multilingual transcription accuracy tests

CREATE tests/services/test_tts.py:
    - TEST voice synthesis with multiple languages
    - ADD audio format validation tests
    - IMPLEMENT caching functionality tests
    - ADD streaming TTS tests with large texts

CREATE tests/services/test_voice_pipeline.py:
    - TEST end-to-end speech-to-speech pipeline
    - ADD conversation state management tests
    - IMPLEMENT fallback mechanism tests
    - ADD performance under load tests

CREATE tests/integration/test_voice_api.py:
    - TEST voice API endpoints with real audio data
    - ADD WebSocket conversation flow tests
    - IMPLEMENT file upload and streaming tests
    - ADD error response validation tests
```

### Per task pseudocode as needed added to each task

```python
# Task 1: Complete Whisper ASR Service
class WhisperService(LoggerMixin):
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: str | None = None,
        model_size: str = "base"
    ) -> TranscriptionResult:
        # PATTERN: Always validate input first (see existing services)
        validated_audio = await self._validate_audio(audio_data)

        # CRITICAL: Handle model loading with device optimization
        model = await self._get_or_load_model(model_size)

        # GOTCHA: Whisper expects 16kHz mono audio
        processed_audio = await self._preprocess_audio(validated_audio)

        # PATTERN: Use existing retry decorator for reliability
        @retry(attempts=3, exponential_backoff=True)
        async def _transcribe():
            # CRITICAL: Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, model.transcribe, processed_audio
            )
            return result

        transcription = await _transcribe()

        # PATTERN: Standardized response format (see other services)
        return self._format_transcription_result(transcription, language)

# Task 4: Voice Pipeline Integration
class VoicePipeline:
    async def process_voice_input(
        self,
        audio_data: bytes,
        target_language: str,
        session_context: VoiceSession
    ) -> VoiceResponse:
        # PATTERN: Multimodal service coordination
        async with self.performance_monitor.track("voice_pipeline"):
            # Step 1: Speech Recognition
            transcription = await self.whisper_service.transcribe_audio(
                audio_data,
                language=session_context.source_language
            )

            # Step 2: Translation (if needed)
            if transcription.language != target_language:
                translation = await self.nllb_service.translate(
                    transcription.text,
                    source_lang=transcription.language,
                    target_lang=target_language
                )
            else:
                translation = transcription.text

            # Step 3: Context-aware response generation
            response_text = await self._generate_contextual_response(
                translation, session_context
            )

            # Step 4: Text-to-Speech
            tts_audio = await self.tts_service.synthesize_text(
                response_text,
                language=target_language,
                voice=session_context.preferred_voice
            )

            return VoiceResponse(
                transcription=transcription.text,
                translation=translation,
                response_text=response_text,
                response_audio_url=tts_audio.url,
                processing_time=self.performance_monitor.get_elapsed()
            )

# Task 7: Voice API Endpoints
@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile,
    language: str | None = None,
    model_size: WhisperModelSize = WhisperModelSize.BASE,
    whisper_service: WhisperService = Depends(get_whisper_service)
) -> TranscriptionResponse:
    # PATTERN: File upload validation (see existing patterns)
    if not audio_file.content_type.startswith("audio/"):
        raise ValidationError("File must be audio format")

    # CRITICAL: Stream large files to avoid memory issues
    audio_data = await audio_file.read()
    if len(audio_data) > settings.max_audio_size:
        raise ValidationError(f"Audio file too large (>{settings.max_audio_size} bytes)")

    # PATTERN: Background task for long processing
    result = await whisper_service.transcribe_audio(
        audio_data=audio_data,
        language=language,
        model_size=model_size.value
    )

    return TranscriptionResponse(
        text=result.text,
        confidence=result.confidence,
        language=result.language,
        processing_time=result.processing_time
    )
```

### Integration Points

```yaml
DATABASE:
    - migration: "Add voice_sessions table with conversation history"
    - index: "CREATE INDEX idx_voice_session_user ON voice_sessions(user_id, created_at)"

CONFIG:
    - add to: app/core/config.py
    - pattern: "WHISPER_MODEL_SIZE = str(os.getenv('WHISPER_MODEL_SIZE', 'base'))"
    - pattern: "TTS_VOICE_PROVIDER = str(os.getenv('TTS_VOICE_PROVIDER', 'system'))"

ROUTES:
    - add to: app/main.py
    - pattern: "app.include_router(voice.router, prefix='/voice', tags=['voice'])"

LIFESPAN:
    - add to: app/main.py lifespan startup
    - pattern: "await voice_services.initialize_models()"
    - add to: app/main.py lifespan shutdown
    - pattern: "await voice_services.cleanup_resources()"

DEPENDENCIES:
    - add to: requirements.txt
    - items:
          [
              "openai-whisper>=20231117",
              "torch>=2.0.0",
              "torchaudio>=2.0.0",
              "librosa>=0.10.0",
              "webrtcvad>=2.0.10",
              "pydub>=0.25.1",
          ]
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check app/services/whisper.py app/services/tts.py --fix
ruff check app/services/audio/ app/api/v1/voice.py --fix
mypy app/services/whisper.py app/services/tts.py
mypy app/services/audio/ app/api/v1/voice.py

# Expected: No errors. If errors, READ the error message and fix systematically.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns

```python
# CREATE tests/services/test_whisper.py with comprehensive test cases:
async def test_transcribe_english_audio():
    """Test English transcription with high confidence."""
    service = WhisperService()
    # Use sample audio file from test fixtures
    with open("tests/fixtures/hello_english.wav", "rb") as f:
        audio_data = f.read()

    result = await service.transcribe_audio(audio_data, language="en")
    assert result.confidence > 0.8
    assert "hello" in result.text.lower()
    assert result.language == "en"

async def test_transcribe_multilingual():
    """Test automatic language detection."""
    service = WhisperService()
    # Test with Spanish audio
    with open("tests/fixtures/hola_spanish.wav", "rb") as f:
        audio_data = f.read()

    result = await service.transcribe_audio(audio_data, language=None)
    assert result.language == "es"
    assert result.confidence > 0.7

async def test_invalid_audio_format():
    """Test error handling for invalid audio."""
    service = WhisperService()
    with pytest.raises(ValidationError, match="Invalid audio format"):
        await service.transcribe_audio(b"not audio data")

# CREATE tests/services/test_tts.py:
async def test_synthesize_multilingual():
    """Test TTS generation in multiple languages."""
    service = TTSService()

    # Test English
    result = await service.synthesize_text("Hello world", language="en")
    assert result.audio_data is not None
    assert result.format in ["wav", "mp3"]

    # Test Spanish
    result = await service.synthesize_text("Hola mundo", language="es")
    assert result.audio_data is not None
    assert result.language == "es"

# CREATE tests/integration/test_voice_api.py:
async def test_voice_transcription_endpoint():
    """Test end-to-end voice transcription API."""
    # Use test client pattern from existing tests
    with open("tests/fixtures/sample_audio.wav", "rb") as f:
        files = {"audio_file": ("test.wav", f, "audio/wav")}
        response = client.post("/voice/transcribe", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "confidence" in data
    assert data["confidence"] > 0.5
```

```bash
# Run and iterate until passing:
uv run pytest tests/services/test_whisper.py -v
uv run pytest tests/services/test_tts.py -v
uv run pytest tests/integration/test_voice_api.py -v

# If failing: Read error, understand root cause, fix code, re-run
# NEVER mock core functionality - fix actual implementation issues
```

### Level 3: Integration Test

```bash
# Start the service
uv run python -m app.main --dev

# Test voice transcription endpoint
curl -X POST http://localhost:8000/voice/transcribe \
  -F "audio_file=@tests/fixtures/sample_english.wav" \
  -F "model_size=base"

# Expected: {"text": "transcribed text", "confidence": 0.95, "language": "en", ...}

# Test TTS endpoint
curl -X POST http://localhost:8000/voice/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test", "language": "en", "voice": "neutral"}'

# Expected: {"audio_url": "/audio/temp/xyz.wav", "format": "wav", ...}

# Test real-time conversation WebSocket
wscat -c ws://localhost:8000/voice/conversation
# Send: {"type": "audio", "data": "<base64-audio>", "session_id": "test"}
# Expected: {"type": "response", "text": "...", "audio_url": "..."}

# If error: Check logs at logs/app.log for stack trace and model loading status
```

## Final validation Checklist

-   [ ] All tests pass: `uv run pytest tests/ -v`
-   [ ] No linting errors: `uv run ruff check app/`
-   [ ] No type errors: `uv run mypy app/`
-   [ ] Voice transcription works: `curl -F "audio_file=@sample.wav" localhost:8000/voice/transcribe`
-   [ ] TTS generation works: `curl -d '{"text":"test"}' localhost:8000/voice/synthesize`
-   [ ] WebSocket conversation flows properly
-   [ ] Performance targets met: <2s speech-to-speech latency
-   [ ] Error cases handled gracefully (no audio, invalid format, model failures)
-   [ ] Logs are informative but not verbose
-   [ ] Audio processing handles noisy environments
-   [ ] Multilingual support working for target languages

---

## Anti-Patterns to Avoid

-   ❌ Don't load Whisper models synchronously in request handlers - preload in lifespan
-   ❌ Don't process large audio files in memory - use streaming/chunking
-   ❌ Don't ignore audio format compatibility - validate and convert as needed
-   ❌ Don't block async functions with heavy CPU tasks - use thread executors
-   ❌ Don't cache raw audio data - cache processed results with TTL
-   ❌ Don't hardcode model paths - use configuration with fallbacks
-   ❌ Don't suppress transcription errors - log and provide meaningful feedback
-   ❌ Don't assume network availability - implement robust offline capabilities

---

## Confidence Score: 9/10

**Reasoning**: This PRP provides comprehensive context including:
✅ Complete technical specifications with performance targets
✅ Detailed implementation blueprint with pseudocode  
✅ Existing codebase patterns and integration points
✅ Comprehensive validation loops with executable tests
✅ Known gotchas and anti-patterns from research
✅ Clear success criteria and quality gates
✅ Real-world audio processing considerations
✅ Multilingual and accessibility requirements

**Risk Mitigation**: The step-by-step approach with validation at each level ensures early detection of issues. Performance monitoring and fallback strategies address real-world deployment challenges.
