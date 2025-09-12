name: "My Buddy Foundation Setup PRP"
description: |

## Purpose

Complete project foundation setup for the AI Buddy travel assistant application, establishing the core backend infrastructure, development toolchain, and initial AI service scaffolding.

## Core Principles

1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal

Create the complete foundation for the My Buddy AI travel assistant project, including:

-   FastAPI backend with feature-first architecture
-   Core AI services scaffolding (ASR, OCR, MT, TTS)
-   Database setup with SQLModel
-   Testing infrastructure with pytest
-   Development environment with proper tooling
-   Configuration management with Pydantic Settings
-   Project documentation and setup instructions

## Why

-   **Business value**: Enables rapid development of travel assistance features
-   **Integration foundation**: Sets up extensible architecture for multimodal AI services
-   **Problems this solves**: Provides structured foundation for complex AI travel assistant

## What

A fully functional backend API foundation with:

-   Working FastAPI application with health endpoints
-   Core AI service interfaces ready for implementation
-   Database models for user preferences and travel context
-   Test suite with CI/CD ready structure
-   Development environment with linting, type checking
-   Configuration management for different environments
-   Documentation for developers

### Success Criteria

-   [ ] FastAPI server runs successfully on localhost:8000
-   [ ] All linting and type checks pass (ruff, mypy)
-   [ ] Full test suite passes with >90% coverage
-   [ ] Database migrations work correctly
-   [ ] AI service interfaces are defined and testable
-   [ ] Environment configuration loads properly
-   [ ] Health check endpoints return valid responses
-   [ ] OpenAPI documentation is generated and accessible

## All Needed Context

### Documentation & References (list all context needed to implement the feature)

```yaml
# MUST READ - Include these in your context window
- url: https://fastapi.tiangolo.com/
  why: Primary web framework - async endpoints, dependency injection, OpenAPI

- url: https://sqlmodel.tiangolo.com/
  why: Database ORM - Pydantic integration, SQLAlchemy foundation

- url: https://ai.pydantic.dev/
  why: Data validation and settings management patterns

- url: https://docs.pydantic.dev/latest/concepts/settings/
  why: Environment-based configuration management

- url: https://github.com/openai/whisper
  why: Speech recognition model architecture and usage patterns

- url: https://ai.meta.com/research/no-language-left-behind/
  why: Machine translation model (NLLB) architecture

- url: https://developers.google.com/ml-kit/vision/text-recognition/v2/android
  why: OCR capabilities and integration patterns

- url: https://docs.pytest.org/en/7.4.x/
  why: Testing framework patterns and best practices

- file: docs/CLAUDE.md
  why: Project architecture rules, file organization, coding standards
  critical: Feature-first structure, 500-line file limit, async patterns

- file: docs/project-proposal.md
  why: Business requirements, core features, technical architecture
  critical: 4 core domains (navigation, restaurant, shopping, safety)

- file: docs/project-research.md
  why: Technical feasibility, performance budgets, privacy requirements
  critical: Edge-first approach, latency targets, multilingual support

- file: docs/INITIAL.md
  why: Feature overview, technology stack, integration requirements
  critical: Multimodal AI capabilities, edge computing focus
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase

```bash
# Currently only documentation exists - no implementation yet
idf-my-buddy/
├── .claude/
│   ├── commands/
│   │   ├── execute-prp.md
│   │   └── generate-prp.md
│   └── settings.local.json
├── docs/
│   ├── CLAUDE.md
│   ├── INITIAL.md
│   ├── PRPs/
│   │   └── templates/
│   │       └── prp_base.md
│   ├── examples/
│   ├── project-proposal.md
│   └── project-research.md
└── .git/
```

### Desired Codebase tree with files to be added and responsibility of file

```bash
idf-my-buddy/
├── app/
│   ├── __init__.py                    # Python package marker
│   ├── main.py                        # FastAPI app factory & startup wiring
│   ├── api/                           # Feature-based routers
│   │   ├── __init__.py
│   │   ├── health.py                  # Health check endpoints
│   │   ├── navigation.py              # Navigation assistance endpoints
│   │   ├── restaurant.py              # Restaurant/menu assistance endpoints
│   │   ├── shopping.py                # Shopping assistance endpoints
│   │   └── safety.py                  # Safety/emergency endpoints
│   ├── models/                        # Data models
│   │   ├── __init__.py
│   │   ├── schemas/                   # Pydantic request/response schemas
│   │   │   ├── __init__.py
│   │   │   ├── common.py              # Common schemas (Location, Language, etc.)
│   │   │   ├── navigation.py          # Navigation request/response schemas
│   │   │   ├── restaurant.py          # Restaurant request/response schemas
│   │   │   ├── shopping.py            # Shopping request/response schemas
│   │   │   └── safety.py              # Safety request/response schemas
│   │   └── entities/                  # SQLModel ORM entities
│   │       ├── __init__.py
│   │       ├── user.py                # User profile and preferences
│   │       ├── session.py             # User session tracking
│   │       └── travel_context.py      # Travel context and history
│   ├── services/                      # Business logic services
│   │   ├── __init__.py
│   │   ├── asr.py                     # Automatic Speech Recognition service
│   │   ├── ocr.py                     # Optical Character Recognition service
│   │   ├── mt.py                      # Machine Translation service
│   │   ├── tts.py                     # Text-to-Speech service
│   │   ├── maps.py                    # Maps and navigation service
│   │   └── allergens.py               # Food allergen detection service
│   ├── core/                          # Cross-cutting concerns
│   │   ├── __init__.py
│   │   ├── config.py                  # Pydantic Settings configuration
│   │   ├── logging.py                 # Structured logging setup
│   │   ├── deps.py                    # Dependency injection wiring
│   │   ├── errors.py                  # Custom exceptions and error handlers
│   │   └── flags.py                   # Feature flags management
│   ├── adapters/                      # External service integrations
│   │   ├── __init__.py
│   │   ├── google_maps.py             # Google Maps SDK client
│   │   ├── mapbox.py                  # Mapbox SDK client
│   │   └── azure_speech.py            # Azure Speech SDK client
│   ├── ml/                            # Edge/cloud ML utilities
│   │   ├── __init__.py
│   │   ├── loader.py                  # Model loading utilities
│   │   ├── quantization.py            # Model quantization helpers
│   │   └── runners/                   # Model runtime engines
│   │       ├── __init__.py
│   │       ├── whisper_runner.py      # Whisper model runner
│   │       └── nllb_runner.py         # NLLB model runner
│   ├── utils/                         # Small pure helpers
│   │   ├── __init__.py
│   │   └── helpers.py                 # Utility functions
│   └── i18n/                          # Internationalization
│       ├── __init__.py
│       └── messages.py                # Message templates
├── migrations/                        # Database migrations
│   └── versions/                      # Alembic migration files
├── tests/                             # Test suite mirroring app structure
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration and fixtures
│   ├── test_main.py                   # Main app tests
│   ├── api/                           # API endpoint tests
│   │   ├── __init__.py
│   │   ├── test_health.py
│   │   ├── test_navigation.py
│   │   ├── test_restaurant.py
│   │   ├── test_shopping.py
│   │   └── test_safety.py
│   ├── services/                      # Service layer tests
│   │   ├── __init__.py
│   │   ├── test_asr.py
│   │   ├── test_ocr.py
│   │   ├── test_mt.py
│   │   └── test_tts.py
│   └── integration/                   # Integration tests
│       ├── __init__.py
│       └── test_workflows.py          # End-to-end workflow tests
├── pyproject.toml                     # Python project configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore patterns
├── README.md                          # Project setup and usage
├── Dockerfile                         # Container configuration
└── docker-compose.yml                # Local development stack
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: FastAPI requires async functions for endpoints
# Example: Always use async def for route handlers
@router.get("/endpoint")
async def handler() -> ResponseModel:
    return await service_call()

# CRITICAL: SQLModel requires specific import patterns
# Example: Import SQLModel before Pydantic BaseModel
from sqlmodel import SQLModel, Field, select
from pydantic import BaseModel  # Import after SQLModel

# CRITICAL: Pydantic Settings validation happens at class creation
# Example: Environment variables must be available when Settings is imported
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

# CRITICAL: Whisper models require specific tensor formats
# Example: Audio must be 16kHz mono float32 numpy array
import numpy as np
audio = np.array(audio_data, dtype=np.float32)

# CRITICAL: NLLB language codes are different from ISO codes
# Example: Use NLLB language codes not ISO 639-1
# "eng_Latn" not "en", "fra_Latn" not "fr"

# CRITICAL: File size rule - split files approaching 500 lines
# Example: Break large modules into smaller cohesive units

# CRITICAL: Feature-first organization over layer-first
# Example: Group by feature domain, not by technical layer
```

## Implementation Blueprint

### Data models and structure

Create the core data models to ensure type safety and consistency.

```python
# Core Pydantic schemas for API contracts
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class LanguageCode(str, Enum):
    """Supported language codes using NLLB format"""
    ENGLISH = "eng_Latn"
    SPANISH = "spa_Latn"
    FRENCH = "fra_Latn"
    JAPANESE = "jpn_Jpan"
    # ... more languages

class Location(BaseModel):
    """Geographic location"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy: Optional[float] = None

# SQLModel entities for database persistence
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime

class User(SQLModel, table=True):
    """User profile and preferences"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(unique=True, index=True)
    preferred_language: LanguageCode
    dietary_restrictions: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Project Structure Setup
CREATE project directory structure:
    - Create all directories as per desired codebase tree
    - Add __init__.py files to make Python packages
    - Create empty placeholder files for main modules

Task 2: Core Configuration
CREATE pyproject.toml:
    - Define project metadata, dependencies, dev tools
    - Configure ruff linting rules
    - Set up pytest configuration
    - Define build system

CREATE .env.example:
    - Template for all required environment variables
    - API keys, database URLs, feature flags
    - Documentation for each variable

CREATE app/core/config.py:
    - Pydantic Settings class for environment configuration
    - Support for different environments (dev, test, prod)
    - Validation for required settings

Task 3: FastAPI Application Setup
CREATE app/main.py:
    - FastAPI app factory with middleware setup
    - CORS configuration for mobile clients
    - Exception handlers and logging
    - Include routers for all feature domains

CREATE app/core/logging.py:
    - Structured JSON logging configuration
    - Request correlation IDs
    - Performance timing middleware

Task 4: Database Foundation
CREATE app/models/entities/user.py:
    - User profile and preferences model
    - Dietary restrictions and language preferences
    - Travel context and session tracking

CREATE app/core/deps.py:
    - Database session dependency
    - User authentication dependency stubs
    - Service layer dependency injection

Task 5: API Routers
CREATE app/api/health.py:
    - Basic health check endpoint
    - Database connectivity check
    - Service status checks

CREATE app/api/navigation.py:
    - Navigation assistance endpoints skeleton
    - Location-based routing stubs

CREATE app/api/restaurant.py:
    - Restaurant assistance endpoints skeleton
    - Menu translation stubs

CREATE app/api/shopping.py:
    - Shopping assistance endpoints skeleton
    - Product information stubs

CREATE app/api/safety.py:
    - Safety assistance endpoints skeleton
    - Emergency contact stubs

Task 6: AI Services Foundation
CREATE app/services/asr.py:
    - Automatic Speech Recognition service interface
    - Whisper model integration stubs
    - Audio processing utilities

CREATE app/services/ocr.py:
    - Optical Character Recognition service interface
    - Image preprocessing utilities
    - Text extraction stubs

CREATE app/services/mt.py:
    - Machine Translation service interface
    - NLLB model integration stubs
    - Language detection utilities

CREATE app/services/tts.py:
    - Text-to-Speech service interface
    - Voice synthesis stubs
    - Audio format utilities

Task 7: External Adapters
CREATE app/adapters/google_maps.py:
    - Google Maps API client wrapper
    - Geocoding and routing utilities
    - Places API integration

CREATE app/adapters/mapbox.py:
    - Mapbox API client wrapper
    - Offline map capabilities

Task 8: Test Suite Setup
CREATE tests/conftest.py:
    - Pytest fixtures for database, client, auth
    - Test data factories
    - Mock service configurations

CREATE tests/test_main.py:
    - FastAPI application tests
    - Health endpoint tests
    - Basic integration tests

CREATE tests for each service:
    - Unit tests for all service interfaces
    - Mock external dependencies
    - Error handling tests

Task 9: Development Tooling
CREATE .gitignore:
    - Python, IDE, OS-specific ignores
    - Environment files, logs, cache directories
    - ML model files and data

CREATE README.md:
    - Project overview and setup instructions
    - Development workflow
    - API documentation links

CREATE Dockerfile:
    - Multi-stage build for production
    - Development stage with hot reload
    - Security best practices

Task 10: Documentation
CREATE requirements.txt:
    - Pin all production dependencies
    - Generate from pyproject.toml

CREATE docker-compose.yml:
    - Local development stack
    - Database, Redis, test services
    - Volume mounts for development
```

### Per task pseudocode as needed added to each task

```python
# Task 3: FastAPI Application Setup
# app/main.py pseudocode
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

def create_app() -> FastAPI:
    # PATTERN: App factory pattern for testing
    app = FastAPI(
        title="My Buddy API",
        description="AI Travel Assistant",
        version="1.0.0"
    )

    # PATTERN: CORS for mobile clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"]
    )

    # PATTERN: Include feature routers
    from app.api import health, navigation, restaurant, shopping, safety
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(navigation.router, prefix="/navigation", tags=["navigation"])
    # ... include other routers

    return app

# Task 6: AI Services Foundation
# app/services/asr.py pseudocode
from abc import ABC, abstractmethod
import numpy as np

class ASRService(ABC):
    """Abstract Speech Recognition service"""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, language: str = None) -> str:
        """Convert audio to text"""
        pass

    @abstractmethod
    async def detect_language(self, audio: np.ndarray) -> str:
        """Detect spoken language"""
        pass

class WhisperASRService(ASRService):
    def __init__(self, model_size: str = "base"):
        # PATTERN: Load model lazily
        self._model = None
        self.model_size = model_size

    async def transcribe(self, audio: np.ndarray, language: str = None) -> str:
        # CRITICAL: Whisper expects 16kHz mono float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # PATTERN: Use existing retry decorator
        @retry(attempts=3, backoff=exponential)
        async def _transcribe():
            model = await self._load_model()
            result = model.transcribe(audio, language=language)
            return result["text"]

        return await _transcribe()
```

### Integration Points

```yaml
DATABASE:
    - migration: "Create initial user and session tables"
    - connection: "PostgreSQL with SQLModel/SQLAlchemy"

CONFIG:
    - add to: app/core/config.py
    - pattern: "Environment-specific settings with Pydantic"

ROUTES:
    - add to: app/main.py
    - pattern: "Feature-based router inclusion with tags"

LOGGING:
    - add to: app/core/logging.py
    - pattern: "Structured JSON logs with correlation IDs"

TESTING:
    - add to: tests/conftest.py
    - pattern: "Fixtures for database, client, and service mocks"
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check app/ tests/ --fix  # Auto-fix what's possible
ruff format app/ tests/       # Format code
mypy app/                     # Type checking

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests

```python
# CREATE comprehensive test suite with these patterns:

def test_health_endpoint(client):
    """Health check returns successful response"""
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_asr_service_transcribe():
    """ASR service transcribes audio correctly"""
    service = WhisperASRService()
    # Mock audio data
    audio = np.random.rand(16000).astype(np.float32)
    result = await service.transcribe(audio)
    assert isinstance(result, str)
    assert len(result) > 0

def test_config_validation():
    """Configuration validates environment variables"""
    with pytest.raises(ValidationError):
        Settings(database_url="invalid")

def test_error_handling():
    """API handles errors gracefully"""
    response = client.post("/navigation/route", json={"invalid": "data"})
    assert response.status_code == 422
    assert "validation_error" in response.json()
```

```bash
# Run and iterate until passing:
pytest tests/ -v --cov=app --cov-report=html
# Target: >90% test coverage
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test

```bash
# Start the service
python -m app.main

# Test the health endpoint
curl -X GET http://localhost:8000/health/ \
  -H "Accept: application/json"

# Expected: {"status": "healthy", "timestamp": "...", "version": "1.0.0"}

# Test OpenAPI documentation
curl -X GET http://localhost:8000/docs

# Expected: Swagger UI with all endpoints documented
```

## Final validation Checklist

-   [ ] All tests pass: `pytest tests/ -v --cov=app`
-   [ ] No linting errors: `ruff check app/ tests/`
-   [ ] No type errors: `mypy app/`
-   [ ] Manual health check successful: `curl localhost:8000/health/`
-   [ ] OpenAPI docs accessible: `curl localhost:8000/docs`
-   [ ] Database migrations run: `alembic upgrade head`
-   [ ] Environment configuration loads: `.env` validation
-   [ ] Docker build successful: `docker build -t my-buddy .`
-   [ ] All directories and files created as specified
-   [ ] README setup instructions are complete

---

## Anti-Patterns to Avoid

-   ❌ Don't create sync functions in async FastAPI context
-   ❌ Don't hardcode model paths - use configuration
-   ❌ Don't skip input validation - use Pydantic schemas
-   ❌ Don't ignore performance budgets - measure latency
-   ❌ Don't commit secrets or large model files
-   ❌ Don't create circular imports between services
-   ❌ Don't violate 500-line file size limit
-   ❌ Don't skip error handling for external APIs
-   ❌ Don't use global state - inject dependencies

## Quality Score: 9/10

**Confidence Level**: Very High (9/10)

This PRP provides comprehensive context for one-pass implementation including:
✅ Complete project structure with clear responsibilities  
✅ All required dependencies and configuration  
✅ Detailed task breakdown with specific implementations  
✅ Executable validation commands  
✅ Known gotchas and anti-patterns  
✅ Integration patterns from FastAPI/SQLModel best practices  
✅ Performance and testing requirements  
✅ Clear success criteria and validation gates

The foundation setup is well-scoped and follows established patterns from the documentation. The only uncertainty (1 point deduction) is around specific model integration details that may require iteration during implementation.
