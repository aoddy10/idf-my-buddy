# Dependency Management Guide

This document explains the dependency management setup for IDF My Buddy Travel Assistant.

## Overview

The project supports multiple dependency management approaches to accommodate different deployment environments and developer preferences:

-   **Poetry** (Recommended): Modern Python dependency management with `pyproject.toml`
-   **pip**: Traditional Python package management with `requirements.txt` files
-   **Docker**: Containerized dependency management
-   **Setup.py**: Legacy compatibility for older systems

## Files Structure

```
├── pyproject.toml          # Poetry configuration (primary)
├── poetry.toml             # Poetry settings
├── requirements.txt        # Production dependencies for pip
├── requirements-dev.txt    # Development dependencies for pip
├── setup.py               # Legacy setuptools configuration
├── Makefile              # Development task automation
└── scripts/
    └── install-deps.sh   # Automated installation script
```

## Dependency Categories

### Production Dependencies (`requirements.txt`)

Core dependencies needed for running the application in production:

#### Web Framework & API

-   **FastAPI**: Modern, fast web framework for building APIs
-   **Uvicorn**: ASGI server for FastAPI applications
-   **Pydantic**: Data validation and serialization

#### Database & ORM

-   **SQLModel**: Modern SQL databases with Python types
-   **Alembic**: Database migration tool
-   **AsyncPG**: Async PostgreSQL driver
-   **psycopg2-binary**: PostgreSQL adapter

#### Authentication & Security

-   **python-jose**: JWT token handling
-   **passlib**: Password hashing utilities
-   **bcrypt**: Secure password hashing

#### AI & Machine Learning

-   **OpenAI**: GPT models and API client
-   **Whisper**: Speech-to-text processing
-   **Transformers**: Hugging Face model library
-   **PyTorch**: Deep learning framework
-   **sentence-transformers**: Semantic text embeddings

#### Computer Vision & OCR

-   **OpenCV**: Computer vision library
-   **Tesseract**: OCR text recognition
-   **Pillow**: Image processing
-   **pdf2image**: PDF to image conversion

#### Audio Processing

-   **librosa**: Audio analysis and processing
-   **soundfile**: Audio file I/O
-   **pydub**: Audio manipulation

#### External Services

-   **Google Maps**: Location and mapping services
-   **httpx**: Async HTTP client
-   **requests**: HTTP client library

### Development Dependencies (`requirements-dev.txt`)

Additional tools for development, testing, and code quality:

#### Testing Framework

-   **pytest**: Testing framework
-   **pytest-asyncio**: Async test support
-   **pytest-cov**: Coverage reporting
-   **factory-boy**: Test data factories
-   **faker**: Fake data generation

#### Code Quality

-   **ruff**: Fast Python linter
-   **black**: Code formatter
-   **isort**: Import sorting
-   **mypy**: Static type checking
-   **bandit**: Security vulnerability scanner

#### Development Tools

-   **IPython**: Enhanced Python shell
-   **Jupyter**: Notebook environment
-   **pre-commit**: Git hooks for code quality
-   **watchdog**: File system monitoring

## Installation Methods

### Method 1: Poetry (Recommended)

Poetry provides the best dependency management experience with automatic virtual environment handling and lock files.

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install production dependencies
poetry install --only=main

# Install with development dependencies
poetry install --with=dev,test,docs

# Run commands in Poetry environment
poetry run uvicorn app.main:app --reload
poetry run pytest

# Activate Poetry shell
poetry shell
```

### Method 2: Automated Script

The installation script handles both Poetry and pip setups automatically:

```bash
# Production installation
./scripts/install-deps.sh

# Development installation
./scripts/install-deps.sh --dev

# Skip system dependencies
./scripts/install-deps.sh --skip-system

# Skip AI model downloads
./scripts/install-deps.sh --skip-models
```

### Method 3: pip with Virtual Environment

Traditional Python package management:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Method 4: Docker

Containerized dependency management (no local Python required):

```bash
# Development environment
./scripts/dev-start.sh

# Production deployment
./scripts/prod-deploy.sh

# Testing environment
./scripts/run-tests.sh
```

### Method 5: Makefile Tasks

Simplified command interface:

```bash
# Complete development setup
make setup

# Install dependencies
make install-dev

# Run development server
make dev

# Run tests
make test

# Code formatting
make format

# View all available tasks
make help
```

## System Dependencies

The application requires several system-level packages:

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    ffmpeg \
    libmagic1 \
    libpq-dev \
    build-essential
```

### macOS

```bash
brew install tesseract tesseract-lang ffmpeg libmagic postgresql
```

### Windows

-   Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
-   Install FFmpeg from: https://ffmpeg.org/download.html
-   Install PostgreSQL from: https://www.postgresql.org/download/windows/

## AI Model Downloads

The application uses several AI models that need to be downloaded:

### Automatic Download (Recommended)

```bash
# Included in installation script
./scripts/install-deps.sh

# Or with Poetry
poetry run python -c "
import nltk
import spacy
nltk.download('punkt')
spacy.cli.download('en_core_web_sm')
"
```

### Manual Download

```bash
# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# spaCy models
python -m spacy download en_core_web_sm

# Whisper models (downloaded automatically on first use)
# Models: tiny, base, small, medium, large
```

## Dependency Updates

### With Poetry

```bash
# Update all dependencies
poetry update

# Update specific dependency
poetry update fastapi

# Add new dependency
poetry add new-package

# Add development dependency
poetry add --group dev new-dev-package
```

### With pip

```bash
# Generate updated requirements
pip freeze > requirements.txt

# Install from updated requirements
pip install -r requirements.txt --upgrade
```

## Lock Files and Reproducibility

### Poetry Lock

-   `poetry.lock`: Exact version lock file (commit to repository)
-   Ensures reproducible builds across environments
-   Updated with `poetry lock` command

### pip Requirements

-   Pin exact versions in requirements.txt for production
-   Use `pip freeze` to generate exact versions
-   Consider using `pip-tools` for better dependency resolution

## Dependency Security

### Security Scanning

```bash
# With Poetry
poetry run safety check

# With pip
pip install safety
safety check

# Bandit security linting
bandit -r app/
```

### Regular Updates

-   Monitor security advisories
-   Update dependencies regularly
-   Test thoroughly after updates
-   Use automated dependency update tools (Dependabot, etc.)

## Troubleshooting

### Common Issues

#### Poetry Installation Issues

```bash
# Clear Poetry cache
poetry cache clear --all pypi

# Reinstall dependencies
rm poetry.lock
poetry install
```

#### Virtual Environment Issues

```bash
# Remove and recreate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### System Dependency Issues

```bash
# Update system packages
sudo apt-get update  # Linux
brew update           # macOS

# Reinstall problematic packages
sudo apt-get install --reinstall tesseract-ocr
```

#### Model Download Issues

```bash
# Manual model installation
python -m spacy download en_core_web_sm --user
python -c "import nltk; nltk.download('punkt', download_dir='~/nltk_data')"
```

### Performance Optimization

#### Production Installations

-   Use `--no-dev` flags to skip development dependencies
-   Consider using `pip install --no-cache-dir` to reduce disk usage
-   Use multi-stage Docker builds for smaller images

#### Development Speed

-   Use Poetry for faster dependency resolution
-   Cache Docker layers for faster rebuilds
-   Use `pip install -e .` for editable installs during development

## Best Practices

### Dependency Management

1. **Pin versions** in production requirements
2. **Regular updates** but test thoroughly
3. **Separate environments** for dev/test/prod
4. **Security scanning** before deployment
5. **Documentation** of critical dependencies

### Environment Setup

1. **Virtual environments** always
2. **Environment variables** for configuration
3. **Requirements files** up to date
4. **Lock files** committed to repository
5. **System dependencies** documented

### CI/CD Integration

1. **Cached dependencies** for faster builds
2. **Security scanning** in pipeline
3. **Multi-environment testing**
4. **Dependency vulnerability checks**
5. **Automated updates** with testing
