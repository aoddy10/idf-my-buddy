# Multi-stage Dockerfile for IDF My Buddy Travel Assistant
# Optimized for production deployment with development support

# Stage 1: Base Python environment
FROM python:3.11-slim as python-base

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    git \
    libmagic1 \
    libmagic-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Stage 2: Development dependencies
FROM python-base as development

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install all dependencies (including dev)
RUN poetry install --with dev && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

# Create non-root user for development
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose development port
EXPOSE 8000

# Development command with auto-reload
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production dependencies
FROM python-base as production-deps

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install only production dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Stage 4: Production runtime
FROM python-base as production

# Copy virtual environment from production-deps stage
COPY --from=production-deps /app/.venv /app/.venv

# Ensure we use venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY app/ /app/app/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/
COPY README.md /app/

# Create uploads and temp directories
RUN mkdir -p /app/uploads /app/temp

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser

# Expose application port
EXPOSE 8000

# Production command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 5: Testing environment
FROM development as testing

# Install additional test dependencies if any
RUN poetry install --with dev,test 2>/dev/null || poetry install --with dev

# Set test environment
ENV ENVIRONMENT=test

# Run tests by default
CMD ["poetry", "run", "pytest", "-v", "--cov=app", "--cov-report=html", "--cov-report=term"]
