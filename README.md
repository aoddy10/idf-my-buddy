# My Buddy - AI Travel Assistant

Your Smart Travel Companion powered by AI.

## Overview

My Buddy is an AI-powered travel assistant that helps travelers navigate, find restaurants, shop, and stay safe while exploring new destinations. The application provides real-time assistance through voice interactions, image recognition, and multilingual support.

## Features

-   **Navigation**: Get directions, find places, and navigate unfamiliar areas
-   **Restaurant Assistant**: Translate menus, identify allergens, get recommendations
-   **Shopping Helper**: Product information, price comparisons, local shopping guides
-   **Safety Assistant**: Emergency contacts, safety tips, and real-time alerts

## Quick Start

### Prerequisites

-   Python 3.11+
-   PostgreSQL (or use Docker Compose for development)
-   Redis (or use Docker Compose for development)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd idf-my-buddy
```

2. Install dependencies:

```bash
pip install -e .
```

3. Set up environment:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run with Docker Compose (recommended for development):

```bash
docker-compose up -d
```

5. Or run locally:

```bash
uvicorn app.main:app --reload
```

### API Documentation

Once running, visit:

-   API docs: http://localhost:8000/docs
-   Health check: http://localhost:8000/health/

## Development

### Running Tests

```bash
pytest tests/ -v --cov=app
```

### Code Quality

```bash
ruff check app/ tests/ --fix
ruff format app/ tests/
mypy app/
```

## Architecture

The application follows a feature-first architecture with:

-   **FastAPI** for the REST API
-   **SQLModel** for database ORM
-   **Pydantic** for data validation
-   **Redis** for caching and sessions
-   **PostgreSQL** for persistent data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
