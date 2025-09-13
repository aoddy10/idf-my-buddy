# Makefile for IDF My Buddy Travel Assistant
# Provides common development and deployment tasks

.PHONY: help install install-dev test lint format clean docker-dev docker-prod migrate backup docs

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	@echo "Installing production dependencies..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install --only=main; \
	else \
		pip install -r requirements.txt; \
	fi

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	@./scripts/install-deps.sh --dev

install-system: ## Install system dependencies
	@echo "Installing system dependencies..."
	@./scripts/install-deps.sh --skip-models

# Development targets
dev: ## Start development server
	@echo "Starting development server..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload; \
	else \
		source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload; \
	fi

shell: ## Activate development shell
	@if command -v poetry >/dev/null 2>&1; then \
		poetry shell; \
	else \
		echo "Activate with: source .venv/bin/activate"; \
	fi

# Testing targets
test: ## Run all tests
	@echo "Running tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest; \
	else \
		source .venv/bin/activate && pytest; \
	fi

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/unit/; \
	else \
		source .venv/bin/activate && pytest tests/unit/; \
	fi

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/integration/; \
	else \
		source .venv/bin/activate && pytest tests/integration/; \
	fi

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest --cov=app --cov-report=html --cov-report=term; \
	else \
		source .venv/bin/activate && pytest --cov=app --cov-report=html --cov-report=term; \
	fi

test-docker: ## Run tests in Docker
	@echo "Running tests in Docker..."
	@./scripts/run-tests.sh

# Code quality targets
lint: ## Run linting
	@echo "Running linters..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run ruff check app/ tests/; \
		poetry run mypy app/; \
	else \
		source .venv/bin/activate && ruff check app/ tests/ && mypy app/; \
	fi

format: ## Format code
	@echo "Formatting code..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run black app/ tests/; \
		poetry run isort app/ tests/; \
	else \
		source .venv/bin/activate && black app/ tests/ && isort app/ tests/; \
	fi

security: ## Run security checks
	@echo "Running security checks..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run bandit -r app/; \
		poetry run safety check; \
	else \
		source .venv/bin/activate && bandit -r app/ && safety check; \
	fi

# Docker targets
docker-build: ## Build Docker images
	@echo "Building Docker images..."
	@docker-compose build

docker-dev: ## Start development environment with Docker
	@echo "Starting development environment with Docker..."
	@./scripts/dev-start.sh

docker-prod: ## Deploy production environment with Docker
	@echo "Deploying production environment with Docker..."
	@./scripts/prod-deploy.sh

docker-test: ## Run tests in Docker environment
	@echo "Running tests in Docker environment..."
	@docker-compose --profile test up --build --abort-on-container-exit

docker-clean: ## Clean Docker resources
	@echo "Cleaning Docker resources..."
	@docker-compose down --volumes --remove-orphans
	@docker system prune -f

# Database targets
migrate: ## Run database migrations
	@echo "Running database migrations..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run alembic upgrade head; \
	else \
		source .venv/bin/activate && alembic upgrade head; \
	fi

migrate-create: ## Create new migration (usage: make migrate-create MESSAGE="description")
	@echo "Creating new migration..."
	@if [ -z "$(MESSAGE)" ]; then echo "Usage: make migrate-create MESSAGE=\"description\""; exit 1; fi
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run alembic revision --autogenerate -m "$(MESSAGE)"; \
	else \
		source .venv/bin/activate && alembic revision --autogenerate -m "$(MESSAGE)"; \
	fi

migrate-history: ## Show migration history
	@echo "Migration history..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run alembic history; \
	else \
		source .venv/bin/activate && alembic history; \
	fi

backup: ## Create database backup
	@echo "Creating database backup..."
	@./scripts/backup-db.sh

# Documentation targets
docs: ## Generate documentation
	@echo "Generating documentation..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run mkdocs build; \
	else \
		source .venv/bin/activate && mkdocs build; \
	fi

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run mkdocs serve; \
	else \
		source .venv/bin/activate && mkdocs serve; \
	fi

# Utility targets
clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	@rm -rf .mypy_cache/ .ruff_cache/

clean-all: clean docker-clean ## Clean everything including Docker
	@echo "Cleaning all artifacts..."
	@rm -rf .venv/ venv/ node_modules/

env-setup: ## Setup environment file
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file. Please edit it with your values."; fi

pre-commit: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pre-commit install; \
	else \
		source .venv/bin/activate && pre-commit install; \
	fi

# CI/CD targets
ci-install: ## Install dependencies for CI
	@echo "Installing CI dependencies..."
	@pip install -r requirements-dev.txt

ci-test: ## Run tests for CI
	@echo "Running CI tests..."
	@pytest --cov=app --cov-report=xml --junit-xml=test-results.xml

ci-lint: ## Run linting for CI
	@echo "Running CI linting..."
	@ruff check app/ tests/
	@mypy app/
	@bandit -r app/ -f json -o security-report.json || true

# Load testing targets
load-test: ## Run load tests
	@echo "Running load tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run locust -f tests/load/locustfile.py --host=http://localhost:8000; \
	else \
		source .venv/bin/activate && locust -f tests/load/locustfile.py --host=http://localhost:8000; \
	fi

# Quick setup target
setup: env-setup install-dev migrate ## Complete development setup
	@echo "Development setup complete!"
	@echo "Run 'make dev' to start the development server"
