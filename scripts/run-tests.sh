#!/bin/bash
# Test runner script for IDF My Buddy Travel Assistant

set -e

echo "🧪 Running tests for IDF My Buddy Travel Assistant"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Create test results directory
mkdir -p test-results

# Stop any existing test containers
echo "🧹 Cleaning up existing test containers..."
docker-compose --profile test down --remove-orphans

# Start test environment
echo "🏗️  Starting test environment..."
docker-compose --profile test up --build -d postgres-test redis-test

# Wait for test services to be ready
echo "⏳ Waiting for test services..."
sleep 5

# Run tests
echo "🔬 Running test suite..."

# Run different test categories
echo "📊 Running unit tests..."
docker-compose --profile test run --rm app-test poetry run pytest tests/unit/ -v --tb=short

echo "🔗 Running integration tests..."
docker-compose --profile test run --rm app-test poetry run pytest tests/integration/ -v --tb=short

echo "🌐 Running API tests..."
docker-compose --profile test run --rm app-test poetry run pytest -m "api" -v --tb=short

echo "🤖 Running AI service tests..."
docker-compose --profile test run --rm app-test poetry run pytest -m "ai" -v --tb=short

echo "🗄️  Running database tests..."
docker-compose --profile test run --rm app-test poetry run pytest -m "database" -v --tb=short

# Run full test suite with coverage
echo "📈 Running full test suite with coverage..."
docker-compose --profile test run --rm app-test \
    poetry run pytest \
    --cov=app \
    --cov-report=html:/app/test-results/coverage \
    --cov-report=term \
    --cov-report=xml:/app/test-results/coverage.xml \
    --junit-xml=/app/test-results/junit.xml \
    -v

# Generate test report
if [ -f test-results/junit.xml ]; then
    echo -e "${GREEN}✅ Test results saved to test-results/junit.xml${NC}"
fi

if [ -d test-results/coverage ]; then
    echo -e "${GREEN}✅ Coverage report saved to test-results/coverage/index.html${NC}"
fi

# Cleanup test environment
echo "🧹 Cleaning up test environment..."
docker-compose --profile test down

echo ""
echo "🎉 Test run completed!"
echo ""
echo "📊 Test artifacts:"
echo "   • JUnit XML: test-results/junit.xml"
echo "   • Coverage HTML: test-results/coverage/index.html"
echo "   • Coverage XML: test-results/coverage.xml"
echo ""

# Check if all tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed. Check the output above.${NC}"
    exit 1
fi
