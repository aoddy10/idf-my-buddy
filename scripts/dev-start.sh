#!/bin/bash
# Development environment startup script for IDF My Buddy Travel Assistant

set -e

echo "🚀 Starting IDF My Buddy Travel Assistant - Development Environment"

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

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env file with your API keys before continuing.${NC}"
    echo "Press any key to continue..."
    read -n 1 -s
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads temp backups test-results

# Build and start development services
echo "🏗️  Building and starting development services..."
docker-compose --profile dev up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
else
    echo -e "${RED}❌ PostgreSQL is not ready${NC}"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is ready${NC}"
else
    echo -e "${RED}❌ Redis is not ready${NC}"
fi

# Check Application
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Application is ready${NC}"
else
    echo -e "${YELLOW}⚠️  Application may still be starting...${NC}"
fi

# Run database migrations
echo "🗄️  Running database migrations..."
docker-compose --profile migrate up migrate

echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📋 Available services:"
echo "   • Application: http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • PostgreSQL: localhost:5432"
echo "   • Redis: localhost:6379"
echo ""
echo "📝 Useful commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart app: docker-compose restart app-dev"
echo "   • Run tests: ./scripts/run-tests.sh"
echo ""
