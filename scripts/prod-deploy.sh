#!/bin/bash
# Production deployment script for IDF My Buddy Travel Assistant

set -e

echo "🚀 Deploying IDF My Buddy Travel Assistant - Production Environment"

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

# Check if .env file exists with production settings
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found. Please create it with production settings.${NC}"
    exit 1
fi

# Verify required environment variables
echo "🔍 Verifying production configuration..."

required_vars=(
    "SECRET_KEY"
    "GOOGLE_MAPS_API_KEY"
    "OPENWEATHER_API_KEY"
    "OPENAI_API_KEY"
    "AZURE_SPEECH_KEY"
    "AZURE_SPEECH_REGION"
)

source .env

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" = "your-${var,,}-here" ] || [ "${!var}" = "your_${var,,}_here" ]; then
        echo -e "${RED}❌ ${var} is not set or contains placeholder value${NC}"
        echo "Please set this variable in your .env file"
        exit 1
    fi
done

echo -e "${GREEN}✅ Configuration verified${NC}"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads temp backups nginx/ssl monitoring/prometheus_data monitoring/grafana_data

# Check SSL certificates
if [ ! -f nginx/ssl/cert.pem ] || [ ! -f nginx/ssl/key.pem ]; then
    echo -e "${YELLOW}⚠️  SSL certificates not found. Generating self-signed certificates...${NC}"
    mkdir -p nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo -e "${YELLOW}⚠️  Using self-signed certificates. Replace with proper SSL certificates for production.${NC}"
fi

# Build production images
echo "🏗️  Building production images..."
docker-compose --profile prod build --no-cache

# Stop any existing services
echo "🛑 Stopping existing services..."
docker-compose down

# Start production services
echo "🚀 Starting production services..."
docker-compose --profile prod up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Run database migrations
echo "🗄️  Running database migrations..."
docker-compose --profile migrate up migrate

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
else
    echo -e "${RED}❌ PostgreSQL is not ready${NC}"
    docker-compose logs postgres
    exit 1
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is ready${NC}"
else
    echo -e "${RED}❌ Redis is not ready${NC}"
    docker-compose logs redis
    exit 1
fi

# Check Application (through nginx)
if curl -f -k https://localhost/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Application is ready${NC}"
else
    echo -e "${RED}❌ Application health check failed${NC}"
    docker-compose logs app-prod
    docker-compose logs nginx
    exit 1
fi

echo ""
echo "🎉 Production deployment successful!"
echo ""
echo "📋 Available services:"
echo "   • Application: https://localhost"
echo "   • API Documentation: https://localhost/docs"
echo "   • Health Check: https://localhost/health"
if docker-compose ps | grep -q monitoring; then
    echo "   • Monitoring (Grafana): http://localhost:3000"
    echo "   • Metrics (Prometheus): http://localhost:9090"
fi
echo ""
echo "📝 Useful commands:"
echo "   • View logs: docker-compose --profile prod logs -f"
echo "   • Stop services: docker-compose --profile prod down"
echo "   • Backup database: ./scripts/backup-db.sh"
echo "   • Update deployment: ./scripts/update-production.sh"
echo ""
echo "🔒 Security reminders:"
echo "   • Replace self-signed SSL certificates with proper ones"
echo "   • Regularly update your API keys and secrets"
echo "   • Monitor application logs for security issues"
echo "   • Keep Docker images updated"
echo ""
