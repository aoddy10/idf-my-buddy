# Docker Quick Start Guide

This guide helps you get IDF My Buddy Travel Assistant running with Docker in different environments.

## Prerequisites

-   Docker Desktop (latest version)
-   Docker Compose v2+
-   Git (for cloning the repository)

## Quick Start Commands

### Development Environment

```bash
# 1. Clone and setup
git clone <repository-url>
cd idf-my-buddy

# 2. Start development environment
./scripts/dev-start.sh

# 3. Access the application
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Production Environment

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your production values

# 2. Deploy to production
./scripts/prod-deploy.sh

# 3. Access the application
# HTTPS: https://localhost
# Docs: https://localhost/docs
```

### Testing

```bash
# Run all tests
./scripts/run-tests.sh

# Run specific test categories
docker-compose --profile test run --rm app-test poetry run pytest tests/unit/
docker-compose --profile test run --rm app-test poetry run pytest -m "api"
```

## Environment Profiles

### Development Profile (`--profile dev`)

-   Hot reload enabled
-   Debug logging
-   Source code mounted as volume
-   Direct database access
-   Development-friendly configuration

Services:

-   `app-dev`: FastAPI application with auto-reload
-   `postgres`: PostgreSQL database
-   `redis`: Redis cache

### Production Profile (`--profile prod`)

-   Optimized Docker images
-   Multiple app replicas
-   Nginx reverse proxy with SSL
-   Production logging level
-   Security headers enabled
-   Resource limits configured

Services:

-   `app-prod`: Production FastAPI application (2 replicas)
-   `postgres`: PostgreSQL database
-   `redis`: Redis cache
-   `nginx`: Reverse proxy with SSL

### Testing Profile (`--profile test`)

-   Isolated test environment
-   In-memory test databases
-   Test-specific configuration
-   Coverage reporting
-   Fast test execution

Services:

-   `app-test`: Test runner container
-   `postgres-test`: Temporary PostgreSQL (tmpfs)
-   `redis-test`: Temporary Redis (no persistence)

### Monitoring Profile (`--profile monitoring`)

-   Prometheus metrics collection
-   Grafana dashboards
-   Application monitoring
-   System metrics
-   Performance tracking

Services:

-   `prometheus`: Metrics collection
-   `grafana`: Visualization dashboards

## Common Commands

### Service Management

```bash
# Start specific profile
docker-compose --profile dev up -d
docker-compose --profile prod up -d

# View logs
docker-compose logs -f app-dev
docker-compose logs -f app-prod

# Stop services
docker-compose --profile dev down
docker-compose --profile prod down

# Restart application
docker-compose restart app-dev
docker-compose restart app-prod

# Scale production app
docker-compose --profile prod up -d --scale app-prod=3
```

### Database Operations

```bash
# Run migrations
docker-compose --profile migrate up migrate

# Backup database
./scripts/backup-db.sh

# Access PostgreSQL
docker-compose exec postgres psql -U postgres -d idf_buddy

# Reset database (development only)
docker-compose down
docker volume rm idf-my-buddy_postgres_data
docker-compose --profile dev up -d
```

### Development Workflow

```bash
# Start development environment
./scripts/dev-start.sh

# View application logs
docker-compose logs -f app-dev

# Run tests during development
./scripts/run-tests.sh

# Reset everything (clean start)
docker-compose down --volumes --remove-orphans
./scripts/dev-start.sh
```

### Production Deployment

```bash
# Initial production deployment
./scripts/prod-deploy.sh

# Update production deployment
docker-compose --profile prod build --no-cache
docker-compose --profile prod up -d

# Check production health
curl -k https://localhost/health

# Monitor production logs
docker-compose --profile prod logs -f
```

## Environment Variables

Key environment variables (see `.env.example` for complete list):

### Required for Production

-   `SECRET_KEY`: JWT secret (generate strong key)
-   `GOOGLE_MAPS_API_KEY`: Google Maps integration
-   `OPENWEATHER_API_KEY`: Weather services
-   `OPENAI_API_KEY`: AI processing
-   `AZURE_SPEECH_KEY`: Text-to-speech services
-   `AZURE_SPEECH_REGION`: Azure region

### Database

-   `POSTGRES_DB`: Database name (default: idf_buddy)
-   `POSTGRES_USER`: Database user (default: postgres)
-   `POSTGRES_PASSWORD`: Database password

### Application

-   `ENVIRONMENT`: development/production
-   `LOG_LEVEL`: DEBUG/INFO/WARNING/ERROR
-   `CORS_ORIGINS`: Allowed origins array

## SSL Configuration

### Development

-   Uses HTTP on port 8000
-   No SSL required

### Production

-   Uses HTTPS on port 443 (via Nginx)
-   Self-signed certificates generated automatically
-   Replace with proper SSL certificates:

```bash
# Place your certificates in nginx/ssl/
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem

# Restart nginx
docker-compose restart nginx
```

## Monitoring

Enable monitoring with Prometheus and Grafana:

```bash
# Start with monitoring
docker-compose --profile prod --profile monitoring up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Troubleshooting

### Application Won't Start

```bash
# Check service status
docker-compose ps

# Check logs for errors
docker-compose logs app-dev
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker-compose exec postgres pg_isready

# Check database logs
docker-compose logs postgres

# Reset database connection
docker-compose restart postgres
sleep 10
docker-compose restart app-dev
```

### Port Conflicts

```bash
# Check what's using ports
lsof -i :8000
lsof -i :5432
lsof -i :6379

# Change ports in .env
APP_PORT=8001
POSTGRES_PORT=5433
REDIS_PORT=6380
```

### Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh

# Fix Docker socket permissions (Linux)
sudo chmod 666 /var/run/docker.sock
```

### Clean Reset

```bash
# Complete cleanup
docker-compose down --volumes --remove-orphans
docker system prune -f
docker volume prune -f

# Remove uploaded files
rm -rf uploads/* temp/*

# Restart fresh
./scripts/dev-start.sh
```

## Performance Optimization

### Production Settings

```bash
# Optimize for production
export ENVIRONMENT=production
export LOG_LEVEL=WARNING

# Scale application instances
docker-compose --profile prod up -d --scale app-prod=4

# Monitor resource usage
docker stats
```

### Development Performance

```bash
# Limit resource usage in development
# Add to docker-compose.override.yml:
services:
  app-dev:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

## Security Checklist

### Before Production Deployment

-   [ ] Set strong `SECRET_KEY`
-   [ ] Use real SSL certificates
-   [ ] Secure API keys in environment
-   [ ] Enable proper CORS origins
-   [ ] Review nginx security headers
-   [ ] Set up database backups
-   [ ] Enable monitoring/logging
-   [ ] Test disaster recovery

### Regular Maintenance

-   [ ] Update Docker images regularly
-   [ ] Rotate API keys periodically
-   [ ] Monitor security logs
-   [ ] Backup database weekly
-   [ ] Test restore procedures
-   [ ] Update SSL certificates before expiry
