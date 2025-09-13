#!/bin/bash
# Database backup script for IDF My Buddy Travel Assistant

set -e

echo "💾 Creating database backup for IDF My Buddy Travel Assistant"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create backup directory if it doesn't exist
mkdir -p backups

# Set backup filename with timestamp
BACKUP_FILE="backups/idf_buddy_backup_$(date +%Y%m%d_%H%M%S).sql"

# Check if PostgreSQL container is running
if ! docker-compose ps postgres | grep -q "Up"; then
    echo -e "${RED}❌ PostgreSQL container is not running${NC}"
    echo "Start the database with: docker-compose up -d postgres"
    exit 1
fi

echo "📊 Creating backup: $BACKUP_FILE"

# Create backup using docker-compose
docker-compose exec -T postgres pg_dump -U postgres -d idf_buddy > "$BACKUP_FILE"

# Check if backup was created successfully
if [ -f "$BACKUP_FILE" ] && [ -s "$BACKUP_FILE" ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo -e "${GREEN}✅ Backup created successfully: $BACKUP_FILE ($BACKUP_SIZE)${NC}"
    
    # Create a compressed version
    COMPRESSED_FILE="${BACKUP_FILE}.gz"
    gzip -c "$BACKUP_FILE" > "$COMPRESSED_FILE"
    COMPRESSED_SIZE=$(du -h "$COMPRESSED_FILE" | cut -f1)
    echo -e "${GREEN}✅ Compressed backup created: $COMPRESSED_FILE ($COMPRESSED_SIZE)${NC}"
    
    # Remove uncompressed version to save space
    rm "$BACKUP_FILE"
    
else
    echo -e "${RED}❌ Backup creation failed${NC}"
    exit 1
fi

# Clean up old backups (keep last 7 days)
echo "🧹 Cleaning up old backups..."
find backups/ -name "idf_buddy_backup_*.sql.gz" -mtime +7 -delete
REMAINING=$(find backups/ -name "idf_buddy_backup_*.sql.gz" | wc -l)
echo -e "${GREEN}✅ Cleanup complete. $REMAINING backup(s) remaining${NC}"

echo ""
echo "💾 Backup completed successfully!"
echo "   File: $COMPRESSED_FILE"
echo ""
echo "🔄 To restore this backup:"
echo "   1. Stop the application: docker-compose down"
echo "   2. Start only PostgreSQL: docker-compose up -d postgres"
echo "   3. Restore: zcat $COMPRESSED_FILE | docker-compose exec -T postgres psql -U postgres -d idf_buddy"
echo "   4. Start application: docker-compose up -d"
echo ""
