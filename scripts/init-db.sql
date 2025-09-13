-- Database initialization script for IDF My Buddy Travel Assistant
-- This script is executed when PostgreSQL container starts for the first time

-- Create application database if it doesn't exist
CREATE DATABASE idf_buddy;

-- Create test database for testing
CREATE DATABASE idf_buddy_test;

-- Create application user with appropriate permissions
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'idf_buddy_user') THEN

      CREATE ROLE idf_buddy_user LOGIN PASSWORD 'idf_buddy_password';
   END IF;
END
$do$;

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON DATABASE idf_buddy TO idf_buddy_user;
GRANT ALL PRIVILEGES ON DATABASE idf_buddy_test TO idf_buddy_user;

-- Connect to application database and set up schema
\c idf_buddy;

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create application schema if needed
CREATE SCHEMA IF NOT EXISTS app;

-- Grant schema permissions
GRANT ALL ON SCHEMA app TO idf_buddy_user;
GRANT ALL ON SCHEMA public TO idf_buddy_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO idf_buddy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO idf_buddy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON SEQUENCES TO idf_buddy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO idf_buddy_user;

-- Connect to test database and apply same setup
\c idf_buddy_test;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE SCHEMA IF NOT EXISTS app;

GRANT ALL ON SCHEMA app TO idf_buddy_user;
GRANT ALL ON SCHEMA public TO idf_buddy_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO idf_buddy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO idf_buddy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON SEQUENCES TO idf_buddy_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO idf_buddy_user;

-- Create indexes that will be useful for the application
-- These will be created by SQLModel/Alembic, but we can prepare the database

-- Switch back to main database
\c idf_buddy;

-- Log successful initialization
DO $$ 
BEGIN 
    RAISE NOTICE 'IDF My Buddy database initialization completed successfully';
END $$;
