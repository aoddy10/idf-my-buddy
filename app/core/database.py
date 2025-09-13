"""Database configuration and session management for My Buddy application.

This module provides async database session management using SQLAlchemy
and SQLModel for the PostgreSQL database connection.
"""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Global variables for lazy initialization
engine = None
AsyncSessionLocal = None


def get_engine():
    """Get or create database engine."""
    global engine
    if engine is None:
        settings = get_settings()
        engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,  # Log SQL queries in debug mode
            pool_size=5,  # Default pool size
            max_overflow=10,  # Default max overflow
            pool_timeout=30,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
    return engine


def get_session_local():
    """Get or create session maker."""
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        AsyncSessionLocal = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
    return AsyncSessionLocal

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for dependency injection."""
    session_local = get_session_local()
    async with session_local() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_db_and_tables():
    """Create database tables."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Import all models to ensure they are registered with SQLModel
            from app.models.entities.user import User
            from app.models.entities.session import Session
            from app.models.entities.travel_context import TravelContext
            
            # Create all tables
            await conn.run_sync(SQLModel.metadata.create_all)
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def drop_db_and_tables():
    """Drop all database tables (for testing/development)."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
            logger.info("Database tables dropped successfully")
            
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session.
    
    Yields:
        AsyncSession: Database session for executing queries
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context():
    """Get database session as async context manager.
    
    Usage:
        async with get_session_context() as session:
            # Use session here
            pass
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db_connection():
    """Close database connection pool."""
    try:
        await engine.dispose()
        logger.info("Database connection pool closed")
    except Exception as e:
        logger.error(f"Failed to close database connection: {e}")


# Health check function
async def check_db_health() -> bool:
    """Check database connection health.
    
    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
        async with AsyncSessionLocal() as session:
            # Simple query to check connection
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
