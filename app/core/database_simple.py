"""Simple database configuration for testing."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel

# Global engine - will be None until initialized
_engine = None
_session_maker = None


def init_database(database_url: str = "sqlite+aiosqlite:///./test.db", debug: bool = False):
    """Initialize database engine and session maker."""
    global _engine, _session_maker
    
    _engine = create_async_engine(
        database_url,
        echo=debug,
        pool_pre_ping=True,
    )
    
    _session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for dependency injection."""
    if _session_maker is None:
        # Initialize with default settings for testing
        init_database()
    
    async with _session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_db_and_tables():
    """Create database tables - stub implementation."""
    # This is a stub - would normally create tables
    pass


async def close_db_connection():
    """Close database connection."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def check_db_health() -> bool:
    """Check if database is accessible."""
    try:
        if _session_maker is None:
            init_database()
            
        async with _session_maker() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
            return True
    except Exception:
        return False
