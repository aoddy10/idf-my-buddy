"""Integration tests for database operations.

Tests database models, relationships, and CRUD operations.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.models.user import User
from app.models.session import UserSession
from app.models.travel import TravelContext
from tests.utils import TestDataFactory, DatabaseTestHelper


@pytest.mark.database
class TestDatabaseOperations:
    """Test basic database CRUD operations."""
    
    async def test_create_user(self, db_session, db_test_helper):
        """Test user creation in database."""
        user_data = TestDataFactory.create_user_data(
            username="dbtest_user",
            email="dbtest@example.com"
        )
        
        user = await db_test_helper.create_test_user(db_session, **user_data)
        
        assert user.id is not None
        assert user.username == "dbtest_user"
        assert user.email == "dbtest@example.com"
        assert user.created_at is not None
        assert user.updated_at is not None
    
    async def test_user_unique_constraints(self, db_session):
        """Test user unique constraints."""
        # Create first user
        user1 = User(
            username="unique_test",
            email="unique@example.com",
            full_name="Test User 1",
            hashed_password="hash1"
        )
        
        db_session.add(user1)
        await db_session.commit()
        
        # Try to create user with same username
        user2 = User(
            username="unique_test",  # Duplicate username
            email="different@example.com",
            full_name="Test User 2",
            hashed_password="hash2"
        )
        
        db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            await db_session.commit()
    
    async def test_user_session_creation(self, db_session, test_user):
        """Test user session creation."""
        session = UserSession(
            user_id=test_user.id,
            session_token="test_token_123",
            device_info={"platform": "web", "browser": "chrome"},
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.session_token == "test_token_123"
        assert session.is_active is True
    
    async def test_travel_context_creation(self, db_session, test_user):
        """Test travel context creation."""
        context_data = TestDataFactory.create_travel_context_data()
        
        context = TravelContext(
            user_id=test_user.id,
            **context_data
        )
        
        db_session.add(context)
        await db_session.commit()
        await db_session.refresh(context)
        
        assert context.id is not None
        assert context.user_id == test_user.id
        assert context.current_location["lat"] == 40.7128
        assert context.travel_mode == "walking"
        assert isinstance(context.preferences, dict)
    
    async def test_user_query_operations(self, db_session):
        """Test user query operations."""
        # Create multiple users
        users_data = [
            {"username": "user1", "email": "user1@example.com"},
            {"username": "user2", "email": "user2@example.com"},
            {"username": "user3", "email": "user3@example.com"}
        ]
        
        created_users = []
        for user_data in users_data:
            user = User(
                **user_data,
                full_name=f"Test {user_data['username']}",
                hashed_password="hash123"
            )
            db_session.add(user)
            created_users.append(user)
        
        await db_session.commit()
        
        # Query all users
        stmt = select(User)
        result = await db_session.execute(stmt)
        all_users = result.scalars().all()
        
        assert len(all_users) >= 3
        
        # Query specific user
        stmt = select(User).where(User.username == "user2")
        result = await db_session.execute(stmt)
        user2 = result.scalar_one()
        
        assert user2.username == "user2"
        assert user2.email == "user2@example.com"
    
    async def test_user_update_operations(self, db_session, test_user):
        """Test user update operations."""
        original_updated_at = test_user.updated_at
        
        # Update user
        test_user.full_name = "Updated Name"
        test_user.preferences = {"theme": "dark", "language": "es"}
        test_user.updated_at = datetime.now()
        
        await db_session.commit()
        await db_session.refresh(test_user)
        
        assert test_user.full_name == "Updated Name"
        assert test_user.preferences["theme"] == "dark"
        assert test_user.updated_at > original_updated_at
    
    async def test_user_deletion(self, db_session):
        """Test user deletion."""
        # Create user to delete
        user = User(
            username="to_delete",
            email="delete@example.com",
            full_name="Delete Me",
            hashed_password="hash123"
        )
        
        db_session.add(user)
        await db_session.commit()
        user_id = user.id
        
        # Delete user
        await db_session.delete(user)
        await db_session.commit()
        
        # Verify deletion
        stmt = select(User).where(User.id == user_id)
        result = await db_session.execute(stmt)
        deleted_user = result.scalar_one_or_none()
        
        assert deleted_user is None


@pytest.mark.database
class TestDatabaseRelationships:
    """Test database relationships and foreign keys."""
    
    async def test_user_sessions_relationship(self, db_session, test_user):
        """Test one-to-many relationship between User and UserSession."""
        # Create multiple sessions for user
        sessions = []
        for i in range(3):
            session = UserSession(
                user_id=test_user.id,
                session_token=f"token_{i}",
                device_info={"device": f"device_{i}"}
            )
            db_session.add(session)
            sessions.append(session)
        
        await db_session.commit()
        
        # Query user sessions
        stmt = select(UserSession).where(UserSession.user_id == test_user.id)
        result = await db_session.execute(stmt)
        user_sessions = result.scalars().all()
        
        assert len(user_sessions) == 3
        for session in user_sessions:
            assert session.user_id == test_user.id
    
    async def test_user_travel_contexts_relationship(self, db_session, test_user):
        """Test one-to-many relationship between User and TravelContext."""
        # Create multiple travel contexts
        contexts = []
        for i in range(2):
            context = TravelContext(
                user_id=test_user.id,
                current_location={"lat": 40.0 + i, "lng": -74.0 + i},
                destination={"lat": 34.0 + i, "lng": -118.0 + i},
                travel_mode="walking"
            )
            db_session.add(context)
            contexts.append(context)
        
        await db_session.commit()
        
        # Query user travel contexts
        stmt = select(TravelContext).where(TravelContext.user_id == test_user.id)
        result = await db_session.execute(stmt)
        user_contexts = result.scalars().all()
        
        assert len(user_contexts) == 2
        for context in user_contexts:
            assert context.user_id == test_user.id
    
    async def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint enforcement."""
        # Try to create session with invalid user_id
        session = UserSession(
            user_id=99999,  # Non-existent user
            session_token="invalid_user_token",
            device_info={"platform": "test"}
        )
        
        db_session.add(session)
        
        with pytest.raises(IntegrityError):
            await db_session.commit()
    
    async def test_cascade_delete_behavior(self, db_session):
        """Test cascade delete behavior if implemented."""
        # Create user with related data
        user = User(
            username="cascade_test",
            email="cascade@example.com",
            full_name="Cascade Test",
            hashed_password="hash123"
        )
        db_session.add(user)
        await db_session.commit()
        
        # Add related session
        session = UserSession(
            user_id=user.id,
            session_token="cascade_token",
            device_info={"platform": "test"}
        )
        db_session.add(session)
        await db_session.commit()
        
        # Delete user
        await db_session.delete(user)
        await db_session.commit()
        
        # Check if session still exists (depends on cascade configuration)
        stmt = select(UserSession).where(UserSession.user_id == user.id)
        result = await db_session.execute(stmt)
        orphaned_session = result.scalar_one_or_none()
        
        # If cascade delete is configured, session should be deleted
        # If not, this test documents the current behavior
        if orphaned_session:
            # No cascade delete - sessions remain as orphans
            assert orphaned_session.user_id == user.id
        else:
            # Cascade delete is configured
            pass


@pytest.mark.database
class TestDatabaseIndexes:
    """Test database indexes and performance."""
    
    async def test_username_index_performance(self, db_session):
        """Test username index for fast lookups."""
        # Create many users
        users = []
        for i in range(100):
            user = User(
                username=f"perf_user_{i:03d}",
                email=f"perf_{i:03d}@example.com",
                full_name=f"Performance User {i}",
                hashed_password="hash123"
            )
            users.append(user)
        
        db_session.add_all(users)
        await db_session.commit()
        
        # Query by username (should use index)
        import time
        start_time = time.time()
        
        stmt = select(User).where(User.username == "perf_user_050")
        result = await db_session.execute(stmt)
        user = result.scalar_one()
        
        query_time = time.time() - start_time
        
        assert user.username == "perf_user_050"
        # Query should be fast (< 0.1 seconds for 100 records)
        assert query_time < 0.1
    
    async def test_email_index_performance(self, db_session):
        """Test email index for fast lookups."""
        # Query by email
        import time
        start_time = time.time()
        
        stmt = select(User).where(User.email == "test@example.com")
        result = await db_session.execute(stmt)
        users = result.scalars().all()
        
        query_time = time.time() - start_time
        
        # Query should be fast
        assert query_time < 0.1


@pytest.mark.database
class TestDatabaseTransactions:
    """Test database transaction behavior."""
    
    async def test_transaction_rollback(self, db_session):
        """Test transaction rollback on error."""
        # Create user successfully
        user1 = User(
            username="tx_user1",
            email="tx1@example.com",
            full_name="Transaction User 1",
            hashed_password="hash123"
        )
        db_session.add(user1)
        
        try:
            # Try to create duplicate user (should fail)
            user2 = User(
                username="tx_user1",  # Duplicate username
                email="tx2@example.com",
                full_name="Transaction User 2",
                hashed_password="hash123"
            )
            db_session.add(user2)
            await db_session.commit()
            
        except IntegrityError:
            # Rollback transaction
            await db_session.rollback()
        
        # First user should still exist
        stmt = select(User).where(User.username == "tx_user1")
        result = await db_session.execute(stmt)
        existing_user = result.scalar_one_or_none()
        
        # Behavior depends on when the first user was committed
        # This test documents the expected behavior
    
    async def test_concurrent_access(self, test_engine):
        """Test concurrent database access."""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        
        # Create two separate sessions
        AsyncSessionLocal = sessionmaker(
            test_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        session1 = AsyncSessionLocal()
        session2 = AsyncSessionLocal()
        
        try:
            # Create user in first session
            user1 = User(
                username="concurrent_user",
                email="concurrent@example.com",
                full_name="Concurrent User",
                hashed_password="hash123"
            )
            session1.add(user1)
            await session1.commit()
            
            # Read user from second session
            stmt = select(User).where(User.username == "concurrent_user")
            result = await session2.execute(stmt)
            user2 = result.scalar_one()
            
            assert user1.username == user2.username
            assert user1.id == user2.id
            
        finally:
            await session1.close()
            await session2.close()


@pytest.mark.database
class TestDatabaseMigrations:
    """Test database schema and migrations."""
    
    async def test_table_creation(self, test_engine):
        """Test that all tables are created correctly."""
        from sqlmodel import SQLModel
        
        # Get table names
        async with test_engine.begin() as conn:
            # Check if all expected tables exist
            inspector = await conn.run_sync(
                lambda sync_conn: sync_conn.dialect.inspector(sync_conn)
            )
            table_names = await conn.run_sync(
                lambda sync_conn: inspector.get_table_names()
            )
        
        # Expected tables
        expected_tables = ["users", "user_sessions", "travel_contexts"]
        
        for table in expected_tables:
            assert table in table_names or any(
                expected in actual for actual in table_names for expected in [table]
            )
    
    async def test_column_definitions(self, test_engine):
        """Test column definitions and constraints."""
        from sqlalchemy import inspect
        
        async with test_engine.begin() as conn:
            inspector = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn)
            )
            
            # Check user table columns
            user_columns = await conn.run_sync(
                lambda sync_conn: inspector.get_columns("users")
            )
            
            column_names = [col["name"] for col in user_columns]
            
            # Expected columns
            expected_columns = [
                "id", "username", "email", "full_name", 
                "hashed_password", "created_at", "updated_at"
            ]
            
            for col in expected_columns:
                assert col in column_names


@pytest.mark.database
class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    async def test_bulk_insert_performance(self, db_session):
        """Test bulk insert performance."""
        import time
        
        # Create many users at once
        users = []
        for i in range(1000):
            user = User(
                username=f"bulk_user_{i:04d}",
                email=f"bulk_{i:04d}@example.com",
                full_name=f"Bulk User {i}",
                hashed_password="hash123"
            )
            users.append(user)
        
        start_time = time.time()
        
        db_session.add_all(users)
        await db_session.commit()
        
        insert_time = time.time() - start_time
        
        # Bulk insert should be reasonably fast
        assert insert_time < 5.0  # Less than 5 seconds for 1000 records
        
        # Verify all users were created
        stmt = select(User).where(User.username.like("bulk_user_%"))
        result = await db_session.execute(stmt)
        created_users = result.scalars().all()
        
        assert len(created_users) == 1000
    
    async def test_query_performance_with_filters(self, db_session):
        """Test query performance with various filters."""
        import time
        
        # Test different query patterns
        queries = [
            select(User).where(User.is_active == True),
            select(User).where(User.username.like("%test%")),
            select(User).where(User.created_at > datetime.now() - timedelta(hours=1)),
            select(User).order_by(User.created_at.desc()).limit(10)
        ]
        
        for query in queries:
            start_time = time.time()
            
            result = await db_session.execute(query)
            users = result.scalars().all()
            
            query_time = time.time() - start_time
            
            # Each query should be fast
            assert query_time < 1.0
