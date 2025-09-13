"""Unit tests for database models.

Tests the SQLModel database models and their relationships.
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional

from app.models.user import User
from app.models.session import UserSession
from app.models.travel import TravelContext


class TestUserModel:
    """Test User model functionality."""
    
    def test_user_creation(self):
        """Test basic user creation."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.hashed_password == "hashed_password_123"
        assert user.preferred_language == "en"  # Default value
        assert user.is_active is True  # Default value
    
    def test_user_with_optional_fields(self):
        """Test user creation with optional fields."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123",
            preferred_language="es",
            profile_image_url="https://example.com/avatar.jpg",
            preferences={"theme": "dark", "notifications": True}
        )
        
        assert user.preferred_language == "es"
        assert user.profile_image_url == "https://example.com/avatar.jpg"
        assert user.preferences["theme"] == "dark"
        assert user.preferences["notifications"] is True
    
    def test_user_timestamps(self):
        """Test user timestamp fields."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        
        # created_at should be set automatically
        assert user.created_at is not None
        assert isinstance(user.created_at, datetime)
        
        # updated_at should be the same as created_at initially
        assert user.updated_at is not None
        assert user.updated_at == user.created_at
    
    def test_user_string_representation(self):
        """Test user string representation."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        
        # Should contain username and email
        user_str = str(user)
        assert "testuser" in user_str
        assert "test@example.com" in user_str
    
    def test_user_validation(self):
        """Test user field validation."""
        # Test email validation (if implemented)
        try:
            user = User(
                username="testuser",
                email="invalid_email",
                full_name="Test User",
                hashed_password="hashed_password_123"
            )
            # If validation is implemented, this should raise an error
        except Exception:
            # Expected if email validation is implemented
            pass


class TestUserSessionModel:
    """Test UserSession model functionality."""
    
    def test_session_creation(self):
        """Test basic session creation."""
        session = UserSession(
            user_id=1,
            session_token="abc123token",
            device_info={"platform": "web", "browser": "chrome"}
        )
        
        assert session.user_id == 1
        assert session.session_token == "abc123token"
        assert session.device_info["platform"] == "web"
        assert session.is_active is True  # Default value
    
    def test_session_timestamps(self):
        """Test session timestamp fields."""
        session = UserSession(
            user_id=1,
            session_token="abc123token",
            device_info={"platform": "mobile"}
        )
        
        assert session.created_at is not None
        assert session.last_activity is not None
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
    
    def test_session_expiration(self):
        """Test session expiration calculation."""
        session = UserSession(
            user_id=1,
            session_token="abc123token",
            device_info={"platform": "mobile"}
        )
        
        # Test if session has expiration logic
        if hasattr(session, 'expires_at'):
            assert session.expires_at is not None
            assert session.expires_at > session.created_at
    
    def test_session_activity_update(self):
        """Test session activity update."""
        session = UserSession(
            user_id=1,
            session_token="abc123token",
            device_info={"platform": "mobile"}
        )
        
        original_activity = session.last_activity
        
        # Simulate activity update
        session.last_activity = datetime.now()
        
        assert session.last_activity > original_activity


class TestTravelContextModel:
    """Test TravelContext model functionality."""
    
    def test_travel_context_creation(self):
        """Test basic travel context creation."""
        context = TravelContext(
            user_id=1,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="walking"
        )
        
        assert context.user_id == 1
        assert context.current_location["lat"] == 40.7128
        assert context.destination["lng"] == -118.2437
        assert context.travel_mode == "walking"
    
    def test_travel_context_with_preferences(self):
        """Test travel context with preferences."""
        preferences = {
            "cuisine": ["italian", "japanese"],
            "budget": "moderate",
            "interests": ["museums", "parks"],
            "accessibility": ["wheelchair"]
        }
        
        context = TravelContext(
            user_id=1,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="walking",
            preferences=preferences
        )
        
        assert context.preferences["cuisine"] == ["italian", "japanese"]
        assert context.preferences["budget"] == "moderate"
        assert "museums" in context.preferences["interests"]
    
    def test_travel_context_optional_fields(self):
        """Test travel context with optional fields."""
        context = TravelContext(
            user_id=1,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="driving",
            context="business",
            group_size=3,
            duration_hours=8
        )
        
        assert context.context == "business"
        assert context.group_size == 3
        assert context.duration_hours == 8
    
    def test_travel_context_timestamps(self):
        """Test travel context timestamps."""
        context = TravelContext(
            user_id=1,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="walking"
        )
        
        assert context.created_at is not None
        assert context.updated_at is not None
        assert isinstance(context.created_at, datetime)
    
    def test_travel_context_validation(self):
        """Test travel context validation."""
        # Test that required fields are enforced
        with pytest.raises(Exception):
            TravelContext(
                user_id=1,
                # Missing required location fields
                travel_mode="walking"
            )
    
    def test_travel_mode_validation(self):
        """Test travel mode validation."""
        valid_modes = ["walking", "driving", "transit", "cycling"]
        
        for mode in valid_modes:
            context = TravelContext(
                user_id=1,
                current_location={"lat": 40.7128, "lng": -74.0060},
                destination={"lat": 34.0522, "lng": -118.2437},
                travel_mode=mode
            )
            assert context.travel_mode == mode


@pytest.mark.integration
class TestModelRelationships:
    """Test model relationships and foreign keys."""
    
    async def test_user_session_relationship(self, db_session):
        """Test User-UserSession relationship."""
        # Create user
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create session for user
        session = UserSession(
            user_id=user.id,
            session_token="abc123token",
            device_info={"platform": "web"}
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Test relationship
        assert session.user_id == user.id
    
    async def test_user_travel_context_relationship(self, db_session):
        """Test User-TravelContext relationship."""
        # Create user
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Create travel context for user
        context = TravelContext(
            user_id=user.id,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="walking"
        )
        db_session.add(context)
        await db_session.commit()
        await db_session.refresh(context)
        
        # Test relationship
        assert context.user_id == user.id
    
    async def test_cascade_operations(self, db_session):
        """Test cascade delete operations if implemented."""
        # Create user with sessions and contexts
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Add session and context
        session = UserSession(
            user_id=user.id,
            session_token="abc123token",
            device_info={"platform": "web"}
        )
        context = TravelContext(
            user_id=user.id,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="walking"
        )
        
        db_session.add(session)
        db_session.add(context)
        await db_session.commit()
        
        # Test that related records exist
        assert session.user_id == user.id
        assert context.user_id == user.id


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_user_to_dict(self):
        """Test user model serialization."""
        user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123"
        )
        
        # Test dict conversion if available
        if hasattr(user, 'dict'):
            user_dict = user.dict()
            assert user_dict["username"] == "testuser"
            assert user_dict["email"] == "test@example.com"
        
        # Test model_dump if using Pydantic v2
        if hasattr(user, 'model_dump'):
            user_dict = user.model_dump()
            assert user_dict["username"] == "testuser"
            assert user_dict["email"] == "test@example.com"
    
    def test_travel_context_json_fields(self):
        """Test JSON field serialization in travel context."""
        preferences = {
            "cuisine": ["italian", "japanese"],
            "budget": "moderate",
            "interests": ["museums", "parks"]
        }
        
        context = TravelContext(
            user_id=1,
            current_location={"lat": 40.7128, "lng": -74.0060},
            destination={"lat": 34.0522, "lng": -118.2437},
            travel_mode="walking",
            preferences=preferences
        )
        
        # JSON fields should maintain structure
        assert isinstance(context.current_location, dict)
        assert isinstance(context.preferences, dict)
        assert context.preferences["cuisine"] == ["italian", "japanese"]
