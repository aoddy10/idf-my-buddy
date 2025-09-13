#!/usr/bin/env python3
"""Validation script for My Buddy foundation setup."""

import sys
import asyncio
from fastapi.testclient import TestClient

def test_app_creation():
    """Test that the FastAPI app can be created successfully."""
    try:
        from app.main import create_app
        app = create_app()
        print(f"✅ App created successfully: {app.title}")
        return True
    except Exception as e:
        print(f"❌ Failed to create app: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint."""
    try:
        from app.main import create_app
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health/")
        print(f"✅ Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health response: {data}")
            return True
        else:
            print(f"❌ Health endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from app.core.config import get_settings
        settings = get_settings()
        print(f"✅ Config loaded: app_env={settings.app_env}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_imports():
    """Test critical imports."""
    tests = [
        ("FastAPI main", "from app.main import create_app"),
        ("Config", "from app.core.config import settings"),
        ("Health API", "from app.api.health import router"),
        ("User model", "from app.models.entities.user import User"),
        ("Common schemas", "from app.schemas.common import Location"),
    ]
    
    success_count = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name} import successful")
            success_count += 1
        except Exception as e:
            print(f"❌ {name} import failed: {e}")
    
    return success_count == len(tests)

def main():
    """Run all validation tests."""
    print("🔍 Running My Buddy Foundation Validation\n")
    
    tests = [
        ("Critical Imports", test_imports),
        ("Configuration", test_config),
        ("App Creation", test_app_creation),
        ("Health Endpoint", test_health_endpoint),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Testing {name}:")
        success = test_func()
        results.append(success)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All foundation tests passed! The My Buddy setup is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
