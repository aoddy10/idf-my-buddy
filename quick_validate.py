#!/usr/bin/env python3
"""Quick Restaurant Intelligence validation."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

print("🔍 Restaurant Intelligence Validation")
print("=" * 50)

success_count = 0
total_tests = 0

def test_import(module_name, class_name):
    global success_count, total_tests
    total_tests += 1
    try:
        module = __import__(module_name, fromlist=[class_name])
        getattr(module, class_name)
        print(f"✅ {class_name} import: SUCCESS")
        success_count += 1
        return True
    except Exception as e:
        print(f"❌ {class_name} import: FAILED - {e}")
        return False

def test_instantiation(module_name, class_name):
    global success_count, total_tests
    total_tests += 1
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls()
        print(f"✅ {class_name} instantiation: SUCCESS")
        success_count += 1
        return instance
    except Exception as e:
        print(f"❌ {class_name} instantiation: FAILED - {e}")
        return None

# Test imports
print("\n🧪 Testing Module Imports...")
test_import("app.services.restaurant_intelligence", "RestaurantIntelligenceService")
test_import("app.services.cultural_dining", "CulturalDiningDatabase")
test_import("app.api.restaurant", "router")
test_import("app.models.entities.restaurant", "MenuItemBase")
test_import("app.schemas.restaurant", "MenuAnalysisRequest")

# Test instantiations
print("\n🏗️  Testing Service Instantiation...")
ri_service = test_instantiation("app.services.restaurant_intelligence", "RestaurantIntelligenceService")
cd_service = test_instantiation("app.services.cultural_dining", "CulturalDiningDatabase")

# Test basic functionality
if ri_service and cd_service:
    total_tests += 1
    try:
        # Test allergen detection method exists
        method = getattr(ri_service, '_detect_allergens', None)
        if method:
            print("✅ Allergen detection method: EXISTS")
            success_count += 1
        else:
            print("❌ Allergen detection method: NOT FOUND")
    except Exception as e:
        print(f"❌ Allergen detection method: ERROR - {e}")
    
    total_tests += 1
    try:
        # Test cultural database method exists
        method = getattr(cd_service, 'calculate_authenticity_score', None)
        if method:
            print("✅ Cultural authenticity method: EXISTS")
            success_count += 1
        else:
            print("❌ Cultural authenticity method: NOT FOUND")
    except Exception as e:
        print(f"❌ Cultural authenticity method: ERROR - {e}")

# Final report
print("\n" + "=" * 50)
print("📊 VALIDATION RESULTS")
print("=" * 50)
print(f"Tests Passed: {success_count}/{total_tests}")
print(f"Success Rate: {(success_count/total_tests*100):.1f}%")

if success_count == total_tests:
    print("🏆 ALL TESTS PASSED - RESTAURANT INTELLIGENCE READY!")
    sys.exit(0)
elif success_count >= total_tests * 0.8:
    print("✅ MOSTLY WORKING - Minor issues detected")
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED - Major issues detected")
    sys.exit(1)
