#!/usr/bin/env python3
"""Direct NLLB service check."""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def direct_nllb_test():
    """Test NLLB service initialization directly."""
    
    print("🧪 Direct NLLB Service Test")
    print("=" * 30)
    
    try:
        print("1. Importing service...")
        from app.services.nllb import NLLBTranslationService
        print("   ✅ Import successful")
        
        print("2. Creating service instance...")
        service = NLLBTranslationService()
        print("   ✅ Service created")
        
        print("3. Checking service attributes...")
        print(f"   Local model: {service._local_model is not None}")
        print(f"   Google translator: {service._google_translator is not None}")
        print(f"   Language codes: {len(service._nllb_codes)} languages")
        
        print("4. Sample language codes:")
        sample_codes = list(service._nllb_codes.items())[:5]
        for lang, code in sample_codes:
            print(f"   {lang} -> {code}")
        
        print("\n✅ Service initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = direct_nllb_test()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
