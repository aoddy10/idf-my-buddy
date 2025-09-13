#!/usr/bin/env python3
"""Simple multilingual test for debugging."""

import asyncio
import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def simple_translation_test():
    """Simple test to check if NLLB service works."""
    
    print("🧪 Simple NLLB Translation Test")
    print("=" * 40)
    
    try:
        from app.services.nllb import NLLBTranslationService
        
        print("1. Creating translation service...")
        service = NLLBTranslationService()
        print("   ✅ Service created successfully")
        
        print("\n2. Testing simple translation (EN → ES)...")
        result = await service.translate_text(
            text="Hello, how are you?",
            source_language="en", 
            target_language="es"
        )
        
        if result:
            print(f"   Original: Hello, how are you?")
            print(f"   Translated: {result.get('translated_text', 'No translation')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print("   ✅ Translation successful!")
            return True
        else:
            print("   ❌ No translation result")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_translation_test())
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
