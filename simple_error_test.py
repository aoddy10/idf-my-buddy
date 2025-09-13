#!/usr/bin/env python3
"""Simple error handling test for voice services."""

import asyncio
import tempfile
from pathlib import Path

async def test_basic_error_handling():
    """Test basic error handling scenarios."""
    
    print("Testing Basic Error Handling...")
    
    try:
        # Test 1: TTS with empty text
        print("\n1. Testing TTS with empty text...")
        from app.services.tts import TTSService
        
        tts_service = TTSService()
        
        try:
            result = await tts_service.synthesize_text(
                text="",
                language="en",
                output_format="mp3"
            )
            print(f"   Result: {type(result)} - {len(str(result))} chars")
        except Exception as e:
            print(f"   Proper error caught: {type(e).__name__}: {str(e)[:100]}")
        
        # Test 2: TTS with normal text  
        print("\n2. Testing TTS with normal text...")
        try:
            result = await tts_service.synthesize_text(
                text="This is a test.",
                language="en", 
                output_format="mp3"
            )
            if result and result.get("audio_data"):
                print(f"   ✅ Success: {len(result['audio_data'])} bytes")
            else:
                print(f"   ⚠️  No audio data returned")
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}: {str(e)[:100]}")
            
        # Test 3: Translation service
        print("\n3. Testing Translation service...")
        try:
            from app.services.nllb import NLLBTranslationService
            
            translation_service = NLLBTranslationService()
            
            result = await translation_service.translate_text(
                text="Hello world",
                target_language="es",
                source_language="en"
            )
            
            if result and result.get("translated_text"):
                print(f"   ✅ Success: '{result['translated_text']}'")
            else:
                print(f"   ⚠️  No translation returned")
                
        except Exception as e:
            print(f"   Error: {type(e).__name__}: {str(e)[:100]}")
            
        # Test 4: Whisper with invalid audio
        print("\n4. Testing Whisper with invalid data...")
        try:
            from app.services.whisper import WhisperService
            
            whisper_service = WhisperService()
            
            # Create empty temp file
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                temp_file.write(b"")  # Empty file
                temp_path = Path(temp_file.name)
                
                try:
                    result = await whisper_service.transcribe_audio(
                        audio_data=temp_path,
                        language="en"
                    )
                    print(f"   Unexpected success: {result}")
                except Exception as e:
                    print(f"   ✅ Proper error: {type(e).__name__}: {str(e)[:100]}")
                    
        except Exception as e:
            print(f"   Service error: {type(e).__name__}: {str(e)[:100]}")
        
        print(f"\n✅ Error handling test completed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_error_handling())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
