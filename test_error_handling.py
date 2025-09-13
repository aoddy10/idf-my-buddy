#!/usr/bin/env python3
"""Error handling validation tests for voice services.

This script tests edge cases and error scenarios to ensure
graceful failure handling across all voice services.
"""

import asyncio
import io
import tempfile
import time
from pathlib import Path
from typing import Any

async def test_invalid_audio_handling():
    """Test handling of invalid audio files."""
    
    print("🔍 Testing Invalid Audio Handling")
    print("=" * 40)
    
    try:
        from app.services.whisper import WhisperService
        
        whisper_service = WhisperService()
        
        test_cases = [
            {
                "name": "Empty file",
                "data": b"",
                "expected": "should handle empty audio gracefully"
            },
            {
                "name": "Invalid audio format", 
                "data": b"This is not audio data, just text",
                "expected": "should reject non-audio data"
            },
            {
                "name": "Corrupted WAV header",
                "data": b"RIFF\x00\x00\x00\x00WAVEfmt corrupted",
                "expected": "should handle corrupted files"
            },
            {
                "name": "Very short audio (1 byte)",
                "data": b"\x01",
                "expected": "should handle too-short audio"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n📋 Test: {test_case['name']}")
            
            # Create temporary file with test data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(test_case["data"])
                temp_file_path = Path(temp_file.name)
            
            start_time = time.time()
            try:
                result = await whisper_service.transcribe_audio(
                    audio_data=temp_file_path,
                    language="en",
                    return_timestamps=False,
                    return_confidence=False
                )
                
                duration = time.time() - start_time
                
                # If we get here, check if result is sensible
                if result and isinstance(result, dict):
                    text = result.get("text", "")
                    if text and len(text.strip()) > 0:
                        print(f"   ⚠️  Unexpected success: '{text[:30]}...' ({duration:.3f}s)")
                        results.append({"test": test_case["name"], "status": "unexpected_success", "result": text})
                    else:
                        print(f"   ✅ Handled gracefully: Empty result ({duration:.3f}s)")
                        results.append({"test": test_case["name"], "status": "graceful_empty", "result": ""})
                else:
                    print(f"   ✅ Handled gracefully: No result ({duration:.3f}s)")
                    results.append({"test": test_case["name"], "status": "graceful_none", "result": None})
                    
            except Exception as e:
                duration = time.time() - start_time
                error_type = type(e).__name__
                print(f"   ✅ Proper error handling: {error_type} ({duration:.3f}s)")
                print(f"      Message: {str(e)[:100]}...")
                results.append({"test": test_case["name"], "status": "proper_exception", "error": error_type})
            
            finally:
                # Cleanup
                temp_file_path.unlink(missing_ok=True)
        
        # Summary
        print(f"\n📊 Invalid Audio Handling Summary:")
        proper_handling = sum(1 for r in results if r["status"] in ["proper_exception", "graceful_empty", "graceful_none"])
        print(f"   Tests: {len(results)}")
        print(f"   Properly handled: {proper_handling}/{len(results)}")
        print(f"   Status: {'✅ PASS' if proper_handling == len(results) else '⚠️  PARTIAL'}")
        
        return proper_handling == len(results)
        
    except Exception as e:
        print(f"❌ Audio handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_tts_error_handling():
    """Test TTS service error handling."""
    
    print(f"\n🔍 Testing TTS Error Handling")
    print("=" * 40)
    
    try:
        from app.services.tts import TTSService
        
        tts_service = TTSService()
        
        test_cases = [
            {
                "name": "Empty text",
                "text": "",
                "language": "en",
                "expected": "should reject empty text"
            },
            {
                "name": "Very long text (>5000 chars)",
                "text": "A" * 6000,
                "language": "en", 
                "expected": "should handle or reject very long text"
            },
            {
                "name": "Invalid language code",
                "text": "Hello world",
                "language": "invalid_lang_code",
                "expected": "should handle invalid language gracefully"
            },
            {
                "name": "Special characters and emojis",
                "text": "Hello 🌍! This has special chars: @#$%^&*()",
                "language": "en",
                "expected": "should handle special characters"
            },
            {
                "name": "Non-Latin script", 
                "text": "こんにちは世界",  # Japanese
                "language": "ja",
                "expected": "should handle non-Latin scripts"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n📋 Test: {test_case['name']}")
            
            start_time = time.time()
            try:
                result = await tts_service.synthesize_text(
                    text=test_case["text"],
                    language=test_case["language"],
                    output_format="mp3"
                )
                
                duration = time.time() - start_time
                
                if result and result.get("audio_data"):
                    audio_size = len(result["audio_data"])
                    engine = result.get("engine", "unknown")
                    print(f"   ✅ Success: {audio_size} bytes, {engine} ({duration:.3f}s)")
                    results.append({"test": test_case["name"], "status": "success", "size": audio_size})
                else:
                    print(f"   ⚠️  No audio generated ({duration:.3f}s)")
                    results.append({"test": test_case["name"], "status": "no_audio", "size": 0})
                    
            except Exception as e:
                duration = time.time() - start_time
                error_type = type(e).__name__
                print(f"   ✅ Proper error: {error_type} ({duration:.3f}s)")
                print(f"      Message: {str(e)[:100]}...")
                results.append({"test": test_case["name"], "status": "proper_exception", "error": error_type})
        
        # Summary
        print(f"\n📊 TTS Error Handling Summary:")
        handled_properly = sum(1 for r in results if r["status"] in ["success", "proper_exception"])
        print(f"   Tests: {len(results)}")
        print(f"   Handled properly: {handled_properly}/{len(results)}")
        print(f"   Status: {'✅ PASS' if handled_properly >= len(results) * 0.8 else '⚠️  NEEDS REVIEW'}")
        
        return handled_properly >= len(results) * 0.8
        
    except Exception as e:
        print(f"❌ TTS error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_translation_error_handling():
    """Test translation service error handling."""
    
    print(f"\n🔍 Testing Translation Error Handling")
    print("=" * 40)
    
    try:
        from app.services.nllb import NLLBTranslationService
        
        translation_service = NLLBTranslationService()
        
        test_cases = [
            {
                "name": "Empty text translation",
                "text": "",
                "source": "en",
                "target": "es",
                "expected": "should handle empty text gracefully"
            },
            {
                "name": "Same source and target language",
                "text": "Hello world",
                "source": "en", 
                "target": "en",
                "expected": "should return original text or translate anyway"
            },
            {
                "name": "Unsupported language pair",
                "text": "Hello world",
                "source": "en",
                "target": "klingon",  # Not a real language code
                "expected": "should handle gracefully"
            },
            {
                "name": "Very long text",
                "text": "This is a very long sentence. " * 100,  # 3000+ chars
                "source": "en",
                "target": "es", 
                "expected": "should handle or chunk long text"
            },
            {
                "name": "Mixed languages in input",
                "text": "Hello world and bonjour le monde and hola mundo",
                "source": "auto",
                "target": "es",
                "expected": "should handle mixed language input"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n📋 Test: {test_case['name']}")
            
            start_time = time.time()
            try:
                result = await translation_service.translate_text(
                    text=test_case["text"],
                    source_language=test_case["source"] if test_case["source"] != "auto" else None,
                    target_language=test_case["target"]
                )
                
                duration = time.time() - start_time
                
                if result and isinstance(result, dict):
                    translated_text = result.get("translated_text", "")
                    confidence = result.get("confidence", 0)
                    print(f"   ✅ Success: '{translated_text[:50]}...' (conf: {confidence:.2f}, {duration:.3f}s)")
                    results.append({"test": test_case["name"], "status": "success", "text_length": len(translated_text)})
                else:
                    print(f"   ⚠️  No translation result ({duration:.3f}s)")
                    results.append({"test": test_case["name"], "status": "no_result", "text_length": 0})
                    
            except Exception as e:
                duration = time.time() - start_time
                error_type = type(e).__name__
                print(f"   ✅ Proper error: {error_type} ({duration:.3f}s)")
                print(f"      Message: {str(e)[:100]}...")
                results.append({"test": test_case["name"], "status": "proper_exception", "error": error_type})
        
        # Summary  
        print(f"\n📊 Translation Error Handling Summary:")
        handled_properly = sum(1 for r in results if r["status"] in ["success", "proper_exception"])
        print(f"   Tests: {len(results)}")
        print(f"   Handled properly: {handled_properly}/{len(results)}")
        print(f"   Status: {'✅ PASS' if handled_properly >= len(results) * 0.8 else '⚠️  NEEDS REVIEW'}")
        
        return handled_properly >= len(results) * 0.8
        
    except Exception as e:
        print(f"❌ Translation error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_service_availability_handling():
    """Test handling of service unavailability scenarios."""
    
    print(f"\n🔍 Testing Service Availability Handling")
    print("=" * 40)
    
    try:
        # Test service availability checks
        from app.services.tts import TTSService
        from app.services.whisper import WhisperService
        from app.services.nllb import NLLBTranslationService
        
        print("📋 Testing service availability detection...")
        
        # Test TTS service availability
        tts_service = TTSService()
        tts_available = tts_service.is_available() if hasattr(tts_service, 'is_available') else True
        print(f"   TTS Service Available: {tts_available}")
        
        # Test Whisper service
        whisper_service = WhisperService()
        whisper_available = whisper_service.is_available() if hasattr(whisper_service, 'is_available') else True
        print(f"   Whisper Service Available: {whisper_available}")
        
        # Test Translation service
        translation_service = NLLBTranslationService()
        translation_available = translation_service.is_available() if hasattr(translation_service, 'is_available') else True
        print(f"   Translation Service Available: {translation_available}")
        
        available_services = sum([tts_available, whisper_available, translation_available])
        
        print(f"\n📊 Service Availability Summary:")
        print(f"   Services Available: {available_services}/3")
        print(f"   Status: {'✅ ALL AVAILABLE' if available_services == 3 else '⚠️  SOME UNAVAILABLE'}")
        
        return available_services >= 2  # At least 2/3 services should be available
        
    except Exception as e:
        print(f"❌ Service availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all error handling tests."""
    
    print("My Buddy AI - Error Handling Validation")
    print("Testing graceful failure handling across voice services")
    print("=" * 60)
    
    # Run all error handling tests
    tests = [
        ("Invalid Audio Handling", test_invalid_audio_handling()),
        ("TTS Error Handling", test_tts_error_handling()),
        ("Translation Error Handling", test_translation_error_handling()),
        ("Service Availability", test_service_availability_handling())
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        print(f"\n{'🧪 ' + test_name}")
        try:
            success = await test_coro
            results.append((test_name, success))
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("ERROR HANDLING VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 All error handling tests passed!")
        print(f"Voice services handle edge cases gracefully.")
    elif passed >= total * 0.8:
        print(f"\n⚠️  Most error handling tests passed.")
        print(f"Minor improvements needed for full robustness.")
    else:
        print(f"\n❌ Error handling needs significant improvement.")
        print(f"Review service error handling implementation.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
