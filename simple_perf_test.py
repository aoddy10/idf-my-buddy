#!/usr/bin/env python3
"""Simple performance test."""

import time
import asyncio

async def test_tts_performance():
    """Quick TTS performance test."""
    
    try:
        print("Testing TTS Performance...")
        
        # Import TTS service
        from app.services.tts import TTSService
        
        # Initialize service
        print("Initializing TTS service...")
        tts_service = TTSService() 
        
        # Test single synthesis
        text = "Hello, this is a performance test for TTS synthesis."
        
        print(f"Synthesizing: '{text}'")
        
        start_time = time.time()
        
        result = await tts_service.synthesize_text(
            text=text,
            language="en",
            output_format="mp3"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"TTS Duration: {duration:.3f}s")
        
        if result.get("audio_data"):
            audio_size = len(result["audio_data"])
            print(f"Audio size: {audio_size} bytes")
            print(f"Engine: {result.get('engine', 'unknown')}")
            
            if duration < 2.0:
                print("✅ FAST - under 2s target")
            elif duration < 5.0:
                print("⚠️  ACCEPTABLE - under 5s")  
            else:
                print("❌ SLOW - over 5s")
        else:
            print("❌ No audio generated")
            
        return duration < 5.0
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tts_performance())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
