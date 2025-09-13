#!/usr/bin/env python3
"""Direct test of TTS service without server."""

import asyncio

async def test_tts_service():
    """Test TTS service directly."""
    try:
        # Import TTS service
        from app.services.tts import TTSService
        
        print("Creating TTS service...")
        tts_service = TTSService()
        
        print("Testing text synthesis...")
        result = await tts_service.synthesize_text(
            text="Hello! This is a direct test of TTS.",
            language="en",
            output_format="mp3"
        )
        
        print(f"TTS Result: {result.keys()}")
        
        if result.get("audio_data"):
            with open("direct_tts_test.mp3", "wb") as f:
                f.write(result["audio_data"])
            print(f"✅ Audio saved: direct_tts_test.mp3 ({len(result['audio_data'])} bytes)")
        else:
            print("❌ No audio data in result")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tts_service())
    print(f"\nDirect TTS test {'passed' if success else 'failed'}")
