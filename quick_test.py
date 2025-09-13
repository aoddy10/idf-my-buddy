import requests
import json

# Test health first
try:
    print("Testing health endpoint...")
    health = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Health: {health.status_code} - {health.text}")
except Exception as e:
    print(f"Health error: {e}")

# Test TTS
try:
    print("\nTesting TTS endpoint...")
    tts_data = {
        "text": "Hello world",
        "language": "en", 
        "voice_gender": "female",
        "speaking_rate": 1.0,
        "output_format": "mp3"
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/voice/text-to-speech",
        json=tts_data,
        timeout=30
    )
    
    print(f"TTS: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        if "audio" in response.headers.get("content-type", ""):
            with open("python_tts_test.mp3", "wb") as f:
                f.write(response.content)
            print(f"Audio saved: {len(response.content)} bytes")
        else:
            print(f"Response: {response.text}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"TTS error: {e}")
