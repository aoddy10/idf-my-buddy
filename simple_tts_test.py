#!/usr/bin/env python3
"""Simple TTS test using Python requests."""

import requests
import time

def test_simple_tts():
    """Test TTS endpoint with a basic request."""
    
    url = "http://localhost:8000/api/v1/voice/text-to-speech"
    
    payload = {
        "text": "Hello, this is a test!",
        "language": "en",
        "voice_gender": "female",
        "speaking_rate": 1.0,
        "output_format": "mp3"
    }
    
    print("Testing TTS endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {payload}")
    
    try:
        # Test health first
        print("\n1. Testing health endpoint...")
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health response: {health_response.json()}")
        
        # Test TTS
        print("\n2. Testing TTS endpoint...")
        start_time = time.time()
        response = requests.post(
            url,
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        print(f"Status code: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f}s")
        print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
        print(f"Content-Length: {response.headers.get('content-length', 'N/A')}")
        
        if response.status_code == 200:
            if 'audio' in response.headers.get('content-type', ''):
                # Audio response
                with open("simple_tts_test.mp3", "wb") as f:
                    f.write(response.content)
                print(f"✅ Audio file saved: simple_tts_test.mp3 ({len(response.content)} bytes)")
                return True
            else:
                # JSON response
                print(f"✅ JSON Response: {response.json()}")
                return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_tts()
    if success:
        print("\n🎉 TTS test completed successfully!")
    else:
        print("\n❌ TTS test failed")
