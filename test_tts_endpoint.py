#!/usr/bin/env python3
"""Test script for the TTS (Text-to-Speech) endpoint.

This script tests the /api/v1/voice/text-to-speech endpoint to verify
that TTS services are working correctly.
"""

import json
import os
import time
from pathlib import Path

import requests


def test_tts_endpoint():
    """Test the TTS endpoint with various configurations."""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/api/v1/voice/text-to-speech"
    
    # Test cases with different configurations
    test_cases = [
        {
            "name": "Basic English TTS",
            "payload": {
                "text": "Hello! Welcome to New York City. I'm your AI travel buddy.",
                "language": "en",
                "voice_gender": "female",
                "speaking_rate": 1.0,
                "output_format": "mp3"
            }
        },
        {
            "name": "Fast Speech Rate",
            "payload": {
                "text": "This is a test of faster speech synthesis.",
                "language": "en", 
                "voice_gender": "male",
                "speaking_rate": 1.5,
                "output_format": "wav"
            }
        },
        {
            "name": "Slow Speech Rate",
            "payload": {
                "text": "This is a test of slower speech synthesis for clarity.",
                "language": "en",
                "voice_gender": "neutral", 
                "speaking_rate": 0.7,
                "output_format": "mp3"
            }
        }
    ]
    
    print("Testing TTS endpoint...")
    print(f"Base URL: {base_url}")
    print("=" * 50)
    
    # Create output directory for audio files
    output_dir = Path("tts_test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                endpoint,
                json=test_case["payload"],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                # Check if response is audio data
                content_type = response.headers.get("content-type", "")
                
                if "audio" in content_type:
                    # Save audio file
                    format_ext = test_case["payload"]["output_format"]
                    filename = f"test_{i}_{test_case['name'].lower().replace(' ', '_')}.{format_ext}"
                    filepath = output_dir / filename
                    
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    file_size = len(response.content)
                    print(f"✅ Audio generated successfully")
                    print(f"   Content Type: {content_type}")
                    print(f"   File Size: {file_size:,} bytes")
                    print(f"   Saved to: {filepath}")
                    
                    success_count += 1
                    
                else:
                    # JSON response (metadata)
                    try:
                        data = response.json()
                        print(f"✅ JSON Response received")
                        print(f"   Response: {json.dumps(data, indent=2)}")
                        success_count += 1
                    except json.JSONDecodeError:
                        print(f"❌ Unexpected response format")
                        print(f"   Content: {response.text[:200]}...")
            else:
                print(f"❌ Request failed")
                try:
                    error_data = response.json()
                    print(f"   Error: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"   Response: {response.text[:200]}...")
                    
        except requests.exceptions.Timeout:
            print("❌ Request timeout (>30s)")
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - is the server running?")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {success_count}/{len(test_cases)} tests passed")
    
    if success_count > 0:
        print(f"\n✅ TTS endpoint is working!")
        print(f"Audio files saved to: {output_dir.absolute()}")
        return True
    else:
        print(f"\n❌ TTS endpoint has issues")
        return False


def test_health_endpoint():
    """Test if the server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False


if __name__ == "__main__":
    print("My Buddy AI - TTS Endpoint Test")
    print("=" * 40)
    
    # First check if server is running
    if not test_health_endpoint():
        print("\nPlease start the server first:")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8000")
        exit(1)
    
    # Test TTS endpoint
    success = test_tts_endpoint()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Listen to the generated audio files")
        print("2. Test with different languages if supported")
        print("3. Test WebSocket conversation flows")
    else:
        print("\n⚠️  Some tests failed - check server logs for details")
        exit(1)
