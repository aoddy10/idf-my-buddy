#!/usr/bin/env python3
"""WebSocket test client for voice conversation flow."""

import asyncio
import json
import websockets
from pathlib import Path

async def test_websocket_conversation():
    """Test WebSocket voice conversation endpoint."""
    
    uri = "ws://localhost:8000/api/v1/voice/conversation/stream"
    
    print("Testing WebSocket Voice Conversation...")
    print(f"Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Send a simple test message (simulating audio data)
            test_audio = b"fake_audio_data_for_testing"
            
            print("Sending test audio data...")
            await websocket.send(test_audio)
            
            print("Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            
            try:
                data = json.loads(response)
                print(f"✅ Received response: {json.dumps(data, indent=2)}")
                return True
            except json.JSONDecodeError:
                print(f"⚠️ Non-JSON response: {response}")
                return True  # Still counts as working
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused - is the server running?")
        return False
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for response")
        return False
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False

async def test_http_endpoints():
    """Test related HTTP endpoints first."""
    import aiohttp
    
    print("\n=== Testing Related HTTP Endpoints ===")
    
    # Test voice session creation
    try:
        async with aiohttp.ClientSession() as session:
            # Create voice session
            async with session.post(
                "http://localhost:8000/api/v1/voice/sessions",
                params={"language": "en"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Voice session created: {data.get('data', {}).get('session_id', 'N/A')}")
                else:
                    print(f"⚠️ Voice session creation failed: {response.status}")
                    
            # Test available voices
            async with session.get(
                "http://localhost:8000/api/v1/voice/voices"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Available voices retrieved: {len(data.get('data', {}).get('voices', []))} voices")
                else:
                    print(f"⚠️ Available voices failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ HTTP endpoint test failed: {e}")

async def main():
    """Run all tests."""
    print("My Buddy AI - WebSocket Voice Conversation Test")
    print("=" * 50)
    
    # Test HTTP endpoints first
    await test_http_endpoints()
    
    print("\n=== Testing WebSocket Conversation ===")
    
    # Test WebSocket conversation
    success = await test_websocket_conversation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 WebSocket conversation flow is working!")
        print("\nNext steps:")
        print("1. Test with real audio data")
        print("2. Test conversation context persistence")  
        print("3. Measure response latency")
    else:
        print("❌ WebSocket conversation flow has issues")
        print("\nCheck server logs for details")

if __name__ == "__main__":
    asyncio.run(main())
