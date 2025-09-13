import asyncio
import websockets
import json

async def simple_ws_test():
    try:
        uri = "ws://localhost:8000/api/v1/voice/conversation/stream"
        print(f"Connecting to {uri}")
        
        async with websockets.connect(uri) as ws:
            print("Connected!")
            
            # Send test data
            await ws.send(b"test")
            print("Sent test data")
            
            # Wait for response 
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(simple_ws_test())
