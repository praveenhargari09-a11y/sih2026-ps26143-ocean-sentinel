from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
from backend.api.pipeline_bus import bus

router = APIRouter()

@router.websocket("/{spill_id}")
async def pipeline_websocket(websocket: WebSocket, spill_id: str):
    await websocket.accept()
    
    queue = bus.subscribe(spill_id)
    try:
        while True:
            # Wait for real events from the detect endpoint
            event = await queue.get()
            await websocket.send_text(json.dumps(event))
            
            # If pipeline is complete or error, we can stay connected
            # but we just wait for more events (or heartbeat could go here)
            if event.get("stage") in ("complete", "error"):
                # We optionally just loop, waiting for potential resets
                pass
                
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        bus.unsubscribe(spill_id)
