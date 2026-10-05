import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import process_manager, ros_bridge

router = APIRouter()


@router.get('/status')
def get_status():
    bridge = ros_bridge.get_bridge()
    return {
        'ros': bridge.get_status(),
        'processes': process_manager.manager.all_status(),
    }


@router.websocket('/ws/status')
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    bridge = ros_bridge.get_bridge()
    cursor = bridge.current_event_seq()      # never replay events from before this connection
    try:
        while True:
            # all_status() walks every process on the machine (~600 here) --
            # running it inline on this coroutine would block the WHOLE
            # asyncio event loop (every other connection: MJPEG streams, the
            # fusion WS, REST requests) for its full duration, every 0.5s,
            # forever. That stall compounding over hours is what was behind
            # the backend going fully unresponsive after sitting idle
            # overnight. to_thread moves it off the event loop thread.
            processes = await asyncio.to_thread(process_manager.manager.all_status)
            events = bridge.get_events_since(cursor)
            if events:
                cursor = events[-1]['seq']
            await websocket.send_json({
                'ros': bridge.get_status(),
                'processes': processes,
                'events': events,
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
